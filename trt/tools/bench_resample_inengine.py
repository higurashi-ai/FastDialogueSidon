#!/usr/bin/env python3
"""Decide: resample inside the encoder engine, or outside on GPU torch?

Builds a 48 kHz-input variant of the wave encoder engine (sinc-conv front)
and compares, per 120 s window:
  A. external: torchaudio GPU resample (48k->16k) + standard 16k engine
  B. in-engine: 48k engine (resample conv baked in)
plus parity (B vs A conditioning SNR) and the 16 kHz skip check
(load_waveform must not touch the resampler for 16 kHz input).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import wave as wavmod
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastdialoguesidon import flexrel_triton, runtime  # noqa: E402
from fastdialoguesidon.engines import (  # noqa: E402
    TRTStage,
    build_wave_engine,
    deserialize,
    errors,
    timed_cuda,
)
from fastdialoguesidon.export_onnx import export_wave_encoder  # noqa: E402
from fastdialoguesidon.modeling import load_bundle  # noqa: E402

SAMPLES = runtime.SAMPLES  # 16 kHz samples / 120 s
SAMPLES48 = SAMPLES * 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True,
                        help="dir with the standard 16 kHz engines")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)

    import torchaudio.functional as AF

    torch.manual_seed(0)
    device = torch.device("cuda")
    flexrel_triton.register_trt_plugin_qr_dyn()
    bundle = load_bundle(args.bundle, device="cuda")

    # 48 kHz test signal (upsampled band-limited noise)
    wave16 = torch.randn(SAMPLES, device=device) * 0.1
    wave48 = AF.resample(wave16.unsqueeze(0), 16_000, 48_000)[0].contiguous()
    assert wave48.numel() == SAMPLES48

    # build the 48k-input variant engine
    onnx48 = args.scratch / "encoder_wave48_fp32.onnx"
    export_wave_encoder(
        copy.deepcopy(bundle.encoder),
        wave48.unsqueeze(0),
        onnx48,
        input_rate=48_000,
    )
    engine48, build_s = build_wave_engine(
        onnx48,
        {"waveform": ((1, 960 * 160), (1, SAMPLES48), (1, SAMPLES48))},
        args.scratch / "encoder_wave48.plan",
    )
    print(f"48k engine built in {build_s:.1f}s", flush=True)
    engine16 = deserialize(args.engine_dir / runtime.ENGINE_FILES["encoder"])

    wave16_buffer = torch.empty(1, SAMPLES, device=device)
    stage16 = TRTStage(engine16, {"waveform": wave16_buffer})
    cond16 = next(iter(stage16.outputs.values()))
    wave48_buffer = wave48.unsqueeze(0).contiguous().clone()
    stage48 = TRTStage(engine48, {"waveform": wave48_buffer})
    cond48 = next(iter(stage48.outputs.values()))

    result = {"gpu": torch.cuda.get_device_name(0), "build_seconds": build_s}
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        # A. external: GPU resample + copy into engine buffer + enqueue
        def external():
            resampled = AF.resample(
                wave48.unsqueeze(0), 48_000, 16_000
            )[:, :SAMPLES]
            wave16_buffer.copy_(resampled)
            stage16.enqueue(torch.cuda.current_stream())

        external()
        torch.cuda.synchronize()
        cond_external = cond16.float().clone()
        result["external_resample_plus_engine"] = timed_cuda(
            external, args.warmup, args.runs
        )
        result["resample_gpu_alone"] = timed_cuda(
            lambda: AF.resample(wave48.unsqueeze(0), 48_000, 16_000),
            args.warmup,
            args.runs,
        )
        result["engine16_alone"] = timed_cuda(
            lambda: stage16.enqueue(torch.cuda.current_stream()),
            args.warmup,
            args.runs,
        )

        # B. in-engine
        def inengine():
            stage48.enqueue(torch.cuda.current_stream())

        inengine()
        torch.cuda.synchronize()
        result["inengine_48k_engine"] = timed_cuda(
            inengine, args.warmup, args.runs
        )
    torch.cuda.synchronize()
    result["parity_inengine_vs_external"] = errors(cond_external, cond48)
    result["delta_ms"] = (
        result["external_resample_plus_engine"]["median_ms"]
        - result["inengine_48k_engine"]["median_ms"]
    )

    # 16 kHz skip check: load_waveform must not call the resampler
    skip_wav = args.scratch / "skip16k.wav"
    data16 = (wave16.clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
    with wavmod.open(str(skip_wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(data16.tobytes())
    original = AF.resample

    def _boom(*_a, **_k):  # noqa: ANN002, ANN003
        raise AssertionError("resample called for 16 kHz input")

    AF.resample = _boom
    try:
        loaded = runtime.load_waveform(skip_wav)
    finally:
        AF.resample = original
    max_dev = (
        (loaded - torch.from_numpy(data16.astype(np.float32) / 32768.0)
         .to(device)).abs().max().item()
    )
    result["skip_16k_check"] = {"resample_skipped": True,
                                "max_abs_vs_file": max_dev}

    print(json.dumps(result, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESAMPLE_BENCH_OK", flush=True)


if __name__ == "__main__":
    main()
