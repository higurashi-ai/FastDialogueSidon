#!/usr/bin/env python3
"""Export ONNX and build the three TensorRT engines from the HF bundle.

Requires a GPU (the FlexRel AOT plugin compiles Triton PTX at build time)
and NVIDIA_TF32_OVERRIDE=0 in the environment. Engines land in
--engine-dir and are validated against the torch fp32 reference chain.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastdialoguesidon import flexrel_triton, runtime  # noqa: E402
from fastdialoguesidon.engines import (  # noqa: E402
    TRTStage,
    build_engine,
    build_wave_engine,
    errors,
)
from fastdialoguesidon.export_onnx import (  # noqa: E402
    ExportableCTM,
    ExportableWaveFitDecoder,
    export_decoder,
    export_dit,
    export_wave_encoder,
)
from fastdialoguesidon.fbank_export import SeamlessFbankFrontend  # noqa: E402
from fastdialoguesidon.modeling import load_bundle  # noqa: E402

SAMPLES, SEQ = runtime.SAMPLES, runtime.SEQ
MIN_SEQ = runtime.MIN_SEQ
MIN_SAMPLES = MIN_SEQ * runtime.SAMPLES_PER_FRAME


def reference_chain(bundle, waveform: torch.Tensor, fixed_noise):
    """fp32 torch reference (true fp32 matmuls — TF32 must be off)."""
    torch.set_float32_matmul_precision("highest")
    front = SeamlessFbankFrontend().to(waveform.device).eval()
    with torch.no_grad():
        features = front(waveform.unsqueeze(0))
        conditioning = bundle.encoder(features, None).float()
        exportable_ctm = (
            ExportableCTM(bundle.ctm, seq_len=SEQ).to(waveform.device).eval()
        )
        latents = exportable_ctm(conditioning, fixed_noise).float()
        exportable_decoder = (
            ExportableWaveFitDecoder(
                bundle.vocoder,
                bundle.latent_mean,
                bundle.latent_std,
                bundle.config["latent_dim"],
            )
            .to(waveform.device)
            .eval()
        )
        audio = exportable_decoder(latents).float().clone()
    return conditioning, latents, audio, exportable_ctm, exportable_decoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument(
        "--wav", type=Path, default=None,
        help="120 s 16 kHz mono wav for validation (random audio if absent)",
    )
    parser.add_argument(
        "--max-batch", type=int, default=1,
        help="largest batch the engines accept; activation memory is "
        "reserved for this at init (default 1 = ~2.9 GB total; build with "
        "a larger value only if you use separate_batch())",
    )
    parser.add_argument(
        "--encoder-precision", choices=["fp16", "bf16"], default="fp16",
        help="encoder engine dtype (fbank front-end stays fp32); fp16 "
        "default: more accurate than bf16 at the same speed",
    )
    parser.add_argument(
        "--dit-precision", choices=["fp16", "bf16"], default="fp16",
        help="DiT engine dtype (LayerNorms stay fp32); fp16 default",
    )
    parser.add_argument(
        "--decoder-precision", choices=["bf16"], default="bf16",
        help="decoder engine dtype: bf16 only — an fp16 decoder overflows "
        "(NaN output), so fp16 is not supported",
    )
    parser.add_argument(
        "--input-rate", type=int, default=16_000,
        help="engine input sample rate; != 16000 bakes a sinc-conv "
        "resample front into the encoder engine (+~0.5 ms faster than "
        "resampling outside)",
    )
    parser.add_argument(
        "--validate-seconds",
        type=lambda s: [int(v) for v in s.split(",")],
        default=[15, 30, 60, 120],
    )
    args = parser.parse_args()
    args.onnx_dir.mkdir(parents=True, exist_ok=True)
    args.engine_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    device = torch.device("cuda")
    # register BEFORE any TensorRT builder/runtime work (segfault otherwise)
    flexrel_triton.register_trt_plugin_qr_dyn()

    bundle = load_bundle(args.bundle, device="cuda")
    if args.wav is not None:
        data = runtime.read_wav_int16(args.wav)[:SAMPLES]
        waveform = (
            torch.from_numpy(data.astype("float32") / 32768.0).to(device)
        )
    else:
        waveform = torch.randn(SAMPLES, device=device) * 0.1
    fixed_noise = torch.randn(1, SEQ, 64, device=device)

    conditioning, latents, audio, exportable_ctm, exportable_decoder = (
        reference_chain(bundle, waveform, fixed_noise)
    )
    print("fp32 reference chain done", flush=True)

    # The validation loop below compares engines against the exportable
    # modules themselves, so it cannot catch a bug in their rewrites
    # (e.g. _upsample_gemm). Guard against the true torch decoder here.
    with torch.no_grad():
        audio_true = bundle.decoder(latents).float()
    rel_rmse = (
        (audio - audio_true).square().mean().sqrt()
        / audio_true.square().mean().sqrt()
    ).item()
    print(f"decoder equivalence guard: rel_rmse={rel_rmse:.3e}", flush=True)
    assert rel_rmse < 1e-5, (
        f"ExportableWaveFitDecoder diverged from bundle.decoder "
        f"(rel_rmse={rel_rmse:.3e}, expected <1e-5; fp32 roundoff "
        f"envelope measured at ~1.3e-6)"
    )

    in_per_frame = runtime.SAMPLES_PER_FRAME * args.input_rate // 16_000
    assert runtime.SAMPLES_PER_FRAME * args.input_rate % 16_000 == 0
    if args.input_rate != 16_000:
        import torchaudio.functional as AF

        waveform_in = AF.resample(
            waveform.unsqueeze(0), 16_000, args.input_rate
        )[0][: SEQ * in_per_frame].contiguous()
    else:
        waveform_in = waveform
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtypes[args.encoder_precision]
    encoder_onnx = args.onnx_dir / "encoder_wave_fp32.onnx"
    export_wave_encoder(
        copy.deepcopy(bundle.encoder),
        waveform_in.unsqueeze(0),
        encoder_onnx,
        input_rate=args.input_rate,
        dtype=dtype,
    )
    dit_onnx = args.onnx_dir / "dit_ctm1step_fp32.onnx"
    st_ctm = (
        __import__("fastdialoguesidon.export_onnx", fromlist=["STExportableCTM"])
        .STExportableCTM(copy.deepcopy(exportable_ctm), dtypes[args.dit_precision])
        .to(device)
        .eval()
    )
    export_dit(st_ctm, conditioning, fixed_noise, dit_onnx)
    decoder_onnx = args.onnx_dir / "decoder_wavefitnext_fp32.onnx"
    export_decoder(exportable_decoder, latents, decoder_onnx)
    print("ONNX exported", flush=True)

    report = {"engines": {}}
    cond_dim = conditioning.shape[2]
    max_batch = args.max_batch
    engines = {}
    engines["encoder"], seconds = build_wave_engine(
        encoder_onnx,
        {
            "waveform": (
                (1, MIN_SEQ * in_per_frame),
                (1, SEQ * in_per_frame),
                (max_batch, SEQ * in_per_frame),
            )
        },
        args.engine_dir / runtime.ENGINE_FILES["encoder"],
    )
    report["engines"]["encoder"] = {"build_seconds": seconds}
    report["input_rate"] = args.input_rate
    report["precision"] = {
        "encoder": args.encoder_precision,
        "dit": args.dit_precision,
        "decoder": args.decoder_precision,
    }
    # strongly-typed DiT (fp16/bf16 weights + fp32 LayerNorms in the ONNX)
    engines["dit"], seconds = build_wave_engine(
        dit_onnx,
        {
            "conditioning": (
                (1, MIN_SEQ, cond_dim), (1, SEQ, cond_dim),
                (max_batch, SEQ, cond_dim),
            ),
            "noise": (
                (1, MIN_SEQ, 64), (1, SEQ, 64), (max_batch, SEQ, 64)
            ),
        },
        args.engine_dir / runtime.ENGINE_FILES["dit"],
    )
    report["engines"]["dit"] = {"build_seconds": seconds}
    engines["decoder"], seconds = build_engine(
        decoder_onnx,
        {
            "latents": (
                (1, MIN_SEQ, 64), (1, SEQ, 64), (max_batch, SEQ, 64)
            )
        },
        args.engine_dir / runtime.ENGINE_FILES["decoder"],
        precision=args.decoder_precision,
    )
    report["engines"]["decoder"] = {"build_seconds": seconds}
    print("engines built", flush=True)

    # validation: chained runs vs the fp32 reference at several lengths
    report["validation"] = {}
    for seconds in args.validate_seconds:
        seq = seconds * 50
        samples = seq * runtime.SAMPLES_PER_FRAME
        wave_short = waveform[:samples]
        wave_short_in = waveform_in[: seq * in_per_frame]
        noise_short = fixed_noise[:, :seq].contiguous()
        if seconds * 50 == SEQ:
            ref = (conditioning, latents, audio)
        else:
            ref = reference_chain(bundle, wave_short, noise_short)[:3]
        wave_buffer = wave_short_in.unsqueeze(0).contiguous().clone()
        encoder_stage = TRTStage(
            engines["encoder"], {"waveform": wave_buffer}
        )
        conditioning_buffer = next(iter(encoder_stage.outputs.values()))
        dit_stage = TRTStage(
            engines["dit"],
            {"conditioning": conditioning_buffer, "noise": noise_short},
        )
        latent_buffer = next(iter(dit_stage.outputs.values()))
        decoder_stage = TRTStage(
            engines["decoder"], {"latents": latent_buffer}
        )
        audio_buffer = next(iter(decoder_stage.outputs.values()))
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for stage in (encoder_stage, dit_stage, decoder_stage):
                stage.enqueue(torch.cuda.current_stream())
        torch.cuda.synchronize()
        report["validation"][f"{seconds}s"] = {
            "conditioning": errors(ref[0], conditioning_buffer),
            "latents": errors(ref[1], latent_buffer),
            "audio": errors(ref[2], audio_buffer),
        }
        print(
            f"{seconds}s:",
            json.dumps(report["validation"][f"{seconds}s"]),
            flush=True,
        )
    (args.engine_dir / "build_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"BUILD_ENGINES_OK engine_dir={args.engine_dir}", flush=True)


if __name__ == "__main__":
    main()
