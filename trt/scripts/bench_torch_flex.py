#!/usr/bin/env python3
"""Comparison extra: the fastest NON-TensorRT configuration.

Pure-torch pipeline with the flex_attention encoder — the strongest
portable setup (~81 ms / 120 s on H100 vs ~32 ms for the TRT chain):
- fbank on GPU (torch, fp32)
- bf16 encoder (compiled flex_attention) captured in a CUDA graph
- torch.compile(max-autotune, fullgraph) CTM one-step and decoder
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastdialoguesidon import runtime  # noqa: E402
from fastdialoguesidon.engines import timed_cuda  # noqa: E402
from fastdialoguesidon.fbank_export import SeamlessFbankFrontend  # noqa: E402
from fastdialoguesidon.modeling import load_bundle  # noqa: E402

SAMPLES, SEQ = runtime.SAMPLES, runtime.SEQ


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    bundle = load_bundle(args.bundle, device="cuda")
    front = SeamlessFbankFrontend().to(device).eval()

    if args.wav is not None:
        data = runtime.read_wav_int16(args.wav)[:SAMPLES]
        waveform = torch.from_numpy(data.astype("float32") / 32768.0).to(
            device
        )
    else:
        waveform = torch.randn(SAMPLES, device=device) * 0.1

    with torch.no_grad():
        static_features = front(waveform.unsqueeze(0)).clone()

    def encoder_eager(features):
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            return bundle.encoder(features, None)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            encoder_eager(static_features)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    encoder_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(encoder_graph):
        static_conditioning = encoder_eager(static_features)

    compiled_ctm = torch.compile(
        bundle.ctm, mode="max-autotune", dynamic=False, fullgraph=True
    )
    compiled_decoder = torch.compile(
        bundle.decoder, mode="max-autotune", dynamic=False, fullgraph=True
    )

    def torch_pipeline():
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            static_features.copy_(front(waveform.unsqueeze(0)))
        encoder_graph.replay()
        noise = torch.randn(1, SEQ, 64, device=device, dtype=torch.bfloat16)
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            latents = compiled_ctm(static_conditioning, noise)
            return compiled_decoder(latents)

    audio = torch_pipeline()
    torch.cuda.synchronize()
    assert audio.float().std().item() > 0
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch_flex_bf16_e2e": timed_cuda(
            torch_pipeline, args.warmup, args.runs
        ),
    }
    result["rtf"] = (
        result["torch_flex_bf16_e2e"]["mean_ms"]
        / 1000.0
        / runtime.WINDOW_SECONDS
    )
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("BENCH_TORCH_FLEX_OK", flush=True)


if __name__ == "__main__":
    main()
