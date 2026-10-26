#!/usr/bin/env python3
"""Benchmark the built TensorRT pipeline: per-stage + chain + CUDA graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastdialoguesidon import runtime  # noqa: E402
from fastdialoguesidon.engines import timed_cuda  # noqa: E402
from fastdialoguesidon.runtime import SAMPLES, TRTPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--seconds",
        type=lambda s: [int(v) for v in s.split(",")],
        default=[120],
        help="window lengths to benchmark, e.g. 15,30,60,120",
    )
    parser.add_argument("--use-graph", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")

    def device_used_mb() -> float:
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1e6

    baseline_mb = device_used_mb()
    pipeline = TRTPipeline(
        args.engine_dir, use_graph=args.use_graph, bucket_seconds=None
    )
    if args.wav is not None:
        data = runtime.read_wav_int16(args.wav)[:SAMPLES]
        waveform = torch.from_numpy(data.astype("float32") / 32768.0).to(
            device
        )
    else:
        waveform = torch.randn(SAMPLES, device=device) * 0.1
    pipeline.wave_buffer.copy_(waveform.view(1, SAMPLES))

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "use_graph": args.use_graph,
        "baseline_device_memory_mb": baseline_mb,
        "after_init_device_memory_mb": device_used_mb(),
        "lengths": {},
    }
    for seconds in args.seconds:
        seq = seconds * 50
        entry = {}
        with torch.cuda.stream(pipeline.compute):
            pipeline._set_length(seq)
            entry["stage_encoder"] = timed_cuda(
                lambda: pipeline.encoder_stage.enqueue(
                    torch.cuda.current_stream()
                ),
                args.warmup,
                args.runs,
            )
            entry["stage_dit"] = timed_cuda(
                lambda: pipeline.dit_stage.enqueue(
                    torch.cuda.current_stream()
                ),
                args.warmup,
                args.runs,
            )
            entry["stage_decoder"] = timed_cuda(
                lambda: pipeline.decoder_stage.enqueue(
                    torch.cuda.current_stream()
                ),
                args.warmup,
                args.runs,
            )
            entry["chain"] = timed_cuda(
                lambda: pipeline.run_chain(seq), args.warmup, args.runs
            )
        torch.cuda.synchronize()
        entry["rtf"] = entry["chain"]["mean_ms"] / 1000.0 / seconds
        entry["peak_device_memory_mb"] = device_used_mb()
        result["lengths"][f"{seconds}s"] = entry
        print(f"{seconds}s done", flush=True)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("BENCH_E2E_OK", flush=True)


if __name__ == "__main__":
    main()
