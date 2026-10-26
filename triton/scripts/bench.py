#!/usr/bin/env python3
"""Benchmark the Triton pipeline (per stage + CUDA-graphed chain), optionally
side by side with the TensorRT engines, plus a kernel-level profile."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fds_triton.runtime import SAMPLES, SEQ, TritonPipeline  # noqa: E402


def timed(fn, warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(runs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {"median_ms": round(statistics.median(samples), 3), "min_ms": round(min(samples), 3)}


def graphed(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


def kernel_profile(fn, reps: int = 5, top: int = 25) -> list:
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    agg = defaultdict(lambda: [0.0, 0])
    for ev in prof.events():
        if ev.device_type.name == "CUDA":
            agg[ev.name][0] += ev.device_time_total / 1000 / reps
            agg[ev.name][1] += 1
    rows = sorted(agg.items(), key=lambda kv: -kv[1][0])
    return [{"name": k[:140], "ms": round(v[0], 4), "calls": v[1] // reps} for k, v in rows[:top]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--seconds", default="15,30,60,120")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--profile", action="store_true", help="kernel breakdown per stage")
    ap.add_argument("--profile-seconds", type=int, default=None, help="length to profile (default: longest)")
    ap.add_argument("--trt-engine-dir", type=Path, default=None)
    ap.add_argument("--trt-repo", type=Path, default=ROOT.parent / "trt")
    ap.add_argument("--gelu-erf", action="store_true")
    ap.add_argument("--accumulate-rows", type=int, default=0, help="-1: never accumulate residual GEMMs")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--attn-impl", choices=["auto", "split", "fused"], default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    free0, total = torch.cuda.mem_get_info()
    pipe = TritonPipeline(args.bundle, bucket_seconds=None, max_batch=args.batch, gelu_tanh=not args.gelu_erf,
                          dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype])
    pipe.model.accumulate_rows = None if args.accumulate_rows < 0 else args.accumulate_rows
    if args.attn_impl is not None:
        pipe.model.attn_impl = args.attn_impl
    pipe.wave_buffer.normal_(std=0.1)
    torch.cuda.reset_peak_memory_stats()
    lengths = [int(s) for s in args.seconds.split(",")]
    report = {"gpu": torch.cuda.get_device_name(0), "batch": args.batch, "attn_impl": pipe.model.attn_impl,
              "triton": {}, "trt": {}}
    for seconds in lengths:
        seq = seconds * 50
        entry = {}
        with torch.cuda.stream(pipe.compute):
            for name, fn in pipe.stage_fns(seq, args.batch).items():
                entry[f"stage_{name}"] = timed(graphed(fn), args.warmup, args.runs)
            entry["chain"] = timed(lambda: pipe.run_chain(seq, args.batch), args.warmup, args.runs)
        entry["rtf"] = round(entry["chain"]["median_ms"] / 1e3 / (seconds * args.batch), 7)
        report["triton"][f"{seconds}s"] = entry
        print(f"[triton] {seconds}s: {json.dumps(entry)}", flush=True)
    free1, _ = torch.cuda.mem_get_info()
    report["triton_device_mem_mb"] = round((free0 - free1) / 1e6, 1)
    report["triton_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    print(f"[triton] device memory used {report['triton_device_mem_mb']} MB "
          f"(peak torch alloc {report['triton_peak_alloc_mb']} MB)", flush=True)

    if args.profile:
        seq = (args.profile_seconds or max(lengths)) * 50
        with torch.cuda.stream(pipe.compute):
            report["profile"] = {name: kernel_profile(fn) for name, fn in pipe.stage_fns(seq, args.batch).items()}
        for name, rows in report["profile"].items():
            print(f"[profile] {name}: sum {sum(r['ms'] for r in rows):.3f} ms (top {len(rows)})", flush=True)
            for r in rows[:15]:
                print(f"    {r['ms']:7.3f} ms x{r['calls']:3d}  {r['name'][:110]}", flush=True)

    if args.trt_engine_dir is not None:
        del pipe
        torch.cuda.empty_cache()
        sys.path.insert(0, str(args.trt_repo / "src"))
        from fastdialoguesidon.runtime import TRTPipeline

        trt = TRTPipeline(args.trt_engine_dir, use_graph=True, bucket_seconds=None, max_batch=args.batch)
        trt.wave_buffer.normal_(std=0.1)
        for seconds in lengths:
            seq = seconds * 50
            entry = {}
            with torch.cuda.stream(trt.compute):
                trt._set_length(seq, args.batch)
                for name, stage in (("encoder", trt.encoder_stage), ("dit", trt.dit_stage),
                                    ("decoder", trt.decoder_stage)):
                    entry[f"stage_{name}"] = timed(lambda s=stage: s.enqueue(torch.cuda.current_stream()),
                                                   args.warmup, args.runs)
                entry["chain"] = timed(lambda: trt.run_chain(seq, args.batch), args.warmup, args.runs)
            entry["rtf"] = round(entry["chain"]["median_ms"] / 1e3 / (seconds * args.batch), 7)
            report["trt"][f"{seconds}s"] = entry
            print(f"[trt] {seconds}s: {json.dumps(entry)}", flush=True)
        for key in report["triton"]:
            if key in report["trt"]:
                t, r = report["triton"][key]["chain"]["median_ms"], report["trt"][key]["chain"]["median_ms"]
                print(f"[compare] {key}: triton {t:.2f} ms vs trt {r:.2f} ms -> {r / t:.2f}x", flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1))
    print("BENCH_OK")


if __name__ == "__main__":
    main()
