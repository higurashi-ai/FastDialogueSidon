#!/usr/bin/env python3
"""Kernel microbenchmarks used to pick implementations (H100, 120 s shapes).

attn:   fused flexrel attention over every tile config, vs the cuDNN-split path
sdpa:   DiT attention backends (flash / cuDNN / efficient) at 12 heads
gemm:   Triton GEMMs (plain and persistent TMA) with fused epilogues vs cuBLAS
dw:     depthwise conv + norm kernels
dwcfg:  every launch config of the depthwise kernels
addmm:  bf16/fp16 GEMM accumulated in place into an fp32 residual
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fds_triton import kernels as K  # noqa: E402


def bench(fn, warmup=5, runs=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def attn(L=6000, B=1, H=16):
    D = 64
    hid = H * D
    qkv = torch.randn(B * L, 3 * hid, device="cuda").bfloat16()
    rel = torch.randn(K.NUM_POS, D, device="cuda").bfloat16() * 0.1
    flops = 4 * B * H * L * L * D
    q, k, v = (qkv[:, i * hid:(i + 1) * hid].contiguous() for i in range(3))
    for layout in ("strided",):
        if layout == "strided":
            tq, tk, tv, width = qkv[:, :hid], qkv[:, hid:2 * hid], qkv[:, 2 * hid:], 3 * hid
        else:
            tq, tk, tv, width = q, k, v, hid
        out = torch.empty(B * L, hid, device="cuda", dtype=torch.bfloat16)
        for _ in (0,):
            for cfg in K._attn_configs():
                bm = cfg.kwargs["BLOCK_M"]

                def run(cfg=cfg, bm=bm):
                    K._flexrel_kernel.fn[(triton_cdiv(L, bm), B * H)](
                        tq, tk, tv, rel, out, L * width, width, L * hid, hid, H, L, 8192,
                        K.LOG2E.value / math.sqrt(D), HEAD_DIM=D, NPOS=K.NUM_POS, NPOS_PAD=K.NUM_POS_PAD,
                        CLAMP_LEFT=K.LEFT, CLAMP_RIGHT=K.RIGHT, num_warps=cfg.num_warps,
                        num_stages=cfg.num_stages, **cfg.kwargs)
                try:
                    ms = bench(run, 3, 15)
                    print(f"[attn] {layout:10s} {cfg.kwargs} w{cfg.num_warps} "
                          f"s{cfg.num_stages}: {ms:.3f} ms {flops / ms / 1e9:.0f} TFLOPS", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[attn] {layout} {cfg}: FAIL {exc!r:.100}", flush=True)
            if layout == "contiguous":
                break
    fused = K.flexrel_attention(qkv, B, H, rel)
    print(f"[attn] autotune pick: {K._flexrel_kernel.best_config}", flush=True)
    t_fused = bench(lambda: K.flexrel_attention(qkv, B, H, rel))
    split = K.flexrel_attention_split(qkv, B, H, rel)
    t_split = bench(lambda: K.flexrel_attention_split(qkv, B, H, rel))
    snr = 20 * torch.log10(fused.float().norm() / (fused.float() - split.float()).norm()).item()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        K.flexrel_attention_split(qkv, B, H, rel)
    t_graph = bench(g.replay)
    print(f"[attn] fused {t_fused:.3f} ms vs cudnn-split {t_split:.3f} ms (graph {t_graph:.3f}) "
          f"split-vs-fused snr {snr:.1f} dB", flush=True)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
    for ev in sorted(prof.key_averages(), key=lambda e: -e.device_time_total)[:8]:
        print(f"[attn]   {ev.device_time_total / 5000:.4f} ms  {ev.key[:90]}", flush=True)
    print(f"[attn] band autotune pick: {K._band_merge_kernel.best_config}", flush=True)




def dw(L=6000, B=1, C=1024):
    g = torch.randn(B * L, C, device="cuda").bfloat16()
    w = torch.randn(31, C, device="cuda") * 0.2
    lw, lb = torch.ones(C, device="cuda"), torch.zeros(C, device="cuda")
    out = K.causal_dw_norm(g, B, w, lw, lb, 1e-5)
    ms = bench(lambda: K.causal_dw_norm(g, B, w, lw, lb, 1e-5))
    print(f"[dw] causal k31: {ms:.4f} ms", flush=True)
    x = torch.randn(2 * B * 2 * L * 2, 512, device="cuda")
    y = torch.randn_like(x).bfloat16()
    o = torch.empty_like(x)
    w7, b7 = torch.randn(7, 512, device="cuda"), torch.randn(512, device="cuda")
    ms = bench(lambda: K.convnext_dw_norm(x, y, 2 * B, w7, b7, lw[:512], lb[:512], 1e-6, out_x=o))
    print(f"[dw] convnext k7 pending R={x.shape[0]}: {ms:.4f} ms", flush=True)


def dwcfg():
    """Every config of both depthwise kernels at the 120 s shapes."""
    for B, L in ((1, 6000), (1, 750)):
        C = 1024
        g = torch.randn(B * L, C, device="cuda").bfloat16()
        w = torch.randn(31, C, device="cuda") * 0.2
        lw, lb = torch.ones(C, device="cuda"), torch.zeros(C, device="cuda")
        out = torch.empty_like(g)
        for cfg in K._causal_dw_configs():
            bt = cfg.kwargs["BT"]
            ms = bench(lambda: K._causal_dw_norm_kernel.fn[(triton_cdiv(L, bt), B)](
                g, w, lw, lb, out, L, 1e-5, C=C, K=31, num_warps=cfg.num_warps, **cfg.kwargs), 3, 20)
            print(f"[dwcfg] causal L={L} BT={bt} w{cfg.num_warps}: {ms:.4f} ms", flush=True)
        S, T, C = 2 * B, 2 * L, 512
        x = torch.randn(S * T, C, device="cuda")
        y = torch.randn_like(x).bfloat16()
        o = torch.empty_like(x)
        xn = torch.empty_like(y)
        w7, b7 = torch.randn(7, C, device="cuda"), torch.randn(C, device="cuda")
        for cfg in K._convnext_dw_configs():
            bt = cfg.kwargs["BT"]
            ms = bench(lambda: K._convnext_dw_norm_kernel.fn[(triton_cdiv(T, bt), S)](
                x, y, o, xn, w7, b7, lw[:C], lb[:C], T, 1e-6, C=C, K=7, PENDING=True,
                num_warps=cfg.num_warps, **cfg.kwargs), 3, 20)
            print(f"[dwcfg] convnext T={T} BT={bt} w{cfg.num_warps}: {ms:.4f} ms", flush=True)


def addmm():
    """bf16 GEMM accumulated straight into an fp32 residual (cuBLAS beta=1)."""
    for M, Kd, N in ((750, 4096, 1024), (750, 1024, 1024), (1500, 4096, 1024), (6000, 4096, 1024),
                     (6000, 1024, 1024)):
        a = torch.randn(M, Kd, device="cuda").bfloat16()
        w = (torch.randn(N, Kd, device="cuda") / math.sqrt(Kd)).bfloat16()
        x = torch.randn(M, N, device="cuda")
        b = torch.randn(N, device="cuda").bfloat16()
        ref = x + a.float() @ w.float().t()
        t_lin = bench(lambda: F.linear(a, w, b))
        for name, fn in (
            ("addmm(out_dtype)", lambda: torch.addmm(x, a, w.t(), out_dtype=torch.float32)),
            ("addmm(out_dtype,out=x)", lambda: torch.addmm(x, a, w.t(), out_dtype=torch.float32, out=x)),
        ):
            try:
                x0 = x.clone()
                got = fn()
                err = 20 * torch.log10(ref.norm() / (ref - got).norm()).item()
                x.copy_(x0)
                ms = bench(fn)
                x.copy_(x0)
                print(f"[addmm] {M}x{Kd}x{N} {name}: {ms:.4f} ms (bf16 linear+bias {t_lin:.4f}) snr {err:.1f}",
                      flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[addmm] {M}x{Kd}x{N} {name}: FAIL {exc!r:.160}", flush=True)


def triton_cdiv(a, b):
    return (a + b - 1) // b


def sdpa(L=6000, B=1, H=12):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    D = 64
    qkv = torch.randn(3, B * L, H * D, device="cuda").bfloat16()
    shape = (B, L, H, D)
    q, k, v = (t.view(shape).transpose(1, 2) for t in qkv)
    flops = 4 * B * H * L * L * D
    ref = None
    for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
        try:
            with sdpa_kernel(be):
                fn = lambda: F.scaled_dot_product_attention(q, k, v)  # noqa: E731
                out = fn()
                ms = bench(fn)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                g.replay()
                torch.cuda.synchronize()
            err = 0.0 if ref is None else (out.float() - ref.float()).abs().max().item()
            ref = out if ref is None else ref
            print(f"[sdpa] {be.name}: {ms:.3f} ms {flops / ms / 1e9:.0f} TFLOPS graph-ok "
                  f"max|diff| {err:.2e} out stride {tuple(out.stride())}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[sdpa] {be.name}: FAIL {exc!r:.150}", flush=True)


def gemm():
    from fds_triton.gemm import linear_act, linear_act_tma

    for M, Kd, N, act in ((6000, 1024, 4096, "silu"), (24000, 512, 1536, "gelu"),
                          (6000, 768, 3072, "gelu"), (6000, 1024, 2048, "none")):
        x = torch.randn(M, Kd, device="cuda").bfloat16()
        w = (torch.randn(N, Kd, device="cuda") / math.sqrt(Kd)).bfloat16()
        b = torch.randn(N, device="cuda").bfloat16()
        flop = 2 * M * N * Kd
        act_fn = {"silu": F.silu, "gelu": F.gelu, "none": lambda t: t}[act]
        t_cublas = bench(lambda: F.linear(x, w, b))
        t_sep = bench(lambda: act_fn(F.linear(x, w, b)))
        ref = act_fn(F.linear(x.float(), w.float(), b.float()))
        out = linear_act(x, w, b, act)
        t_tri = bench(lambda: linear_act(x, w, b, act))
        snr = 20 * torch.log10(ref.norm() / (ref - out.float()).norm()).item()
        out_t = linear_act_tma(x, w, b, act)
        t_tma = bench(lambda: linear_act_tma(x, w, b, act))
        snr_t = 20 * torch.log10(ref.norm() / (ref - out_t.float()).norm()).item()
        line = (f"[gemm] {M}x{Kd}x{N} {act}: cublas {t_cublas:.4f} ({flop / t_cublas / 1e9:.0f} TF) "
                f"cublas+act {t_sep:.4f} triton {t_tri:.4f} ({flop / t_tri / 1e9:.0f} TF) snr {snr:.1f} "
                f"tma {t_tma:.4f} ({flop / t_tma / 1e9:.0f} TF) snr {snr_t:.1f}")
        if act == "gelu":
            t_lt = bench(lambda: torch._addmm_activation(b, x, w.t(), use_gelu=True))
            line += f" cublasLt-gelu(tanh) {t_lt:.4f}"
        print(line, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="attn")
    args = ap.parse_args()
    for part in args.parts.split(","):
        {"attn": attn, "sdpa": sdpa, "gemm": gemm, "dw": dw, "dwcfg": dwcfg, "addmm": addmm}[part]()
    print("MICROBENCH_OK")
