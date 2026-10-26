"""Triton GEMM with a fused bias + activation epilogue: act(x @ w^T + b).

cuBLASLt only offers ReLU / tanh-GELU epilogues; this kernel fuses SiLU and
the exact erf GELU so the [M, N] hidden is written once, already activated.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SQRT1_2 = tl.constexpr(0.7071067811865476)


def _configs():
    return [
        triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP": 8}, num_warps=w, num_stages=s)
        for bm, bn, bk, w, s in (
            (128, 256, 64, 8, 3),
            (256, 128, 64, 8, 3),
            (128, 128, 64, 8, 4),
            (128, 128, 64, 4, 4),
            (128, 256, 64, 8, 4),
            (64, 256, 64, 4, 4),
            (128, 64, 64, 4, 4),
        )
    ]


def _prune(configs, named_args, **kwargs):
    n = {**named_args, **kwargs}["N"]
    k = {**named_args, **kwargs}["K"]
    keep = [c for c in configs if n % c.kwargs["BN"] == 0 and k % c.kwargs["BK"] == 0]
    return keep or configs[-1:]


@triton.autotune(configs=_configs(), key=["M_BUCKET", "N", "K", "ACT"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _linear_act_kernel(x_ptr, w_ptr, b_ptr, y_ptr, M, M_BUCKET, N: tl.constexpr, K: tl.constexpr,
                       ACT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                       GROUP: tl.constexpr):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n: tl.constexpr = N // BN
    width = GROUP * grid_n
    group = pid // width
    first_m = group * GROUP
    size_m = tl.minimum(grid_m - first_m, GROUP)
    pid_m = first_m + (pid % width) % size_m
    pid_n = (pid % width) // size_m

    rm = pid_m.to(tl.int64) * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mmask = (rm < M)[:, None]
    a_ptrs = x_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = w_ptr + rn[None, :] * K + rk[:, None]  # [BK, BN] view of w^T
    acc = tl.zeros([BM, BN], tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(a_ptrs, mask=mmask, other=0.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK
    acc += tl.load(b_ptr + rn).to(tl.float32)[None, :]
    if ACT == "silu":
        acc = acc * tl.sigmoid(acc)
    elif ACT == "gelu":
        acc = 0.5 * acc * (1.0 + tl.math.erf(acc * SQRT1_2))
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(y_ptr.dtype.element_ty), mask=mmask)


def linear_act(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, act: str = "none") -> torch.Tensor:
    """x [M, K] bf16 contiguous, w [N, K] bf16, b [N] -> act(x w^T + b) bf16."""
    M, K = x.shape
    N = w.shape[0]
    assert x.is_contiguous() and w.is_contiguous() and act in ("none", "silu", "gelu")
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    bucket = 1 << max(0, (M - 1).bit_length())
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * (N // meta["BN"]),)  # noqa: E731
    _linear_act_kernel[grid](x, w, b, y, M, bucket, N=N, K=K, ACT=act)
    return y


# ---------------------------------------------------------------- TMA + persistent

_ALLOCATOR_SET = False


def _ensure_allocator() -> None:
    """Device-side tensor descriptors need a scratch allocator (graph-capture safe)."""
    global _ALLOCATOR_SET
    if not _ALLOCATOR_SET:
        triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
        _ALLOCATOR_SET = True


def _tma_configs():
    return [
        triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP": 8, "SUBTILE": st}, num_warps=w, num_stages=s)
        for bm, bn, bk, w, s in (
            (128, 256, 64, 8, 3),
            (128, 256, 64, 8, 4),
            (128, 128, 64, 8, 4),
            (128, 128, 64, 4, 4),
            (128, 128, 128, 8, 3),
            (256, 128, 64, 8, 3),
            (128, 64, 64, 4, 4),
        )
        for st in (False, True)
    ]


@triton.autotune(configs=_tma_configs(), key=["M_BUCKET", "N", "K", "ACT"],
                 prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _linear_act_tma_kernel(x_ptr, w_ptr, b_ptr, y_ptr, M, M_BUCKET, N: tl.constexpr, K: tl.constexpr,
                           ACT: tl.constexpr, NUM_SMS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                           BK: tl.constexpr, GROUP: tl.constexpr, SUBTILE: tl.constexpr):
    """Persistent TMA GEMM (tutorial-09 style) with bias + activation epilogue."""
    a_desc = tl.make_tensor_descriptor(x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    b_desc = tl.make_tensor_descriptor(w_ptr, shape=[N, K], strides=[K, 1], block_shape=[BN, BK])
    if SUBTILE:
        c_desc = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN // 2])
    else:
        c_desc = tl.make_tensor_descriptor(y_ptr, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    grid_m = tl.cdiv(M, BM)
    grid_n: tl.constexpr = N // BN
    num_tiles = grid_m * grid_n
    k_tiles: tl.constexpr = K // BK
    width = GROUP * grid_n
    for tile in tl.range(tl.program_id(0), num_tiles, NUM_SMS, flatten=True):
        group = tile // width
        first_m = group * GROUP
        size_m = tl.minimum(grid_m - first_m, GROUP)
        pid_m = first_m + (tile % width) % size_m
        pid_n = (tile % width) // size_m
        om = pid_m * BM
        on = pid_n * BN
        acc = tl.zeros([BM, BN], tl.float32)
        for ki in range(k_tiles):
            a = a_desc.load([om, ki * BK])
            b = b_desc.load([on, ki * BK])
            acc = tl.dot(a, b.T, acc)
        rn = on + tl.arange(0, BN)
        acc += tl.load(b_ptr + rn).to(tl.float32)[None, :]
        if ACT == "silu":
            acc = acc * tl.sigmoid(acc)
        elif ACT == "gelu":
            acc = 0.5 * acc * (1.0 + tl.math.erf(acc * SQRT1_2))
        out = acc.to(y_ptr.dtype.element_ty)
        if SUBTILE:
            out = tl.reshape(out, (BM, 2, BN // 2))
            out = tl.permute(out, (0, 2, 1))
            o0, o1 = tl.split(out)
            c_desc.store([om, on], o0)
            c_desc.store([om, on + BN // 2], o1)
        else:
            c_desc.store([om, on], out)


_NUM_SMS: dict = {}


def linear_act_tma(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, act: str = "none") -> torch.Tensor:
    """Persistent TMA version of :func:`linear_act` (Hopper+)."""
    M, K = x.shape
    N = w.shape[0]
    assert x.is_contiguous() and w.is_contiguous() and act in ("none", "silu", "gelu")
    _ensure_allocator()
    dev = x.device.index or 0
    if dev not in _NUM_SMS:
        _NUM_SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    sms = _NUM_SMS[dev]
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    bucket = 1 << max(0, (M - 1).bit_length())
    grid = lambda meta: (min(sms, triton.cdiv(M, meta["BM"]) * (N // meta["BN"])),)  # noqa: E731
    _linear_act_tma_kernel[grid](x, w, b, y, M, bucket, N=N, K=K, ACT=act, NUM_SMS=sms)
    return y
