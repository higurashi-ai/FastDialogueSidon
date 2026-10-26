#!/usr/bin/env python3
"""Fused flash-attention kernel with in-kernel w2v-BERT relative_key bias.

scores[b,h,i,j] = (Q.K^T)/sqrt(d) + (Q[i].R[clamp(j-i,-64,8)+64])/sqrt(d)

The [L, L] bias is never materialized: each row tile computes
QR = Q_tile @ R^T once (R is 73x64, SMEM-resident) and the column-tile loop
gathers QR[m, clamp(n-m,-64,8)+64] in registers. Forward only, fp16/bf16
in/out, fp32 softmax/accumulate.

Also provides the TensorRT quickly-deployable-plugin (IPluginV3 QUICK)
wrappers; the runtime uses `register_trt_plugin_qr_dyn()` (op
`sidon::FlexRelAttnQRD`: QR bias table from the graph, dynamic L and B).
"""

import math
import os
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

LEFT, RIGHT = 64, 8
NUM_POS = LEFT + RIGHT + 1  # 73
NUM_POS_PAD = 128  # tl.arange needs a power of 2; rows >= NUM_POS are zero
LOG2E = tl.constexpr(1.4426950408889634)

_PLUGIN_OP = "sidon::FlexRelAttn"


def _configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
        for bm, bn, w, s in (
            (128, 64, 8, 3),
            (128, 128, 8, 3),
            (128, 128, 8, 2),
            (128, 64, 4, 3),
            (64, 128, 4, 3),
            (64, 64, 4, 3),
            (64, 64, 4, 4),
            (64, 64, 8, 3),
        )
    ]


@triton.jit
def _softmax_update(s, m_i, l_i, acc, v_tile):
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.math.exp2((m_i - m_new) * LOG2E)
    p = tl.math.exp2((s - m_new[:, None]) * LOG2E)
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
    return m_new, l_i, acc


@triton.jit
def _flexrel_body(
    Q, K, V, R, O,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    H, L, scale,
    HEAD_DIM: tl.constexpr,
    NPOS: tl.constexpr,
    NPOS_PAD: tl.constexpr,
    CLAMP_LEFT: tl.constexpr,
    CLAMP_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_valid = offs_m < L

    q_base = Q + b * stride_qb + h * stride_qh
    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=m_valid[:, None],
        other=0.0,
    )

    # per-row-tile relative projection: QR = Q @ R^T * scale  [BLOCK_M, NPOS_PAD]
    offs_p = tl.arange(0, NPOS_PAD)
    r = tl.load(
        R + offs_p[:, None] * HEAD_DIM + offs_d[None, :],
        mask=offs_p[:, None] < NPOS,
        other=0.0,
    )
    qr = tl.dot(q, tl.trans(r)) * scale
    # far-from-diagonal tiles: clamp saturates, bias is a column vector
    qr_left = tl.sum(tl.where(offs_p[None, :] == 0, qr, 0.0), 1)
    qr_right = tl.sum(tl.where(offs_p[None, :] == NPOS - 1, qr, 0.0), 1)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    m_lo = pid_m * BLOCK_M
    m_hi = m_lo + BLOCK_M - 1

    # n-range split so the two bulk loops have no mask, branch, or gather:
    #   [0, n_left)        pure-left clamp  (bias = qr_left, unmasked)
    #   [n_left, band_end) diagonal band    (gather + L mask)
    #   [n_right, n_full)  pure-right clamp (bias = qr_right, unmasked)
    #   [n_tail, L)        pure-right tail  (bias = qr_right, L mask)
    n_left = tl.maximum(
        (m_lo - CLAMP_LEFT - BLOCK_N + 1 + BLOCK_N) // BLOCK_N, 0
    ) * BLOCK_N
    n_right = ((m_hi + CLAMP_RIGHT + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    n_full = (L // BLOCK_N) * BLOCK_N
    band_end = tl.minimum(n_right, L)
    n_tail = tl.maximum(n_right, n_full)

    for n0 in tl.range(0, n_left, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        k_tile = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :]
        )
        s = tl.dot(q, tl.trans(k_tile)) * scale + qr_left[:, None]
        v_tile = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :]
        )
        m_i, l_i, acc = _softmax_update(s, m_i, l_i, acc, v_tile)

    for n0 in tl.range(n_left, band_end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_valid = offs_n < L
        k_tile = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :],
            mask=n_valid[:, None],
            other=0.0,
        )
        rel = (
            tl.minimum(
                tl.maximum(offs_n[None, :] - offs_m[:, None], -CLAMP_LEFT),
                CLAMP_RIGHT,
            )
            + CLAMP_LEFT
        )
        s = tl.dot(q, tl.trans(k_tile)) * scale + tl.gather(qr, rel, axis=1)
        s = tl.where(n_valid[None, :], s, float("-inf"))
        v_tile = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :],
            mask=n_valid[:, None],
            other=0.0,
        )
        m_i, l_i, acc = _softmax_update(s, m_i, l_i, acc, v_tile)

    for n0 in tl.range(n_right, n_full, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        k_tile = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :]
        )
        s = tl.dot(q, tl.trans(k_tile)) * scale + qr_right[:, None]
        v_tile = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :]
        )
        m_i, l_i, acc = _softmax_update(s, m_i, l_i, acc, v_tile)

    for n0 in tl.range(n_tail, L, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_valid = offs_n < L
        k_tile = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :],
            mask=n_valid[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k_tile)) * scale + qr_right[:, None]
        s = tl.where(n_valid[None, :], s, float("-inf"))
        v_tile = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :],
            mask=n_valid[:, None],
            other=0.0,
        )
        m_i, l_i, acc = _softmax_update(s, m_i, l_i, acc, v_tile)

    acc = acc / l_i[:, None]
    o_base = O + b * stride_ob + h * stride_oh
    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_d[None, :],
        acc.to(O.dtype.element_ty),
        mask=m_valid[:, None],
    )


@triton.autotune(configs=_configs(), key=["B", "H", "L"])
@triton.jit
def _flexrel_fwd_kernel(
    Q, K, V, R, O,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    B, H, L, scale,
    HEAD_DIM: tl.constexpr,
    NPOS: tl.constexpr,
    NPOS_PAD: tl.constexpr,
    CLAMP_LEFT: tl.constexpr,
    CLAMP_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    _flexrel_body(
        Q, K, V, R, O,
        stride_qb, stride_qh, stride_qm,
        stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn,
        stride_ob, stride_oh, stride_om,
        H, L, scale,
        HEAD_DIM, NPOS, NPOS_PAD, CLAMP_LEFT, CLAMP_RIGHT, BLOCK_M, BLOCK_N,
    )


@triton.jit
def _flexrel_fwd_kernel_aot(
    Q, K, V, R, O,
    H: tl.constexpr,
    L: tl.constexpr,
    scale: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NPOS: tl.constexpr,
    NPOS_PAD: tl.constexpr,
    CLAMP_LEFT: tl.constexpr,
    CLAMP_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # contiguous [B, H, L, D] layout: strides are B-independent constants
    _flexrel_body(
        Q, K, V, R, O,
        H * L * HEAD_DIM, L * HEAD_DIM, HEAD_DIM,
        H * L * HEAD_DIM, L * HEAD_DIM, HEAD_DIM,
        H * L * HEAD_DIM, L * HEAD_DIM, HEAD_DIM,
        H * L * HEAD_DIM, L * HEAD_DIM, HEAD_DIM,
        H, L, scale,
        HEAD_DIM, NPOS, NPOS_PAD, CLAMP_LEFT, CLAMP_RIGHT, BLOCK_M, BLOCK_N,
    )


def flexrel_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """q, k, v: [B, H, L, D] (last dim contiguous), r: [NUM_POS, D] contiguous."""
    batch, heads, length, dim = q.shape
    assert dim == 64 and r.shape == (NUM_POS, dim), (q.shape, r.shape)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    assert r.is_contiguous()
    if out is None:
        out = torch.empty_like(q, memory_format=torch.contiguous_format)
    scale = 1.0 / math.sqrt(dim)
    grid = lambda meta: (triton.cdiv(length, meta["BLOCK_M"]), batch * heads)
    _flexrel_fwd_kernel[grid](
        q, k, v, r, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        batch, heads, length, scale,
        HEAD_DIM=dim,
        NPOS=NUM_POS,
        NPOS_PAD=NUM_POS_PAD,
        CLAMP_LEFT=LEFT,
        CLAMP_RIGHT=RIGHT,
    )
    return out


def explicit_core_fp32(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, r: torch.Tensor
) -> torch.Tensor:
    """Reference attention core: explicit-bias relative_key attention (fp32)."""
    q, k, v, r = (t.float() for t in (q, k, v, r))
    batch, heads, length, dim = q.shape
    qr = torch.einsum("bhld,rd->bhlr", q, r)
    pos = torch.arange(length, device=q.device)
    idx = torch.clamp(pos[None, :] - pos[:, None], -LEFT, RIGHT) + LEFT
    bias = torch.gather(
        qr, 3, idx.view(1, 1, length, length).expand(batch, heads, -1, -1)
    )
    scores = (q @ k.transpose(-2, -1) + bias) / math.sqrt(dim)
    return torch.softmax(scores, dim=-1) @ v


def torch_core_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
    out: torch.Tensor,
    chunk: int = 512,
) -> torch.Tensor:
    """Pure-torch fallback (fp32 math, chunked rows) for plugin debugging."""
    batch, heads, length, dim = q.shape
    scale = 1.0 / math.sqrt(dim)
    pos = torch.arange(length, device=q.device)
    kf, vf, rf = k.float(), v.float(), r.float()
    for start in range(0, length, chunk):
        stop = min(start + chunk, length)
        qc = q[:, :, start:stop].float()
        idx = (
            torch.clamp(pos[None, :] - pos[start:stop, None], -LEFT, RIGHT)
            + LEFT
        )
        bias = torch.gather(
            torch.einsum("bhld,rd->bhlr", qc, rf),
            3,
            idx.view(1, 1, stop - start, length).expand(
                batch, heads, -1, -1
            ),
        )
        scores = (qc @ kf.transpose(-2, -1) + bias) * scale
        out[:, :, start:stop] = (torch.softmax(scores, dim=-1) @ vf).to(
            out.dtype
        )
    return out


def best_config() -> str:
    cfg = getattr(_flexrel_fwd_kernel, "best_config", None)
    return str(cfg)


# ---- TensorRT quickly-deployable plugin (IPluginV3 QUICK) -------------------

IMPL_BACKEND = "triton"  # or "torch" (debug fallback inside the TRT plugin)
DEBUG_PRINT = False

# Kernel blob format the AOT impls hand to TensorRT ("ptx" or "cubin").
# cubin cannot be param-stripped (see below), and the sm_120 segfault sits
# in libnvinfer's launch-param setup either way, so ptx stays the default.
_AOT_FORMAT = os.environ.get("FLEXREL_AOT_FORMAT", "ptx")

# triton >= 3.4 appends hidden trailing kernel params (global / profile
# scratch pointers, both size 0 for these kernels: declared, never read).
# The TRT AOT runner builds the launch param buffer from the *declared
# plugin signature*, so the extra params shift the layout and segfault
# libnvinfer at create_execution_context() on sm_120. Stripping the
# unused trailing .param declarations from the PTX makes the kernel
# signature match exactly. FLEXREL_STRIP_HIDDEN=0 restores raw PTX.
_STRIP_HIDDEN = os.environ.get("FLEXREL_STRIP_HIDDEN", "1") == "1"


def _aot_blob(compiled) -> Union[str, bytes]:
    """Kernel blob for a trtp.aot_impl return, honoring the knobs above."""
    if _AOT_FORMAT == "cubin":
        return compiled.asm["cubin"]
    ptx = compiled.asm["ptx"]
    if not _STRIP_HIDDEN:
        return ptx
    import re

    indices = sorted(
        {int(i) for i in re.findall(r"\.param[^\n]*_param_(\d+)", ptx)},
        reverse=True,
    )
    for index in indices:
        if len(re.findall(rf"_param_{index}\b", ptx)) > 1:
            break  # used param: everything before it stays
        ptx = re.sub(rf",\s*\.param[^\n]*_param_{index}\b", "", ptx)
    return ptx

_registered = False


AOT_HEADS = 16
AOT_SEQ = 6000
AOT_BLOCK_M, AOT_BLOCK_N, AOT_WARPS, AOT_STAGES = 64, 64, 4, 3


def register_trt_plugin(aot: Optional[bool] = None) -> None:
    """Register sidon::FlexRelAttn in the global TRT plugin registry.

    aot=True (default, override with FLEXREL_AOT=0): the Triton kernel is
    compiled to PTX at build time and TRT launches it natively — no Python
    in the enqueue path.  aot=False: JIT python impl (debuggable, slower).
    """
    global _registered
    if _registered:
        return
    import os

    if aot is None:
        aot = os.environ.get("FLEXREL_AOT", "1") == "1"
    import tensorrt.plugin as trtp

    @trtp.register(_PLUGIN_OP)
    def _flexrel_desc(
        q: trtp.TensorDesc,
        k: trtp.TensorDesc,
        v: trtp.TensorDesc,
        r: trtp.TensorDesc,
    ) -> trtp.TensorDesc:
        return q.like()

    @trtp.autotune(_PLUGIN_OP)
    def _flexrel_autotune(q, k, v, r, outputs):
        return [
            trtp.AutoTuneCombination(
                "FP16, FP16, FP16, FP16, FP16", "LINEAR"
            )
        ]

    if aot:

        @trtp.aot_impl(_PLUGIN_OP)
        def _flexrel_aot(
            q, k, v, r, outputs, tactic
        ) -> Tuple[
            Union[str, bytes],
            Union[str, bytes],
            trtp.KernelLaunchParams,
            trtp.SymExprs,
        ]:
            constexprs = dict(
                H=AOT_HEADS,
                L=AOT_SEQ,
                scale=1.0 / math.sqrt(64),
                HEAD_DIM=64,
                NPOS=NUM_POS,
                NPOS_PAD=NUM_POS_PAD,
                CLAMP_LEFT=LEFT,
                CLAMP_RIGHT=RIGHT,
            )
            signature = {name: "*fp16" for name in ("Q", "K", "V", "R", "O")}
            signature.update({name: "constexpr" for name in constexprs})
            signature.update(BLOCK_M="constexpr", BLOCK_N="constexpr")
            # the TRT AOT runner cannot raise the 48KiB static smem limit,
            # so pick the first candidate config that fits
            compiled = None
            for bm, bn, warps, stages in (
                (64, 64, 4, 3),
                (64, 64, 4, 2),
                (128, 64, 8, 2),
                (64, 32, 4, 3),
            ):
                constexprs["BLOCK_M"], constexprs["BLOCK_N"] = bm, bn
                src = triton.compiler.ASTSource(
                    fn=_flexrel_fwd_kernel_aot,
                    signature=signature,
                    constexprs=dict(constexprs),
                    # TRT tensors are >=256B aligned; without these hints
                    # vectorized loads are lost (2x slower)
                    attrs={
                        (index,): [["tt.divisibility", 16]]
                        for index in range(5)
                    },
                )
                candidate = triton.compile(
                    src,
                    options={"num_warps": warps, "num_stages": stages},
                )
                print(
                    f"[flexrel aot] bm={bm} bn={bn} warps={warps} "
                    f"stages={stages} shared={candidate.metadata.shared}",
                    flush=True,
                )
                if candidate.metadata.shared <= 48 * 1024:
                    compiled = candidate
                    block_m, block_warps = bm, warps
                    break
            if compiled is None:
                raise RuntimeError("no AOT config fits 48KiB shared memory")
            launch = trtp.KernelLaunchParams()
            launch.grid_x = (AOT_SEQ + block_m - 1) // block_m
            launch.grid_y = q.shape_expr[0] * AOT_HEADS
            launch.block_x = block_warps * 32
            launch.shared_mem = compiled.metadata.shared
            return (
                compiled.metadata.name,
                _aot_blob(compiled),
                launch,
                trtp.SymIntExprs(0),
            )

        _registered = True
        return

    @trtp.impl(_PLUGIN_OP)
    def _flexrel_impl(q, k, v, r, outputs, stream):
        ext = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(ext):
            qt = torch.as_tensor(q, device="cuda")
            kt = torch.as_tensor(k, device="cuda")
            vt = torch.as_tensor(v, device="cuda")
            rt = torch.as_tensor(r, device="cuda")
            ot = torch.as_tensor(outputs[0], device="cuda")
            if DEBUG_PRINT:
                print(
                    "[flexrel impl]",
                    "backend", IMPL_BACKEND, "stream", stream,
                    "q", qt.shape, qt.stride(), qt.dtype, hex(qt.data_ptr()),
                    "k", kt.shape, kt.stride(), hex(kt.data_ptr()),
                    "v", vt.shape, vt.stride(), hex(vt.data_ptr()),
                    "r", rt.shape, rt.stride(), rt.dtype, hex(rt.data_ptr()),
                    "o", ot.shape, ot.stride(), ot.dtype, hex(ot.data_ptr()),
                    flush=True,
                )
            if IMPL_BACKEND == "torch":
                torch_core_chunked(qt, kt, vt, rt, ot)
            else:
                flexrel_attention(qt, kt, vt, rt, out=ot)

    _registered = True


class FlexRelAttnOp(torch.autograd.Function):
    """ONNX-exportable custom op mapping to the TRT plugin (trace-only)."""

    @staticmethod
    def forward(ctx, q, k, v, r):  # noqa: ARG004 - shapes only, TRT runs the real kernel
        return torch.zeros_like(q)

    @staticmethod
    def symbolic(g, q, k, v, r):
        return g.op(
            "sidon::FlexRelAttn",
            q, k, v, r,
            plugin_namespace_s="sidon",
            plugin_version_s="1",
        )


# ---- QR variant: bias table precomputed by the graph (v2) -------------------
#
# The Q @ R^T projection is moved OUT of the kernel into a plain MatMul node
# that TensorRT executes itself; the plugin receives QR [B, L, H, 73] and
# q/k/v/out stay in [B, L, H, D] layout (no transposes around the plugin).
# This removes the R smem staging + NPOS_PAD dot from the kernel, freeing
# shared memory for larger tiles under the TRT AOT 48KiB cap.

_PLUGIN_OP_QR = "sidon::FlexRelAttnQR"

# (BLOCK_M, BLOCK_N, num_warps, num_stages) forced for the AOT build; None
# selects the first candidate that fits 48KiB.
AOT_QR_CONFIG = None
_QR_CANDIDATES = (
    (128, 64, 8, 2),
    (64, 64, 4, 2),
    (128, 32, 8, 2),
    (64, 32, 4, 3),
)


@triton.jit
def _flexrel_qr_body(
    Q, K, V, QR, O,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_rb, stride_rh, stride_rm,
    stride_ob, stride_oh, stride_om,
    H, L, scale,
    HEAD_DIM: tl.constexpr,
    NPOS: tl.constexpr,
    CLAMP_LEFT: tl.constexpr,
    CLAMP_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_valid = offs_m < L

    q_base = Q + b * stride_qb + h * stride_qh
    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=m_valid[:, None],
        other=0.0,
    )

    qr_base = QR + b * stride_rb + h * stride_rh
    qr_left = tl.load(
        qr_base + offs_m * stride_rm, mask=m_valid, other=0.0
    ).to(tl.float32) * scale
    qr_right = tl.load(
        qr_base + offs_m * stride_rm + (NPOS - 1), mask=m_valid, other=0.0
    ).to(tl.float32) * scale

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    m_lo = pid_m * BLOCK_M
    m_hi = m_lo + BLOCK_M - 1

    n_left = tl.maximum(
        (m_lo - CLAMP_LEFT - BLOCK_N + 1 + BLOCK_N) // BLOCK_N, 0
    ) * BLOCK_N
    n_right = ((m_hi + CLAMP_RIGHT + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    n_full = (L // BLOCK_N) * BLOCK_N
    band_end = tl.minimum(n_right, L)
    n_tail = tl.maximum(n_right, n_full)

    for n0 in tl.range(0, n_left, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        k_tile = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :]
        )
        s = tl.dot(q, tl.trans(k_tile)) * scale + qr_left[:, None]
        v_tile = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :]
        )
        m_i, l_i, acc = _softmax_update(s, m_i, l_i, acc, v_tile)

    for n0 in tl.range(n_left, band_end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_valid = offs_n < L
        k_tile = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :],
            mask=n_valid[:, None],
            other=0.0,
        )
        rel = (
            tl.minimum(
                tl.maximum(offs_n[None, :] - offs_m[:, None], -CLAMP_LEFT),
                CLAMP_RIGHT,
            )
            + CLAMP_LEFT
        )
        bias = tl.load(
            qr_base + offs_m[:, None] * stride_rm + rel,
            mask=m_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        s = tl.dot(q, tl.trans(k_tile)) * scale + bias * scale
        s = tl.where(n_valid[None, :], s, float("-inf"))
        v_tile = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :],
            mask=n_valid[:, None],
            other=0.0,
        )
        m_i, l_i, acc = _softmax_update(s, m_i, l_i, acc, v_tile)

    for n0 in tl.range(n_right, n_full, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        k_tile = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :]
        )
        s = tl.dot(q, tl.trans(k_tile)) * scale + qr_right[:, None]
        v_tile = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :]
        )
        m_i, l_i, acc = _softmax_update(s, m_i, l_i, acc, v_tile)

    for n0 in tl.range(n_tail, L, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_valid = offs_n < L
        k_tile = tl.load(
            k_base + offs_n[:, None] * stride_kn + offs_d[None, :],
            mask=n_valid[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k_tile)) * scale + qr_right[:, None]
        s = tl.where(n_valid[None, :], s, float("-inf"))
        v_tile = tl.load(
            v_base + offs_n[:, None] * stride_vn + offs_d[None, :],
            mask=n_valid[:, None],
            other=0.0,
        )
        m_i, l_i, acc = _softmax_update(s, m_i, l_i, acc, v_tile)

    acc = acc / l_i[:, None]
    o_base = O + b * stride_ob + h * stride_oh
    tl.store(
        o_base + offs_m[:, None] * stride_om + offs_d[None, :],
        acc.to(O.dtype.element_ty),
        mask=m_valid[:, None],
    )


@triton.autotune(configs=_configs(), key=["B", "H", "L"])
@triton.jit
def _flexrel_qr_kernel(
    Q, K, V, QR, O,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_rb, stride_rh, stride_rm,
    stride_ob, stride_oh, stride_om,
    B, H, L, scale,
    HEAD_DIM: tl.constexpr,
    NPOS: tl.constexpr,
    CLAMP_LEFT: tl.constexpr,
    CLAMP_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    _flexrel_qr_body(
        Q, K, V, QR, O,
        stride_qb, stride_qh, stride_qm,
        stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn,
        stride_rb, stride_rh, stride_rm,
        stride_ob, stride_oh, stride_om,
        H, L, scale,
        HEAD_DIM, NPOS, CLAMP_LEFT, CLAMP_RIGHT, BLOCK_M, BLOCK_N,
    )


@triton.jit
def _flexrel_qr_kernel_aot(
    Q, K, V, QR, O,
    H: tl.constexpr,
    L: tl.constexpr,
    scale: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NPOS: tl.constexpr,
    CLAMP_LEFT: tl.constexpr,
    CLAMP_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # [B, L, H, D] contiguous layout (no transposes in the graph):
    #   q/k/v/o strides: b = L*H*D, l = H*D, h = D
    #   qr [B, L, H, NPOS]: b = L*H*NPOS, l = H*NPOS, h = NPOS
    _flexrel_qr_body(
        Q, K, V, QR, O,
        L * H * HEAD_DIM, HEAD_DIM, H * HEAD_DIM,
        L * H * HEAD_DIM, HEAD_DIM, H * HEAD_DIM,
        L * H * HEAD_DIM, HEAD_DIM, H * HEAD_DIM,
        L * H * NPOS, NPOS, H * NPOS,
        L * H * HEAD_DIM, HEAD_DIM, H * HEAD_DIM,
        H, L, scale,
        HEAD_DIM, NPOS, CLAMP_LEFT, CLAMP_RIGHT, BLOCK_M, BLOCK_N,
    )


def flexrel_attention_qr(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    heads_last: bool = True,
) -> torch.Tensor:
    """q/k/v/out [B, L, H, D] (heads_last) or [B, H, L, D]; qr matches."""
    if heads_last:
        batch, length, heads, dim = q.shape
        bi, hi, mi = 0, 2, 1
    else:
        batch, heads, length, dim = q.shape
        bi, hi, mi = 0, 1, 2
    assert dim == 64 and qr.shape[-1] == NUM_POS
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == qr.stride(-1) == 1
    if out is None:
        out = torch.empty_like(q, memory_format=torch.contiguous_format)
    scale = 1.0 / math.sqrt(dim)
    grid = lambda meta: (triton.cdiv(length, meta["BLOCK_M"]), batch * heads)
    _flexrel_qr_kernel[grid](
        q, k, v, qr, out,
        q.stride(bi), q.stride(hi), q.stride(mi),
        k.stride(bi), k.stride(hi), k.stride(mi),
        v.stride(bi), v.stride(hi), v.stride(mi),
        qr.stride(bi), qr.stride(hi), qr.stride(mi),
        out.stride(bi), out.stride(hi), out.stride(mi),
        batch, heads, length, scale,
        HEAD_DIM=dim,
        NPOS=NUM_POS,
        CLAMP_LEFT=LEFT,
        CLAMP_RIGHT=RIGHT,
    )
    return out


_registered_qr = False


def register_trt_plugin_qr() -> None:
    """Register sidon::FlexRelAttnQR (AOT only; QR computed by the graph)."""
    global _registered_qr
    if _registered_qr:
        return
    import tensorrt.plugin as trtp

    @trtp.register(_PLUGIN_OP_QR)
    def _flexrel_qr_desc(
        q: trtp.TensorDesc,
        k: trtp.TensorDesc,
        v: trtp.TensorDesc,
        qr: trtp.TensorDesc,
    ) -> trtp.TensorDesc:
        return q.like()

    @trtp.autotune(_PLUGIN_OP_QR)
    def _flexrel_qr_autotune(q, k, v, qr, outputs):
        return [
            trtp.AutoTuneCombination(
                "FP16, FP16, FP16, FP16, FP16", "LINEAR"
            )
        ]

    @trtp.aot_impl(_PLUGIN_OP_QR)
    def _flexrel_qr_aot(
        q, k, v, qr, outputs, tactic
    ) -> Tuple[
        Union[str, bytes],
        Union[str, bytes],
        trtp.KernelLaunchParams,
        trtp.SymExprs,
    ]:
        constexprs = dict(
            H=AOT_HEADS,
            L=AOT_SEQ,
            scale=1.0 / math.sqrt(64),
            HEAD_DIM=64,
            NPOS=NUM_POS,
            CLAMP_LEFT=LEFT,
            CLAMP_RIGHT=RIGHT,
        )
        signature = {name: "*fp16" for name in ("Q", "K", "V", "QR", "O")}
        signature.update({name: "constexpr" for name in constexprs})
        signature.update(BLOCK_M="constexpr", BLOCK_N="constexpr")
        candidates = (
            (AOT_QR_CONFIG,) if AOT_QR_CONFIG is not None else _QR_CANDIDATES
        )
        compiled = None
        for bm, bn, warps, stages in candidates:
            constexprs["BLOCK_M"], constexprs["BLOCK_N"] = bm, bn
            src = triton.compiler.ASTSource(
                fn=_flexrel_qr_kernel_aot,
                signature=signature,
                constexprs=dict(constexprs),
                attrs={
                    (index,): [["tt.divisibility", 16]] for index in range(5)
                },
            )
            candidate = triton.compile(
                src, options={"num_warps": warps, "num_stages": stages}
            )
            print(
                f"[flexrel qr aot] bm={bm} bn={bn} warps={warps} "
                f"stages={stages} shared={candidate.metadata.shared}",
                flush=True,
            )
            if candidate.metadata.shared <= 48 * 1024:
                compiled = candidate
                block_m, block_warps = bm, warps
                break
        if compiled is None:
            raise RuntimeError(
                f"no QR AOT config fits 48KiB smem (tried {candidates})"
            )
        launch = trtp.KernelLaunchParams()
        launch.grid_x = (AOT_SEQ + block_m - 1) // block_m
        launch.grid_y = q.shape_expr[0] * AOT_HEADS
        launch.block_x = block_warps * 32
        launch.shared_mem = compiled.metadata.shared
        return (
            compiled.metadata.name,
            _aot_blob(compiled),
            launch,
            trtp.SymIntExprs(0),
        )

    _registered_qr = True


class FlexRelAttnQROp(torch.autograd.Function):
    """ONNX custom op for the QR-variant plugin (trace-only)."""

    @staticmethod
    def forward(ctx, q, k, v, qr):  # noqa: ARG004
        return torch.zeros_like(q)

    @staticmethod
    def symbolic(g, q, k, v, qr):
        return g.op(
            "sidon::FlexRelAttnQR",
            q, k, v, qr,
            plugin_namespace_s="sidon",
            plugin_version_s="1",
        )


# ---- dynamic-length QR variant --------------------------------------------
#
# Same math as sidon::FlexRelAttnQR but L is a runtime kernel argument
# (passed by TRT via AOT extra_args from the input shape) and the grid is
# symbolic in both B and L, so one engine serves any sequence length.

_PLUGIN_OP_QRD = "sidon::FlexRelAttnQRD"
_registered_qrd = False


@triton.jit
def _flexrel_qr_kernel_aot_dyn(
    Q, K, V, QR, L, O,
    H: tl.constexpr,
    scale: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NPOS: tl.constexpr,
    CLAMP_LEFT: tl.constexpr,
    CLAMP_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # [B, L, H, D] contiguous; only the batch stride depends on runtime L
    _flexrel_qr_body(
        Q, K, V, QR, O,
        L * H * HEAD_DIM, HEAD_DIM, H * HEAD_DIM,
        L * H * HEAD_DIM, HEAD_DIM, H * HEAD_DIM,
        L * H * HEAD_DIM, HEAD_DIM, H * HEAD_DIM,
        L * H * NPOS, NPOS, H * NPOS,
        L * H * HEAD_DIM, HEAD_DIM, H * HEAD_DIM,
        H, L, scale,
        HEAD_DIM, NPOS, CLAMP_LEFT, CLAMP_RIGHT, BLOCK_M, BLOCK_N,
    )


def register_trt_plugin_qr_dyn() -> None:
    """Register sidon::FlexRelAttnQRD (AOT, dynamic L and B)."""
    global _registered_qrd
    if _registered_qrd:
        return
    import tensorrt.plugin as trtp

    @trtp.register(_PLUGIN_OP_QRD)
    def _flexrel_qrd_desc(
        q: trtp.TensorDesc,
        k: trtp.TensorDesc,
        v: trtp.TensorDesc,
        qr: trtp.TensorDesc,
    ) -> trtp.TensorDesc:
        return q.like()

    @trtp.autotune(_PLUGIN_OP_QRD)
    def _flexrel_qrd_autotune(q, k, v, qr, outputs):
        return [
            trtp.AutoTuneCombination(
                "BF16|FP16, BF16|FP16, BF16|FP16, BF16|FP16, BF16|FP16",
                "LINEAR",
            )
        ]

    @trtp.aot_impl(_PLUGIN_OP_QRD)
    def _flexrel_qrd_aot(
        q, k, v, qr, outputs, tactic
    ) -> Tuple[
        Union[str, bytes],
        Union[str, bytes],
        trtp.KernelLaunchParams,
        trtp.SymExprs,
    ]:
        constexprs = dict(
            H=AOT_HEADS,
            scale=1.0 / math.sqrt(64),
            HEAD_DIM=64,
            NPOS=NUM_POS,
            CLAMP_LEFT=LEFT,
            CLAMP_RIGHT=RIGHT,
        )
        import tensorrt as trt

        element = "*bf16" if q.dtype == trt.bfloat16 else "*fp16"
        signature = {
            "Q": element, "K": element, "V": element, "QR": element,
            "L": "i32", "O": element,
        }
        signature.update({name: "constexpr" for name in constexprs})
        signature.update(BLOCK_M="constexpr", BLOCK_N="constexpr")
        candidates = (
            (AOT_QR_CONFIG,) if AOT_QR_CONFIG is not None else _QR_CANDIDATES
        )
        compiled = None
        for bm, bn, warps, stages in candidates:
            constexprs["BLOCK_M"], constexprs["BLOCK_N"] = bm, bn
            src = triton.compiler.ASTSource(
                fn=_flexrel_qr_kernel_aot_dyn,
                signature=signature,
                constexprs=dict(constexprs),
                attrs={
                    (index,): [["tt.divisibility", 16]]
                    for index in (0, 1, 2, 3, 5)
                },
            )
            candidate = triton.compile(
                src, options={"num_warps": warps, "num_stages": stages}
            )
            print(
                f"[flexrel qrd aot] bm={bm} bn={bn} warps={warps} "
                f"stages={stages} shared={candidate.metadata.shared}",
                flush=True,
            )
            if candidate.metadata.shared <= 48 * 1024:
                compiled = candidate
                block_m, block_warps = bm, warps
                break
        if compiled is None:
            raise RuntimeError(
                f"no QRD AOT config fits 48KiB smem (tried {candidates})"
            )
        seq = q.shape_expr[1]
        launch = trtp.KernelLaunchParams()
        launch.grid_x = trtp.cdiv(seq, block_m)
        launch.grid_y = q.shape_expr[0] * AOT_HEADS
        launch.block_x = block_warps * 32
        launch.shared_mem = compiled.metadata.shared
        extra_args = trtp.SymIntExprs(1)
        extra_args[0] = trtp.SymInt32(seq)
        return (
            compiled.metadata.name,
            _aot_blob(compiled),
            launch,
            extra_args,
        )

    _registered_qrd = True


class FlexRelAttnQRDynOp(torch.autograd.Function):
    """ONNX custom op for the dynamic-length QR plugin (trace-only)."""

    @staticmethod
    def forward(ctx, q, k, v, qr):  # noqa: ARG004
        return torch.zeros_like(q)

    @staticmethod
    def symbolic(g, q, k, v, qr):
        return g.op(
            "sidon::FlexRelAttnQRD",
            q, k, v, qr,
            plugin_namespace_s="sidon",
            plugin_version_s="1",
        )
