# FastDialogueSidon — Triton backend

TensorRT-free inference for the FastDialogueSidon speech-separation pipeline
(raw 16 kHz dialogue in, two separated 24 kHz speaker tracks out), written as
Triton kernels around cuBLAS GEMMs. Same weights (the
[higurashi-ai/FastDialogueSidon](https://huggingface.co/higurashi-ai/FastDialogueSidon)
HF bundle) and the same runtime interface as the
[TensorRT backend](../trt/README.md). Part of
[FastDialogueSidon](../README.md). There is no engine build step: it runs on
any CUDA GPU that PyTorch + Triton support. All commands below run from this
`triton/` directory.

## Results (H100 SXM, batch 1, CUDA graphs)

Default precision: fp16 GEMM operands / activations, fp32 residual streams,
norms, softmax and accumulation (`dtype=torch.bfloat16` / `--dtype bf16`
is available and runs at the same speed).

| window | encoder | DiT | decoder | **chain** | TRT chain (default) | speed-up |
|---|---|---|---|---|---|---|
| 15 s | 2.25 ms | 0.66 ms | 0.79 ms | **3.71 ms** | 5.72 ms | 1.54x |
| 30 s | 3.68 ms | 1.09 ms | 1.28 ms | **6.08 ms** | 8.62 ms | 1.42x |
| 60 s | 7.22 ms | 2.09 ms | 2.33 ms | **11.7 ms** | 15.0 ms | 1.28x |
| 120 s | 15.8 ms | 4.81 ms | 4.46 ms | **25.3 ms** (RTF 2.1e-4) | 32.3 ms | 1.27x |

Both columns from one run (`scripts/bench.py --trt-engine-dir`, 2026-10-26);
the TRT column is its default build (fp16 encoder + DiT, bf16 decoder) from
[`../trt`](../trt/README.md).
Batch 4 at 120 s: ~23 ms per window (+7 % throughput). Device memory: ~2.2 GB
for batch 1 (CUDA context, cuBLAS workspace and the shared graph pool included).

**Where the speed-up comes from** (20 s window, H100, batch 1) — both the
single DiT step and the lightweight decoder matter. Against an optimized
implementation of the DialogueSidon architecture (bf16, torch.compile + CUDA
graphs), going from 30 steps to one step gives ~2.4x; the lightweight decoder
and fused kernels give another ~6x.

| 20 s window | encoder | DiT | decoder | total |
|---|---|---|---|---|
| DialogueSidon architecture, 30 steps (optimized) | 8.7 ms | 38.4 ms | 15.8 ms (DAC) | 63.0 ms |
| DialogueSidon architecture, 1 step (optimized) | 8.7 ms | 1.5 ms | 15.8 ms (DAC) | 26.0 ms |
| FastDialogueSidon (Triton fp16) | 2.6 ms | 0.75 ms | 0.94 ms | **4.3 ms** |

**Accuracy** — SNR vs the fp32 torch reference with the same input and noise
(CallFriend 120 s x 5 for the separated audio; one 120 s window for the
stage columns — CallFriend ja for the unmarked rows):

| | conditioning | latents | separated audio (5 windows) |
|---|---|---|---|
| **Triton fp16** (default) | 60.0 dB | 57.7 dB | **30.6-35.7 dB** |
| Triton bf16 | 46.6 dB† | 42.5 dB† | 15.8-18.3 dB |
| TRT default (fp16 enc+DiT, bf16 dec) | 52.6 dB | 49.0 dB | 16.7-18.8 dB |
| TRT all-bf16 | 38.8 dB* | 26.0 dB* | 9.9-12.9 dB |

† CallFriend de window; \* a different (non-CallFriend) 120 s input, so
compare those stage numbers only roughly. An all-fp16 TRT build is not usable: its decoder
emits NaN for 7-38 % of the samples. Separation failures (any 10 s segment
whose speaker-assignment margin drops below 0.5) over 5 windows x 20 noise
seeds: 0/100 for every working configuration (`tools/seed_sweep.py`); the
noise seed barely matters for this 1-step CTM (margin sd 0.01-0.05).

The chained audio SNR is lower than the stage numbers because the decoder
amplifies small latent differences. (Example: batch 2 vs batch 1 differs
only by GEMM rounding in the encoder, yet the final audio of the two runs
agrees far less; the DiT and decoder are bitwise identical across batch
sizes.)

## What runs where

All activations are token-major `[rows, C]`. Residual streams are fp32; GEMM
operands are fp16 (default) or bf16 with fp32 accumulation (cuBLAS);
LayerNorm / softmax / RoPE / depthwise convs run in fp32 registers. Keeping
every accumulation and residual in fp32 is what makes fp16 safe here (no
overflow) while giving ~15 dB more accuracy than bf16.

- **Encoder** (13-layer w2v-BERT conformer): fp32 fbank front-end (torch, no
  TF32) -> per layer the residual GEMMs accumulate straight into the fp32
  stream (`addmm(out_dtype=fp32, out=x)`, cuBLAS beta=1; their biases are
  carried as a running vector into the next norm), `res_norm` (bias +
  LayerNorm, optionally two in a row) feeds the next GEMM, the
  `relative_key` attention (below), `glu` and `causal_dw_norm`
  (k=31 causal depthwise conv + LayerNorm + SiLU in one kernel).
- **DiT** (CTM one step): speaker heads + `cond_proj` + the constant time
  embedding are folded into one projection; adaLN modulation fused into the
  norm kernels; `rope_split` for q/k; attention via cuDNN SDPA.
- **Decoder** (SAN-WaveNeXt2 (WaveFitNeXt), 2 iterations): `convnext_dw_norm` (k=7 depthwise
  conv + LayerNorm with the previous block's residual folded into the taps),
  cuBLASLt bias+GELU epilogue, layer-scale folded into pwconv2,
  `linear2 . linear1` folded, STFT as one GEMM, the k=3 spec projection as
  three row-shifted GEMMs accumulated in place.

Exact weight folds are computed in fp64 at load time (`model.py`).

### Encoder attention with the relative_key bias

Outside the 71-column diagonal band the bias is a per-row constant (the clamp
saturates), so for windows >= 85 s (`attn_impl="auto"`) it is split exactly:
the left region `j - i <= -64` is a plain causal attention of
`(Q[64:], K[:L-64], V[:L-64])` and the right region `j - i >= 8` one of the
time-reversed `(Q[:L-8], K[8:], V[8:])`, both run by cuDNN with their
log-sum-exp; `_band_merge_kernel` computes the band (bias Q.R^T gathered from
an L1-resident scratch tile) and merges the three parts in log space with the
row constants. 0.48 ms per layer at 120 s vs 0.58 ms for the single fused
Triton kernel (`flexrel_attention`, FA2-class ~270 TFLOPS, clamp-saturated
tiles bias-free with a running-max shift), which stays the default below
85 s (fixed launch costs) and on GPUs without cuDNN SDPA.

## Usage

```bash
cd triton
pip install -e .                   # package: fds_triton (or: export PYTHONPATH=$PWD/src)
source configs/env.local.sh        # copy of configs/env.example.sh
python scripts/run.py --bundle /path/to/weights \
  --wav dialogue.wav --output-dir out/           # any length <= 120 s
python scripts/bench.py --bundle /path/to/weights --seconds 15,120
python scripts/verify.py --bundle /path/to/weights --wav dialogue.wav
python tests/test_kernels.py                     # kernel unit tests (GPU)
```

```python
from fds_triton.runtime import TritonPipeline, load_waveform
pipe = TritonPipeline("/path/to/weights")   # fp16, bucket_seconds=10
audio = pipe.separate(load_waveform("dialogue.wav"))      # [2, samples] 24 kHz
```

`TritonPipeline` keeps fixed max-size buffers and one CUDA graph per
(bucketed length, batch); the first call at a new length autotunes and captures
(a few seconds), later calls replay. `separate_batch` serves same-bucket
windows when built with `max_batch > 1`; `stream_files` overlaps file I/O
with compute for 120 s windows.

Requirements: torch (with Triton), torchaudio, transformers, safetensors,
numpy. Tested with torch 2.8 / Triton 3.4 / CUDA 12.8 on H100. Longer inputs
than 120 s must be chunked by the caller.

## Layout

```
src/fds_triton/
  kernels.py     Triton kernels (norms, GLU, depthwise convs, FlexRel attention, RoPE)
  model.py       weight re-layout / folds + encoder, DiT, decoder forwards
  runtime.py     TritonPipeline (buffers, CUDA graphs, bucketing, batching, streaming)
  gemm.py        Triton GEMM with fused activation (benchmark reference, unused)
  reference/     fp32 reference modules + bundle loader (copied from ../trt)
scripts/         run / bench / verify
tools/           microbench.py (kernel variants), seed_sweep.py, run/compare_paperbench.py
tests/           GPU kernel unit tests (pytest or plain python)
```

## Things measured and rejected (H100)

- Own Triton GEMM with SiLU/erf-GELU epilogue: 234-362 TFLOPS vs cuBLAS
  470-580 — slower than cuBLAS + a separate activation pass.
- Parking the Q.R^T tile in global scratch inside the fused kernel: no gain
  there (the band is a small part of it).
- `warp_specialize` on the attention loops: fails to compile / no gain
  (Triton 3.4, Hopper).
- Two-pass channel-tiled causal depthwise conv: 1.3x slower than the fused
  full-row kernel.
- Persistent TMA Triton GEMM with bias+SiLU epilogue (tutorial-09 style):
  373 TFLOPS, still slower than cuBLAS + a separate SiLU (0.135 vs 0.124 ms).
- Autotuning the depthwise kernels: the pick depended on the first length
  run and BT>=16 / 4 warps spills (3-16x slower at 120 s) -> fixed BT=8
  (smaller when the grid would not cover the SMs).
- Accumulating the decoder residual in the GEMM: no net gain (the fp32 C
  read/write costs what the norm kernel saves). In the encoder it is
  speed-neutral but +4 dB conditioning SNR, so it is the default.
- Exact erf GELU (`--gelu-erf`): +0.6 ms at 120 s with no accuracy benefit
  (tanh epilogue measured equal or better SNR vs the erf reference).
- Register `tl.gather` for the band bias: the band kernel ran 0.13 ms instead
  of 0.08 ms (pointer gather from scratch).
