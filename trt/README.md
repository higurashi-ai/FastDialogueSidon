# FastDialogueSidon — TensorRT backend

TensorRT inference for the FastDialogueSidon speech-separation pipeline:
raw 16 kHz dialogue in, two separated 24 kHz speaker tracks out.
Part of [FastDialogueSidon](../README.md); the engine-free Triton backend
with the same interface lives in [`../triton`](../triton/README.md).
All commands below run from this `trt/` directory.

- **Encoder**: 13-layer w2v-BERT encoder with a `relative_key`
  attention bias, executed by a fused Triton flash-attention TensorRT
  plugin (`sidon::FlexRelAttnQRD`, AOT PTX — no Python at enqueue). The
  SeamlessM4T fbank front-end is folded into the engine as fp32 ONNX ops,
  so the engine input is the raw waveform. fp16 engine (default).
- **Head**: CTM one-step DiT (t=999 -> s=0), fp16 strongly-typed engine
  (default; fp32 LayerNorms) with precomputed RoPE tables / time embeddings.
- **Vocoder**: SAN-WaveNeXt2 (WaveFitNeXt) latent decoder, bf16 engine with
  the iteration-2 STFT expressed as a fixed conv1d. **fp16 is not
  supported for the decoder** (it overflows to NaN).
- **Runtime**: zero-copy engine chain + per-length CUDA graphs +
  overlapped streaming (reader thread, pinned double buffers, async
  H2D/D2H, writer thread).

## GPU support

The pipeline targets any CUDA GPU with tensor cores that TensorRT 10.x
supports (Ampere / Ada / Hopper / consumer Blackwell). Two things are
GPU-specific, both handled automatically:

- **Engines are architecture-specific** — run `scripts/build_engines.py`
  once on the GPU you deploy on (a few minutes; the FlexRel plugin
  compiles its Triton kernel to PTX for that arch and auto-selects a tile
  configuration that fits the 48 KiB shared-memory limit of the TRT AOT
  runner).
- **Throughput scales with the GPU's fp16/bf16 tensor throughput.** Device
  memory is reserved at init for the engines' maximum batch profile
  (`build_engines.py --max-batch`, default 1): measured **~2.9 GB total**
  on H100 (incl. ~0.6 GB CUDA context and all bucket CUDA graphs).
  Building with a larger `--max-batch` (for `separate_batch`) grows the
  reservation by roughly ~1.4 GB per additional batch slot.

Reference measurements (120 s window, batch 1) on an H100 SXM with the
default precision (fp16 encoder + DiT, bf16 decoder) — treat these as an
upper bound and scale by your GPU's fp16/bf16 tensor TFLOPS for a rough
estimate:

| stage | ms (H100, default precision) |
|---|---|
| encoder (fbank in-engine, 13 layers) | 19.0 |
| DiT CTM 1-step | 4.7 |
| SAN-WaveNeXt2 (WaveFitNeXt) decoder | 9.6 |
| **e2e chain** | **~32.3** (RTF 2.7e-4) |
| non-TRT torch reference (`scripts/bench_torch_flex.py`) | ~81 |

Example: a GPU with ~10x less tensor throughput than H100 should land
around RTF 2-3e-3 — still hundreds of times faster than real time.
On GPUs without TensorRT, `scripts/bench_torch_flex.py` is the fastest
pure-torch configuration (bf16 CUDA-graphed flex_attention encoder +
torch.compile head/decoder) and runs anywhere torch 2.8 does.

## Precision

- **Master weights are fp32** (the HF bundle). Default engine dtypes:
  **encoder fp16, DiT fp16, decoder bf16** (`build_engines.py
  --encoder-precision / --dit-precision / --decoder-precision`; all-bf16
  is `--encoder-precision bf16 --dit-precision bf16`).
- **fp16 decoder: not supported.** An fp16 SAN-WaveNeXt2 (WaveFitNeXt) engine
  overflows: 7-38 % of the output samples are NaN on
  CallFriend 120 s (2026-10-26). `--decoder-precision` only accepts bf16.
- **Accuracy vs the fp32 torch reference** (same input and noise,
  CallFriend 120 s x 5): separated-audio SNR 16.7-18.8 dB with the default
  vs 9.9-12.9 dB all-bf16; at 120 s conditioning 52.6 dB / latents 49.0 dB.
  Separation failures over 5 samples x 20 noise seeds: 0/100 for both.
  Speed is the same (32.3 vs 32.0 ms at 120 s).
- The **fbank front-end and softmax/accumulation run fp32** inside the
  engines (strongly-typed ONNX; `NVIDIA_TF32_OVERRIDE=0` required — see
  SETUP).
- **fp8 is not supported**: its quantization error is too large for
  this pipeline (the separated-audio degradation is audible), so no fp8
  path is provided.

## Input handling & resampling

The default engine takes **16 kHz mono waveform** directly (fbank is
inside the engine; 16 kHz files skip resampling entirely).

- Fixed non-16 kHz source: build with `--input-rate 48000` etc. —
  resampling is baked into the engine (bit-exact vs torchaudio, slightly
  faster than resampling outside).
- Mixed/unknown rates: `runtime.load_waveform` resamples on the GPU
  (never on CPU — ~200x slower). `scripts/run.py` picks the right path
  automatically.

## Variable-length input

**120 s is the maximum window, not a requirement.** One engine set serves
any length from 3.2 s to 120 s: the FlexRel plugin takes the sequence
length as a runtime kernel argument, RoPE tables are sliced in-graph, and
the runtime caches one CUDA graph per encountered length. Inputs are
padded to the next 20 ms boundary; cost scales with length (attention
~O(L²), everything else ~O(L)), so shorter windows are strictly faster.
Accuracy is length-independent (validated at 15/30/60/120 s during
engine build, `--validate-seconds`).

Measurements on an RTX PRO 6000 Blackwell Server Edition (sm_120,
torch 2.10/cu129, TRT 10.16, batch 1, **all-bf16 engines**
(measured before the fp16 encoder/DiT default; the default has not been
re-validated on sm_120), CUDA graphs; peak device memory ~2.8 GB):

| window | encoder | DiT | decoder | **e2e chain** | RTF |
|---|---|---|---|---|---|
| 15 s | 7.8 ms | 1.8 ms | 2.0 ms | **10.9 ms** | 7.3e-4 |
| 30 s | 10.1 ms | 2.1 ms | 2.6 ms | **13.7 ms** | 4.6e-4 |
| 60 s | 14.1 ms | 3.5 ms | 3.8 ms | **20.7 ms** | 3.5e-4 |
| 120 s | 21.8 ms | 6.7 ms | 7.1 ms | **34.8 ms** | 2.9e-4 |

sm_120 needs two export-level rewrites (BF16
ConvTranspose -> GEMM in the decoder, bias-folded qkv GEMMs in the
encoder — see docs/SETUP.md pitfalls 8-9). Both are mathematically
equivalent reformulations of the same fp32 weights. The decoder rewrite
is used on every GPU. QKV bias folding is selected automatically for
Blackwell (sm_100 and newer), while Hopper and older GPUs retain the
faster native GEMM+bias path.

Reference measurements per window length (H100 SXM, batch 1, default
precision, CUDA graphs):

| window | encoder | DiT | decoder | **e2e chain** | RTF | peak device mem |
|---|---|---|---|---|---|---|
| 15 s | 4.1 ms | 0.8 ms | 1.8 ms | **5.7 ms** | 3.8e-4 | 2.9 GB |
| 30 s | 5.5 ms | 1.2 ms | 2.7 ms | **8.6 ms** | 2.9e-4 | 2.9 GB |
| 60 s | 9.1 ms | 2.2 ms | 4.6 ms | **15.0 ms** | 2.5e-4 | 2.9 GB |
| 120 s | 19.0 ms | 4.7 ms | 9.6 ms | **32.3 ms** | 2.7e-4 | 2.9 GB |

(all-bf16 engines: 5.3 / 8.0 / 15.0 / 32.0 ms — the fp16 encoder is
0.4-0.7 ms slower, the fp16 DiT slightly faster.)

Peak device memory (total, incl. ~0.6 GB CUDA context) is
**length-independent by design**: TensorRT contexts reserve activation
memory for the maximum profile shape (`--max-batch`, 120 s) at
initialization — the figure is the default `--max-batch 1` build.
(reproduce with `scripts/bench_e2e.py --seconds 15,30,60,120 --use-graph`;
the graph-free mode costs ~0.8 ms more per window)

**Any input length up to 120 s is fine as-is**: for efficiency, windows
are internally padded up to 10 s buckets (one cached CUDA graph per
bucket, ~120 MB total) and the output is trimmed back to the true input
length. Zero padding leaves the true-length part unaffected (validated at
15-120 s). The bucket size is configurable
(`--bucket-seconds N`; `0` disables bucketing for exact-length,
graph-free execution).

**Batching**: sequential B=1 + bucketing is the recommended mode — the
GPU is already saturated at batch 1, and it accepts arbitrary mixed
lengths (verified stable over 20k consecutive windows). **If your
windows all share the same length, build the engines with a larger
batch profile** (e.g. `build_engines.py --max-batch 8`; +~1.4 GB
reserved per batch slot) and run them batched:

```python
pipeline = TRTPipeline(engine_dir, max_batch=8)
audio = pipeline.separate_batch(waveforms)        # [B, 2, samples]
```

This trades Bx latency for ~8% more throughput.

Longer than 120 s: chunk into windows (the streaming runtime processes a
window list; chunk-overlap stitching is up to the caller).

## Quick start

See [docs/SETUP.md](docs/SETUP.md) for the full environment recipe
(plain `pip`/`uv`, no source builds). Weights live on the Hugging Face
Hub: [higurashi-ai/FastDialogueSidon](https://huggingface.co/higurashi-ai/FastDialogueSidon)
(fp32 safetensors, ≈1.73 GB).

```bash
cd trt
hf download higurashi-ai/FastDialogueSidon --local-dir weights/
pip install -e .                      # package: fastdialoguesidon
cp configs/env.example.sh configs/env.local.sh   # edit FDS_PYTHON
source configs/env.local.sh           # NVIDIA_TF32_OVERRIDE=0 etc. (REQUIRED)
python scripts/build_engines.py --bundle weights/ \
  --engine-dir artifacts/$(hostname)-trt10 --onnx-dir /tmp/fds-onnx
python scripts/run.py --engine-dir artifacts/$(hostname)-trt10 \
  --wav dialogue_16k_mono.wav --output-dir out/     # any length <= 120 s
python scripts/bench_e2e.py --engine-dir artifacts/$(hostname)-trt10 \
  --seconds 15,30,60,120
```

## Repository layout

```
src/fastdialoguesidon/
  modeling.py          slim inference-only modules + bundle loader
  flexrel_triton.py    fused rel-pos flash-attention kernel + TRT plugin
  fbank_export.py      ONNX-exportable SeamlessM4T fbank front-end
  export_onnx.py       engine export wrappers (dynamic length)
  engines.py           strongly-typed fp16/bf16 + bf16 decoder builds, TRTStage
  runtime.py           chain + per-length CUDA graphs + streaming
  wavefit_next.py      vendored WaveFitNeXt (yukara-ikemiya/wavefit-pytorch)
scripts/               build_engines / bench_e2e / run / bench_torch_flex
tools/                 run_paperbench (benchmark runner), bench_resample_inengine
```
