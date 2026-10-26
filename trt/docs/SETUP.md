# Setup (fresh machine)

Target: CUDA 12.x GPU (engines are arch-specific — build on the GPU you deploy on),
Python 3.10-3.12, driver >= 550.

## 1. Environment

```bash
uv venv ~/venvs/fastdialoguesidon --python 3.12
source ~/venvs/fastdialoguesidon/bin/activate
uv pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu129
uv pip install -e /path/to/FastDialogueSidon/trt
```

`tensorrt-cu12==10.16.1.11` comes in via the package deps (pin < 11: TRT 11
drops the FP16 builder flags and measured ~5% slower on this model).
torch 2.10 ships triton 3.6, which the FlexRel AOT plugin supports
(3.4/3.5 also work; hidden trailing scratch params are stripped from the
AOT PTX — see pitfall 9).

## 2. Get the weights

```bash
hf download higurashi-ai/FastDialogueSidon --local-dir weights/
```

(≈1.73 GB fp32.)

## 3. Runtime environment variables

```bash
cp configs/env.example.sh configs/env.local.sh   # edit FDS_PYTHON
source configs/env.local.sh
```

`NVIDIA_TF32_OVERRIDE=0` is NOT optional — see pitfalls below. Engines must
be built and run under the same setting.

(The env file sets `HF_HUB_OFFLINE=1`, so download the weights first.)

## 4. Build engines + run

```bash
python scripts/build_engines.py --bundle weights/ \
  --engine-dir artifacts/$(hostname)-trt10 --onnx-dir /tmp/fds-onnx \
  [--wav some_120s_16k_mono.wav]
python scripts/bench_e2e.py --engine-dir artifacts/$(hostname)-trt10
python scripts/run.py --engine-dir artifacts/$(hostname)-trt10 \
  --wav input.wav --output-dir out/
```

## Known pitfalls (all enforced or documented in code)

1. **NVIDIA_TF32_OVERRIDE=0 required.** TensorRT runs "fp32" GEMMs as TF32
   even in strongly-typed networks; the resulting noise makes quiet-bin
   log-mel chaotic and per-bin normalization amplifies it to unit variance
   (conditioning drops from ~50 dB to ~17 dB). Asserted in
   `engines.build_wave_engine` / `runtime.TRTPipeline`.
2. **Engines must be built and run under the same TF32 setting** — TRT
   refuses to create execution contexts otherwise.
3. **Register the FlexRel plugin before ANY TensorRT builder/runtime
   work.** Importing `tensorrt.plugin` after engines exist segfaults.
4. **Enqueue on a non-default stream.** With the legacy default stream TRT
   runs layers on an internal stream and races the plugin kernel
   (silently wrong outputs). `TRTPipeline` owns a dedicated stream.
5. **AOT plugin configs are capped at 48 KiB shared memory** (the TRT AOT
   runner cannot raise the static limit); `flexrel_triton` picks the first
   fitting tile config automatically.
6. Weakly-typed per-layer precision constraints are ignored inside myelin
   fusions — that is why the encoder engine is strongly typed with fp32
   front-end + fp16 encoder dtypes carried in the ONNX itself.
7. Point `TRITON_CACHE_DIR` at node-local storage (NFS lock contention).
8. **sm_120 (Blackwell RTX): Myelin has no BF16 `ConvTranspose` tactic**
   (TRT 10.16/11.2) — the decoder engine build fails. The exportable
   decoder expresses the upsampler as GEMM + phase interleave instead
   (`ExportableWaveFitDecoder._upsample_gemm`), equivalent to fp32
   roundoff (guarded in `build_engines.py` against the true torch
   decoder, threshold 1e-5 vs measured ~1.3e-6).
9. **Blackwell: a GEMM+bias epilogue feeding the FlexRel AOT plugin
   segfaults `create_execution_context()`** (inside libnvinfer;
   independent of PTX/cubin blob and of TRT 10.16 vs 11.2). The qkv
   projections are exported as bias-folded GEMMs on sm_100 and newer
   GPUs (ones column appended to the activations, bias row appended to
   the weight) so no Add feeds the plugin. sm_90 and older GPUs keep the
   faster native GEMM+bias graph. Related knobs, both default-on-safe:
   `FLEXREL_AOT_FORMAT`
   (ptx|cubin) and `FLEXREL_STRIP_HIDDEN` (drops triton >= 3.4's hidden
   trailing scratch params from the PTX so the kernel signature matches
   the declared plugin signature).
