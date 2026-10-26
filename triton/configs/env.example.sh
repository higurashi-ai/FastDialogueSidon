# FastDialogueSidon Triton backend — runtime environment; source before any script.
# Copy to env.local.sh and adjust paths for your site.

# Python env with torch (+ triton), transformers, safetensors, torchaudio
export FDS_PYTHON="${FDS_PYTHON:-$HOME/venvs/fastdialoguesidon/bin/python}"

# Optional: TensorRT libs for the side-by-side comparison against the
# FastDialogueSidon engines (scripts/{verify,bench}.py --trt-engine-dir).
# The TRT engines also require NVIDIA_TF32_OVERRIDE=0 (harmless here: the
# Triton pipeline runs its fp32 fbank GEMMs without TF32 anyway).
export NVIDIA_TF32_OVERRIDE=0
TRT_LIBS="$("${FDS_PYTHON}" - <<'PY' 2>/dev/null || true
import pathlib, tensorrt_libs
print(pathlib.Path(tensorrt_libs.__file__).parent)
PY
)"
if [ -n "${TRT_LIBS}" ]; then
  export LD_LIBRARY_PATH="${TRT_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Triton kernel cache off shared filesystems
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$USER/triton-cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
