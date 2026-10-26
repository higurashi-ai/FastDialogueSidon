# FastDialogueSidon runtime environment — source this before any script.
# Copy to env.local.sh and adjust paths for your site.

# REQUIRED: TF32 must be off. The fbank front-end runs true-fp32 inside the
# strongly-typed encoder engine; with TF32, quiet-bin log-mel turns into
# noise and per-bin normalization amplifies it to unit variance.
# TensorRT also refuses to run engines built under a different setting.
export NVIDIA_TF32_OVERRIDE=0

# Python env with this package installed (see docs/SETUP.md)
export FDS_PYTHON="${FDS_PYTHON:-$HOME/venvs/fastdialoguesidon/bin/python}"

# tensorrt-cu12 pip wheels put libnvinfer under site-packages/tensorrt_libs
TRT_LIBS="$("${FDS_PYTHON}" - <<'PY'
import pathlib, tensorrt_libs
print(pathlib.Path(tensorrt_libs.__file__).parent)
PY
)"
export LD_LIBRARY_PATH="${TRT_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Triton kernel cache off shared filesystems (NFS quota / lock contention)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/$USER/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/$USER/torchinductor}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
