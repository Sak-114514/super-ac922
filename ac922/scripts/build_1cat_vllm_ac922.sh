#!/bin/sh
set -eu

# Build the pinned 1Cat-vLLM CUDA/SM70 wheel entirely in AC922's RAM image.
# Invoke only after a functional PyTorch 2.10 CUDA wheel is installed there.
root=${AC922_BUILD_ROOT:-/home/debian/ac922-vllm/ram-build/fs}
source_dir=$root/1Cat-vLLM
venv=$root/venv
headers=$root/toolchains/python-dev-sysroot/usr/include/python3.11

test -s "$headers/Python.h"
test -s "$headers/powerpc64le-linux-gnu/python3.11/pyconfig.h"
test "$(git -C "$source_dir" rev-parse HEAD)" = \
    db292f9a49318459f064075bfdbbed438b4e77a3
test "$(cat /sys/block/sda/device/queue_depth)" = 1
test "$(cat /sys/devices/system/memory/auto_online_blocks)" = offline
"$venv/bin/python" - <<'PY'
import torch
assert torch.__version__.startswith('2.10.0'), torch.__version__
assert torch.version.cuda and torch.version.cuda.startswith('12.4'), torch.version.cuda
assert torch._C.__file__.endswith('.so'), torch._C.__file__
print('PyTorch build prerequisite:', torch.__version__, torch.version.cuda)
PY

export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$venv/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nccl/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0 CMAKE_CUDA_ARCHITECTURES=70
export VLLM_TARGET_DEVICE=cuda VLLM_USE_PRECOMPILED=0
export VLLM_USE_PRECOMPILED_RUST=0 VLLM_REQUIRE_RUST_FRONTEND=0
export MAX_JOBS=${MAX_JOBS:-2} CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS:-2}
export CARGO_BUILD_JOBS=1
export CMAKE_INCLUDE_PATH=$headers
export CFLAGS="-O2 -I$headers -I${headers%/python3.11}"
export CXXFLAGS=$CFLAGS
export TMPDIR=$root/tmp PIP_CACHE_DIR=$root/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$root/wheels"

cd "$source_dir"
exec numactl --cpunodebind=0 --membind=0 "$venv/bin/python" -m pip wheel \
    --no-build-isolation --no-deps --no-index \
    --wheel-dir "$root/wheels" .
