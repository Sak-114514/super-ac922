#!/bin/sh
set -eu

# Resume the isolated 1Cat source build from restored RAM/Ninja objects.
root=${AC922_BUILD_ROOT:-/home/debian/ac922-vllm/ram-build/fs}
source_dir=$root/third-party-test/1Cat-vLLM
venv=$root/venv-third-party
headers=$root/toolchains/python-dev-sysroot/usr/include

test -d "$source_dir"
test -s "$headers/python3.11/Python.h"
test "$(git -C "$source_dir" rev-parse HEAD)" = \
    db292f9a49318459f064075bfdbbed438b4e77a3
test "$(cat /sys/devices/system/memory/auto_online_blocks)" = offline

export PYTHONPATH=$venv/lib/python3.11/site-packages:$root/venv/lib/python3.11/site-packages
export LD_LIBRARY_PATH=$root/third-party-test/extra-libs:/usr/local/nccl/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=7.0 CMAKE_CUDA_ARCHITECTURES=70
export VLLM_TARGET_DEVICE=cuda VLLM_USE_PRECOMPILED=0
export VLLM_USE_PRECOMPILED_RUST=0 VLLM_REQUIRE_RUST_FRONTEND=0
export CMAKE_BUILD_TYPE=Release
export MAX_JOBS=${MAX_JOBS:-1} CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS:-1}
export CARGO_BUILD_JOBS=1 CMAKE_INCLUDE_PATH=$headers/python3.11
export CFLAGS="-O2 -I$headers/python3.11 -I$headers"
export CXXFLAGS=$CFLAGS
# The nested flash-attention-v100 setup.py invokes nvcc directly. CPATH also
# reaches its host compiler, which otherwise looks only in /usr/include.
export CPATH=$headers/python3.11:$headers/python3.11/powerpc64le-linux-gnu/python3.11:${CPATH:-}
export TMPDIR=$root/tmp PIP_CACHE_DIR=$root/pip-cache
export TORCH_EXTENSIONS_DIR=$root/third-party-test/torch_extensions
export CARGO_HOME=$root/toolchains/cargo RUSTUP_HOME=$root/toolchains/rustup
export PATH=/usr/local/cuda/bin:$venv/bin:$root/venv/bin:$CARGO_HOME/bin:$PATH
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$root/third-party-test/wheels"

cd "$source_dir"
exec nice -n 10 numactl --cpunodebind=0 --membind=0 "$venv/bin/python" \
    -m pip wheel --no-build-isolation --no-deps --no-index \
    --wheel-dir "$root/third-party-test/wheels" .
