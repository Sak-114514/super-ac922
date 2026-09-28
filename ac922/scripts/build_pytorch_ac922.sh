#!/bin/sh
set -eu

# Run on AC922 after the pinned v2.10.0 source submodules and Python build
# requirements are complete.  `configure` performs CMake checks only.
mode=${1:-configure}
case "$mode" in configure|wheel) ;; *) echo 'usage: build_pytorch_ac922.sh configure|wheel' >&2; exit 2 ;; esac

root=${AC922_WORK_ROOT:-/home/debian/ac922-vllm}
source_dir=${PYTORCH_SOURCE_DIR:-$root/pytorch}
venv=$root/venv
python_dev_include=${PYTHON_DEV_INCLUDE:-$root/ram-build/fs/toolchains/python-dev-sysroot/usr/include/python3.11}
[ -s "$python_dev_include/Python.h" ] || {
    echo "Python development headers missing: $python_dev_include/Python.h" >&2
    exit 1
}
[ -s "$python_dev_include/powerpc64le-linux-gnu/python3.11/pyconfig.h" ] || {
    echo "POWER Python configuration header missing under $python_dev_include" >&2
    exit 1
}
[ "$(cat /sys/block/sda/device/queue_depth)" = 1 ] || {
    echo 'SATA NCQ is active; refusing large build' >&2; exit 1;
}
[ "$(cat /sys/devices/system/memory/auto_online_blocks)" = offline ]
systemctl is-active --quiet ac922-memory-guard.service
[ "$(git -C "$source_dir" rev-parse HEAD)" = 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0 ]
if git -C "$source_dir" submodule status | grep -E '^[+-]' >/dev/null; then
    echo 'PyTorch submodules are incomplete or at the wrong commit' >&2
    exit 1
fi
"$venv/bin/python" -c 'import numpy, yaml, typing_extensions, sympy, jinja2'

export PATH=/usr/local/cuda/bin:"$venv/bin":$PATH
export CUDA_HOME=/usr/local/cuda
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
export TMPDIR=${TMPDIR:-$source_dir/.tmp}
wheel_stage=${PYTORCH_WHEEL_STAGE:-$root/ram-build/wheel-stage}
mkdir -p "$TMPDIR" "$wheel_stage"
[ "$(findmnt -n -o FSTYPE -T "$wheel_stage")" = tmpfs ] || {
    echo "Wheel staging must use tmpfs: $wheel_stage" >&2
    exit 1
}
export LD_LIBRARY_PATH=/usr/local/nccl/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export NCCL_INCLUDE_DIR=/usr/local/nccl/include
export NCCL_LIB_DIR=/usr/local/nccl/lib
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS=${MAX_JOBS:-2}
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
export BLAS=OpenBLAS
export USE_CUDA=1 USE_DISTRIBUTED=1 USE_GLOO=1 USE_NCCL=1 USE_SYSTEM_NCCL=1
export USE_CUDNN=0 USE_CUSPARSELT=0 USE_CUDSS=0 USE_CUFILE=0 USE_MAGMA=0
export USE_XPU=0 USE_ROCM=0 USE_NVSHMEM=0 USE_MPI=0 USE_TENSORPIPE=0
export USE_FBGEMM=0 USE_FBGEMM_GENAI=0 USE_KINETO=0 USE_MKLDNN=0
export USE_NNPACK=0 USE_PYTORCH_QNNPACK=0 USE_XNNPACK=0
export USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 BUILD_TEST=0
export PYTORCH_BUILD_VERSION=2.10.0 PYTORCH_BUILD_NUMBER=1
export BUILD_LIBTORCH_WHL=0 BUILD_PYTHON_ONLY=0
export CMAKE_INCLUDE_PATH=$python_dev_include
# Setuptools compiles torch/csrc/stub.c separately from CMake. It needs the
# same RAM-extracted Python headers, including the POWER pyconfig.h directory.
export CFLAGS="-I$python_dev_include -I$python_dev_include/powerpc64le-linux-gnu/python3.11 ${CFLAGS:-}"
# Refresh the old BUILD_PYTHON=OFF cache once. Keep the now-correct cache on
# incremental retries so existing C++/CUDA objects are not invalidated.
if grep -Eq '^BUILD_PYTHON:BOOL=(ON|True|TRUE|1)$' "$source_dir/build/CMakeCache.txt" 2>/dev/null; then
    unset CMAKE_FRESH
else
    export CMAKE_FRESH=1
fi

if [ "$mode" = configure ]; then export CMAKE_ONLY=1; fi
cd "$source_dir"
exec numactl --cpunodebind=0 --membind=0 "$venv/bin/python" \
    setup.py bdist_wheel --bdist-dir "$wheel_stage"
