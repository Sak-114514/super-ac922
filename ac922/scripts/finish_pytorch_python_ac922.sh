#!/bin/sh
set -eu

# Finish the pinned Torch 2.10 Python wheel from existing RAM-built C++/CUDA
# objects. Run only after the first BUILD_PYTHON=OFF build has stopped.
workroot=${AC922_WORK_ROOT:-/home/debian/ac922-vllm}
root=${AC922_BUILD_ROOT:-$workroot/ram-build/fs}
source_dir=$root/pytorch
pid_file=$root/pytorch-wheel.pid

case "$(findmnt -n -o SOURCE -T "$source_dir")" in
    /dev/loop*) ;;
    *) echo 'PyTorch source is not on the RAM-backed loop filesystem' >&2; exit 1 ;;
esac
if [ -r "$pid_file" ] &&
    ps -p "$(cat "$pid_file")" -o args= 2>/dev/null |
    grep -F 'setup.py bdist_wheel' >/dev/null; then
    echo 'The first PyTorch build is still running' >&2
    exit 1
fi

headers=$root/toolchains/python-dev-sysroot/usr/include/python3.11
test -s "$headers/Python.h"
test -s "$headers/powerpc64le-linux-gnu/python3.11/pyconfig.h"

export PYTORCH_SOURCE_DIR=$source_dir
export TMPDIR=$root/tmp
alt_pid_file=$root/third-party-test/1cat-torch212-build.pid
if [ -r "$alt_pid_file" ] &&
    ps -p "$(cat "$alt_pid_file")" -o args= 2>/dev/null |
    grep -F 'pip wheel' >/dev/null; then
    default_jobs=2
else
    default_jobs=3
fi
export MAX_JOBS=${MAX_JOBS:-$default_jobs}
echo "Building Python bindings with MAX_JOBS=$MAX_JOBS"
mkdir -p "$TMPDIR"
touch "$root/pytorch-python-build-start"
echo $$ > "$root/pytorch-python-wheel.pid"
sh "$workroot/build_pytorch_ac922.sh" wheel

grep -Eq '^BUILD_PYTHON:BOOL=(ON|True|TRUE|1)$' "$source_dir/build/CMakeCache.txt"
wheel=$(find "$source_dir/dist" -maxdepth 1 -type f \
    -name 'torch-2.10.0*.whl' -newer "$root/pytorch-python-build-start" \
    -print -quit)
test -n "$wheel"
unzip -Z1 "$wheel" | grep -E '^torch/_C[^/]*\.so$' >/dev/null
unzip -Z1 "$wheel" | grep -Fx 'torch/lib/libtorch_python.so' >/dev/null
sha256sum "$wheel"
echo "Verified Python/CUDA Torch wheel: $wheel"
