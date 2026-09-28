#!/bin/sh
set -eu

# Rebuild the ppc64le extensions needed by the Qwen/1Cat-vLLM text path.
# Run on AC922 after copying toolchains/ into the RAM-backed build root.
root=${AC922_BUILD_ROOT:-/home/debian/ac922-vllm/ram-build/fs}
venv=$root/venv
artifacts=$root/toolchains
headers=$artifacts/python-dev-sysroot/usr/include

cd "$artifacts"
sha256sum -c SHA256SUMS >/dev/null
if [ ! -s "$headers/python3.11/Python.h" ]; then
    mkdir -p "$artifacts/python-dev-sysroot"
    dpkg-deb -x "$artifacts/libpython3.11-dev_3.11.2-6+deb12u8_ppc64el.deb" \
        "$artifacts/python-dev-sysroot"
fi
if [ ! -e "$headers/python3.11/powerpc64le-linux-gnu" ]; then
    ln -s ../powerpc64le-linux-gnu "$headers/python3.11/powerpc64le-linux-gnu"
fi
test -s "$headers/python3.11/powerpc64le-linux-gnu/python3.11/pyconfig.h"

export CFLAGS="-O2 -I$headers/python3.11 -I$headers"
export CARGO_HOME=$artifacts/cargo
export RUSTUP_HOME=$artifacts/rustup
export CARGO_BUILD_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1
export TMPDIR=$root/tmp PIP_CACHE_DIR=$root/pip-cache
export PATH=$venv/bin:$CARGO_HOME/bin:$PATH
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
cd "$root"

for source in \
    psutil-7.2.2.tar.gz \
    msgspec-0.21.1.tar.gz \
    uvloop-0.22.1.tar.gz \
    safetensors-0.8.0.tar.gz \
    tokenizers-0.22.2.tar.gz; do
    case "$source" in
        psutil-*) wheel='psutil-7.2.2-*.whl' ;;
        msgspec-*) wheel='msgspec-0.21.1-*.whl' ;;
        uvloop-*) wheel='uvloop-0.22.1-*.whl' ;;
        safetensors-*) wheel='safetensors-0.8.0-*.whl' ;;
        tokenizers-*) wheel='tokenizers-0.22.2-*.whl' ;;
    esac
    if find "$artifacts" -maxdepth 1 -name "$wheel" -print -quit | grep -q .; then
        echo "Already built: $source"
        continue
    fi
    export CARGO_TARGET_DIR=$artifacts/cargo-target-${source%%-*}
    nice -n 10 "$venv/bin/python" -m pip wheel \
        --no-build-isolation --no-deps --no-index \
        --wheel-dir "$artifacts" "$artifacts/$source"
done
