#!/bin/sh
set -eu

# Build tiktoken 0.14.0 on AC922 while keeping Cargo's many files off the
# inode-limited ext4 loop image. The source tarball is in toolchains/.
root=${AC922_BUILD_ROOT:-/home/debian/ac922-vllm/ram-build/fs}
outer=${AC922_RAM_ROOT:-$(dirname "$root")}
source_tar=$root/toolchains/tiktoken-0.14.0.tar.gz
work=$outer/tiktoken-build
expected=231dec90efcdccf1b565a1416107736f1e09b1a08fe736ef9d6363e626d03874

test -s "$source_tar"
echo "$expected  $source_tar" | sha256sum -c -
test "$(findmnt -n -o FSTYPE -T "$outer")" = tmpfs
mkdir -p "$work" "$outer/cargo-tiktoken" "$outer/tiktoken-tmp"
tar -xzf "$source_tar" -C "$work"

export CARGO_HOME=$outer/cargo-tiktoken
export RUSTUP_HOME=$root/toolchains/rustup
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2}
export TMPDIR=$outer/tiktoken-tmp
export PATH=$root/toolchains/cargo/bin:$PATH

"$root/venv/bin/python" -m pip wheel --no-build-isolation --no-deps \
    --no-index --no-cache-dir --wheel-dir "$root/toolchains" \
    "$work/tiktoken-0.14.0"
