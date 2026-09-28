#!/bin/sh
set -eu

# Isolated Rust toolchain for ppc64le Python extensions used by vLLM.
root=${AC922_TOOLCHAIN_ROOT:-/home/debian/ac922-vllm}
export RUSTUP_HOME=$root/rustup
export CARGO_HOME=$root/cargo
url=https://static.rust-lang.org/rustup/dist/powerpc64le-unknown-linux-gnu/rustup-init
installer=$root/rustup-init-powerpc64le
checksum=$root/rustup-init-powerpc64le.sha256

if [ ! -s "$installer" ] || [ ! -s "$checksum" ]; then
    curl --fail --location --retry 3 --silent --show-error -o "$installer" "$url"
    curl --fail --location --retry 3 --silent --show-error -o "$checksum" "$url.sha256"
fi
expected=$(cut -d ' ' -f 1 "$checksum")
actual=$(sha256sum "$installer" | cut -d ' ' -f 1)
[ "$actual" = "$expected" ] || {
    echo 'Rustup installer checksum mismatch' >&2
    exit 1
}
chmod 0755 "$installer"
"$installer" -y --profile minimal --default-toolchain 1.95.0 --no-modify-path
"$CARGO_HOME/bin/rustc" -Vv
"$CARGO_HOME/bin/cargo" -V
