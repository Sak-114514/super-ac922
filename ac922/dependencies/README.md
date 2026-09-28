# Dependency inventory

[`WHEEL_CACHE.sha256`](WHEEL_CACHE.sha256) is a verified snapshot of 98
artifacts used or staged while building on AC922. It includes source archives,
downloaded Power wheels, locally compiled wheels, Python development headers,
and toolchain installers. It is **not** a complete `pip freeze`, and some
listed alternatives were evaluated without entering the final runtime.

## Sources

| Category | Source | Examples |
| --- | --- | --- |
| Locally compiled on AC922 | Upstream PyPI/source releases | safetensors 0.8.0, tokenizers 0.22.2, tiktoken 0.14.0, msgspec 0.21.1, uvloop 0.22.1, psutil 7.2.2 |
| IBM Power wheels | [IBM ppc64le wheel index](https://wheels.developerfirst.ibm.com/ppc64le/linux-v2026.06.0/+simple/) | llguidance 1.7.5, xgrammar 0.2.1, apache-tvm-ffi 0.1.11; earlier index builds supplied Pillow, sentencepiece, pyzmq |
| PyPI wheels | [PyPI](https://pypi.org/) | pydantic, transformers, openai-harmony, mistral-common, jsonschema and pure Python support packages |
| Build headers and tools | [Debian packages](https://packages.debian.org/bookworm/libpython3.11-dev), [Rust](https://static.rust-lang.org/) | `libpython3.11-dev` extracted into RAM, Rust 1.95, CMake, Ninja |
| Experimental Torch 2.12 | [RomanMoz release](https://github.com/RomanMoz/power-torch-cuda124/releases/tag/v2.12.0a1) | Linked by provenance only; binary not redistributed here |

The experimental Torch 2.12 import required runtime libraries from NVIDIA
cuDNN 9/CUPTI 12.4 and Debian `libomp`. These libraries are **not** in the
release bundle. CUDA and NCCL availability, `LD_LIBRARY_PATH`, Python 3.11,
ppc64le, and glibc compatibility must be checked on each host.

Use separate virtual environments for the pinned PyTorch 2.10 lane and the
experimental PowerTorch 2.12 + 1Cat lane. The latter distribution exports the
`torch` Python module but is named `power-torch-cuda124` in package metadata;
`pip check` therefore reports the fork's `torch` requirement as unsatisfied.
The source-built 1Cat wheel was installed with `--no-deps` for isolated
compatibility work; that metadata mismatch is not a claim of supported
dependency resolution.

The wheel cache omits model weights, credentials, machine logs, firmware,
proprietary NVIDIA downloads, and the downloaded third-party Torch binary.

`tiktoken 0.14.0` has no matching Power wheel in the consulted IBM index; the
local build used the PyPI source archive and RAM-installed Rust 1.95. The
[`build_tiktoken_ac922.sh`](../scripts/build_tiktoken_ac922.sh) script keeps its
Cargo build tree in the outer tmpfs to avoid exhausting ext4 loop-image
inodes. The produced wheel's SHA256 is in the release manifest.
