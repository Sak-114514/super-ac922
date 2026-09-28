"""Compile 1Cat's actual SM70 GDN extension against an isolated Torch 2.12.

This is a compatibility probe, not a vLLM service build. Run on AC922 using
the separate RAM virtualenv and the original pinned 1Cat source file.
"""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

ROOT = Path("/home/debian/ac922-vllm/ram-build/fs")
SOURCE = (
    ROOT
    / "1Cat-vLLM/flash_qla/ops/gated_delta_rule/chunk/sm70/csrc/gdn_forward.cu"
)
HEADERS = ROOT / "toolchains/python-dev-sysroot/usr/include"
BUILD = ROOT / "third-party-test/extension-build"

assert torch.__version__.startswith("2.12.0a1"), torch.__version__
assert torch.version.cuda == "12.4", torch.version.cuda
assert SOURCE.is_file()
assert (HEADERS / "python3.11/Python.h").is_file()
BUILD.mkdir(parents=True, exist_ok=True)

extension = load(
    name="gdn_torch212_probe",
    sources=[str(SOURCE)],
    build_directory=str(BUILD),
    extra_include_paths=[str(HEADERS / "python3.11"), str(HEADERS)],
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O2", "-gencode=arch=compute_70,code=sm_70"],
    with_cuda=True,
    verbose=True,
)
assert callable(extension.gdn_forward)
assert callable(extension.resolve_column_groups_per_block)
print("PROBE_OK", extension.__file__)
print("COLUMN_GROUPS", extension.resolve_column_groups_per_block(16, 1, 1))
