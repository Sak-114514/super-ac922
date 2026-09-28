"""Run one tiny 1Cat SM70 GDN kernel built against isolated Torch 2.12."""

import importlib.util
from pathlib import Path

import torch

ROOT = Path("/home/debian/ac922-vllm/ram-build/fs")
MODULE = ROOT / "third-party-test/extension-build/gdn_torch212_probe.so"
assert MODULE.is_file()
assert torch.__version__.startswith("2.12.0a1"), torch.__version__
assert torch.cuda.is_available()

spec = importlib.util.spec_from_file_location("gdn_torch212_probe", MODULE)
assert spec and spec.loader
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)

torch.manual_seed(922)
device = "cuda:0"
q = torch.randn(1, 16, 1, 128, dtype=torch.float16, device=device) * 0.01
k = torch.randn(1, 16, 1, 128, dtype=torch.float16, device=device) * 0.01
v = torch.randn(1, 16, 1, 128, dtype=torch.float16, device=device) * 0.01
gate = torch.zeros(1, 16, 1, dtype=torch.float16, device=device)
beta = torch.full((1, 16, 1), 0.5, dtype=torch.float16, device=device)

outputs = extension.gdn_forward(q, k, v, gate, beta, None, 128**-0.5, True, False)
torch.cuda.synchronize()
assert outputs
for tensor in outputs:
    assert tensor.is_cuda and torch.isfinite(tensor).all().item()
print("SM70_GDN_OK", [tuple(tensor.shape) for tensor in outputs])
