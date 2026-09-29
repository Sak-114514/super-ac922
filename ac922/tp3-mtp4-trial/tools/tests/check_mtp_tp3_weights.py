# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stream the local BF16 MTP checkpoint through AC922 TP3 head replication."""

import sys

import torch
from safetensors import safe_open

from vllm.model_executor.models.qwen3_5_tp3_heads import replicate_tp3_mtp_weights


def main(path: str) -> None:
    expected = {
        "mtp.layers.0.self_attn.q_proj.weight": (18432, 5120),
        "mtp.layers.0.self_attn.k_proj.weight": (1536, 5120),
        "mtp.layers.0.self_attn.v_proj.weight": (1536, 5120),
        "mtp.layers.0.self_attn.o_proj.weight": (5120, 9216),
        "mtp.layers.0.mlp.gate_proj.weight": (17664, 5120),
        "mtp.layers.0.mlp.up_proj.weight": (17664, 5120),
        "mtp.layers.0.mlp.down_proj.weight": (5120, 17664),
    }
    seen: set[str] = set()
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        weights = ((name, checkpoint.get_tensor(name)) for name in checkpoint)
        for name, expanded in replicate_tp3_mtp_weights(weights):
            if name not in expected:
                continue
            assert expanded.dtype == torch.bfloat16
            assert tuple(expanded.shape) == expected[name], name
            original = checkpoint.get_tensor(name)
            if name.endswith("o_proj.weight"):
                assert torch.equal(expanded[:, :3072] * 2, original[:, :3072])
                assert torch.equal(expanded[:, 3072:6144], original[:, 3072:])
                assert torch.equal(expanded[:, 6144:], expanded[:, :3072])
            elif name.endswith("q_proj.weight"):
                assert torch.equal(expanded[12288:], original[:6144])
            elif name.endswith(("k_proj.weight", "v_proj.weight")):
                assert torch.equal(expanded[1024:], original[:512])
            elif name.endswith(("gate_proj.weight", "up_proj.weight")):
                assert not torch.count_nonzero(expanded[17408:])
            elif name.endswith("down_proj.weight"):
                assert not torch.count_nonzero(expanded[:, 17408:])
            seen.add(name)
    assert seen == set(expected), seen
    print("PASS MTP BF16 checkpoint: seven TP3 tensors and half-output invariant")


if __name__ == "__main__":
    main(sys.argv[1])
