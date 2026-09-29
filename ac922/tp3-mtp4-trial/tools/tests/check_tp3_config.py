# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration-only smoke for the AC922 TP3 trial (does not load weights)."""

import os
import sys

from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.models.qwen3_5_tp3_heads import validate_tp3_head_config


def main(model: str, draft: str | None = None) -> None:
    overrides = {
        "ac922_tp3_head_replication": True,
        "text_config": {
            "num_attention_heads": 36,
            "num_key_value_heads": 6,
            "linear_num_key_heads": 18,
            "linear_num_value_heads": 54,
            "intermediate_size": 17664,
        },
    }
    speculative = (
        {
            "method": "dflash",
            "model": draft,
            "num_speculative_tokens": int(os.environ.get("AC922_DFLASH_DEPTH", "7")),
            "draft_tensor_parallel_size": 3,
            "draft_sample_method": "probabilistic",
            "kv_cache_dtype": "auto",
        }
        if draft is not None
        else None
    )
    args = EngineArgs(
        model=model,
        tensor_parallel_size=3,
        quantization="compressed-tensors",
        dtype="float16",
        max_model_len=200000,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.95,
        limit_mm_per_prompt={"image": 0, "video": 0},
        hf_overrides=overrides,
        speculative_config=speculative,
    )
    config = args.create_engine_config()
    assert config.parallel_config.tensor_parallel_size == 3
    assert config.model_config.hf_config.ac922_tp3_head_replication
    validate_tp3_head_config(config.model_config.hf_config)
    print("PASS TP3 config: 36 Q / 6 KV / 18 GDN key / 54 GDN value")
    if draft is not None:
        assert config.speculative_config is not None
        draft_config = config.speculative_config.draft_model_config.hf_config
        assert draft_config.ac922_tp3_draft_replication
        assert (draft_config.num_attention_heads, draft_config.num_key_value_heads) == (
            36,
            9,
        )
        print(f"PASS speculative config: {config.speculative_config.method}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
