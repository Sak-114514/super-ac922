# Exclusion-only GPU argmax metadata candidate (2026-10-15)

## Source correction and current behavior

`llm_graph_context::build_argmax()` builds device argmax when `LLAMA_GPU_ARGMAX` is enabled and `samplers.empty()`; the earlier note that inverted this guard was incorrect. In tensor-split server mode, backend sampling is rejected and CPU `common_sampler` remains external, so this raw argmax graph is available.

With exclusion bias, `common_sampler_sample()` skips `llama_get_argmax_ith()` and calls `llama_get_logits_ith()`. Context flushes pending logits to host before returning the row, then the CPU helper scans the entire vocabulary and avoids building/sorting a `llama_token_data` candidate vector. With no bias/suppress list, the sampler uses the raw device token ID. `--ignore-eos` contributes EOG `-INFINITY` entries to active bias parameters.

Existing reuse:

- `llama-graph.cpp:4243–4266`: raw `GGML_OP_ARGMAX` / `GGML_OP_ARGMAX_SPLIT` graph construction.
- `argmax.cu:9–64, 77–125`: CUDA raw max/index reductions and tie-to-lowest-token behavior.
- `ggml-backend-meta.cpp:1203, 1787`: `ARGMAX_SPLIT` partial slots/rank/offset handling; no shape-width assertion in the meta split contract.
- `llama-context.cpp:3254–3299, 3680–3704`: lazy logits copy and per-row argmax ID extraction.
- `llm_graph_input_sampling::set_input()` and `can_reuse()` show the per-sequence dynamic graph-input pattern, but the exclusion-only candidate does not need to upload IDs.
- `llama_sampler_logit_bias_backend_apply()` uses `ggml_set_rows`, but it is backend sampling, which tensor split rejects; it is not directly reusable for this external CPU sampler path.

## Narrow candidate

Keep the current raw GPU best ID. Extend existing `GGML_OP_ARGMAX` and `GGML_OP_ARGMAX_SPLIT` with optional metadata output, using op parameters and expanded output tensors rather than adding an op enum or public `llama.h` API:

- Single device: output raw token ID plus `unique_and_valid`.
- Tensor split: each device emits `(max, token_id, second_max, invalid)` in its existing partial candidate slots. AllReduce combines those slots. A tiny final reduction yields raw token ID plus `unique_and_valid`, detecting both local and cross-rank ties.
- Keep raw token feedback separate and unchanged for MTP. A private metadata getter is used only by the exclusion-only CPU sampler branch.
- If metadata is missing/invalid, the raw winner is tied, or the raw winner appears in the active `-INFINITY` bias IDs / vocab suppress list, use the existing full-logit CPU scan and full-chain tie fallback. If the unique raw winner is not excluded, return it without copying the full logits row.
- Existing guards for grammar, finite bias, penalties, temperature, sampler order, reasoning budget, and backend sampling remain in force. This candidate does not add graph ID inputs, exclusion-mask kernels, or per-slot device copies.

Provisional CUDA-only source scope (no implementation authorized yet): `src/ggml/src/ggml-impl.h`, `src/ggml/src/ggml.c`, `src/ggml/src/ggml-cuda/argmax.cu`, `src/src/llama-graph.h`, `src/src/llama-graph.cpp`, `src/src/llama-context.h`, `src/src/llama-context.cpp`, and `src/common/sampling.cpp`. The meta-backend split contract appears generic for candidate-slot width. The context must gate metadata mode to a CUDA output backend; if CPU-backend execution of metadata mode is required, CPU dispatch adds two more source files and the scope must be reconsidered. Root approval is required before any of these core edits.

## CPU regression fixture

`fixtures/greedy-exclusion-argmax-fixture.cpp` tests the contract using existing CPU helpers: all-`-INF` bias admission, duplicate bias/suppress IDs, unique raw winner admission only when unexcluded, excluded raw winner fallback, tie fallback through the full sampler reference, all-masked fallback, and NaN/+Inf fallback.

Build/run result:

```text
PASS raw_argmax_admission=7 exclusion_ids_dedup_semantics=1 excluded_top_fallback=1 tie_fallback=1 all_masked=1 nonfinite_fallback=1
```

Artifacts: `fixtures/greedy-exclusion-argmax-fixture-20261015`, `notes/greedy-exclusion-argmax-fixture-{build,run}-20261015.log`.

The fixture is CPU-only. No GPU, model, engine build, or core-source modification was performed for this candidate. It validates the admission/fallback contract only; no GPU metadata path or performance gain has been measured.
