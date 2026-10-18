# Greedy suppression audit-budget gate (2026-10-15)

## Change

`common_sampler_greedy_audit_take_slot` now passes the existing `has_exclusion_bias` condition to a small shared allocator. Calls without an exclusion bias return before `fetch_add`; eligible biased calls retain the original atomic 32-slot rule. The shadow audit remains observational and unchanged. The fast greedy algorithm and default-off promotion state are unchanged.

This means ordinary unmasked greedy traffic cannot exhaust the finite audit budget before an `ignore_eos` request with active EOG `-INFINITY` logit biases is seen.

## CPU fixture verification

Built and ran the standalone `greedy-suppression-fixture` against the current `runtime-perf-v9b` sampler library. It retains all 161 reference comparisons and adds interleaved budget checks: 1,000 no-bias attempts before and between biased calls consume zero slots; the first 32 biased calls receive slots 0–31; later biased calls are denied; ordinary calls still do not affect the count.

Result:

```text
PASS exact_reference_cases=161 unique_masked_scans=160 nonunique_random=0 finite_bias_rejected=5 audit_budget=32 interleaved_unbiased_consumed=0
```

This was a CPU-only fixture run. No engine build, model load, GPU work, or performance test was performed. The fixture's libllama shared object carries CUDA libraries as dependencies, but the test only calls the CPU sampler chain and does not initialize the backend.

## Fast-path route evidence

- `common_sampler_greedy_eligible()` allows exclusion biases only when `LLAMA_GREEDY_SUPPRESSION_FAST=1` and all configured bias values are negative infinity (`sampling.cpp:626–639`, `sampling-greedy-suppression.h:21–30`).
- `ignore_eos` appends the model's EOG `-INFINITY` biases to `params.sampling.logit_bias` (`common.cpp:1383–1395`), so it qualifies as an exclusion-bias audit case.
- With an exclusion bias, `common_sampler_sample()` bypasses `llama_get_argmax_ith()` and calls `llama_get_logits_ith()` followed by one CPU pass through `common_greedy_suppressed_argmax()` (`sampling.cpp:829–845`). This preserves the masking and tie check while avoiding construction and partial sorting of a full `llama_token_data` candidate vector.
- `llama_get_logits_ith()` synchronizes and flushes pending logits before returning the host logits row (`src/llama-context.cpp:5416–5429`, `3680–3691`). With GPU argmax enabled, raw logits are queued and copied only when read (`src/llama-context.cpp:3254–3271`), so the biased CPU route can incur a full vocabulary-row device-to-host copy before its scan.
- With no explicit exclusion bias and no model suppress tokens, the sampler uses `llama_get_argmax_ith()` (`sampling.cpp:831–838`). That returns the already-computed argmax ID; the context reads `argmax_ids` populated from the device result (`src/llama-context.cpp:3275–3299`, `3694–3704`) while raw logits remain deferred.

These source facts explain the two routes; they are not a speed measurement or a claim that masked greedy wins end to end.

## Artifacts

- Source: `src/common/sampling-greedy-suppression.h`, `src/common/sampling.cpp`, `fixtures/greedy-suppression-fixture.cpp`
- CPU fixture binary: `fixtures/greedy-suppression-fixture-auditbudget-20261015`
- Build log: `notes/greedy-audit-budget-fixture-build-20261015.log`
- Run log: `notes/greedy-audit-budget-fixture-run-20261015.log`
- Plan: `notes/greedy-audit-budget-plan-20261015.md`

Audit eligibility intentionally follows the existing `has_exclusion_bias = !params.logit_bias.empty()` variable. A vocabulary-only suppress list with no explicit logit bias still takes a CPU scan path, but does not consume this bias-specific audit budget.

The current V9b candidate profile keeps `LLAMA_GREEDY_SUPPRESSION_FAST=0`; this audit-budget fix does not enable or promote the fast path by itself.
