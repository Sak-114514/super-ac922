# Greedy suppression audit budget gate plan (2026-10-15)

## Finding

`common_sampler_sample()` currently allocates one of the global 32 shadow-audit slots for every eligible greedy fast-path call. It only later computes `has_exclusion_bias`; consequently ordinary unmasked greedy calls can consume the budget before an `ignore_eos` request with actual EOG `-INFINITY` logit biases is audited.

## Narrow change

1. Add a small testable header helper that takes the existing atomic counter and an explicit `has_exclusion_bias` boolean. Calls without exclusion bias return without incrementing; eligible bias calls preserve the current atomic sequence and 32-slot limit.
2. Pass the existing `has_exclusion_bias` value from `sampling.cpp` into the slot allocator. Keep the greedy shortcut, shadow clone, failure fallback, and token selection unchanged.
3. Extend `greedy-suppression-fixture.cpp` with an interleaved budget test: many no-bias attempts consume no slots; the first 32 bias attempts are admitted; later bias attempts are denied and ordinary attempts still do not change the counter.

## Verification boundary

No engine build, GPU, model, or runtime test is planned for this task. Review the patch and fixture source statically only. Do not change `common/speculative.cpp`, the CUDA path, or fast-path promotion state.

## Verification update

Root authorized a CPU-only standalone fixture compile and run (still no engine build, GPU, or model). The existing 161 full-chain comparison cases must remain intact; the new interleaved audit-budget case is added to that same fixture.

## Run outcome

The standalone CPU sampler fixture was compiled and run as authorized. All 161 full-chain comparisons passed; the interleaved audit budget test confirmed unbiased calls consume no slots and the first 32 bias-eligible calls do. See `notes/greedy-audit-budget-results-20261015.md`.
