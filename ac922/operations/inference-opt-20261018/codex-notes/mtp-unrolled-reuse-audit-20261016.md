# V13 unrolled MTP graph reuse audit

Date: 2026-10-16

## Scope

Read-only review of the V13 four-request P4 wave. This note records the observed unrolled-graph reuse counter and the source that emits it. It does not claim a speed gain or make a new runtime change.

## Log evidence

Run:

`fixtures/dual-context-v13-prep-20261016/results/p4-d3-20261015T200425Z/`

The `real8k-p4-2x2` wave spans `runtime.log` lines 64944–109264. In that slice:

- `MTP unrolled graph active` appears **151** times.
- `rebuilt=0` appears **146** times.
- `rebuilt=1` appears **5** times.
- Thus the logged reuse rate for this wave is **146/151 = 96.69%**.

The five rebuild records are at `runtime.log` lines 103004, 103041, 105937, 106006, and 106022. Their logged `n_kv` values are respectively 16640, 16640, 16640, 16896, and 16896. The evidence shows occasional graph-family rebuilds, not a rebuild on every unrolled invocation.

## Source locations

Source root: `/home/debian/ac922-monster/experiments/standard-opt-20261014/src`

- `src/llama-context.cpp:2721` begins `process_ubatch_mtp_unrolled`.
- `src/llama-context.cpp:2843-2857` obtains the prior result, builds graph parameters with that result, and attaches the unrolled step snapshots.
- `src/llama-context.cpp:2859` is the exact reuse predicate: `gf_res_prev_active == res && res->can_reuse(gparams)` (when graph reuse is not disabled).
- `src/llama-context.cpp:2864-2883` resets/builds/allocates only on a reuse miss; `2884-2889` updates parameters on a hit.
- `src/llama-context.cpp:2903-2905` emits the per-invocation `rebuilt` field used for the count above.
- `common/speculative.cpp:2415-2419` states that the fixed-prefix unrolled graph contains target-hidden catch-up; `2448-2450` submits that path through one `llama_decode(ctx_dft, batch)`. The separate per-depth fallback loop is at `2452-2464` when unrolling is not selected.
- `src/llama-context.cpp:2691-2695` defines `llama_step_timing` as timing for `process_ubatch`; `3287` increments its `n_build`; `3371-3376` prints its periodic aggregate. `process_ubatch_mtp_unrolled` is a separate function and emits its own exact per-call counter at line 2903.

## Interpretation boundary

The V13 `step-timing rebuilt` aggregate counts graph builds inside ordinary `process_ubatch`; it is not the per-call unrolled MTP rebuild counter. For this P4 wave, use the `MTP unrolled graph active rebuilt=` records when assessing unrolled graph-family reuse. The 96.69% result falsifies the specific premise that this unrolled graph rebuilt on every draft step. It does not explain the remaining five rebuilds or rule out other scheduling costs.
