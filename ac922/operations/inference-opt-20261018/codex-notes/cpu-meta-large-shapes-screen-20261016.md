# Large CPU-to-Meta upload screen

Date: 2026-10-16  
Scope: inspect the existing CPU-to-Meta adapter and qualify 16/20/32/33 MiB copies once. No engine edits or model run.

## V36 evidence and stride gate

The V36 clean runtime shows repeated CPU-to-Meta copies falling back at the 10 MiB policy:

- 16 MiB `attn_inp_kq_mask (view)` fallbacks: `fixtures/dual-context-v36-combo-20261016/results/root-p4-v36-v85-onebank4-ep6-d3-16k-20261016t091525z-remote/runtime.log:36922, 70615`.
- 20 MiB `mtp_h_state-48` fallbacks: same log, lines 38801 and 71194.
- Summary counters: mask owner `copy_policy_limit count=480` at line 74983; MTP owner `count=80` at line 74957.

The callback tests `ggml_nbytes(source) > max_copy_bytes` before it sets `req.source_contiguous = ggml_is_contiguous(source)` and `same_layout` (`src/ggml/src/ggml-backend-meta.cpp:5115-5134`). Thus the 16 MiB cap-fallback log does not prove the source view is contiguous. The ring rejects genuinely noncontiguous source/destination tensors or a layout mismatch (`ggml-meta-cpu-upload-private.h:280-293`). Contiguous VIEW/RESHAPE tensors can pass; a noncontiguous view cannot.

The CPU-only view check established that the chosen 16 MiB mask-shaped VIEW and 20 MiB MTP-shaped RESHAPE are contiguous and layout-equal. Its intentionally strided mask-shaped view has a 33,521,664-byte physical span, is rejected by the ring as `source_not_contiguous`, and would hit the production size guard first. The saved V36 log alone does not tell us which case its named `(view)` source is.

On a decline, `ggml_backend_tensor_copy_async` synchronizes the source and destination and performs the existing synchronous copy (`src/ggml/src/ggml-backend.cpp:529-548`). The fixture verifies exact fallback output; it does not claim strided views qualify for the async adapter.

## Fixture and host checks

`fixtures/cpu-meta-large-shapes-20261016/` extends the nested Meta2 A/B/A copy fixture. It retains event reuse, source mutation/free after each API call, F32 identity values, and exact/canary readback on all six leaves. The 64 MiB process-wide budget and 32 MiB ring limit remain unchanged.

Host-only checks passed:

- `python3 fixtures/cpu-meta-large-shapes-20261016/run_cpu_tests.py` → PASS, including size boundaries and a simulated two-owner 64 MiB `pinned_budget_full` fallback.
- V80 fixture compile/link passed against manifest SHA `be87339463070c579d226cf13085c5a6b71a3f1eb60591c6178f9476f23a0b5b`; binary SHA-256 `aa913f586b146b81dd4a5e911758a63550275483842ce4e28d1cda17c92fbd5b`.
- Device-hidden metadata check passed for the contiguous mask VIEW and MTP RESHAPE and confirmed rejection of a noncontiguous view.
- Host result: `fixtures/cpu-meta-large-shapes-20261016/results/host-build-v80-20261016T101354Z-202775/host-build.log`.

## Bounded GPU run and observed outcome

One six-GPU V80 run completed: `fixtures/cpu-meta-large-shapes-20261016/results/gpu-v80-large-shapes-20261016T100423Z-201536/run.log`. Pre/post guards showed 38–41°C and at least 16,119 MiB free on each card; a follow-up query found no compute processes remaining.

Observed adapter totals:

- Owner A: 7 async copies / 108 MiB.
- Baseline owner B: 3 async copies / 30 MiB.
- Fresh B-budget owner: its 16 MiB copy was also queued asynchronously (1 / 16 MiB), so **the GPU run did not observe `pinned_budget_full`**.
- The mask VIEW, MTP RESHAPE, and 32 MiB copy were accepted. The 33 MiB copy used `copy_policy_limit` and the synchronous fallback.
- All 72 six-leaf readbacks were exact with canaries intact; all 12 sources were overwritten and freed after the call.

The run wrapper initially returned nonzero because its validator incorrectly required a B budget fallback. That was a fixture assumption: an earlier sync after each A transfer allowed the ring to reuse its arena. This is not a backend failure. The raw log is preserved. One `DESTINATION_READBACK` line was interleaved with a backend log message; the fixture's `PASS` marker was printed after all exact comparisons succeeded. The corrected validator was run against the preserved log and passed: 72 readback attempts (71 complete structured records plus one interleaved line), 12 source releases, and B classified as `budget_not_full_b_copy_queued`. The fixture now flushes each readback marker before any future run.

Root directed that no further GPU run is needed. The host-only helper already covers cross-owner `pinned_budget_full` at the fixed 64 MiB limit. No budget-forcing GPU attempt will be made. These results qualify sizes and transfer correctness, not a model-level speedup.
