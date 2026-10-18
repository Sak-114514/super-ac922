# P1: Restore opt-in PLE page-batched GET_ROWS

Status: restored in the isolated `standard-opt-20261014/src` copy only.
No shared source, GPU code, server runtime, or model inference was changed.

## Source evidence

The prior experiment header was verified before copying:

- `hardware-next-20261014/src/ggml/src/ggml-cpu/ple-row-cache-api.h`
  SHA256: `d065ad8f5f02bfa35f805295a902962c68fb99160b56fef8352991c2a390b0cc`
- It contains the fixed `max_batch_rows = 64` and tested
  `row_cache::read_many` implementation.
- `hardware-next-20261014/notes/build-completed.json` records run
  `20261014T101113.159568Z`; the CPU `ops.cpp` and
  `ple-row-cache-api.cpp` compile objects both returned 0, with zero CUDA
  objects affected. That record also contains an unrelated
  `llama-context.cpp` compile error from its broader build; this P1 did not
  rely on that full-server build or any model output.

An audit found the older batch API code was present but the real
`ggml_cpu_ple_cached_rows` wrapper still used scalar reads and had no
`GGML_CPU_PLE_PAGE_BATCH` switch. This change connects that existing API to
the real CPU `GGML_OP_GET_ROWS` dispatch.

## Behavior

- `GGML_CPU_PLE_PAGE_BATCH=1` opts into the page-batched row-cache path and
  enables the cache even when `GGML_CPU_PLE_ROW_CACHE=0`.
- With the page-batch gate off, `GGML_CPU_PLE_ROW_CACHE=1` retains scalar
  cache reads. With both variables unset or zero, the helper returns false
  and the stock CPU GET_ROWS implementation runs.
- Each worker uses `read_many` when its assigned PLE rows are at least 8.
  Smaller worker chunks use the scalar row-cache path and emit one
  `ple-page-batch: scalar bypass` message per process.
- The batch callback dequantizes each packed IQ4_NL row into its original
  output row using `dst->nb[1]`. Row indices are asserted in range before
  dispatch.
- Worker completion is synchronized before the single worker logs the batch
  counters. The existing shared-cache workspace lifetime barriers remain.
- The first and then each
  `GGML_CPU_PLE_PAGE_BATCH_LOG_PERIOD` batch calls log counters; the default
  period is 16.

The full 64-row `read_many` implementation was copied byte-for-byte from the
verified prior header. `ple-row-cache-api.cpp` is unchanged; it continues to
provide the process-wide out-of-line `acquire` entry point.

## Build scope

The standard-opt dry-run plan includes unrelated pre-existing source edits and
would compile 16 objects / relink 8 targets. To keep this task inside P1, the
fixture build compiled only `ggml-cpu/ops.cpp` and
`ggml-cpu/ple-row-cache-api.cpp`, then linked a private CPU DSO from those
objects plus the existing stable CPU objects. It wrote the test DSO and new
objects under `fixtures/P1-build`. It did not write into `build-reuse`, the
shared source tree, or any CUDA build object.

## CPU GET_ROWS fixture

The fixture used a private sparse temporary IQ4_NL table with shape
`[160, 745655]` (67,108,950 logical bytes, only 24 rows written, about 4 MiB
allocated). It invokes real `ggml_get_rows` through the CPU backend scheduler
and compares output against direct IQ4_NL `to_float`. It does not load a
model. The three 32-row outputs from stock GET_ROWS, scalar row-cache, and
page-batched GET_ROWS are byte-identical:

- 5,120/5,120 F32 values exact per graph execution
- maximum absolute error 0
- no nonfinite values or first mismatch
- all output dumps have SHA256
  `1067566ff461a9393196651d485f88b968e4bb87314c71cd6ad1ee3ccacc0949`

Additional path checks:

- Four CPU workers, 32 rows: page-batch stats report `workers=4/4`.
- Four CPU workers, 20 rows: per-worker chunks are 5 rows and the scalar-bypass
  message appears.
- One CPU worker, 128 rows: `read_many` crosses two fixed 64-row chunks.
- Existing 256 KiB temporary-file page-batch API fixture passes cold direct
  grouping, duplicate rows, row order, repeated hits, mmap/fallback and
  16-worker cache eviction. The fixture's fadvise/madvise calls apply only to
  its own `/tmp` file.

All test paths use CPU only. No global cache eviction, large file scan, full
build, model load, or GPU work was performed.

## Patch and artifacts

- `P1-final.patch` applies the two PLE source-file changes relative to the
  `source-before` baseline; `patch --dry-run -p1` passes.
- `P1-build-cpu.py` and `P1-build-cpu.json` record the CPU-only compile and
  link commands.
- `P1-incremental-build-plan.json` records why the broader 16-object
  incremental plan was not run.
- `P1-source-manifest.json` records before/after hashes for the PLE files.
- `P1-run-commands.txt` gives fixture build and run commands.
- Raw logs, F32 dumps, sparse fixture inputs and both fixture sources remain
  in `fixtures/P1-*`; no final result was inferred from model semantic
  behavior.
