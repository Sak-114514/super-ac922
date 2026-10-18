# Graph reuse rejection diagnostics — 2026-10-15

## Effective behavior

With `LLAMA_GRAPH_RESULT_DEBUG=2`, the diagnostic records graph reuse rejections for `LLM_GRAPH_TYPE_DEFAULT` and `LLM_GRAPH_TYPE_DECODER`. Encoder and decoder-MTP graphs are excluded. Output is one-line, starts with `graph-reuse-reject`, and is capped at 128 records per process. Detailed legacy graph-result messages at debug level 2 follow the same graph-type filter.

The reports include old/new batch scalars (`tokens`, `seq_tokens`, `n_seqs`, `n_seqs_unq`, `equal_seqs`, `n_outputs`), graph/scheduler/memory pointer identities, MTP and sampler pointer differences, and failing Hybrid/QSA input classes with cached shapes/state and current expected values where safe.

`llm_graph_context` now retains `params.gtype`. QSA input snapshot collection checks that actual graph type as well as the debug level, so MTP does not collect target diagnostics.

Diagnostics never dereference old `mctx` values or old transient ubatch arrays. They report pointer identities and scalar copies; QSA old state comes from debug snapshots captured when input buffers are refreshed. Sampler output and sequence-ID array contents are marked as not dereferenced.

No reuse guard, graph topology, tensor shape, cache mask, or execution behavior was changed.

## Audit correction

The first implementation filtered for `LLM_GRAPH_TYPE_DECODER` only. Root's V5 run reported zero reason lines. The server's ordinary target graph is actually `LLM_GRAPH_TYPE_DEFAULT`; `DECODER` is the encoder-decoder graph path. The predicate now accepts DEFAULT and DECODER and excludes ENCODER and DECODER_MTP. This fixes the filter; it has not yet been exercised in a server run.

## Changed files

- `src/src/llama-graph.cpp`: applies the shared graph-type predicate to record emission and debug filtering; copies `params.gtype` into graph contexts.
- `src/src/llama-graph.h`: defines the shared target predicate and stores graph type in `llm_graph_context`.
- `src/src/models/qwen4exp.cpp`: passes actual graph type to QSA input objects and gates diagnostic snapshots on it.
- `fixtures/graph-reuse-debug-policy.cpp`: CPU fixture using the production enum and predicate.

## Verification

- POWER9 syntax-only compilation passed for `src/src/llama-graph.cpp` and `src/src/models/qwen4exp.cpp` using the experiment source and existing generated-header layout.
- CPU fixture compiled and ran: `graph-reuse-debug-policy PASS`. It asserts the actual enum values (DEFAULT=0, ENCODER=1, DECODER=2, DECODER_MTP=3) and verifies the predicate accepts DEFAULT/DECODER and rejects ENCODER/DECODER_MTP.
- Earlier byte-level guard comparison showed the two Hybrid and three QSA `can_reuse` predicates unchanged. The graph-type filter correction does not touch them.
- The QSA source still has a pre-existing signedness warning in untouched `qwen4exp_qsa_mblk_fast` (`n_tokens` versus gather limit).
- No engine build, server launch, GPU work, or model request was run for this correction. V5 runtime coverage is specifically absent; root can include this filter fix in the next combined V6 build/run.

## Backups and rollback

Full pre-instrumentation copies are in `notes/backups/graph-reuse-diag-20261015/`. Restore these three files to remove the instrumentation completely:

```sh
cp notes/backups/graph-reuse-diag-20261015/llama-graph.cpp src/src/llama-graph.cpp
cp notes/backups/graph-reuse-diag-20261015/llama-graph.h src/src/llama-graph.h
cp notes/backups/graph-reuse-diag-20261015/qwen4exp.cpp src/src/models/qwen4exp.cpp
```

Pre-correction copies of the instrumented sources are in `notes/backups/graph-reuse-gtype-fix-20261015/` if only the graph-type correction needs inspection.

## Current source hashes

- `src/src/llama-graph.cpp` — `5d8f3679936dbfc370bf7037b1da79ac495d720b8b58d184c9d43e851767863d`
- `src/src/llama-graph.h` — `a454a66c8d1715029997188fcbc1b9b2bb2f10ca680be8d628c12d551eed192b`
- `src/src/models/qwen4exp.cpp` — `b097e573dc9de15e73f341d7b8c531fe7179b40e54fb7d62ebaab238e7270561`
- `fixtures/graph-reuse-debug-policy.cpp` — `46b0e2d3f686da920898ab033f76d31eead7f7a9f44bc917aed80fabf7f672d9`
