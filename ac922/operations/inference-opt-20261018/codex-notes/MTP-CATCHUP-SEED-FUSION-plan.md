# MTP catch-up and seed fusion plan

Status: implementation plan recorded before source edits. Default remains disabled.

1. Add `LLAMA_MTP_FUSE_CATCHUP_SEED`, opt-in only. Leave `LLAMA_MTP_GPU_CHAIN_SINGLE` unchanged.
2. Before the existing catch-up decode, snapshot the original catch-up plus seed batch. Fuse only when every active lane has the same positive catch-up length, the per-lane rows are contiguous through the seed, only the seed row requests output, all rows fit the draft ubatch, and chain-layout validation passes. Unequal or unsupported layouts continue through the existing separate catch-up decode.
3. Teach the active chain validator to accept that uniform multi-row shape at chain step zero. Later steps remain one row per lane. On a fused chain failure, remove draft KV from each lane's `first_pos` and replay the snapshot through the generic path. On begin rejection, replay without having decoded the fused rows.
4. Add CPU layout/rollback regressions for catch-up lengths 0..8, unequal prefixes, output flags, sequence mapping, first-position rollback and snapshot replay. Run only fixture and syntax checks; do not build the engine or use GPU/model.

Risks: The graph path must consume all prefix rows while producing one output per lane at the seed. The static fixture and syntax checks do not validate CUDA graph/runtime semantics; the flag stays off until the parent runs an end-to-end candidate.
