# Ordered relay6 static audit

Date: 2026-10-16 UTC  
Scope: read-only review of the local candidate at `src/ggml/src/ggml-cuda/allreduce-p2p.cu`. The AC922 remote source/runtime remained on the previous V91 relay implementation during this review. No engine edits, build, model load, or GPU run was made for this audit.

## Dataflow and phase ordering

The kernel partitions ranks as island A `0,1,2` and island B `3,4,5`. Its staging layout is the existing double-buffered `[half][rank slot][cap]` array. For each block and call epoch:

1. **Phase 0, local arrivals:** ranks 1 and 2 write their raw partials into slots 1 and 2 of rank 0's stage; ranks 4 and 5 write into slots 4 and 5 of rank 3's stage. Rank 0 waits for ranks 1/2. Rank 3 waits for ranks 4/5. Each producer thread executes `__threadfence_system()`, then a block barrier completes before thread 0 publishes its phase 0 flag.
2. **Phase 1, rank ordered prefix:** rank 0 starts from its own partial, then computes `x0 + x1`, followed by `(x0 + x1) + x2`. It writes the prefix to slot 0 of rank 3's stage, fences, synchronizes the block, and publishes the phase 1 flag to rank 3.
3. **Phase 2, final result:** rank 3 waits for that prefix, appends its own `x3`, then `x4`, then `x5` in order. It writes the final value to slot 3 of its local stage and slot 3 of rank 0's stage, fences both writes, synchronizes the block, and publishes phase 2 to all six ranks.
4. Ranks 0/1/2 read final slot 3 from rank 0's stage; ranks 3/4/5 read slot 3 from rank 3's stage.

The cross-island data traffic is one prefix transfer from rank 0 to rank 3 and one final transfer from rank 3 to rank 0: 2V. The remaining gathers and reads stay within their three-rank island, assuming communicator rank order matches the physical 3+3 grouping.

## Synchronization and buffer reuse

I found no missing producer fence in the new phase sequence. Phase 0, phase 1, and phase 2 all fence the writing threads before the publishing flag is written. The destination slots are disjoint: prefix uses rank 3 slot 0, final uses slot 3, and ranks 4/5 partials use slots 4/5.

The half-reuse proof is valid under one shared per-rank collective order. A rank can enter call `k+1` only after its call `k` kernel has completed on that rank's stream. The phase 2 publication for `k+1` depends on rank 0's phase 1 prefix, which depends on phase 0 arrivals from ranks 1/2, and rank 3's local phase 0 arrivals from ranks 4/5. Thus every rank has entered `k+1` after completing `k` before the final `k+1` flags are published. Each rank's call `k+2` begins only after it consumed call `k+1`'s final value, so its writes to the reused half cannot overtake another rank still reading call `k`.

Kind 4 shares the existing per-block epoch and `flags_seg` array with the segmented kernel. Its phases 0/1/2 reuse segmented flag cells 0/1/2. That reuse is safe for sequential, rank-consistent calls because each kernel increments the same epoch once and waits for the new epoch value. One-shot remains on the separate `flags` array. The fixed grid is part of the invariant: `P2P_AR_BLOCKS` reserves 64 rows, while the communicator grid is parsed once from `GGML_CUDA_P2P_AR_BLOCKS` and must be identical across ranks; the current parser accepts 8, 16, 32, or 64.

The local Meta implementation provides an enqueue ordering path when context views are enabled: `ggml_meta_submit_order` locks the physical device cohort, and the shared submit-tail events order later contexts on the same device streams. The relay kernel itself assumes each communicator sees the same collective sequence on all ranks. If V39 permits overlapping callers to share one `p2p_ar` outside that Meta ordering path, its non-atomic per-block epoch and reused flag cells would not identify which call published a flag; verify that caller configuration before enabling such overlap.

## Rank order and signed zero

The ordered kernel preserves the one-shot rank-0-to-rank-5 left fold. It does not compute `(x0+x1+x2) + (x3+x4+x5)`. `x0` initializes the accumulator directly when present, and each later rank is added with `__fadd_rn` in sequence. If rank 0 is absent, both paths initialize from positive zero; absent later ranks likewise contribute positive zero. Loads and staging writes copy float bits without arithmetic. Under the same V100 round-to-nearest F32 semantics as the one-shot kernel, this preserves signed-zero results as well as the intended addition order.

## Concrete gates and diagnostic limits

- Kind 4 is selected only for six ranks and payloads from 204,800 bytes through the configured relay maximum. `GGML_CUDA_P2P_AR_RELAY6=1` enables the relay family (default maximum 4 MiB); `GGML_CUDA_P2P_AR_RELAY6_ORDERED=1` selects kind 4. These environment values are cached on first use, so set them before backend initialization. An ordered flag by itself does not select the kernel.
- A deployment must preserve rank order `0,1,2 | 3,4,5` for the 2V cross-island claim. `ggml_cuda_p2p_ar_init` verifies peer access for every pair, but that does not prove the first and second triples are the intended local islands.
- With `GGML_CUDA_P2P_AR_DEBUG=1`, spin limits are 2^24. A timeout prints `p2p relay6 stuck ...` and breaks out of the wait instead of trapping or returning an error status. Treat any such line as a failed collective even if the process exits normally or output canaries happen to pass.
- Runtime path logs should show `algorithm=relay6_ordered` and `cross-payload=2V` on all six devices. The host source uses the same kind selection for direct dispatch and CUDA graph append.

## Conclusion

Static review found no concrete phase-order or producer-fence defect in the local ordered kernel, and its source preserves signed-zero rank order. This is not runtime validation: the ordered candidate was not built or run, and the remote V91 runtime did not contain it.
