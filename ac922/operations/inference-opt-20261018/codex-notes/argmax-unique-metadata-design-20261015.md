# Greedy exclusion-only argmax metadata candidate (2026-10-15)

## Status

Core graph/ggml/CUDA source is prepared in the isolated `standard-opt-20261014` source tree. The path is default-off. No engine build, CUDA compilation, GPU run, model request, or semantic/performance claim has been made. The CPU-only hierarchy fixture passed; it checks the host algebra contract, not the CUDA implementation.

## Intended contract

When `LLAMA_GPU_ARGMAX_UNIQUE=1` is admitted by context initialization for a CUDA output device, the existing raw GPU argmax node also emits a one-bit `unique-and-valid` side plane. The raw token-ID plane remains byte-for-byte the existing argmax ID and is still the only plane fed back into MTP. The common sampler may use the raw ID for exclusion-only greedy sampling only if the metadata bit is true and the raw winner is not in either the dynamic `-INF` logit-bias IDs or model suppress-token IDs. If it is excluded, tied, non-finite, or invalid, the current full-logit CPU fallback remains authoritative.

The GGML operation enum is unchanged. `GGML_OP_ARGMAX` has opt-in private modes for direct row metadata and split finalization; `GGML_OP_ARGMAX_SPLIT` has a private mode for four split channels. The private functions are declared only in `ggml/src/ggml-argmax-unique.h`, never in public `include/llama.h`.

Single-device output is I32 `[nrows, 2]`, laid out as contiguous IDs then contiguous valid-unique flags. Tensor split partial output is F32 `[4*n_dev, nrows]`, with each rank's slot `(max, global_id, runner_up, invalid)` and all non-rank partial slots exactly zero in all four fields. The existing meta AllReduce sums these rows. Finalization writes the same I32 two-plane layout. For non-partial mirrored inputs, unused slots use `(-inf, -1, -inf, 0)` and are ignored.

A uniqueness flag is true only when the raw winner exists, every input logit is finite, and the runner-up is strictly below the best value. NaN, either infinity, all-invalid rows, and any maximum tie produce false. The raw argmax behavior continues to ignore NaN/-Inf in the same way as the old CUDA row kernel and includes +Inf as the raw winner, so the fallback can inspect the original logits.

## Context/sampling integration patch for root

Do not apply until the current context/sampler owner has handed off those files. Required changes:

1. In `llama-context.cpp`, set `cparams.gpu_argmax_unique` only when all are true: `LLAMA_GPU_ARGMAX_UNIQUE=1`, existing `cparams.gpu_argmax`, and `model.dev_output()` resolves to CUDA. If the output device is META, recurse through `ggml_backend_meta_device_n_devices()` / `ggml_backend_meta_device_member()` and require every leaf registry name to equal `CUDA`. Otherwise leave false so graph construction uses the original raw argmax path and never creates the metadata shape on CPU/SYCL/Metal.
2. Add `cp.gpu_argmax_unique` to the context cache/hash at the existing `add(cp.gpu_argmax)` block.
3. Add uniqueness flags beside the pinned argmax output buffer and `get_argmax_unique_ith()` accessor to `llama-context.{h,cpp}`. Do not use a pageable `std::vector<int32_t>` for these flags: Nsight measured pageable argmax-ID D2H at 15.98 ms. After the pinned-output owner lands, copy `n_outputs` I32 flags into the matching pinned range. Keep the existing `t_argmax` ID copy untouched. Initialize the relevant range to zero when metadata is absent; clear/swap the flags with the pinned IDs in row reset/reorder; clear them when MTP chain completion overwrites raw argmax IDs from host chain tokens.
4. Declare the getter in internal `src/src/llama-ext.h` and include that staging header from `common/sampling.cpp`; do not add it to `include/llama.h`.
5. In `common_sampler_sample`, leave the existing bias-free raw GPU argmax route unchanged. For exclusion-only greedy, read the raw ID and metadata bit. If metadata is true and the ID is not in any active `-INF` bias or model suppress list, accept the ID without calling `llama_get_logits_ith`. Otherwise use the existing full-logit fallback unchanged. This path must not bypass any non-exclusion sampler.

The available context owner must review/apply those hunks and verify output-row swaps and MTP completion lifecycle. The graph source exposes `get_argmax_unique()` and `llama_gpu_argmax_unique_enabled()` for this integration.

## Fixture evidence

Command:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  fixtures/greedy-exclusion-argmax-metadata-fixture.cpp \
  -o fixtures/greedy-exclusion-argmax-metadata-fixture-20261015
fixtures/greedy-exclusion-argmax-metadata-fixture-20261015
```

Result: `PASS rows=4803 modes=single+split(2,3,6) cases=finite+ties+nan+inf+all-inf exact_raw_id=1 uniqueness=1 gpu=not-run`.

The fixture compares the old first-index raw argmax against the new top-two merge for single rows and split chunk/rank hierarchy. It also simulates four-channel partial tensors, verifies non-rank slots are zero, sums rank contributions, and finalizes IDs/uniqueness. This is not a CUDA kernel parity check.

## Source scope / build closure review

Owned modified source: `ggml/src/ggml.c`, `ggml/src/ggml-cuda/argmax.cu`, new `ggml/src/ggml-argmax-unique.h`, `src/llama-cparams.h`, `src/llama-graph.{h,cpp}`. Context and sampling remain a handoff patch only.

The new private header is included by only `ggml.c`, `argmax.cu`, and `llama-graph.cpp`; no `ggml-impl.h`, `ggml.h`, or public `llama.h` edit was made. A source grep in this tree finds 6 direct consumers of `llama-cparams.h` and 10 of `llama-graph.h`. The present `build-reuse/compile_commands.json` is stale: its source entries point to `/home/debian/ac922-monster/src`, not this isolated tree. Do not use it to claim an incremental build closure. No build was run.

## Risks / gates remaining

- CUDA source has not been compiled; SM70 NVCC compatibility and template/device-call rules remain unverified.
- Meta partial graph/allreduce shape has been statically traced, while only host simulation verified channel algebra. Root must inspect real scheduler placement and run the isolated CUDA fixture before enabling it.
- Unsupported backends must keep `gpu_argmax_unique=false`. A CPU execution of the private metadata shape is intentionally not supported.
- The full logits fallback remains necessary whenever the metadata is false or the winning token is excluded.
- Raw token IDs are checked for ordinary finite rows and ties in the host fixture. All-`-FLT_MAX` multi-rank logits are pathological: legacy split argmax can produce an invalid slot lookup; the new metadata reports invalid and returns no candidate rather than claiming a usable ID.
