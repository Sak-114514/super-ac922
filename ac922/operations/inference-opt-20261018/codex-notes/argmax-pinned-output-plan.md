# Opt-in pinned host staging for GPU argmax D2H

## Evidence and goal

The V11 full trace identifies the greedy argmax copy in `llama-context.cpp` as a host-side blocker: 54 asynchronous D2H transfers of only 32 bytes each use a pageable `std::vector<int32_t>` destination and account for 15.98 ms in the measured span; the pinned-device-host comparison is about 1.3 microseconds. Keep this as a lossless, default-off host-memory path change.

## Implementation boundary

- Change only private `src/llama-context.h` and `src/llama-context.cpp` in the experiment tree, plus this note and a bounded CPU fixture.
- Opt in with `LLAMA_ARGMAX_PINNED_OUTPUT=1`; default behavior remains the current `argmax_ids` destination.
- Reserve two small context-owned int32 staging planes in the existing output buffer, which already chooses the output device's host buffer type: IDs and the per-output metadata-unique flags that the root is integrating. Check the **allocated buffer's actual type name**; enable staging only for `CUDA_Host`. If allocation falls back to CPU, log that and use `argmax_ids` directly; provide the flags-plane pointer/capacity for the root's metadata integration.
- Queue GPU D2H argmax copies into the IDs plane, then after scheduler synchronization copy the completed prefix into the existing `argmax_ids` vector. Keep current output row ordering and token getter semantics. Leave the second plane available for the metadata flag path so that integration does not reintroduce pageable D2H.
- Before a new decode resets `argmax_ids`, before output-buffer reallocation/repointing, and before output reordering of pending rows, flush the previous pinned copy. Do not resize or free staging memory while a D2H copy is in flight.
- Preserve the existing MTP GPU-chain scratch path; it remains separate from regular argmax output staging.

## CPU regression and syntax verification

- Exercise stage-to-vector copy, output reorder after the deferred copy, capacity rebind only after pending data is committed, two independent context staging objects, and non-CUDA buffer fallback without CUDA initialization.
- Run a C++ syntax-only compile of the edited context translation unit using private headers; no engine link/build, CUDA initialization, GPU use, or model run in this task.
- Root owns runtime validation and full-model measurement.

## Preparation results (2026-10-15)

- Implemented the opt-in path in the private llama-context.h and llama-context.cpp. The extra two int32 planes are appended to the existing context output buffer only when GPU argmax is enabled and LLAMA_ARGMAX_PINNED_OUTPUT=1.
- The allocated output buffer's actual type is checked against CUDA_Host. On any other type, the context logs the fallback and retains the existing pageable argmax_ids transfer destination.
- Pinned IDs are copied into the existing CPU vector only after ggml_backend_sched_synchronize(). The code flushes pending IDs before decode resets, output-buffer reserve/rebind, and output reorder that could swap rows. MTP GPU-chain token transfers remain on their existing pinned buffer.
- The second plane has flags(), queue_flags(), and commit_flags() helper operations for the root's metadata integration. The metadata producer is not wired to the plane here, and synchronize() currently commits IDs only; do not enable metadata flag D2H into this plane until the root adds its pending-flag commit/output-reorder hook.
- CPU fixture passed without CUDA initialization. It covered actual type-name classification, CPU fallback, deferred ID commit, post-commit output reorder, refusing to rebind while a copy is pending, capacity growth after commit, separate context buffers, and the flags plane helper.
- llama-context.cpp passed a private-header C++ syntax-only compile. No engine link/build, CUDA initialization, GPU use, or model inference was performed.
- Runtime proof is still required: logs must show actual allocated type CUDA_Host on the target server. If it reports CPU, this candidate falls back to the old pageable path.

### CPU fixture command and result

    /usr/bin/c++ -std=c++17 -O2 -Wall -Wextra -Werror \
      -I.../src/common -I.../src/vendor -I.../src/src -I.../src/include -I.../src/ggml/include \
      fixtures/argmax-pinned-output-fixture.cpp -o /tmp/argmax-pinned-output-fixture
    /tmp/argmax-pinned-output-fixture

    PASS pinned argmax staging: fallback, deferred commit, reorder, growth, context ownership, flags plane; no CUDA initialized

### Syntax-only command

    /usr/bin/c++ -std=c++17 -fsyntax-only \
      -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_CUDA \
      -DLLAMA_BUILD -DLLAMA_SHARED -DMONPACK_AVAILABLE=1 -Dllama_EXPORTS \
      -I.../src/src -I.../build-reuse/src -I.../src/include -I.../src/ggml/include \
      src/src/llama-context.cpp

The command completed with exit code 0 against the private experiment tree.

## Risks to check during implementation

- `output_reserve()` recalculates view offsets even when reusing a larger output buffer; pending argmax work must be synchronized before moving the scratch pointer.
- `synchronize()` must commit the pending host prefix only after backend copies complete, and the public getter must continue to expose `argmax_ids` after that commit.
- CUDA host-buffer allocation can silently fall back to pageable CPU memory; the actual `ggml_backend_buffer_get_type()` must be checked, not just the requested buffer type.
