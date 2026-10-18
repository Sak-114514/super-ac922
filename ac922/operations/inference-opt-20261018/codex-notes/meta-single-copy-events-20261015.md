# Single-copy Meta event path and FIFO pipeline candidate

Date: 2026-10-15. Scope: private V13-derived `ggml-backend.cpp` plus the synthetic island-frame fixture. Root built and froze runtime-perf-v15; the fixture was separately compiled against its libraries. Both native and Nsight fixture gates passed on six V100s. This is scheduler/copy-path fixture evidence, not full-model correctness or production throughput.

## Scheduler change

`ggml_backend_sched_new()` normally allocates per-backend events only when `parallel` is true and `n_copies > 1`. The model's FIFO scheduler deliberately has one graph copy, so `sched->events[meta_id][0]` stays null. At a split boundary with `split->n_inputs == 0`, `ggml_backend_sched_compute_splits()` therefore calls `ggml_backend_synchronize(previous_backend)`: the host cannot submit the next island graph until the previous island has finished.

The private opt-in `GGML_SCHED_META_SINGLE_COPY_EVENTS=1` allocates one event for each of the two exact Meta3 backends while leaving `n_copies == 1` untouched. It fails closed unless:

- `GGML_META_FIFO_PIPELINE=1` and `GGML_META_GPU_EVENT_WAIT=1` were set before backend initialization;
- the scheduler is non-parallel, has exactly three backends, exactly two distinct three-member Meta devices, and a trailing CPU backend;
- the six Meta members are distinct CUDA devices and the Meta/member async and event callbacks/capabilities are available;
- both Meta event allocations succeed.

Any rejection logs a reason and leaves the existing host-synchronize path in force. Successful creation logs `GGML_SCHED_META_SINGLE_COPY_EVENTS active`; reaching a no-input Meta-to-Meta boundary logs `GPU wait active ... n_inputs=0`. If a matching scheduler did not enable FIFO, the scheduler logs `host-wait fallback` rather than silently using a host synchronization. The normal Meta event-wait path then queues CUDA stream waits. This does not alter tensors, arithmetic, split selection, graph-copy count, or the FIFO assertion.

## What it does and does not pipeline

The change only removes the host barrier at a **no-input** cross-Meta split. A split with cross-backend inputs follows the existing tensor-copy/Meta event path. A single scheduler still walks its graph splits in order and submits the A and B portions of each ubatch; the event gives the consumer stream a dependency on the producer stream without waiting on the host.

This is a plausible building block for two-island ubatch overlap if one scheduler and the same two Meta backend objects/streams are reused across fixed-shape ubatches:

- island A's graph and activation bridge copy are ordered on A's own member streams;
- the next A graph is queued after those operations, so it cannot overwrite A scratch before the bridge copy has read it;
- island B waits for the matching A event and then consumes its copied activation on B's streams;
- successive B graphs are ordered on B's streams, preserving B-local KV/GDN recurrence;
- the host input FIFO snapshots alternate buffers and waits on a pending slot event before overwriting that host snapshot.

Under those conditions, the activation bridge's internal staging-slot events and stream order can protect the copy source; duplicating the **whole graph scratch arena** is not automatically required. This is not proven for the production executor. The existing fixture uses separate backend contexts per frame, which have separate streams and cannot be treated as one scheduler's stream-ordering proof. Reusing one scheduler also loses overlap if each ubatch forces reserve/reallocation, a graph-shape change, a host result callback, explicit scheduler synchronization, or another host-side state transition. In particular, `ggml_backend_sched_alloc_splits()` synchronizes all backends when it must reserve/reallocate.

### Activation source and destination lifetime evidence

The existing cross-island async path provides the required source fence when it is selected. `ggml_backend_sched_compute_splits()` (private `ggml-backend.cpp`, around the input-copy loop) asks the destination Meta backend's `cpy_tensor_async` callback to copy the split input. `ggml_backend_meta_cpy_tensor_async()` (`ggml-backend-meta.cpp:3155-3255`) only accepts contiguous, same-layout, mirrored 3+3 CUDA tensors; it selects the CUDA copy callback and logs `async MIRRORED island copy active` on first success. For the peer-fanout route, `ggml_backend_cuda_meta_copy_async()` queues the source D2D peer read on the source member-0 stream and records `source_ready` (`ggml-cuda.cu:3824-3828`); the destination member streams wait for that event before writing the B-side split input (`:3829-3841`). Since the next A graph is enqueued later on the same scheduler and same MetaA backend, it is ordered after the A-side copy read on each A member stream. Therefore the shared A scratch output cannot be overwritten before the copy has read it. B's prior graph and the next B-side input overwrite are ordered on B's same member streams.

The two-slot staging buffer has an additional reuse fence: with `GGML_CUDA_META_COPY_GPU_WAIT=1`, reuse waits on every `dst_done` event from the source stream before overwriting that staging slot (`ggml-cuda.cu:3789-3803`). Those destination-done events mark staging-to-B copies, not completion of B's graph; B-side graph/input-buffer safety instead follows B-stream order. With GPU wait disabled, the existing copy helper host-synchronizes the slot, which is safe but serial. If `GGML_META_SHARED_DEVICE_COPY=1`, its path explicitly synchronizes the destination backend before copy (`ggml-backend-meta.cpp:3177-3197`), also safe but host-blocking.

If async copy initialization/callback returns false, the existing scheduler fallback synchronizes source and destination before the ordinary copy (`ggml-backend.cpp` input-copy fallback near the end of `ggml_backend_sched_compute_splits()`). This preserves correctness but removes the desired overlap. The activation-copy runtime gate below requires the async-copy and GPU-event-wait logs plus a bounded hot submission; absence of async-copy logs or a host wait is a failed pipeline gate. The event feature itself must not be used as evidence that the activation copy succeeded.

Before model integration, verify that the real executor can queue the next ubatch through the same scheduler/backend objects without per-ubatch synchronization or allocator reallocation, and keep the QSA/KV/GDN input metadata immutable for each queued graph. Distinct layer ownership across the two islands is required for A1 to overlap B0; any dependency from a later A chunk to the earlier B chunk must retain its event edge.

## Runtime gate added to the fixture

The existing stateful fixture gains two isolated scheduler tests using the same `[Meta3-A, Meta3-B, CPU]` scheduler object with `parallel=false` and FIFO enabled. First, independent branches exercise the no-input event wait. Each branch chains 32 FP32 identity `MUL_MAT` nodes at K=2048, M=2048, N=8192. Each multiply represents 68.7 GFLOP of nominal arithmetic; this makes a host wait visible. The output is checked exactly. After the first graph reaches replay-ready, a second graph uses a real A-to-B dependency and chains 32 matmuls on each side. It queues three calls without scheduler synchronization, capturing each B output to pinned host snapshots. This forces reuse of the two-slot async bridge while the same single-copy graph scratch and backend streams are reused. The test sets `GGML_CUDA_META_COPY_TOPOLOGY=1`, `GGML_CUDA_META_COPY_GPU_WAIT=1`, and `GGML_META_SHARED_DEVICE_COPY=0` for this isolated copy gate.

Both hot phases require a submission time below 100 ms and exact output data. That threshold is a diagnostic guard against the expected host-blocking path, not a speed target. The later Nsight run also measures whether A's following-frame kernels execute while B's previous-frame kernels are still active.

Expected runtime log evidence:

```text
GGML_SCHED_META_SINGLE_COPY_EVENTS active ... copies=1 ...
GGML_SCHED_META_SINGLE_COPY_EVENTS GPU wait active ... n_inputs=0
meta cross-device event wait active: CUDA stream dependencies
SINGLE_COPY_EVENT_GATE_PASS ... replay_ready=1 ...
SINGLE_COPY_EVENT_EXACT_OUTPUT island=A ...
SINGLE_COPY_EVENT_EXACT_OUTPUT island=B ...
ggml_backend_meta_cpy_tensor_async: async MIRRORED island copy active ...
meta copy slot reuse: GPU event waits; allocation growth still drains on host
SINGLE_COPY_ACTIVATION_COPY_GATE_PASS copies=1 splits=2 queued=3 replay_ready=1 ...
SINGLE_COPY_ACTIVATION_COPY_EXACT frame=0 ...
SINGLE_COPY_ACTIVATION_COPY_EXACT frame=1 ...
SINGLE_COPY_ACTIVATION_COPY_EXACT frame=2 ...
```

The fixture also reports split count, warm-up readiness, submit time, and drain time. If the async-copy or GPU-wait log is absent, treat the copy path as synchronous/fallback and stop pipeline integration. It must not be described as model-level proof or a throughput result.

## V15 fixture result

- Runtime: `/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v15`; native driver 550.54.15, CUDA 12.4; `LD_LIBRARY_PATH` and `/proc/self/maps` both resolved all GGML libraries to V15. The fixture linked only against V15 and CUDA/NCCL/NVTX libraries; no engine rebuild was done in this lane.
- Provenance hashes: private `ggml-backend.cpp` `da463a61c23ece78bb9af205f033e1361df60e09a3de8debe117475d828b4196`; fixture source `3d64bdc51bb4627a7c3a059021b0045a908de7db7c7cdda1ee68526959ae1264`; V15 fixture executable `ca9fc959c3ed0899fffedcddbc9e68fb85c19c64a2f2b498cf5eed1a9397b484`.
- Native fixture run: 6 s, event gate hot submit 242 us, activation-copy three-frame submit 1.074 ms. Replay-ready after two warm passes in each gate. The no-input event marker and Meta CUDA stream-wait marker appeared. The 64 MiB activation copy selected peer fanout, logged `GPU event waits`, and logged async mirrored island copy. Both independent branch outputs and all three copied-frame outputs were exact.
- Nsight run: 11 s, event gate hot submit 580 us, activation-copy three-frame submit 1.994 ms. The report contained 1,536 `volta_sgemm_128x64_tn` kernels, 256 per GPU, matching eight dispatches of 32 matmuls per island. Dispatches 1–3 were the independent event gate; 4–5 warmed the dependent copy graph; 6–8 were the three queued activation frames.
- Cross-island overlap: dispatch 7 (A frame 1) overlapped dispatch 6 (B frame 0) on all 9 A/B GPU pairs, by 150.577–155.282 ms. Dispatch 8 (A frame 2) overlapped dispatch 7 (B frame 1) on all 9 pairs, by 147.349–156.652 ms. This confirms the synthetic one-scheduler schedule can overlap later A work with prior B work while preserving copy/output canaries.
- Safety/cleanup: 1 s GPU monitors peaked at 48 C and recorded minimum free memory of 15,183 MiB (native) and 15,171 MiB (Nsight). After each run, all GPUs returned to 0–1 MiB used at 39–41 C; no model/server process remained. Both runs had `rc=0` and no CUDA error, OOM, or fixture `FAIL` line.
- Artifacts: `fixtures/island-frame-state-20261015/results/single-copy-v15-native.log`, `single-copy-v15-native-gpu.csv`, `single-copy-v15-nsys.log`, `single-copy-v15.nsys-rep`, `single-copy-v15.sqlite`, `single-copy-v15-stats.txt`, and pre/post GPU snapshots.

## Remaining integration proof

The scheduler/copy path itself now has fixture-level evidence, including actual cross-island kernel overlap. Production integration still must confirm that real fixed-shape prompt ubatches reuse the same scheduler and two Meta backend objects without explicit synchronization or allocator reallocation, that the target activation is contiguous/mirrored and the async callback remains selected, and that each queued QSA/KV/GDN input snapshot is immutable. No full model was run with this scheduler flag, so do not claim model-level correctness or throughput yet.
