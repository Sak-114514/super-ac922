# EP + Monpack readiness review

Date: 2026-10-15. Scope: read-only source and existing-result review for the
current expert-parallel layout. No core code or GPU state was changed.

## Verdict

EP makes the Monpack **storage slice** easier than the old TP slice: the sidecar
stores expert E outermost, with each expert's K/N tile grid contiguous. With
full K and N and only a contiguous E range per rank, a local expert range is a
direct byte span; no format revision or K/N tile gather is needed.

Monpack is still **not usable with EP in the current runtime**. Its loader only
accepts Meta axis-0 K and axis-1 N splits; its CUDA MMID dispatch explicitly
aborts for a registered packed EP tensor; GEMV and the grouped-TC work list
treat global route IDs as local expert indices; and the documented Monpack
contract excludes fused gate/up+GLU. Removing the abort or setting the env flag
would be unsafe.

The missing integration is bounded to loader slicing, EP route mapping/foreign
zero behavior, dispatch, and a safe fused-path gate. It is a small/medium port,
not a rewrite. The existing prefill Tensor Core evidence is negative, so the
first validation should be a small decode consumer fixture; do not repeat the
same TC schedule expecting higher route density alone to reverse the result.

## Why axis-2 is simpler than the old TP slice

The current model placement code chooses `GGML_BACKEND_SPLIT_AXIS_2` for EP
expert weights and sets expert granularity to 1
(`src/src/llama-model.cpp:386-391,477-480`). Meta then writes the actual prefix
offset and global expert count into each device-local MMID node
(`ggml/src/ggml-backend-meta.cpp:1928-1947`). This is the same contiguous
axis-2 contract exercised by the V26 stock-kernel EP consumer fixture; that
fixture does not test Monpack (`fixtures/ep-consumer-validation-20261015/README.md:71-109`).

The Monpack format already has the necessary byte order:

- `monpack_format.h:3-7,85-97` specifies E outermost and all tiles for one expert
  contiguous; the tile address is
  `((e * n_tiles_n + n_tile) * n_tiles_k + k_tile) * tile_bytes`.
- One expert stride is `n_tiles_n * n_tiles_k * tile_bytes`. The sidecar span for
  `[e0,e0+Elocal)` is `blob_off + e0 * expert_stride`, length
  `Elocal * expert_stride`; it is already contiguous.
- The actual family descriptor is **88 bytes** and the index entry is **104
  bytes**, asserted by the loader (`src/src/llama-monpack-loader.cpp:27-33`).
  The family descriptor is type metadata; the index already carries the full
  tensor K/N/E and blob range. The EP owner offset is runtime state, so there is
  no need to change the sidecar layout revision.

The old TP failure was a different problem. P3 split down-K into 192/224/224
and gate/up-N into 256/192/192; these cuts do not map to full-tensor tile bytes
and required slice repacking. The P3 report records the 224 alignment fallback
(`P3-WARREPORT-20261013.md:28-35`). With EP, K/N stay whole, so those tile
alignment constraints disappear.

## Current blockers in the integrated path

| Stage | Source evidence | Result for EP |
|---|---|---|
| Sidecar activation | `llama_monpack::init()` checks `<GGUF>.monpack`; `MONSTER_MONPACK=0` returns before open (`src/src/llama-monpack-loader.cpp:300-320`). The 45,586,996,800-byte sidecar is currently named `.monpack.disabled-by-p3`; its manifest says 147 packed tensors. P3 launcher default is `MONPACK_ARM=0` (`launch-p3-arm.sh:13`). | No sidecar is active in the current recipe. CMake builds the optional Monpack TUs because they exist, but `MONPACK_AVAILABLE` is compile-time availability, not activation (`src/src/CMakeLists.txt:79-84`; `ggml/src/ggml-cuda/CMakeLists.txt:118-133`). |
| Plain weight override | `tensor_src_override()` requires full index E and full `ggml_nbytes(t)` to match the tensor, then returns the whole blob (`src/src/llama-monpack-loader.cpp:560-591`). | A plain local `[K,N,Elocal]` tensor fails the full-E/full-byte check. It needs its owner offset and local span. |
| Meta slice loader | `tensor_meta_override()` accepts only axis 0 K or axis 1 N; other axes, including E/axis 2, return to GGUF (`src/src/llama-monpack-loader.cpp:746-755`). The current repacker gathers K/N tiles across every expert and registers each resulting member (`:649-689,872-938`). | Add a direct E-span branch. It should not call the existing K/N repacker. |
| MMID dispatch | `ggml_cuda_mul_mat_id()` explicitly logs and aborts on `is_registered(src0->data) && ep`; only non-EP packed tensors call `try_mul_mat_id()` (`ggml/src/ggml-cuda/ggml-cuda.cu:2847-2858`). | This is an intentional safety guard. Do not remove it until the Monpack EP route path is complete. |
| GEMV expert lookup | The packed GEMV reads `channel_x = ids[...]` and sets `e = channel_x` (`ggml/src/ggml-cuda/monpack/monpack-kernels.cuh:704-716`). `mp_view_t` carries tile geometry but no owner offset (`:82-104`). | Global IDs can index the wrong local expert or access beyond the local E bank. No foreign-slot zero path exists. |
| Grouped TC routing | `mp_run_tc_moe()` passes `meta.ne_e` as the work-list expert count (`ggml/src/ggml-cuda/monpack/monpack-kernels.cu:688-727`). The work-list indexes `cnt[ids[...]]` directly and iterates `[0,n_experts)` (`monpack-kernels.cuh:1318-1363`). | For a nonzero E offset, a global ID may index past `cnt[]`; foreign slots must be skipped before any array access and explicitly written as exact zero. |
| Fused route | The Monpack API says fused gate/up+GLU is outside its contract (`ggml/src/ggml-cuda/monpack/monpack.h:38-46`). The EP fused wrapper preserves MMID metadata but calls the standard `ggml_cuda_mul_mat_vec_q()` path (`ggml/src/ggml-cuda/ggml-cuda.cu:5905-5926`). | An unfused Monpack MMID hook does not cover the production fused graph. Disable that fusion for Monpack EP weights or add a fused Monpack consumer. |

Read-only runtime check during this audit found `llama-server` PID 255962 running
`runtime-perf-v28` with `LLAMA_EXPERT_PARALLEL=1` and `MONSTER_MONPACK=0`.
The target directory had no active `*.gguf.monpack`; it only had
`IQ3_XXS-MTP-Q8.gguf.monpack.disabled-by-p3` and its manifest. So the current
server is using the normal GGUF EP slices, not Monpack.

There is also a fail-closed concern before re-enabling any sidecar. The
documented `GGML_CUDA_MONPACK_DISABLE=1` makes `try_mul_mat_id()` return false
at entry (`monpack-kernels.cu:798-805`); the caller can then proceed into stock
MMVQ/MMQ. The contract itself warns that stock kernels cannot read packed bytes
and demands that every registered tensor hit a covered Monpack callsite
(`monpack/monpack.h:38-39`). A safe fallback must stay on a Monpack kernel or
abort; it must not silently read packed bytes as GGUF data.

## Existing correctness and performance evidence

The sidecar's byte layout passed its conversion checks, but that does not close
runtime quality. The P3 warreport records 147/147 T0 packed-byte round trips,
then: IQ4_NL-only and IQ4_NL+Q8_0 GEMV arms were exact; enabling grid-family
IQ2/IQ3 weights produced garbage output (`P3-WARREPORT-20261013.md:14-24,33-35`).
The report attributes the remaining mismatch to Monpack grid GEMV not matching
the engine's actual fused/rework arithmetic path.

Recorded full-model prefill numbers were baseline UB512 1478.9 tok/s, IQ4_NL
only 1474.6 tok/s, and IQ4_NL+Q8_0 1380.5 tok/s. The full grid GEMV arm was
999–1120 tok/s with invalid output. Monpack TC v1 was 712–863 tok/s at UB512
and 825–863 for its FAM-B arm; the report notes the TC run occurred during a
weight-misalignment period, but still judged the kernel slower than the tuned
baseline (`P3-WARREPORT-20261013.md:8-18,37-41`).

A separate 2026-10-15 **non-Monpack** streamed-TC fixture used actual packed-row
samples and synthetic routes. It also lost to grouped MMQ: TC/MMQ core ratio
5.54x at 20, 3.16x at 40, and 2.74x at 80 assignments per active expert; with
GPU preparation included, ratios were 4.73x, 2.74x, and 2.35x. The schedule lost
60/60 core comparisons and 59/60 prepared comparisons
(`notes/streamed-tc-iq-large-route-20261015-results.md:20-47`). This is not a
Monpack benchmark, but it is evidence against rerunning the same TC schedule
expecting route density alone to turn it into a win.

Monpack is a byte reordering, not a smaller quant. P3's earlier 9.5x HBM-traffic
claim assumed dequant writeback; the warreport corrects that because the
deployed grouped-dp4a baseline already reads compressed weights directly
(`P3-WARREPORT-20261013.md:37-40`). No sidecar EP prefill/decode gain has been
measured. The recent V26 EP fixture proves the stock IQ2_S EP range and fused
wrapper; it does not load Monpack weights
(`fixtures/ep-consumer-validation-20261015/README.md:81-109`).

## Shortest safe patch and local gates

1. **CPU-only sidecar span fixture.** For an existing entry, split E into
   `[0,170)`, `[170,341)`, `[341,512)` and prove the three sidecar byte ranges
   concatenate byte-for-byte to the original packed tensor; each length must be
   `Elocal * n_tiles_n * n_tiles_k * tile_bytes`. Cover all eight packed types.
2. **Loader axis-2 branch.** Keep full parent entry/hash checks, require one
   segment on split axis 2, unchanged full K/N, positive local E shares summing
   to index E, and prefix offsets derived from the Meta split state. Copy the
   local contiguous sidecar span directly to each member tensor, register its
   local shape, and carry the global E offset to execution. Keep existing K/N
   repack behavior unchanged; reject invalid shapes before uploading anything.
3. **Offset-aware MMID.** Pass the private MMID `{offset,global_E}` into Monpack
   dispatch. Every GEMV variant—including IQ2_S Volta and Q2_0 SM70-rework—must
   translate owned IDs to local E and skip foreign weights. Ensure each foreign
   output slot becomes exact `+0.0`. Make the TC work-list compact owned rows and
   initialize output before its sparse writes. Fail closed if a registered
   tensor cannot use the selected kernel.
4. **Fused route.** First keep Monpack EP weights out of the fused stock
   MMVQ path; use two offset-aware Monpack MMID ops plus the ordinary GLU. Add a
   fused Monpack kernel only after unfused correctness passes.
5. **Bounded decode gate.** One V100, IQ2_S first, then one grid family and
   IQ4_NL; test T=1/4/8 across all owner ranges. Compare against full-bank
   Monpack at `[0,512)` so both sides use the same arithmetic; require exact
   owned output, exact foreign zero, A/B/A, and Nsight/dispatch proof of the
   Monpack EP kernel. Expand to all eight types only after this passes.
6. **Prefill gate.** Only after decode correctness, test T=512/1024 with the
   existing actual EP routes; compare Monpack TC against EP grouped MMQ including
   route setup, activation/Q8 preparation, output clearing, and graph time. If
   the integrated path loses, do not take it to full-model testing.

## Bottom line

Axis-2 EP removes the old K/N tile-alignment obstacle: Monpack's expert-major
blob can be sliced directly. However, the current integrated loader rejects
axis 2 and CUDA explicitly aborts Monpack+EP MMID. A viable port still needs an
E-slice loader branch, global-to-local route mapping/foreign zeroing, and a safe
fused-path policy. The TC branch has no positive existing evidence; Monpack
mixed-grid decode also lacks a clean runtime-quality record. The format can be
reused, but there is no current path to enable and no supported performance
claim.

All evidence above came from source, manifests, and existing fixture results.
No core files, model files, sidecars, services, or GPU state were modified.
