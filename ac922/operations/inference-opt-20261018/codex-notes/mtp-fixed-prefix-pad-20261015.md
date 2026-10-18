# Fixed-prefix D2 KV metadata fixture (2026-10-15)

## Scope

CPU-only exact metadata test for a 4-sequence, D=2 fixed first step. It uses the actual `llama_kv_cache`, `llama_batch_allocr`, `llama_kv_cache_context`, `llama_kv_cells`, and `seq_rm` implementation from the built V15 runtime. The model is loaded in vocabulary-only mode; only scalar `block_count` and `embedding_length` values from the loaded GGUF metadata are copied into the vocabulary-only model's otherwise-empty `hparams`, because its loader intentionally returns before populating those two fields. The cache is constructed with an always-false layer filter, so it retains real cell/head/sequence metadata but creates no per-layer KV tensor or buffer.

This fixture does not run attention/GEMV, construct a CUDA graph, or compare hidden-state values. It verifies the allocator and mask/index inputs required by the proposed graph plan. Duplicate seed token identity is checked; hidden-state duplication remains a graph input/property for the graph-level test.

## Fixture sequence

Each round creates one 24-row batch in stable lane-major order: four catch-up positions, seed at row 4, and a dummy future row at row 5. The dummy repeats the seed token and uses the unique position `seed+1`. Only seed rows 4, 10, 16, and 22 are marked as outputs.

The real `llama_kv_cache::init_batch()` path reserves and applies step 0. The fixture snapshots its exported K indices, output mask, and causal KQ mask. It then calls `seq_rm(lane, seed+1, -1)` for each lane, confirms the four dummy cells are absent and the maximum position returns to the seed, and confirms the saved step-0 indices, output mask, and mask bytes remain unchanged. A four-row real step 1 at the same `seed+1` positions is then prepared and applied. Its four slots must be unique, must not overlap the 20 real prefix/seed slots, and must equal the just-freed dummy slots.

## Results

- Vocabulary-only model metadata: Qwen4Exp, 49 layers, embedding length 2560, vocab 248320. `CUDA_VISIBLE_DEVICES` was empty; no model weights were loaded.
- Real cache: capacity 4096 cells, one unified stream, four sequence owners, 49 layer-filter callbacks, zero KV tensor layers and zero KV buffers.
- Every round passed the 24-row output-mask check, unique step-0 cell indices, dummy-token duplication, causal mask check (seed cannot see its future dummy), stable saved step-0 metadata after rollback, step-1 index uniqueness, no real prefix/seed overwrite, and `seq_pos_max == seed+1`.
- All 100 rounds passed. Used cells and the allocator's occupied range grew together from 24 to 2400; `used_min=0` and `used_max_p1=used_count` at each sampled checkpoint. The step-1 end index plus one matched `used_max_p1`, and each next step 0 began at the previous end. No allocator holes accumulated.
- Positions remained contiguous per sequence from 0 through 599. The test models one rejected dummy row per lane per round, immediately replaced by the actual next token.
- The initial attempt used `vocab_only=false,no_alloc=true`; that traversed the tensor metadata and was stopped. It is not evidence. The passing run used `vocab_only=true` and skipped tensors.

## Verification command

Build only the isolated fixture against the existing V15 libraries:

`fixtures/mtp-fixed-prefix-pad-20261015/build.sh`

Run with CUDA hidden:

`CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=/home/debian/ac922-monster/experiments/standard-opt-20261014/runtime-perf-v15/lib OMP_NUM_THREADS=1 fixtures/mtp-fixed-prefix-pad-20261015/mtp-fixed-prefix-pad-fixture /home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf`

The final run ended with `PASS real_vocab_only_fixed_prefix_d2 rounds=100 ... cuda_used=0`. The fixture itself did not use a GPU. During the run, the separate V15 server was active on the six GPUs.

## Source path verified

- `llama-kv-cache.cpp`: constructor initializes real `v_cells`/heads before applying `layer_filter`, then skips tensor creation for every filtered layer.
- `llama-kv-cache.cpp::init_batch`: uses the production batch allocator, calls `prepare`, and returns the actual KV memory context.
- `llama-kv-cache.cpp::apply_ubatch`: commits the selected cells and advances the internal head to `last_cell + 1`.
- `llama-kv-cache.cpp::seq_rm`: removes dummy rows by sequence/position and moves the head to the first released cell when appropriate.
- `llama-kv-cache.cpp::set_input_k_idxs` and `set_input_kq_mask`: the test reads the same slot indices and causal mask data used as graph inputs.
- `llama-model.cpp::load_hparams`: vocabulary-only loading stores scalar GGUF keys and then returns before filling model hparams; the fixture fills only the two required scalars from those actual keys.

## Artifacts

- `fixtures/mtp-fixed-prefix-pad-20261015/fixture.cpp`
- `fixtures/mtp-fixed-prefix-pad-20261015/build.sh`
- `fixtures/mtp-fixed-prefix-pad-20261015/cpu-only.log`


## D4 extension (2026-10-15)

The common-side activation keeps D2 as the default. `LLAMA_MTP_GPU_CHAIN_UNROLLED_DEPTH` is parsed exactly: unset or `2` selects depth 2; exact `4` selects depth 4; any other spelling/value rejects the unrolled shape. Four lanes, `LLAMA_MTP_GPU_CHAIN_UNROLLED=1`, and `LLAMA_MTP_FIXED_PREFIX_ROWS=6` remain required. A depth mismatch continues through the existing generic fallback.

The allocator fixture now uses four real catch-up rows plus the seed, pads one future row in each lane, removes that row, then allocates three accepted rows at `seed+1`, `seed+2`, and `seed+3`. It checks the actual allocator's row indices, uniqueness, sequence maxima, prefix preservation, step-0 mask/index stability, and 100 rounds of contiguous capacity accounting. It uses the existing vocab-only/layer-filtered setup and allocates no KV tensors.

The common/context graph depth loop and CUDA unroll are owned by the root task; this change only admits explicit common depth 4. The fixture does not prove graph-output equivalence or model semantics.
