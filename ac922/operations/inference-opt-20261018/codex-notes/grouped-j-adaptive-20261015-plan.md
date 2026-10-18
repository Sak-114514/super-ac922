# Adaptive grouped MMQ J fixture plan (2026-10-15)

## Goal

Test whether the previous uniform-20 result can be preserved for low-count experts while avoiding J32's hot-expert penalty, without changing engine code. The prototype invokes the existing compact grouped MMQ kernel through a standalone fixture.

## Workload and policy

Use 512 experts and 10,240 synthetic token assignments with this intentionally skewed, Zipf-like histogram:

- experts 0–15: 128 assignments each
- 16–63: 64 each
- 64–175: 24 each
- 176–335: 12 each
- 336–495: 3 each
- 496–511: 2 each

The counts sum to 10,240. Adaptive policy sends experts with count <=32 to one compact J32 queue and experts with count >32 to a compact J64 queue. A baseline queue runs all assignments through J64. This tests two grouped kernel graph nodes versus one and includes graph-launch overhead, but excludes CPU classification, quantization, and prefix construction time.

For each of the seven sampled packed expert formats, reuse the prior actual GGUF row samples and synthetic output-row dimensions. Generate each token's Q8_1_MMQ activation once, using non-power-of-two random FP32 scale mantissas, then repack the same rows into the baseline and two adaptive queues. Output IDs map back to global token positions. Compare the complete adaptive output against J64 baseline bitwise.

## Execution gate

- Write only under `fixtures/grouped-j-adaptive-20261015/` and `notes/grouped-j-adaptive-20261015*`.
- Compile the standalone fixture on CPU with at most two jobs; do not modify or rebuild the engine.
- Do not initialize CUDA or launch until the root owner explicitly releases a GPU. Then use only the assigned GPU UUID and release it immediately after this bounded run.
- If cold/hot MMQ configs cannot share a per-expert kernel invocation, retain the two-queue/two-kernel design. No new engine path is implied by this fixture.

## Evidence boundary

Packed rows are real; repeated rows, scale inputs, and Zipf-like routing are synthetic. This is a candidate filter only, not observed production routing or full-model performance. A win must be followed by a guarded full-model A/B before any runtime default changes.
