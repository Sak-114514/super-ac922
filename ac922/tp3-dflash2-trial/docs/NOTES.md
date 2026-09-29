# Field notes: every blocker, in the order it was hit

Each blocker below cost one failed server start. Recorded so the next person
skips straight to the fix.

## Target model (TP3, no drafter)

1. **Vision tower: 16 heads % 3 ≠ 0.** The checkpoint ships a 16-head vision
   tower even though this deployment is text-only (`limit-mm-per-prompt 0/0`
   still constructs it). Fix: `language_model_only` + skip vision construction
   in the TP3 path.
2. **Vocabulary: 248320 % 3 ≠ 0.** lm_head/embedding sharding asserts. Fix:
   internal padding to 248448 (multiple of 192); original rows preserved,
   pad rows zero.
3. **MLP intermediate: 17408 % 3 ≠ 0.** Fix: in-memory zero-padding to 17664
   — new gate/up input columns and down output rows filled with mathematical
   zeros, verified before loading. Checkpoint files untouched.
4. **CUDA Graph capture OOM at gpu_memory_utilization=0.95** despite weights
   fitting (449,142 KV tokens reported). Capture needed ~810 MiB more. Fix:
   0.90 → 398,857 KV tokens and capture succeeds. On 16GB cards the last
   ~800 MiB is graph-capture headroom, not free capacity.
5. **Head replication itself is boring if you keep one rule**: copy the head
   weights bitwise, then halve BOTH source and copied output-projection
   column blocks. Halving only the copy doubles that head's contribution and
   destroys the model. int4/int8 scales halve exactly in fp16, so the surgery
   survives quantization. There is a per-head output gate on the Q projection
   in this checkpoint — the copy block is 512 channels per head, gate
   included; packed int4 zero-point blocks are copied along the output axis.

## DFlash2 drafter

6. **The drafter is built on the target's TP3 process group even when
   `draft_tensor_parallel_size=1` is accepted by config.** First drafter load
   dies on `248320 is not divisible by 3`. Fix: expand the drafter in memory
   with the same replication rule (32 Q / 8 KV → 36 Q / 9 KV, half-scaling,
   vocab padding); it shares the target embedding and LM head.
7. **KV page-size unification**: vLLM's generic planner raises
   `NotImplementedError` ("page size of the layer is not divisible by the
   maximum page size") for the hybrid GDN + drafter layout. Fix: a bounded,
   opt-in common-page scheme (KV cache group size 8→16) where bounded
   sliding-window/GDN layers absorb page padding instead of full-context
   layers. Covered by a regression test
   (`test_tp3_dflash_kv_page.py`).
8. **Depth 7 caps concurrency at 3.** With depth 7 only 3 sequences ever ran
   (`Running=3, Waiting=1` in per-second engine metrics). Depth 3 + group 16
   unlocked `Running=4`. If your 4-way numbers look like 3-way numbers, check
   the engine's own `running` counter, not the client.

## Measurement traps (all bit us)

- **Prefix cache silently turns a "fresh 64k prefill" into a 4s cache hit.**
  Use a per-request `cache_salt` when you mean to measure prefill.
- **Multi-token stream chunks undercount decode tokens.** With speculative
  decoding one `stream_options` chunk can carry 6+ tokens; counting chunks or
  deltas as tokens reported 16 tok/s for a 110 tok/s run. Count
  `generated_tokens` from usage, or reassemble from token ids.
- **TP2 vs TP3 baselines were captured with different batch-token settings**;
  prefill deltas cannot be attributed to topology alone. Say so in the report
  instead of claiming the whole 2.2×.
- **Greedy parity ≠ bitwise equality.** Logprob diffs (median 0.00135, max
  0.259 at 640 paired tokens) are expected from sharded summation order.
  Token identity is the contract.

## Machine-specific (AC922 8335-GTW)

- **NVLink topology is two triangles, not one hexagon**: GPUs 0/1/2 and
  3/4/5 are NV2 full triangles on NUMA 0 and 8; across islands only PCIe/SMP.
  Any 4-GPU TP crosses sockets — TP3 on one island is the sane shape.
- **BMC quiet-mode false alarm**: pushing fan policy via legacy `systemctl`
  timed out once and the BMC read low RPM as triple-fan failure and cut power
  mid-compile. Keep the watchdog enabled, keep fan overrides out of the
  serving path, and monitor temperatures from the trial instead (they stayed
  43–55 °C through the entire campaign).
- 4-GPU capacity note: TP4 on GPUs 2/3/4/5 (3+1 split, fewest cross-socket
  pairs) holds 1.09M fp8 KV tokens and passes all construction assertions with
  zero surgery — worth an A/B if capacity matters more than NVLink locality.
