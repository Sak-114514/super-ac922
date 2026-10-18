# Adaptive grouped MMQ integration plan

Scope: private experiment only, default off. Preserve stored weights, quantization, routing and arithmetic kernels.

1. Reuse the current globally sorted activation buffer, `expert_bounds`, and `ids_dst`. Avoid CPU classification and activation queue copies.
2. One GPU prefix scan builds two compact offsets arrays. Experts with <=32 assignments use original J32 kernel; larger experts use original J64. Empty experts have zero tiles in both queues.
3. Guard SM70, seven validated formats, prompt batches >=512, compatible J32/J64 shared-memory configs, and no explicit tile override. Other shapes keep existing grouped launch.
4. Regression fixture must include GPU classification/prefix plus both original kernels in timing, exact randomized FP32-scale outputs, shuffled output IDs, and empty/skew routing. No full-model run before this gate.
5. Only promote after actual dispatch logs and full-model semantic/real-text performance gates. Roll back by unsetting GGML_CUDA_MOE_ADAPTIVE_J.
