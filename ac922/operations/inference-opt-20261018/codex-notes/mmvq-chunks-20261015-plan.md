# SM70 MMVQ 12/16-row chunk fixture plan

## Source finding

The frozen V12 CUDA backend already contains a non-overlapping chunk route behind the environment setting GGML_SM70_DECODE_LINEAR_CHUNKS=8:

- ggml-cuda.cu:2798-2825: quantized MUL_MAT_ID with CUDA weight buffers, F32 activations/output, contiguous first dimensions and ID shape [topk,T], where 8 < T <= 64; views each chunk as min(8, remaining) rows and calls existing ggml_cuda_mul_mat_vec_q.
- ggml-cuda.cu:2613-2633: ordinary quantized MUL_MAT similarly divides ne11 columns into 8-column views.
- mmvq.cuh:3 caps MMVQ at 8. V100's get_mmvq_mmid_max_batch() returns 8 for all supported quantized formats. Do not change that shared header constant.

The ID chunk view preserves original order and layout by offsetting input with src1->nb[2], IDs with ids->nb[1], and output with dst->nb[2]; ids_stride remains ids->nb[1]/sizeof(int32_t). MMVQ treats ne2 as token count and ne1 as top-k channels. For four active lanes, D2 means T=4*(2+1)=12 (8+4), and D3 means T=4*(3+1)=16 (8+8). Top-k remains 10.

All seven model expert formats appear in the real packed manifest and the MMVQ type switch: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, Q2_0. Existing TP notes give gate/up IQ3_XXS K=2560/N=224 and IQ4_NL down K=224/N=2560. K=224 is valid for IQ4_NL's 32-element blocks; do not impose it on Q2_0, whose blocks are 64 elements. Use its actual block-aligned local K.

The V11 runtime proof recorded GGML_SM70_DECODE_LINEAR_CHUNKS=0; a prior 2026-10-14 trace-target run with the flag set logged moe tokens=35 chunk=8 and MUL_MAT_ID shapes [640,10,35,1] and [2560,10,35,1]. That proves the existing branch was reached at T=35, but does not validate 12/16-row output parity or speed.

## Fixture

- Use the frozen V12 ggml CUDA backend and a CUDA graph fixture on GPU2 only. Do not load a model.
- Generate small, nonzero packed weights for E=16 (or reuse real packed row samples) and deterministic F32 inputs. Use N=224 for local gate/up shapes; use N=2560 for IQ4_NL down. Keep Q2_0's K block-aligned.
- Test the seven expert formats on MUL_MAT_ID at T=8, T=12, and T=16, with top-k 10 unique expert IDs. Include both balanced and hot-plus-tail routing. Run separate ordinary MUL_MAT cases so the ID and non-ID dispatches are independently visible.
- In one GGML_SM70_DECODE_LINEAR_CHUNKS=8 process, execute graph A (explicit MMVQ chunks, 8+tail), graph B (full 12/16 rows through the dispatcher, requiring its actual moe tokens=T chunk=8 marker), then A again with changed input/ID canaries. Compare outputs in original [N,topk,T] order; all IDs must remain within E.
- For a separate GGML_SM70_DECODE_LINEAR_CHUNKS=0 run, execute full-T grouped MMQ as the engine baseline. Also compute a CPU FP32 dequantized-dot reference for all output slots. Require bitwise identity between the chunked path and explicit <=8 MMVQ composition; report exact match and numerical error against MMQ/CPU FP32 separately because reducers can accumulate in a different order.
- Check CUDA graph capture/replay for A/B/A, no dispatch fallback, no NaN/Inf, and peak allocation below 2 GiB. Record CUDA-event timings for MMVQ chunks and full-T MMQ.

## Run constraints and stop gates

- Use only GPU2's UUID and CUDA_VISIBLE_DEVICES; verify GPU2 maps to logical device 0 in the fixture log.
- Check temperature before launch and stop if it reaches 58 C. Release GPU2 immediately after the bounded run; do not reset the GPU or alter system settings.
- This fixture tests kernel/stride/ID behavior and performance only. It is not a full-model semantic or throughput claim.
- If the actual dispatcher marker is absent, stop and inspect shape guards before editing anything. Do not alter mmvq.cuh, MMVQ_MAX_BATCH_SIZE, weight precision, quantization, or expert count.
