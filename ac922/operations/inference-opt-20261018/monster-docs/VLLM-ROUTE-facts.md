# vLLM 路线事实清单（VLLM-ROUTE-facts）

> 2026-10-12 编写。为"AC922 推理引擎路线辩论"提供硬事实：上游 vLLM、1Cat-vLLM fork、本机历史实验三条线。
> 结论速览：**ppc64le 红灯、sm70 红灯、Flash-Next 线性注意力+MoE 在 sm70 红灯**；三者叠加时唯一活路是 1Cat-vLLM 源码自建 fork——而这条路本机已经走过，Flash-Next GGUF 最好成绩 **29 tok/s（无 MTP、无 parity）**，远低于 monster llama.cpp 的 134-140。
> 服务器 ssh 本次未连通（端口超时，网络闪断），第 4 节以 Mac 侧完整镜像目录为准。

---

## 1. 上游 vLLM 对 V100/sm70 的支持现状：红灯（事实性弃用）

| # | 事实 | 来源 |
|---|---|---|
| 1.1 | 官方安装文档明确 **NVIDIA GPU 最低 compute capability 7.5**（T4/RTX20xx/A100+ 为例），V100=7.0 低于门槛；默认二进制 CUDA 12.9 | https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html |
| 1.2 | 主干 CMakeLists：CUDA≥12.8 时 `CUDA_SUPPORTED_ARCHS="7.5;8.0;…"`，**7.0 只在 CUDA<12.8 分支**——现代工具链下官方根本不编译 sm70 | https://github.com/vllm-project/vllm/blob/main/CMakeLists.txt（112-132 行） |
| 1.3 | 新版实测报错原文："vLLM requires Compute Capability 7.5 or higher. SM70 (Volta) is deprecated" | https://www.speediyo.com（vLLM Legacy GPU Setup） |
| 1.4 | FLASH_ATTN 后端仅支持 **sm80+**（Ampere 起），FA3 需 sm90；V100 历史上靠 XFormers/V0 引擎 | https://github.com/vllm-project/vllm/issues/29743 |
| 1.5 | XFormers 新版不再提供 CC 7.0 算子（"No operator found… too old"）；xformers FA 内核本身要求 sm80+ | https://github.com/vllm-project/vllm/issues/10662 、https://github.com/facebookresearch/xformers/issues/795 |
| 1.6 | V1 引擎下 V100 残存可用后端是 `TRITON_ATTN_VLLM_V1`（社区 workaround）；v0.11.0 起该取值又变更 | https://github.com/vllm-project/vllm/issues/25528 、https://discuss.vllm.ai/t/which-attention-backend-for-gpt-oss-in-version-0-11-0/6356 |
| 1.7 | **实测 0.11.0 在 V100 上比 0.8.1 慢约 10 倍**（Torch 2.7+/XFormers 0.0.30+ 弃 sm70 所致）——降级维护的量化证据 | https://github.com/vllm-project/vllm/issues/27400 |
| 1.8 | Marlin（W4A16 快速内核）**最低 sm80**，V100 不可用；1Cat fork 致谢列表里存在独立的 `marlin_v100` 第三方移植，即上游无此路径 | https://discuss.vllm.ai/t/why-i-feel-cuda-kernel-marlin-run-not-fast/712 、https://github.com/1CatAI/1Cat-vLLM |
| 1.9 | 与 AC922 相关的 sm70 特性矩阵：fp16 TC ✓；**bf16 ✗、fp8 KV ✗（硬件无 FP8）、cp.async ✗、TMA ✗**。官方 Flash-Next recipe 大量依赖 fp8 KV/indexer fp8/FlashInfer autotune——在 V100 上全部无意义 | 本机 ENGINE-MASTER.md §2 + recipe（见 §3） |

**小结**：上游 vLLM 对 V100 = 事实弃用（文档门槛 7.5、构建不编 sm70、注意力后端全线要求 sm80+、性能 10×回退）。可用特性只剩 Triton 注意力后端 + Triton fused MoE + chunked prefill（调度器层、GPU 无关）+ CUDA graph（V100 支持）。

## 2. 上游 vLLM 对 POWER9/ppc64le：红灯（无官方轮子、无 CI、全源码自建）

| # | 事实 | 来源 |
|---|---|---|
| 2.1 | PyPI 无官方 ppc64le 轮子；`pip install vllm` 触发源码构建且依赖链（outlines_core 等）缺 PowerPC 轮子而失败 | https://github.com/vllm-project/vllm/issues/11616 |
| 2.2 | 社区 ppc64le 包全是 **CPU-only**：SIXE 首个 ppc64le .deb（POWER9/10 无 GPU 推理）、LibrePower（POWER10/11 MMA 指令，无 GPU）；IBM 官方 wheel 索引的 torch 也是 CPU-only | https://sixe.be / https://sixe.eu 、https://librepower.org 、本机 THIRD_PARTY_WHEEL_ASSESSMENT.md |
| 2.3 | IBM 官方存在 GPU-on-Power9 的 vLLM 教程/repo（`llm-pt-ibm/vllm_gpu`，Podman 容器）：**LLVM→Triton→PyTorch→vLLM 四层全部源码编译**，无性能数据、无 V100 专项提及——证明可行但零官方维护 | https://github.com/llm-pt-ibm/vllm_gpu 、IBM Community "LLM Inference with vLLM Using GPU on Power9"(2026-04-30) |
| 2.4 | OpenAI Triton 官方轮子只有 x86_64/aarch64，**无 linux_ppc64le**；Triton 在 ppc64le 需自编译 | https://pypi.org/project/triton/ |
| 2.5 | 唯一验证过的 ppc64le+CUDA12.4 Triton 是第三方 fork **RomanMoz/power-triton-cuda124**（3.6.0，明确写着"Tesla V100 sm70 kernels for PyTorch and SGLang"）；本机已用其轮子在 V100 跑通真实 add kernel | https://github.com/RomanMoz/power-triton-cuda124 、本机 STATUS.md/THIRD_PARTY_WHEEL_ASSESSMENT.md |
| 2.6 | PyTorch 同理：官方无 CUDA ppc64le 轮子；本机走的是 RomanMoz/power-torch-cuda124 2.12.0a1（Torch 2.12 preview，非稳定 ABI）+ 自编译 Torch 2.10（SM70 only，RAM 盘构建，SATA 盘禁写） | https://github.com/RomanMoz/power-torch-cuda124/releases/tag/v2.12.0a1 、本机 STATUS.md |

**小结**：ppc64le 上的 vLLM = "四层全自建 + 三方二进制拼装"（Torch/Triton 靠 RomanMoz，vLLM 靠 1Cat fork 或自己编）。本机 2026-09 月已经付出过全套装机成本（约 3 天、含两次 SATA EEH 硬件事故），产物还在 wheels/ 与 runtime-wheelhouse/。

## 3. Qwen3-Next 家族在上游 vLLM 的实现：核心全是 Triton/FlashInfer，验证硬件无 sm70；Flash-Next 官方 recipe 甚至不卖 PyPI

| # | 事实 | 来源 |
|---|---|---|
| 3.1 | `qwen3_next` 模型实现：GDN 线性注意力来自 `vllm/third_party/flash_linear_attention/ops`（**FLA Triton 内核**：chunk_gated_delta_rule 等）+ causal_conv1d + fused_qk_rmsnorm_rope（Triton）+ fused_moe（Triton）；ROCm 另有 aiter 路径 | https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next.py 、…/layers/mamba/gdn/qwen_gdn_linear_attn.py |
| 3.2 | Qwen3-Next 官方 recipe：建议 **4×H200/H20 或 4×A100/A800**；MTP 用 `{"method":"qwen3_next_mtp","num_speculative_tokens":2}`；MoE 为 Triton fused kernel（E=512、N=128 调优文件） | https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3-Next.md |
| 3.3 | **Flash-Next 官方 recipe（2026-09-30 更新）**：`min_vllm_version: 0.29.0`、`nightly_required: true`、专用 Docker 镜像 `vllm/vllm-openai:qwen38-flash-next`，**"PyPI installation is not supported for this recipe"** | https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-Flash-Next.yaml |
| 3.4 | 同 recipe 验证硬件清单：h200/h100/gb200/gb300/mi355x/**rtx_pro_6000_4x**/dgx_station_gb300——最低端是 Blackwell sm120 工作站卡；**无任何 sm70/sm80 消费级或 Volta 条目** | 同上 |
| 3.5 | 同 recipe 关键参数：176B 参数（125B 主干+51B PLE）；BF16 335GiB / FP8 173GiB / **NVFP4 最小 130GiB**（每变体 vram_minimum_gb）；MTP=`{"method":"mtp","num_speculative_tokens":3}`；H100-80G 都要 `VLLM_PLE_CPU_OFFLOAD=1`（PLE 表 51GB）；**N-gram Embedding 不支持 pipeline parallelism**；Hopper 用 `--moe-backend triton`，RTX PRO 6000 用 `--moe-backend marlin` | 同上 |
| 3.6 | 容量推论（本机算术）：6×V100-16GB = 96GiB 显存 < 官方最小 NVFP4 变体 130GiB。**官方任何精度变体都装不进这台机器**，唯一装得下的是 IQ3_XXS GGUF 78.8GB（llama.cpp 生态量化） | 3.5 + ENGINE-MASTER.md §3 |
| 3.7 | 上游 MTP/投机解码（DeepSeek 式 EAGLE、qwen3_next_mtp）内核同样以 Triton/FlashInfer 为主；FlashInfer 预编译内核面向 sm75/80+，无 sm70 cubin | https://github.com/flashinfer-ai/flashinfer （flashinfer-cubin 支持架构列表） |
| 3.8 | 1Cat fork 内 MTP 在 sm70 的实际状况（本机实测）：MTP4 TP2 单流 66 tok/s 但 **4 并发崩溃**（draft tokens list/tensor bug）；**MTP4 单条 ~12k prompt 死锁**（GPU 100% 零进展，MTP3@16k 正常）——已写成待提交上游 issue | 本机 upstream-issue-mtp4-12k-hang.md 、bench-results/2026-09-28-1cat/REPORT.md |

## 4. 服务器/本机摸底：ac922-vllm 目录 = 完整的 1Cat 路线历史（2026-09-27 → 10-05）

ssh 未连通（100.66.42.47:22 超时，符合"网络可能闪断"预告）；以下来自 Mac 侧镜像 `/Users/lyx/Desktop/ssh/ac922-vllm/`（与服务器同步，最后更新 10-05）。

### 4.1 "1cat" 是什么

- **1CatAI/1Cat-vLLM**：GitHub 上的 vLLM 工程化 fork，口号 "Make Volta Fast Again"，把 **V100/SM70 当一等公民**。~1.3k star，最新版 v1.5.1（2026-10-03）。
- sm70 专属内核族：FlashAttention-V100（Volta 原生 dataflow 重写）、**FlashQLA/GDN**（gated delta net）、QSA 稀疏注意力、grouped MoE、custom all-reduce、TurboMind 派生 W4A16/compressed-tensors 内核、FP8 E4M3/E5M2 KV、NVFP4/MXFP4、MTP4 与 DFlash2 投机、GGUF 元数据/tokenizer 适配器（Qwen3.5 dense/MoE、Flash-Next、DFlash2 draft）与 **IQ3/IQ4 混合位宽算子**。
- **发布轮子只有 `linux_x86_64`（cp312, Torch 2.10+CUDA 12.8）**：ppc64le 必须源码自建（本机已建过 Torch2.10 与 Torch2.12 两套）；GGUF/混合加载特性晚于轮子，也需源码构建。来源：https://github.com/1CatAI/1Cat-vLLM
- README 示例：Qwen3.8-27B NVFP4+DFlash2 在 4×V100 ≈260 tok/s（27B 小模型，非 Flash-Next）。

### 4.2 该路线已经跑过什么（时间线与成绩）

| 阶段 | 模型/拓扑 | 最好成绩 | 状态 |
|---|---|---|---|
| 2026-09-28 | Qwen3.8-**27B**-AWQ（nicosuter W4A16，GDN 16/48 头，非 Flash-Next）TP2 eager | 单流 7.5 tok/s、c8 聚合 42.5 | 可用但极慢（eager） |
| 2026-09-28 | 同上 + CUDA Graph | 单流 37.1；**c8 聚合 186** | TP2 无投机最快 |
| 2026-09-28 | 同上 + MTP3/MTP4 graph | MTP3 c4 130.9；**MTP4 单流 66.0**，c4 崩溃 | MTP4 不稳 |
| 2026-09-29 | 27B PP3/TP3 头复制手术 + **DFlash2** draft | 1k/8k/64k 解码 78/67/62，接受率 80.1%，30 分钟 4 路稳定 | 200k 装不下 |
| 2026-09-29 | 27B 原生 MTP4 TP3+FP8 KV | 1k 61.5、64k 69.7、**4×1k all-live 367** | 曾上生产 8000 口 |
| 2026-09-30 | MRV2 runner + FULL_DECODE_ONLY + 5.5GiB 固定 KV | 1k 80 tok/s、128k 输入 60 tok/s | 最后一代生产 |
| **2026-10-03** | **Flash-Next IQ3_XXS GGUF TP4（跨岛）** | **5.18 tok/s**，输出第 1 token 即与 llama.cpp oracle 分岔 | 判死 |
| **2026-10-03** | **Flash-Next GGUF PP2×TP3（3+3 岛）** | **29.19 tok/s**（batch-1 FULL decode graph，无 MTP）；前 5 token 对齐后分岔 | 未过 parity，研究态 |

来源：本机 `ac922-vllm/bench-results/*/REPORT.md`、`onecat-flashnext-experiment/{RESULTS.md, RESULTS-3X3.md, PP3X3-STATE.md}`。

### 4.3 Flash-Next on 1Cat 的死因解剖（对辩论最有用的部分）

1. **GGUF→HF 数值变换深坑**：llama.cpp 转换器对 norm +1 折叠、GDN `A_log→-exp(A_log)`、QKV/Z/V tiled 重排——1Cat loader 直读这些值全部读错。run `l` 用原版 BF16 shard 字节级反演修复后才得到"前 5 token 对齐、29 tok/s"，**512-token parity 至今未过**（index 5 分岔）。（RESULTS-3X3.md "Correctness repair"）
2. **tilelang 缺位**：GDN prefill 的 TileLang 路径在本机不可 import（512-token 请求即 ModuleNotFoundError 退出），只能用 wheel 内预编译 SM70 FlashQLA VLK 路径。（2026-09-28 REPORT + RESULTS.md）
3. **加载极慢**：模型加载 537-550 秒/次（GGUF mmap 流式 + POWER9 CPU），6 并发读者会把主机压进 swap（3 次尝试死于换页门限），必须岛间 stagger 分阶段加载。
4. **结构不整除**：GDN 10240 不被 TP3 整除、QSA KV 头复制、专家宽 640 padding——全部要运行时手术，且手术又引入正确性风险。
5. **eager→graph 差 5.5×**（4.89→27 tok/s）：POWER9 上 kernel launch 开销让一切 eager 路径不可用，而 graph 形状又受 Mamba align 块 784/FP8 页 1568 与 batched-token 上限互相打架。
6. MTP：Flash-Next 的 MTP 头（GGUF block-48）**从未在 1Cat 上加载成功**（"adapt the GGUF block-48 MTP head" 列为 next work）；DFlash2 checkpoint 只属于 27B 模型，不是 Flash-Next drafter。

### 4.4 服务器可用模型（Mac 镜像可证）

- `/home/debian/models/qwen3.8-27b-awq-nicosuter`（W4A16+MTP shard，已双 SHA256 校验）——1Cat 路线的老朋友。
- `/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf`（78.8GB，monster 现用）。
- DFlash2 drafter：`z-lab/Qwen3.8-27B-DFlash2`（3.85GB，仅配 27B）。
- ssh 不通，`/home/debian/models/` 其余内容无法核实。

## 5. 与 monster llama.cpp 现状的对比锚点

| 指标 | monster（llama.cpp fork） | 1Cat-vLLM 历史最好 | 官方 vLLM |
|---|---|---|---|
| Flash-Next 单流 decode | **134-140**（MTP-4） | 29.19（无 MTP、无 parity） | 无法运行（§1/§3） |
| 4 路聚合 | 93-100 | 27B 模型 4×1k 367（all-live）；Flash-Next 未测并发 | 无法运行 |
| prefill | ~1120 | Flash-Next 稳态窗 ~819 | 无法运行 |
| 正确性 | fixture 12/12 + natural ×3 | 前 5 token 后分岔 | N/A |

## 6. 三角红绿灯（辩论结论用）

- **ppc64le 角**：🔴 上游零支持（无轮子/CI），全链自建；唯一证明可行的是本机已付出的 RomanMoz+1Cat 组合（沉没成本已付，wheels 还在）。
- **sm70 角**：🔴 上游事实弃用（7.5 门槛 + 不编译 + 后端 sm80+ + 10× 回退）；FP8/Marlin/FlashAttn 全灭；Triton/MoE/chunked-prefill/CUDA-graph 这几样理论可用。
- **Flash-Next(线性注意力+512 专家 MoE) 角**：🔴 上游实现全 Triton/FlashInfer 且官方 recipe 验证硬件最低 sm120 工作站卡、专用 nightly 镜像、不支持 PP；6×16GB 连最小官方变体都装不下；GGUF 路线在 1Cat 上实测 29 tok/s + parity 未过 + MTP 头从未跑通。
- 综合：**换模型（如 27B-AWQ）时 1Cat 路线是真实存在过的生产 API（4×1k 聚合 367 tok/s 曾上线）**；但**要在 Flash-Next 上走 vLLM 系，等于在三条红灯上同时叠 GGUF 数值反演、TileLang 缺位、加载 9 分钟等未解决问题**，当前证据全面劣于 monster 树。
