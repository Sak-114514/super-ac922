# 硬件调度层只读审计（2026-10-14）

> 用户假设："算子已非瓶颈，瓶颈是硬件调度"。**判定：成立（带精确化）**——不是调度代码写得差，而是这套调度层为另一种负载形状（单请求流式、形状稳定、小规模同构卡）而写，遇到本机的双岛拓扑 + 低并发形态时暴露结构性空转。
> 证据四路合流：① ggml-cuda 调度层代码探查 ② meta 岛/服务器/MTP 调度代码探查 ③ 本地实测档案 ④ 业界方案对照。
> 注：岛/MTP/FIFO 代码在 `sources/llama-v100-v3/` 快照（非 llama-volta-jusko）；服务器活体树另有 ~6129 行未提交。

## 1. llama 的硬件调度是怎么写的（三层架构）

### 1.1 ggml-cuda 通用层（设备内）
- **发射**：严格逐节点 launch（`ggml-cuda.cu:4363-4443`），无跨节点合并发射；launch 开销全靠 CUDA graph 消化；融合靠节点级 try_fuse（fork 加了 topk-moe/QSA-PP/MoE 归约等十余处）
- **CUDA graph**：key=首节点指针；属性 diff=逐节点 memcmp 整个 tensor 结构+指针+形状（`:2657-2675`）；**两次 warmup 无变化才捕获**（`:4520-4528`）；属性再变回 eager。无上游 capture_sizes 机制
- **流**：每设备 8 流惰性、**无专用拷贝流**；唯一多流机制是 opt-in 的 QKV 三分支 fork/join（硬编码找 attn_norm fan-out=3，默认关 GGML_CUDA_GRAPH_OPT）
- **多卡通信**：内部 AR 自旋内核**仅支持 2 卡**（allreduce.cu:400-406）；>2 卡走 NCCL（F32 小张量/BF16 压缩大张量）；上游"按行 split+主设备聚合"已被 fork 删除
- **内存**：权重逐 tensor cudaMalloc 不池化；compute 临时量 per-(device,stream) 池（VMM 32GB 窗/best-fit 256 槽）；池稳定=图指针稳定的来源

### 1.2 meta 岛后端（fork 专用层，本机拓扑核心）
- **岛**：`LLAMA_ISLAND_SPLIT=3,3` = 两岛各 3 卡，每岛一个 meta 虚拟设备（llama.cpp:158-169）；层按**连续段**分配（岛0=前半层，llama-model.cpp:1762-1772）
- **岛内 TP**：加载期按张量名正则静态切（llama-model.cpp:425-1005）：gate/up=N 切、down=K 切 → 布局代数保证**每层恰好 2 次 AllReduce**（attn out + MoE down），相邻 partial 还能合并（ggml-backend-meta.cpp:2919-3075）
- **岛间**：一次 MIRRORED 激活复制（可选异步桥，硬编码 3 成员岛）；无其它通信
- **FIFO_PIPELINE**：ubatch 间 host-GPU 重叠（输入快照进双缓冲 pinned 槽，host 不等 GPU 就准备下一 ubatch）——是**步间**流水，不是层间
- **MTP GPU chain**：单 seq+greedy 时 draft 链零 host 回读（GPU argmax 直写、hidden 由 GPU carry），链尾一次同步

### 1.3 服务器层
- slot 混批（decode 行优先 + prompt 吃剩余空间）、**无抢占**、KV 不够时 n_batch 减半重试；cache_prompt 走 LCP 前缀相似度挑 slot

## 2. 不适合本设备/负载的点（按伤害排序，全部有实证）

| # | 不适配点 | 代码事实 | 本机实测伤害 |
|---|---|---|---|
| 1 | **单请求假设全面硬化**：FIFO/MTP-FIFO/FOLD/GPU-chain 全部硬性 `n_seq==1`/`n_drafting==1`（llama-context.cpp:1736-1740; speculative.cpp:1512,1859） | 并发>1 时全部静默退化：每步 host 回读 hidden、CPU 采样、batch 形状随活跃序列抖动、深度锁步快序列被拖 | **4 路 MTP 接受率 3/4→2.3→1.6 崩塌的机制根源**；结构图组合 5^L 图税 |
| 2 | **岛间串行依赖链**：层连续段+单向交接，上游 pipeline_parallel 机制存在（graph_compute_async 错峰+事件）但被 `:568 split_mode==LAYER` 挡死，tensor-split 双岛永不激活 | 单流 step = T_岛A + T_岛B | **每岛空转 ~50%**（NVML 1312-1350MHz=空闲降频档；交接带宽占用 <0.5%；带宽下限 ~3ms vs compute 107ms） |
| 3 | **形状抖动 vs 图复用战争**：verify 行数天然 =1+接受数抖动 → rebuild 慢路（host 全图重遍历+split-state 重算+gallocr 全 sync 重分配+图重捕获两次） | fork 用 LLAMA_MTP_PAD/LLAMA_SPEC_PAD_MULTI 硬钉形状治标 | **v8 干净 decode 窗口 155.2ms = compute 107.7 + alloc 39.4（100% 图重建慢路）** = 25% 步时 |
| 4 | **预填独占无分块**：无 chunked prefill/混合批，长 prompt 一步接一步占满，decode 全停 | server 层 cont_batching 混批仅限 n_batch 空间 | **4 路整波 49.9s 中 decode 仅 9-10s，波墙 80% 是预填**（Sarathi-Serve 所治之病） |
| 5 | **默认假设无 P2P/弱互联**：内部 AR 仅 2 卡；peer access 要 env 显式开；AR 主路径按 PCIe staging 设计 | XBus 恢复后 15 对全 peer（实测）但代码未跟进 | TP6 单岛化（decode +20-35% 算术潜力）从未被启用；异步桥在 peer 可用时会静默失效（需 GGML_CUDA_META_COPY_TOPOLOGY=1 防回归，已采纳） |
| 6 | **POWER9 弱单核零补偿**：发射=逐节点 launch，PDL 仅 Hopper+，无批量发射 | ggml-cuda 内零 POWER9 引用 | host 侧发射延迟高于 x86；graph 关闭时代价放大 |
| 7 | 小项：异步桥硬编码 3 成员岛；岛大小必须相等；NCCL 阈值启发式只写到 4 卡；预取器有效率仅 11.6%（激活≠有效） | — | — |

## 3. 业界方案对照（联网调研 2026-10-14）

| 我们的不适配点 | 业界对应方案 | 状态 |
|---|---|---|
| 岛空转 50%（#2） | 1F1B/interleaved/Seq1F1B 微批流水（GPipe→Megatron→[Seq1F1B](https://arxiv.org/html/2406.03488v5)）；[Primus-Pipeline](https://rocm.blogs.amd.com/software-tools-optimization/primus-pipeline/README.html) | TWO-TO-FOUR-STREAM-PLAN §B' 已规格化（lane 拆 2 微批错峰进岛，~2× 上限）；挡板= :568 条件放宽（中工程） |
| 预填独占（#4） | [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf) chunked prefill + stall-free（OSDI'24，32K 最差 TTFT -68%）；vLLM/DeepSpeed-FastGen 已内置 | 旧线 FAIR_SHARE 预填聚合 +83% 零成本件未并入 |
| 计算通信重叠 | [TBO/Dual Batch Overlap](https://docs.vllm.ai/en/latest/design/dbo)（vLLM：批拆两微批，算一个通信一个）；[DualPipe](https://github.com/deepseek-ai/dualpipe)（DeepSeek 双向流水近零气泡） | TP6 化后的 AR 隐藏可参考；批小时 TBO 无效（需投机扩批配合） |
| MTP 并发崩塌（#1） | [TETRIS](https://arxiv.org/html/2502.15197v1)（位置相关接受率→大块先填的 draft 分配）；DSpark（投机=调度问题，V4 上 +51-400%）；[Batch Speculative Done Right](https://arxiv.org/html/2510.27656v2) | 需先把 n_seq==1 硬化拆掉（red-team 级改动）；goodput 自适应深度已列第二梯队 |
| 图税（#3） | vLLM piecewise/形状桶化、TensorRT-LLM capture | 重建慢路消灭已列第二梯队#1（高水位 reserve+桶键控，+5-15%） |
| MoE 分布式 | [EP+PP 混合](https://www.premai.io/blog/multi-gpu-llm-inference-tp-vs-pp-vs-ep-parallelism-guide-2026)、[vLLM MoE Playbook](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)、[通信高效 MoE 推理](https://arxiv.org/html/2503.04398v3) | 专家不跨岛约束可再评估（XBus 已通） |

## 4. 结论

用户判断与代码/实测/业界三面证据一致：**算子层（dp4a/TC）在当前负载形状下已接近带宽饱和，剩余大头是调度结构性的**——单请求硬化×岛串行链×形状抖动税×预填独占，四个都是调度层属性。修复杠杆全部已在 TWO-TO-FOUR-STREAM-PLAN §3/§6 规格化（乘法账 78.6 → ~247 聚合），与业界路线一一对应。本审计只读未改任何文件。
