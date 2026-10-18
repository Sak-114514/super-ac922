# 跨岛流量敏感度探索分析（CROSS-ISLAND EXPLORATION）

> 2026-10-12。任务：检验用户假说——"XBus 恢复后跨岛已达 PCIe4.0×16 级别，极端的'避免跨岛流量'设计（层流水双岛+专家不跨岛）会不会才是 decode/prefill 慢的元凶？专家跨岛也可以接受。"
> 方法：全部基于实测（拓扑矩阵、一次性 GPU 微基准、源码勘察、历史 server.log 取证）。服务器只读，未起服务，未改任何服务器文件；微基准产物已清理。

---

## 0. 核心结论（TL;DR）

1. **带宽假说不成立**：实测跨岛交接每步仅 ~400KB（日志实证 81920B = 4 token × 10240 × 2B），走 26-28GB/s 的路径只占 decode 步时（21.9ms）的 **0.1-0.2%**、prefill ubatch（915ms）的 **0.1-0.2%**。"避免跨岛流量"没有让我们在**带宽**上付出可见代价。
2. **但结构假说部分成立**：层流水双岛的代价不是流量，而是**结构性 ~50% 岛空转**（依赖链串行 A→B）+ 4 路 MTP 接受率坍塌——这是**调度问题，不是带宽问题**（NVML 1312-1350MHz 恰为 V100 空闲降频档）。
3. **XBus 恢复确实改变了格局**：`cudaDeviceCanAccessPeer` 现在 15 对全 =1（微基准实测），跨岛 P2P 直拷 26-28GB/s。TP6 单岛化（过去因无 peer 而判死）**重新可行**，算术上有 decode 步时 -20~35%、prefill +30~50% 的潜力——这是最值得跑的实验（env 级，零改码）。
4. **"专家跨岛"（折中方案 C）不建议**：跨岛流量比 TP6 更大、还保留串行注意力流水，且 split-state 架构不支持跨 meta 设备分片（真实工程量大）。被方案 B 支配。
5. **意外发现（需要立即行动）**：XBus 恢复后，`GGML_META_ASYNC_COPY=1` 的异步交接桥会在 init 时**静默失效**（代码规定：跨岛有 peer access 且未设 `GGML_CUDA_META_COPY_TOPOLOGY=1` → init 拒绝 → 回退同步 host 中转）。下次重启服务会无声地变慢一点。修复 = 加一个 env。

---

## 1. 实测拓扑（2026-10-12，ssh ac922）

### 1.1 连接矩阵（nvidia-smi topo -m）

| | GPU0 | GPU1 | GPU2 | GPU3 | GPU4 | GPU5 | NUMA |
|---|---|---|---|---|---|---|---|
| **GPU0** | X | NV2 | NV2 | SYS | SYS | SYS | 0（CPU0, 核 0-87）|
| **GPU1** | NV2 | X | NV2 | SYS | SYS | SYS | 0 |
| **GPU2** | NV2 | NV2 | X | SYS | SYS | SYS | 0 |
| **GPU3** | SYS | SYS | SYS | X | NV2 | NV2 | 8（CPU1, 核 88-175）|
| **GPU4** | SYS | SYS | SYS | NV2 | X | NV2 | 8 |
| **GPU5** | SYS | SYS | SYS | NV2 | NV2 | X | 8 |

- 岛内 = **NV2**（每对 2 条 NVLink × 25GB/s = 50GB/s/方向/对）；每卡 6 条 NVLink 全 25GB/s（nvlink -s 实测），岛内两 peer 各占 2 条。
- 跨岛 = **SYS**（GPU→本地 PCIe4.0×16 → XBus → 对侧 PCIe → GPU）。
- 与既有认知一致：GPU0-2/CPU0、GPU3-5/CPU1，双 3 卡 NVLink 岛。

### 1.2 跨岛带宽微基准（一次性，nvcc CUDA 12.4，跑前确认 6 卡显存 0 MiB，~30s）

cudaDeviceCanAccessPeer 矩阵：**15 对全部 = 1（含跨岛 9 对）**，且 cudaDeviceEnablePeerAccess 双向全部成功 → XBus 确已恢复，跨岛 P2P 直通可用。

| 路径 | 64MB | 2MB | 256KB（有效带宽，含延迟）|
|---|---|---|---|
| GPU0→GPU1（岛内 NV2）| —（事件计时毛刺）| **41.0 GB/s** | 21.4 GB/s |
| GPU0→GPU3（跨岛 SYS）| **27.8 GB/s** | 25.9 GB/s | 17.8 GB/s |
| GPU3→GPU0（反向）| ~26 | 26.6 GB/s | 19.7 GB/s |
| GPU2→GPU4（跨岛）| ~26 | 26.0 GB/s | 18.5 GB/s |
| GPU1→GPU5（跨岛）| ~26 | 26.0 GB/s | 18.5 GB/s |
| host 中转：GPU0→pinned→GPU3 | — | D2H 46.9 / H2D 46.9 GB/s，**串行有效 23.4 GB/s** | — |

结论：
- 用户"跨岛 ≈ PCIe4.0×16（~32GB/s）级"的说法**基本属实**：大块 P2P 直拷实测 26-28GB/s（32GB/s 的 85-88%），小块（激活尺寸级）有效 18-20GB/s。
- P2P 直通比 host 中转（23.4GB/s）只快 ~15%，但**免掉同步阻塞**（异步、省一次 host 往返延迟）。
- 岛内 NV2（41-50GB/s/对）与跨岛（26GB/s）差距约 1.6-1.9×，不是数量级差距——这是"单岛化重新可行"的物理基础。

---

## 2. 代码勘察（源码树 /home/debian/ac922-monster/src）

### 2.1 LLAMA_ISLAND_SPLIT 语义与"单岛 6 卡"配置（src/llama.cpp:158-235）

- 解析：逗号分隔的正整数列表，**必须等长、总和 = 设备数**（llama.cpp:182-222）。
- **合法的单岛 6 卡配置**：
  - `unset LLAMA_ISLAND_SPLIT`（或空串）→ `island_sizes = [n_devs]` 单岛（llama.cpp:184-187），即"以前的 behavior"；
  - `LLAMA_ISLAND_SPLIT=6` 同效；`=0` 非法（总和 ≠ 6，报错退出）。
  - 附注：`LLAMA_DRAFT_DETP=1` 也会强制单岛（islands 被忽略），但那会同时改变 MTP 行为，不是干净路径。
- 单岛时 `island_mode = false`（llama-model.cpp:1434，需 `n_devices()>1`）→ 层分配退化为普通 TENSOR 模式：全部 49 块落在唯一 meta 设备上，TP6。
- meta 后端对跨岛 P2P 的处理：meta 设备内的 AllReduce 走 CUDA reg 的 **in-graph P2P AllReduce**（`ggml_backend_p2p_ar_init`，allreduce-p2p.cu:140-170 要求**每对** peer access；失败才回退 NCCL → 再不行 host）。6 卡单岛时 15 对 peer 全 1（实测）→ **in-graph P2P AR 可以直接跑 6 rank，不需要 NCCL**。

### 2.2 岛分配逻辑与"专家跨岛"的最小改动点（src/llama-model.cpp:1425-1505）

- 层→岛：`get_layer_buft_list()`（1460-1470）按 `splits` 边界把**整层**（含全部 512 专家张量）钉在某岛的 buffer type 上。岛界默认 = 块中点（25/24），`LLAMA_ISLAND_LAYERS=<n>` 可显式钉（1437-1443，"blocks 0..n-1 on island 0"）。
- 专家随层钉岛是**按构造成立**的：专家张量是层张量子集，没有独立的设备路由。
- **"专家可跨岛"没有小改动点**：split-state 回调（llama-model.cpp:124-160）是"每张量归属一个 meta 设备"的单设备模型；让 ffn_*_exps 跨两个 meta 设备需要新的跨 meta 分片语义 + MoE 归约路径改造（token 按专家归属 scatter/gather + 输出 AR）≈ 等于给 MoE 单独做 TP6。工程量远超收益（见 §4 方案 C 算术）。
- 最便宜的"岛相关"改动是 env 级的 `LLAMA_ISLAND_LAYERS` 再平衡（岛0 20.2GiB vs 岛1 25.9GiB 专家字节，28% 失衡 → 边界右移 3 块可拉平）。

### 2.3 跨岛激活交接的实现与实际流量（ggml-backend-meta.cpp:2742-2880 + ggml-cuda.cu:3390-3720）

- 门控：`GGML_META_ASYNC_COPY=1`（run-candidate.sh 已开）+ 双方都是 3 成员 meta 设备 + MIRRORED 激活。
- 两条实现路径（ggml-cuda.cu）：
  - **默认（host 中转）**：src 成员0 → pinned host（2 槽双缓冲）→ 异步 H2D 扇出到对岛 3 卡。串行有效 ~23.4GB/s（实测）。
  - **peer 直通（`GGML_CUDA_META_COPY_TOPOLOGY=1`）**：`cudaMemcpyPeerAsync` src[0]→对岛 leader（跨岛 26GB/s）→ leader 岛内 NVLink 扇出 2 卡。`peer_route` 要求 leader 跨岛 peer + 岛内 peer——**当前硬件全满足**（ggml-cuda.cu:3472-3524）。
- **实际每步流量（日志实证）**：`async MIRRORED island copy active ..., bytes=81920`（多个 e0-*/model/server.log）= 4 token × 10240 × 2B。即：
  - decode：每 verify 前向 **1 次 A→B 交接**（draft/MTP 层全在岛1，草稿链不过界），尺寸 = n_tok × 20KB。4 路 × 5 token = 400KB/步。
  - prefill：每 1024-token ubatch 1 次交接 = 1024 × 10240 × 2B = **20.97MB**。
- 岛内 AR：server.log 实证 `in-graph P2P AllReduce on 3 devices`（×3 个 meta 上下文）+ `cold collectives use rank-ordered P2P AllReduce`（GGML_META_P2P_COLD=1 生效）——**稳态 AR 根本不走 NCCL**，走 NVLink P2P 核内自旋。

### 2.4 意外发现：XBus 恢复导致异步交接桥静默失效（回归风险）

`ggml_backend_cuda_meta_copy_init`（ggml-cuda.cu:3567-3574）：**若任一跨岛对 canAccessPeer=1 且未设 TOPOLOGY=1 → init 返回 null → 该源/目的对被永久标记不合格 → 每次交接都走"已同步的 backend 回退"**（阻塞式 host 往返）。

- 历史 e0 日志（v14-v33，XBus 恢复前）显示 `async MIRRORED island copy active`——当时 CanAccessPeer=0，host 中转异步桥正常。
- 我现在实测 CanAccessPeer 跨岛全 1 → **下一次重启服务，异步桥会静默关闭**，交接变成同步阻塞路径。交接字节小（400KB/步），直接损失 ~0.1ms 级 + 每步一次 host 同步的调度抖动，但完全没必要白付。
- **修复（env 级，一行）**：`GGML_CUDA_META_COPY_TOPOLOGY=1` → 启用 peer 直通异步桥（更快且异步）。

---

## 3. 三方案带宽算术

常量：hidden 10240，激活 f16（20KB/token，日志实证）；跨岛 P2P 26GB/s（大块）/ 18-20GB/s（小块有效）；host 中转串行 23.4GB/s；岛内 NV2 41-50GB/s。decode 步 = 21.9ms（5-token 趟）；prefill = 1024-token ubatch ≈ 915ms（1120 tok/s）。AR 次数 = 2/层（attn/GDN 输出投影后 + MoE down 后）× 48 层 = 96/前向。环状 AR 跨岛边界流量 ≈ 2×V/次（RS+AG 各 ~V）。

### 3.1 decode（步预算 21.9ms；4 路 × 5 token → V = 400KB）

| 方案 | 跨岛流量/步 | 纯带宽耗时 | 占步时 | 结构性代价/收益 |
|---|---|---|---|---|
| **A 现状层流水双岛** | 1 次交接 ≈ 0.4MB | 17-42µs（+同步回退 ~0.1ms）| **0.1-0.5%** | 岛空转 ~50%（依赖串行）；岛间负载失衡 28% |
| **B 全 6 卡 TP 单岛** | 96 AR × 2V ≈ 76.8MB | **2.95ms** + AR 延迟 ~1-4ms → 合计 **4-7ms** | 18-32% | 计算减半（每层 6 卡分摊专家权重读取：每层专家字节/GPU 减半）→ busy ~11ms + AR → **步时 ~15-18ms，预期 +20~45%** |
| **C 岛内 TP3 + 专家全局** | MoE AR 48×2V ≈ 38MB + 路由 scatter/gather ~2V/层 ≈ 38MB → **~76MB** | ~3ms | ~14% | 注意力仍串行流水 + MoE 部分 TP6 → **两头不讨好**，且需大改 split-state |

关键洞见：decode 的边际成本 ~3.25ms/token 由**专家权重读取流量**主导（MASTER §1.2）。层流水下每层只有 owning 岛的 3 卡 HBM 在读专家（3×900GB/s）；TP6 下每层 6 卡分摊（6×900GB/s）→ **每层专家读取时间理论上减半**。TP6 不是"多付 AR 换什么"，而是"把空转的另一半 HBM 带宽用起来付 AR"。

### 3.2 prefill（1024-token ubatch，V = 20.97MB，当前 915ms/ubatch）

| 方案 | 跨岛流量/ubatch | 耗时 | 占比 | 判词 |
|---|---|---|---|---|
| **A 现状** | 1 次交接 20.97MB | 0.9ms（P2P）/ 1.8ms（host 串行）| **0.1-0.2%** | 不是瓶颈（与 TWO-TO-FOUR-STREAM-PLAN 的"~3ms vs 107ms"一致）|
| **B TP6** | 96 AR × 2V ≈ 4.03GB | **155ms** @26GB/s（+延迟）| 17-25% | 计算若 915→~460ms（TP6 分摊），总 ~615-660ms → **PP 1550-1670（+38-49%）**；若 NCCL/延迟拉胯（<10GB/s 有效）则 AR 400ms+ → 无收益 |
| **C 折中** | MoE AR+scatter ≈ 4GB | ~155ms | ~17% | 注意力部分仍串行 → 上限 ~+15-30%，工程量最大 |

### 3.3 显存与布局影响（方案 B）

- 专家 46.1GiB：÷3（现状，岛1 每卡 ~10-11GiB 不均衡）→ ÷6 = **7.7GiB/卡均匀**，每卡多出 ~3GiB 余量给 KV/图变体。
- GDN 状态 108MiB/流：MIRRORED ×3 → ×6（648MiB/流），无压力。
- KV（--kv-unified）：25 层存岛0 / 23 层存岛1 → 48 层均分 6 卡，总量不变。
- PLE 表 26.8GiB 在 CPU（行缓存），不受拓扑影响。

---

## 4. 判定：用户假说成立性

| 假说成分 | 判定 | 依据 |
|---|---|---|
| "我们对跨岛**流量**过于敏感" | **不成立** | 交接流量实测占步时 0.1-0.5%、占 prefill 0.1-0.2%；26-28GB/s 的路径跑 400KB 根本不构成瓶颈 |
| "极端避跨岛设计是 decode/prefill 慢的**元凶**" | **半成立——元凶是结构不是流量** | 层流水把 48 层切成两条串行依赖链 → 单流每岛空转 ~50%（岛A 1312-1350MHz = 空闲降频档，非带宽拥堵）；4 路慢的大头另有其人（MTP 接受率 3/4 → 1.6-2.3，MASTER §1.2）。prefill 1120 vs 锚点 1350 的差主要是测量协议（MASTER 已定案），别记在拓扑头上 |
| "专家跨岛也可以接受" | **技术上可行但被支配，不建议** | 跨岛带宽够（26GB/s），但方案 C 流量 ≥ TP6、保留串行注意力、split-state 需大改；同样的工程预算投 TP6 或 B' 岛交错回报更高 |
| （隐含）"XBus 恢复应该改变拓扑决策" | **成立，且是最重要的可行动发现** | CanAccessPeer 全 1（实测）→ 6-rank in-graph P2P AR 可用、peer 直通交接桥可用；TP6 从"判死"变"待验证"，decode +20-45%、prefill +38-49% 的算术上限值得一个 env 级 A/B |

**等待空泡 = 调度问题还是带宽问题？** 调度。三重证据：降频档位恰为空闲特征；交接带宽占用 <0.5%；依赖链串行本质（单流 step = T_A + T_B，两岛各自只忙一半）。解法两族：①让两岛同时忙不同 token（B' 岛交错 / micro-batch 1F1B，TWO-TO-FOUR-STREAM-PLAN 已有规格，不依赖跨岛带宽）；②让每层用全部 6 卡（TP6 单岛化，依赖跨岛 AR——现在可行）。

**已知风险（TP6）**：96 AR/步的跨岛延迟是唯一未知数（P2P 自旋 flag 过 XBus、NCCL 在 POWER9 SYS 上历史口碑差——但稳态可全程走 in-graph P2P AR 不碰 NCCL）；TP3 调优过的内核（GDN_X4、MOE_GROUPED、QSA_SPARSE）在 shard 宽度减半（48 v-heads/6=8）下的小 token GEMV 效率未验证；MTP fold/chain 与 6-rank meta 组合未验证。→ 这正是 E1 要一次性回答的。

---

## 5. 实验方案（按性价比排序，均 env 级、零改码）

### E1. TP6 单岛 A/B（最高优先级：直接判决用户假说的"解法半边"）

- **配置**：feature-env.sh 加 `unset LLAMA_ISLAND_SPLIT`（其余基座 env 全不动，--tensor-split 1,1,1,1,1,1 不变）。对照臂 = 现状 3,3。
- **协议**：新协议（warm 4×128 弃 → 2× 4×512）+ 单流 n=512 流式分段（3/8 段后）+ PP 4×8192。两臂交错。
- **预期信号与判据**：
  - 加载日志须出现 `island mode: 1 islands of 6 devices`（或 island_mode 关闭、单 meta 设备）+ `in-graph P2P AllReduce on 6 devices`（若后者没出现而走 NCCL，记录下来——那是次级发现）。
  - **通过**：单流稳态 TG ≥ 155（vs 134-140，+10% 起）或 4 路聚合 ≥ 110（vs 89-100）。LLAMA_STEP_TIMING 显示步时 ≤ 18ms 且 collective 时间占比 < 6ms。
  - **失败**：步时 ≥ 24ms 或 AR/等待时间 > 8ms → 跨岛 AR 延迟吃掉收益，假说的"解法半边"证伪，资源转投 B' 岛交错。
  - 顺手记录：NVML 是否不再出现岛 A 1312-1350MHz（结构性空转消失的直接证据）。
- **成本**：一次服务重启 + 两波测试，~30 分钟。
- **副臂（可选）**：`GGML_CUDA_P2P_AR_DISABLE=1` 强制 NCCL AR 对照 in-graph P2P，二选一留快的。

### E2. peer 直通交接桥（防回归 + 免费延迟收益，5 分钟）

- **配置**：现状岛拓扑不变，env 加 `GGML_CUDA_META_COPY_TOPOLOGY=1`。
- **动机**：§2.4——XBus 恢复后异步桥会静默失效退化为同步 host 往返；TOPOLOGY=1 启用 peer 直通（26GB/s、异步、免 host 往返）。
- **判据**：日志出现 `meta copy topology: peer source=... staging leader=...` 与 `async MIRRORED island copy active ... bytes=...`；decode 步时 -0.1~0.3ms、PP +0~1%。收益小但确定，且消除一个潜伏回归源。

### E3. 岛负载再平衡（治标空转失衡，10 分钟）

- **配置**：现状拓扑 + `LLAMA_ISLAND_LAYERS=28`（blocks 0-27 归岛0：20.2→~24.7GiB，岛1 25.9→~21.4GiB+output/MTP，拉平 T_A≈T_B；可再试 27）。
- **判据**：NVML 岛 A 空闲降频事件减少；若 FIFO 流水真在填（4 路），聚合 +5-15%；单流预期不变（总 T_A+T_B 不变——不变不算失败）。与 E2 可叠加。

### 不推荐投入

- 方案 C（专家跨岛 split-state 改造）：被 E1 支配，E1 若败它更败（流量更大且仍串行）。
- E1 通过后，B' 岛交错降级为备选（TP6 下无"岛"可交错）；E1 失败则 B' 升回主线。

---

## 6. 证据索引

| 证据 | 位置 |
|---|---|
| 拓扑矩阵 / NVLink 状态 | `nvidia-smi topo -m`、`nvidia-smi nvlink -s -i 0/3`（2026-10-12 实测，本文件 §1.1）|
| CanAccessPeer 全 1、跨岛 26-28GB/s、host 中转 23.4GB/s | 一次性微基准（nvcc sm_70，64MB/2MB/256KB × 10-30 rep，事件计时；产物已清理）|
| 交接尺寸 81920B | `experiments/graph-max-20261011-1757/e0-20261012-v14..v33/model/server.log`：`async MIRRORED island copy active Meta(CUDA0,1,2) -> Meta(CUDA3,4,5), bytes=81920` |
| 岛内 AR 模式 | 同上日志：`ggml_cuda_p2p_ar_init: in-graph P2P AllReduce on 3 devices` + `cold collectives use rank-ordered P2P AllReduce` |
| ISLAND_SPLIT 语义/单岛合法性 | src/llama.cpp:158-235（unset/空=单岛，"6" 合法，"0" 非法，DETP 强制单岛）|
| 层→岛分配 / LLAMA_ISLAND_LAYERS / 岛守卫 | src/llama-model.cpp:1425-1505 |
| 交接两路径 + 静默失效条件 | ggml/src/ggml-backend-meta.cpp:2742-2880；ggml/src/ggml-cuda/ggml-cuda.cu:3390-3720（init 拒绝条件 :3567-3574，peer_route :3472-3524）|
| in-graph AR 的 peer 全对要求 | ggml/src/ggml-cuda/allreduce-p2p.cu:140-170 |
| 运行时 env 基座 | experiments/graph-max-20261011-1757/run-candidate.sh:7-16 |
| 空转 50% / 带宽非瓶颈旧分析 | monster-engine/TWO-TO-FOUR-STREAM-PLAN.md §2.4；DESIGN-BLUEPRINT-v2.md（"跨岛带宽【未实测】13GB/s 推算"——本次已补实测）|

---

## 7. 上机验证结果（2026-10-12 晚，E1/E2/E3 全部执行）

| 实验 | 配置 | AGG m1/m2 | 判决 |
|---|---|---|---|
| E1 v36-tp6 | unset LLAMA_ISLAND_SPLIT | 84.7 / 97.1 | **in-graph P2P AllReduce on 6 devices 激活**（96 AR/步跨岛延迟≈0）；decode 持平双岛基线（88.8/97.4）→ §5 预期的 +20-45% 未兑现，"避跨岛伤害 decode"证伪；PP 149-151（vs 178-186，shard 48heads/6=8 的预填代价）。**副产品：拓扑自由度确认**——6 卡任意切分/再平衡可行，跨岛 P2P 直拷 26-28GB/s 免费可用 |
| E2 v37-d2topo | d2 + GGML_CUDA_META_COPY_TOPOLOGY=1 | 93.3 / 100.1 | peer 直通桥激活（"peer source=0 staging leader=3"）；性能中性 → **采纳为标准配置**（防 XBus 状态变化导致的异步桥静默失效回归） |
| E3 v38-d2bal | d2 + TOPOLOGY + ISLAND_LAYERS=28 | 93.2 / 100.0 | 岛负载再平衡中性 → 稳态 4 路 decode 下岛间失衡空转不是因素（NVML 偶发降频是瞬态等待而非结构性 50% 空转） |

**综合判定**：用户假说的"带宽半边"证伪（交接 0.4MB/步仅占 0.1-0.2%，TP6 也持平），"结构半边"部分成立但空转泡在稳态多路 decode 中不主导——真正的主导仍是 MTP 接受率坍塌 + MoE 边际成本（见 ENGINE-MASTER §1.2）。跨岛探索的净产出：① TOPOLOGY=1 直通桥入栈；② 6 卡拓扑自由度（未来专家再平衡/异构切分的选项空间）；③ 排除一个方向，资源集中到接受率攻击。
