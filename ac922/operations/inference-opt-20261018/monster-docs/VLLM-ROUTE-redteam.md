# 红队报告：攻击"换模型 / 换 1cat-vllm"路线

> 2026-10-12。任务：找出该提案死在哪、多快死、代价多大。只调研+推理，不构建不运行。
> **取证说明**：ssh ac922 三次连接 + Tailscale ping 全部超时（闪断），本报告基于**本地完整镜像**取证——`/Users/lyx/Desktop/ssh/ac922-vllm/`（服务器 `~/ac922-vllm` 的镜像，含 STATUS.md 40KB 全程记录、bench-results 六轮基准）、`opensource/super-ac922-clone/`（1Cat-vLLM 完整 git 克隆，含 AC922 试验提交）、`reports/`（1cat-vllm-flash-next-eval、FLASHNEXT-FULL-EXPERIMENT-REPORT、onecat-six-run-20261007 等）——证据等级足够；web 佐证另注。服务器侧待验证项在 §6.3 列出。

---

## 0. 结论速览

| 路线 | 判决 | 死因一句话 | 置信度 |
|---|---|---|---|
| A. Flash-Next 换 1cat-vllm | **死透（两次弃线，尸检报告齐全）** | 正确性门未过（首 token 即错）+ 16GB 显存墙（TP4 需 24GB/rank）+ PLE 728× I/O 放大 | 95% |
| B. 换更小的 MoE 模型（留守 monster 树） | **不划算** | 边际成本只线性降，量化推演仅 ~1.55×，够不到 2×；3000 行 per-model 定制 + 3 天正确性关账全部重做 | 80% |
| C. 换 dense 小模型（如 27B）留守 monster 树 | **反向** | MoE 稀疏激活在本机是资产不是负债：dense 27B 每 token 读全量权重（~5× 字节/token），HBM 3-7% 效率下更慢 | 85% |
| D. 换模型 = 退回 27B-AWQ @ 1Cat 生产配置 | **唯一活口，但它是产品决策不是工程决策** | 质量降级换吞吐；实测 4 路 ~180 聚合（30 分钟压测）/116 @4×50k；仅占 3 卡 | 实测事实，无需置信度 |
| E. 留守 monster + 接受率/内核线 | **推荐** | v39 已证瓶颈=内核效率+接受率，路线图有乘法账到 62/流；无未知深度的正确性地雷 | 75% |

**最终排序：E > D（仅当用户接受 27B 质量）> B > A ≈ C（并列死）。**

---

## 1. 三重角落逐层攻击：ppc64le × sm70 × GDN-MoE-PLE

三层各自有独立致命伤，任意一层死 = 整线死。**本案三层全死。**

### 1.1 ppc64le 层：整条 Python GPU 栈无官方轮子，全靠自建 + 临时兼容层

- **PyTorch 无官方 CUDA ppc64le 轮子**：IBM 公开轮子索引的 torch 是 **CPU-only**（本地 THIRD_PARTY_WHEEL_ASSESSMENT.md 实测）。本案曾用第三方 `RomanMoz/power-torch-cuda124` 2.12.0a1 **预览版**轮子——缺 cuDNN9/libomp/CUPTI，要手工从 NVIDIA ppc64le 归档+Debian 包拼 LD_LIBRARY_PATH；另一条 torch 2.10 线是**本机 RAM 内源码编译**（18GiB tmpfs ext4 镜像，MAX_JOBS=2-3，多天）。
- **vLLM 官方只有 linux_x86_64 轮子**（THIRD_PARTY_WHEEL_ASSESSMENT 原文："The 1Cat-vLLM release wheel is linux_x86_64 and cannot be installed on this ppc64le host"）。ppc64le 上 vLLM 必须源码编，且上游 metadata 对 ppc64le **直接把 triton 依赖标记排除**（`platform_machine != "ppc64le"`）。
- **Triton 无 ppc64le 轮子**：PyPI 只发 x86_64/aarch64。本机的 "POWER Triton 3.6" 是自建轮（bench-results/2026-09-28-1cat/REPORT.md 记载在役）。
- **numba 无 ppc64le 轮子**：DFlash2 的 ngram 助手 `from numba import njit` 在导入链上硬依赖，只能 stub（TP3-DFLASH-PLAN §7 明载三选一全是妥协）。
- **NVIDIA 已弃 ppc64le CUDA 镜像**，驱动生态靠 525 兼容栈硬撑（本案 `with-ac922-cuda525` 包装器 + cuda-compat-12-4 用户态库拼装）。
- **运维死穴：构建环境在 tmpfs，重启即蒸发**。编译期间的 SATA EEH 事故 3 次（console-failure ×2、model-load-failure ×1，Marvell 88SE9235 DMA 病），每次 ext4 日志中止、强制 fsck。**每次依赖升级 = 重启多天 RAM 编译 + 再赌一次盘**。
- Web 佐证：vLLM 无 ppc64le GPU 支持（仅 CPU 后端列 Power9）；[ppc64le build-scripts 社区仓](https://github.com/ppc64le/build-scripts)、[IBM 轮子索引](https://wheels.developerfirst.ibm.com) 均为源码/半官方路线。

**小结**：这不是"装个 vLLM"，是维护一整套冻在 2026-09-27（db292f9a4）/2026-10-07（bc01f2899）快照上的自建发行版。任何上游 bump 全部重来。好消息：核心轮子已存档发布（torch 155MB + 1cat 223MB，`ppc64le-artifacts-20261007` release）；坏消息：**冻结 = 永远追不上上游修复**。

### 1.2 sm70 层：上游全线路弃 Volta

- **vLLM 0.20.0+ 官方不再支持 V100/sm70**（[官方论坛确认](https://discuss.vllm.ai)），V1 引擎依赖的 SymmMem 通信器在 capability 7.0 不存在（"Device capability 7.0 not supported"）；**CUDA 13 整体移除 Volta 目标**（[daily.dev 报道](https://daily.dev)）。社区只剩 [vllm-v100](https://github.com) 类专门 fork。
- **FlashAttention-2 要求 SM80+**，V100 不支持（[Dao-AILab/flash-attention #1760](https://github.com/Dao-AILab/flash-attention/issues/1760)）——vLLM 主力注意力后端直接缺席，只能靠 fork 自带的 Flash-V100/FlashQLA 特化核（= 又回到 1Cat 私有内核）。
- **Marlin sm75+ 编不过 Volta**（本地 SASS 审计实测原话："1Cat marlin 是 sm_75+ 编不过 Volta"）——W4A16 量化 GEMM 快路径缺席。
- **custom allreduce 在 V100 有挂死历史**（fork 已知 issue #652/#430，TP3-DFLASH-PLAN §6 明令 SM70 必须走 NCCL allreduce）。
- **fla（flash-linear-attention）gated-delta-rule Triton 内核是 pre-Ampere 常规雷区**——社区靠专门 patch 才能在 V100 跑（[dg1kjd/vllm-v100-sxm2-qwen3.5-397b](https://github.com/dg1kjd/vllm-v100-sxm2-qwen3.5-397b)——注意它也是 1Cat-vLLM 下游 fork，但目标是 **8×V100-32GB、x86**；[haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100) 同类）。
- 本机实测的 sm70 专属 bug 存量：**MTP4 @~12k prompt 死锁**（GPU 100% 空转，11 分钟零进度，只能杀进程；MTP3@16k 正常——已起草上游 issue，upstream-issue-mtp4-12k-hang.md）；**MTP 输出坍缩质量 bug**（2026-06 审计：长输出退化成 `555...`，靠 5 个 env 门修）；GDN prefill >16k 挂死（上游 #488 同签名）。

### 1.3 qwen4exp（GDN+QSA+512 专家+PLE+MTP）层：三个硬墙

1. **显存墙（TP4 dual-bank）**：1Cat 上游 Flash-Next headline（C1 92.3 / C4 280.4 聚合）的账本是 **4×V100-32GB**：模型 24.18 GiB/rank，且 dual-bank（原始字节块 + canonical 转码块并存，每 rank ~24GB）。**16GB 卡物理装不下**——zcode-tp3-catchup 四连实测判死："TP4=双银行墙……1Cat 用 32GB 卡正因此"。TP6 也不行（vocab 248320%6≠0 + 512 专家/6 需结构适配）；DP2×TP3 显存爆。唯一能装下的拓扑 = **TP3×PP2**。
2. **PLE 死结（vLLM GGUF 路径）**：PLE 表 26.8GiB 只能留 CPU。vLLM 路径 = 单 CPU 进程持整表 mmap、无页合并无预取，**每新行 64KiB 缺页（728× I/O 放大），冷 8K 请求 845MB I/O**；为小 RAM 设计的 cascade 三层被 harness 两行配置堵死（VLLM_PLE_CPU_OFFLOAD=1 + ple_disk_cascade:False）。修复链 = cascade 开关→页合并内核(20-40 行)→行缓存(~100 行)，全部未落。
3. **正确性墙（GGUF 布局逆转）**：Flash-Next GGUF→vLLM 的权重映射有一族未解分岔：parity 追踪已排除 PLE 哈希/卷积/行号/输出头/竞态/编译轨道/权重变换，收敛到 **GGUF 布局逆转**（148 个 norm 预加 1、36 个 ssm_norm 不能减、ssm_a 负指数→A_log、V 头 tiled 重排、in_proj_qkvz 拆分例外）——run l 集成验证**前 5 token 与 llama.cpp oracle 一致、第 6 token 起分岔未定位**；终局六卡 run 直接 `stopped_first_token_mismatch`（见 §2.3）。最坏假说若成立（共享 GGUF 反变换错误），**上游 TP4 也从未 token 对拍正确**——上游 92.3/280 只是吞吐数。

---

## 2. "1cat-vllm" 是什么：身份、家谱、两次死亡

### 2.1 身份

**1Cat-vLLM = GitHub `1CatAI/1Cat-vLLM`（计划改名 1Cat-LLM）**：vLLM 社区 fork，专攻 SM70/V100 + Qwen3.x（qwen3_5 GDN 含 SM70 Triton split kernel、qwen3_next/qwen3_5 MTP、DFlash2 投机解码、GGUF loader 支持 IQ3_XXS/IQ4_NL、SM70 TurboMind/Marlin 后端、Flash-V100/FlashQLA/flashinfer-sm70 特化内核）。RELEASE 1.5.0 = "Qwen3.8 DFlash2 实用化"。本地克隆 `opensource/super-ac922-clone` = 公开镜像 `Sak-114514/super-ac922`（上游 db292f9a4 + 本机 5 个 AC922 提交，含 AC922 可复现构建、TP3 试验全套）。命名注：**"1cat/onecat" 同源**——monster 树里的 ONECAT 内核（ple-conv-sm70.cu）就是从这份遗产移植的。

服务器 `/home/debian/ac922-vllm/` = 该线的主工作区（本地有完整镜像）：torch 源码编译脚本、RAM 构建体系、memory/PSU/存储三套守护、27B 服务脚本（serve_1cat_525.sh 等）、runtime-wheelhouse（90 项）。

### 2.2 它曾经活着：27B 生产线的辉煌（2026-09-28~30，全部实测）

- **栈**：1Cat `0.1.dev1+gdb292f9a4` + torch 2.12.0a1（第三方预览轮）+ POWER Triton 3.6（自建）+ NCCL 2.27.7，ppc64le 源码编译，RAM tmpfs 构建。
- **Qwen3.8-27B AWQ（nicosuter W4A16）TP3（GPU0/1/2，NVLink 岛内）+ 无损头复制 + 原生 MTP4 生产**：
  - 单流 64k 解码 **69.74 tok/s**（接受率 83.9%）；1k prefill 1531 tok/s。
  - **4×50k 全活解码 116.30 tok/s 聚合**（每路 ~29）；4×1k 波次 **367-370 tok/s 聚合（每路中位 81）**。
  - DFlash2 前代：110 单流（82.2% 接受），**4 路持续 30 分钟 ~180 tok/s 聚合（每路 ~45）**。
  - 正确性已关账：greedy parity 20/20 vs TP2、OpenAI+Anthropic 端点 200、KV 池 227,555 tokens。
  - 显存极限操作：0.88 配额、CUDA Graph 生产 1.13GiB/卡 vs 试车 0.37 的惊喜、余量 183-577MiB/卡。
- **但**：MTP4 在 ~12k prompt 死锁（上游 issue 草稿在案）；eager 模式下基线只有 7.5-42.5 tok/s（图模式才可用）；10 条环境雷区（USE_LIBUV=0/tilelang 缺失/flash_qla ABI/E4M3 rev4/mamba block 2080/async scheduling 竞态…TP3-DFLASH-PLAN §8）。

### 2.3 它为 Flash-Next 死了两次（2026-10-02~08）

**第一次（Codex，10-02~03，`onecat-flashnext-experiment`）**：
- 评估乐观（"可行，唯一大缺口是 PLE"，预估数天-一周）→ 实际打进 29.19 tok/s（run l）就停摆：**集成验证第 6 token 起与 oracle 分岔，未定位**；prefill TP4 线 819 但 decode 5.18。**Codex 额度耗尽，用户裁决弃线**（FLASHNEXT-FULL-EXPERIMENT-REPORT §2 原文）。

**第二次（ZCode，10-07~08，`onecat-six-run-20261007`，同步上游 bc01f2899 后六卡 TP3×PP2 总攻）**：
- 拓扑判决：TP4/TP6/DP2×TP3 全灭，**只剩 TP3×PP2**。
- 基线里程碑：**decode 19.8 tok/s**（每轮 50.5ms 含 draft+verify）、prefill 196 tok/s（PLE 冷缺页 ~16s + JIT 尖峰）。归因：**decode ×3.8 差距 = 纯 MTP 接受率 0%**。
- parity 攻关未竟：静默型结构 bug，嫌疑收敛到 GDN FlashQLA prefill@TP3 几何 / MoE@768 / PP2 传输 / **共享 GGUF 反变换（若成立则上游从未正确）**。
- **终局实锤**：两次计时 run 的 status 分别是 **`stopped_first_token_mismatch`（09:16）** 和 **`graph_replay_unverified`（10:22）**（benchmark-091642.json / benchmark-102252.json）——**正确性门都没过，从未产出一条有效计时**。此后全线转 monster。

**结论**：1cat-vllm 不是"没试过的方案"，是**在这台机器上为这个模型专门攻过两轮、留有完整尸检的废案**。27B 生产配置仍可一条命令回滚复活（DFlash/MTP unit 都在），但 Flash-Next on 1Cat 的尸体上有三发独立子弹（正确性、显存、PLE I/O）。

### 2.4 上游数字的不可迁移性（防"抄近道"幻想）

1Cat 上游 Flash-Next 92.3/280.4 的契约：**4×V100-SXM2-32GB、TP4、驱动 580、Torch 2.10/CUDA 12.8、x86**；C4 用 **128-token 短 prompt 合成重复文本 + ignore_eos**（自然负载回落 ~217-227）。换到本机 = 16GB 卡（TP4 物理不可能）+ ppc64le（自建栈冻在 10-07）+ 真实 8K 语料 + 正确性未过。**每一个契约维度都对我们不利。**

---

## 3. 换模型的隐性成本清单（攻击项 3）

1. **正确性重新关账**：本线花了 3 天（GDN head_offset 索引 bug 修复 + FP-noise 容差门三重证据 + fixture 12/12）才拿到"fixed 逐位 → natural ×3 → 单流锚点"的验证链。任何新模型 = 新 fixture 全套重做。而 1Cat 线的 GGUF 布局逆转问题说明：**换框架不豁免正确性成本，反而把它放大成未知深度**（run l 第 6 token 分岔追了两天没追到根因）。
2. **3000 行 monster 定制全是 per-model 的**：GDN_X4、QSA sparse/gather fused-indexer、MTP fold/gpu_chain/gpu_carry、双岛层切分（48 层=36 GDN+12 QSA 的特定布局）、MOE_GROUPED/REWORK（512 专家 top-10 特化）、PLE 行缓存、结构图……换模型 ≈ 重做大半。对照：这些在 1Cat 线的对应物同样是 per-model 的（qwen3_5.py 特化内核、mamba_align、TP3 头复制手术），**两边都没有"免费换模型"**。
3. **IQ 系列对 GDN 层不友好有实证**：Strata 线的 MTP 转换器 Q8→Q2 双重量化 lineage 使接受率从 0.662 掉到 0.376（收档报告实测）。新 GGUF 的量化 lineage 必须重新验证接受率与质量，不是拿来就能跑。
4. **KV/状态显存重算**：GDN 状态 108MiB/流（岛内 MIRRORED ×2）是头几何决定的；换成全注意力模型则 64K 上下文 KV 成为新大头（K8V4 扩容实测 3.51GiB/卡 @262K）；换 MoE 则 PLE 表（26.8GiB）这类结构或消失或变形——**当前显存账本全部作废重算**。
5. **数据集/锚点作废**：kv262k-realtext prompt 库、单流锚点、4 路协议全部按 Flash-Next 建立，跨模型不可比，性能回归链断裂。

---

## 4. 沉没成本反诉（攻击项 4）：物理账不撒谎

**v39 判决：瓶颈 = 小批量内核效率（HBM 3-7%、SM 100%）。换框架不改变 sm70 内核物理。**

1. **1Cat 自己的内核也是手搓 sm70，长上下文 decode 同样撞墙**：27B 在 TP2/TP3 上 64k 单流 62-70 tok/s、每轮 ~50ms 量级——与 monster 每轮 21.9ms（5-token 轮）同数量级。六卡 Flash-Next TP3×PP2 每轮 50.5ms。**这不是框架问题，是 V100 小批量 GEMV/GDN 递推的物理**。vLLM 的 continuous batching 优势在批多请求时释放（4×1k 370 聚合），但 4×50k 长上下文只剩 116——**恰好在用户的目标负载（长上下文 4 路）上优势最小**。
2. **换模型只线性缩每 token 字节，而账是"固定+边际"两项**：步时模型 = 固定 ~6ms + ~3.25ms/token（MoE 专家流量主导）。换一个"每 token 字节减半"的同构小 MoE：边际降到 ~1.6ms/token，4 路轮从 4×21.9=87.6ms → 4×(6+1.6×5)=56ms，每路 ~40 tok/s（接受率 2.3 不变）——**仍不到 50**。ENGINE-MASTER 的判词成立："只靠批不能到——必须同时救接受率"。**换模型与救接受率不互替，而接受率工程（n_seq==1 门放宽）只存在于 monster 树**。
3. **换 dense 模型是反向操作**：Flash-Next 是 167B 总参/6B 激活——每 token 只读 top-10 激活专家。27B dense 每 token 读全量 ~13GB 权重（monster 树口径下字节/token **升 ~5×**）。在 HBM 3-7% 效率的机器上，**MoE 稀疏激活是资产不是负债**。1Cat 线 27B 跑得动靠的是 vLLM 批处理架构 + AWQ Marlin/TurboMind 路径，不是字节更少。
4. **留守路线的账面**：接受率线（单流 ~3 vs 多路 1.6-2.3 的差距）+ SPEC_PAD_MULTI + d2 深度（已实测 99.9-100.1 历史最佳）+ B' 岛交错（理论上限 ×2，v36 已证跨岛 AR 免费）——乘法账 78.6×1.11×1.3×1.8×1.21 ≈ **247 聚合 ≈ 62/流**。内核线（P3：IQ 离线 LUT 重打包 + CUTLASS 2.x Volta 寄存器双缓冲，预算 4-6 周）是最后的大杠杆。**对比：1Cat 复活 = 未知深度的正确性地雷（最坏情形上游从未对）+ PLE 修复链（cascade/页合并/行缓存）+ MTP 接受率 0% 归因，三口井都不知道多深。**

---

## 5. 反方平衡（攻击项 5）：换路线最可能赢的三种情形

红队要公允。以下是"换"能赢的**具体**情形，全部有实测支撑：

1. **【最强】接受 27B 质量 → 一条命令复活 1Cat 27B 生产配置**：这是**唯一已被 30 分钟压测验证过的接近目标的配置**——4 路持续 ~180 聚合（每路 ~45）、4×1k 每路中位 81、4×50k 聚合 116。若目标负载偏 8K 级上下文，每路大概率落在 45-65 区间（4×1k 81 与 4×50k 29 之间插值），**有机会直接达标 4×50**。代价：模型能力从 167B/6B MoE 降到 27B dense（这是产品决策）；占用 GPU0-2（monster 需让路或共存调试）；运维回到 torch2.12 预览轮+525 兼容栈的脆弱世界。**判词：作为"第二生产线复活"值得一次 1-2 天的验证波，作为"替换 monster 主线"不成立。**
2. **若 1Cat-LLM 上游出现 ppc64le+16GB+token 对拍三全其美的路径**（例如 dual-bank 改单 bank 换取 16GB 可装、GGUF 反变换被官方修掉、PLE cascade 落地）：其 mamba_align 批量 verify 架构对多路并发的根本优势（C4 每流保持单流 76% vs monster 多路每路 15-19%）是 monster 树结构性学不来的。**监控点：1Cat-LLM repo + dg1kjd fork 的 ppc64le/正确性动向。两周内出现则重评。**
3. **若存在"每 token 字节 ≤ 一半且质量可接受"的同构小 MoE GGUF 现成检查点**（如 80B-A3B 级）：量化推演每路 ~40 tok/s（§4.2），叠加 d2 深度+接受率修复可望 >50。但前提苛刻：现成高质量 GGUF（IQ 系列对 GDN 层的 lineage 验证过）+ 重做 per-model 定制的预算。**这实际上是"留守路线的模型降级子选项"，不是换框架。**

---

## 6. 最终裁决（攻击项 6）

### 6.1 排序

1. **留守 monster + 接受率/内核线（置信度 75%）**——正确性已关账是唯一无地雷资产；v39 已把问题定位到内核效率+接受率两个可工程化的点；乘法账到 62/流虽乐观但每一段都有实测锚。P3 内核线 4-6 周是最后大杠杆。
2. **27B@1Cat 生产配置复活（仅当质量可接受；验证成本 1-2 天）**——不是替换主线，是"快速第二生产线"选项；GPU0-2 与 monster 的共存/分时是前置。
3. **换小 MoE 模型留守 monster（置信度低）**——只在"现成验证过的 GGUF"存在时值得评估，否则正确性+定制重做的成本吞掉收益。
4. **Flash-Next on 1cat-vllm（死，置信度 95%）**——三发子弹（正确性门 first_token_mismatch、TP4 显存墙、PLE 728× I/O）+ 两次弃线史 + 上游数字契约四维全不利。**重启条件只有一个：上游修掉 GGUF 反变换且有人做出 16GB 可装的拓扑。**

### 6.2 两周内的决定性信号

| 信号 | 判什么 | 期限 |
|---|---|---|
| v31 PATH_LOG 干净配置判决（eager/replay 比） | eager 高 → 形状稳定化（SPEC_PAD_MULTI/桶化）是快钥匙；replay 为主 → 确认走接受率+P3 内核 | 本周 |
| MTP 接受率统计（verbosity 4 / /metrics） | 单流 ~3 vs 多路 1.6-2.3 的差距若定位到 n_seq==1 门 → 中工程放宽后 4 路能否 ≥120 聚合 | 3-5 天 |
| d2 深度 + SPEC_PAD_MULTI=24 联测 | 固定形状收益是否叠上 d2 的 +4-5% | 3 天 |
| 27B@1Cat 复活验证波（monster 暂停，GPU0-2，wave-par 同协议 4×8K/4×50K） | 每路 45-65 是否成立 → 决定第二生产线 | 1-2 天 |
| 1Cat-LLM / dg1kjd 上游动向（GGUF 反变换、16GB、ppc64le） | 出现"三全"路径才重开 Flash-Next 线 | 持续 |

### 6.3 ssh 恢复后的服务器侧验证清单（本地镜像未覆盖项）

- `/home/debian/ac922-vllm/experiments/upstream-six-20261007/results/` 六卡 run 的完整 journal（本地只有两个失败 JSON 的头部）；
- 27B 生产 unit（ac922-coding-api-tp3-mtp.service）与模型文件是否完好、GPU0-2 是否可空出；
- runtime-wheelhouse 与 ram-build 是否还在（重启即失）；
- 1Cat runtime 是否随 10 月以来任何系统变更（kernel/驱动）失效。

---

## 附录：证据索引

**本地（均为绝对路径）**
- 身份与克隆：`/Users/lyx/Desktop/ssh/opensource/super-ac922-clone/`（origin github.com/Sak-114514/super-ac922；RELEASE.md、ac922/tp3-dflash2-trial/README.md、ac922/tp3-mtp4-trial/README.md、SM70_MTP_OUTPUT_QUALITY_AUDIT_20260616.md）
- 全程状态：`/Users/lyx/Desktop/ssh/ac922-vllm/STATUS.md`、`THIRD_PARTY_WHEEL_ASSESSMENT.md`、`TP3-DFLASH-PLAN.md`、`upstream-issue-mtp4-12k-hang.md`、`bench-results/2026-09-28-1cat/REPORT.md`
- Flash-Next 尸检：`/Users/lyx/Desktop/ssh/reports/FLASHNEXT-FULL-EXPERIMENT-REPORT.md`（§1.4 vLLM catch-up、§2 1Cat 线）、`1cat-vllm-flash-next-eval.md`、`onecat-six-run-20261007/benchmark-091642.json`（stopped_first_token_mismatch）、`benchmark-102252.json`（graph_replay_unverified）、`onecat-followup-scouting.md`（上游 32GB 契约/C4 280 拆解）、`onecat-six-run-20261007/TP4-REFERENCE.md`
- monster 现状：`/Users/lyx/Desktop/ssh/monster-engine/ENGINE-MASTER.md`、`ALL-EXPERIMENTS.md`（v26c-v38）、`TWO-TO-FOUR-STREAM-PLAN.md`（乘法账 247）、`DESIGN-BLUEPRINT-v2.md`（P3 预算 4-6 周、Marlin sm75+ 原话）

**Web**
- [vLLM 论坛：0.20+ 弃 V100/sm70](https://discuss.vllm.ai) · [CUDA 13 移除 Volta](https://daily.dev) · [vllm-v100 fork：SymmMem 不支持 7.0](https://github.com)
- [Dao-AILab/flash-attention #1760：FA2 需 SM80+](https://github.com/Dao-AILab/flash-attention/issues/1760)
- [flashinfer-ai/flashinfer：最低 sm70](https://github.com/flashinfer-ai/flashinfer)（但无 ppc64le 轮子）
- [triton 无 ppc64le 轮子 / vLLM 在 ppc64le 排除 triton 依赖](https://github.com/triton-lang/triton)、[IBM ppc64le 轮子索引（torch 仅 CPU）](https://wheels.developerfirst.ibm.com)、[ppc64le build-scripts](https://github.com/ppc64le/build-scripts)
- [dg1kjd/vllm-v100-sxm2-qwen3.5-397b（1Cat 下游，8×V100-32GB x86）](https://github.com/dg1kjd/vllm-v100-sxm2-qwen3.5-397b)、[haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100)——gated-delta-rule 是 pre-Ampere 常规雷区的旁证
