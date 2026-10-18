# 特化引擎设计纲要 v2（三路审核后修订版）
单架构 · 单 checkpoint · 单机器 · 4-8 流延迟优先
v1→v2 变更：并入三路审核的全部修正（服务器代码实证 / 网络源码论证 / 本地报告交叉核对）。
审核方法：每条承重断言按"确认（file:line 或 URL）/ 修正（真值+证据）/ 无法核验"三裁。
**未实测项单独标注，禁止当事实用。**

## 0. 定位（不变）
从 llama.cpp fork（1afa565 winner，单流 1350.8 PP / 150.7 TG）瘦身出单架构引擎：
砍 69%（852k→~265k 行），保住 V100 特化内核，调度层重写为多流延迟优先，
PLE = CPU 常驻字节 + GPU 计算 + 行缓存/预取，专家随层驻岛。

## 1. 硬件事实表（全部实测核验）
| 资源 | 事实（证据） | 含义 |
|---|---|---|
| GPU | 6×V100-SXM2-16GB SM70：fp16 TC(m8n8k4)✓ dp4a✓；bf16/fp8/int8-TC✗ cp.async✗（TRT #993：V100 int8 慢于 fp16 TC；Marlin #4：cp.async=sm80+） | 内核全按 CUTLASS-2.x-sm70 寄存器级流水（examples/**07**_volta_tensorop_gemm，BSD-3） |
| 互联 | 双 NVLink 岛；岛内 P2P AR（门=GGML_CUDA_P2P_AR_DISABLE，负逻辑；AC922_TEST_P2P_AR 只是 harness 标签）；跨岛=激活交接（meta 后端，GGML_META_ASYNC_COPY） | **跨岛解锁后带宽【未实测】**：13GB/s 是报告推算上限、27.1 是 CPU 侧 X-bus 数字（GPU 路径到不了）。→ P0 加一项微基准实测 |
| 容量 | 每岛 3×16384MiB=48GiB raw / ~47.3 可用 | 专家 42.456 GiB（GGUF 实测求和）随层分岛：岛0(blocks 0-24)=20.24GiB，岛1(blocks 25-48+output+MTP)=25.89GiB，合计 46.13 ✓ |
| 内存 | 31GB RAM；PPC64 64KiB 页（mmap 缺页 728×理论/412×实测） | PLE 行级缓存+4KiB O_DIRECT 必须 |
| 盘 | 唯一 SATA BX500：425MB/s 顺序 / 10-25MB/s 随机 4K。无 NVMe | 只做冷尾+空闲流快照 |
| CPU | POWER9 VSX；powerpc/quants.c 有 19 个 vec_dot（vec_msum，无 vmsummbm 字面量） | PLE 路径 CPU 零 FLOP，无需 VSX 新内核 |

## 2. 模型事实（checkpoint 钉死，GGUF 元数据实测）
- 对象：IQ3_XXS-MTP-Q8-ssmq8.gguf 78.8GB，1541 张量；fork 枚举 Q2_0=42（ggml.h:431）
- **PLE 表实测**：`per_layer_token_embd.weight`，type 20=IQ4_NL，[160, 320001536]，26.822 GiB，**90 B/行**，数据偏移 **44.00 GiB**（43.81 是旧文件 dd 数，别用）。每 token 16 行 × 90B = **1,440B**。blk.1 另有 65.74MB 小件（key/value/conv1d/norms）
- **GDN 状态实测**：36 层 × **48** v-heads × 128×128 × **F32** = **108 MiB/流**（MIRRORED 复制岛内各卡；服务端每 seq ~450 MiB 含 4 条 MTP lane）。8 流 ~3.6GB，HBM 无压力
- 线性注意力多流优势不变（Kimi Linear arXiv **2510.26692**：TPOT 6.3× vs MLA）

## 3. 拓扑（三路一致确认，file:line 在案）
- **3+3 层流水线**：每层整层归属单岛（llama.cpp:158-163 注释 + llama-model.cpp:1735-1770 分配 + 1785-1804 守卫）；岛内 TP3、岛间层切。**MoE 路由/专家永不跨岛**（按构造）
- 跨岛流量 = 复制激活交接（meta async copy，n_island_members=3）；P2P AR 只在岛内（无 peer 即关）
- ~~岛内专家全复制~~ **删除**（v1 错项）：流水线下天然成立；每岛空闲 21-27 GiB 改投 KV 池/GDN 状态/图变体/UB

## 4. 带宽账（复核通过，误差 <15%）
- 8 流×60tok/s 活跃专家走盘 = 5.2-6.6 GB/步（310-400 GB/s）> 任何存储链路 25-50×，甚至超主机 DRAM 聚合 → 死路确认
- **注意项**：MTP4 draft 若命中不相交专家，每步专家流量趋近 5×（25-30GB/步）——跨流+跨 draft 去重是调度器必修课
- PLE 热时 1,440B/token；冷尾 29MB/请求；X-bus 只承担激活交接（KB 级）
- 专家缓存经济性（#29887 实证）：命中 <75% 时 GPU LRU 缓存反而更慢（47%/12.9t/s 是失败案例）；全驻留设计下不适用此件

## 5. 分阶段施工（修订版）

### P0 地基（1 周）
- 砍树（清单不变：15 死后端 222-267k + miniaudio/mtmd 123k + 别家架构 37k + CPU 外国架构 56k + 杂项）→ ~265k 行
- 硬编码张量表（1541 项）+ PLE 偏移 44.00 GiB
- **新增：跨岛带宽微基准实测**（P2P copy + 激活交接路径），把"未实测"变成数字
- 验收：1350/150 复现 ±1%

### P1 多流核心（2-4 周；前置=Codex per-lane MTP，实验进行中）
1. per-lane MTP 快链（fold 门在 speculative.cpp:1516 `n_seq==1`；行布局 l*n_pad+r；等行数 batch；双形状双图）
2. Sarathi 分块 prefill + decode 不让路（OSDI'24 确认）
3. VTC 公平队列（**OSDI'24**，不是 NSDI）+ 准入控制
4. 固定 lanes 1-8 + exact-batch CUDA Graph（ninfer 证实该技术，注意其数字是 sm_120a 的）
- 验收：4×8K 每流 ≥40 tok/s、最差/最好 ≥0.8、64K 不拖垮 8K

### P2 PLE 三件套（1-2 周，零件全在设计稿里）
- 行缓存 64-128MiB + 批内去重 + 4KiB O_DIRECT（GGML_CPU_PLE_ROW_CACHE 已在树）
- page-merge 内核接 GGUF 路径（~20-40 行）
- layer-ahead 预取器（designs/PLE-PREFETCH-DESIGN.md：~200+30 行，冷 TTFT 45.6→13-15s，row-id 离线可重算）
- 开 LLAMA_PLE_ONECAT=1（ple-conv-sm70.cu，parity A/B 后）
- vLLM 侧参考升级：hash+gather 全是 GPU Triton 内核读 pinned（CPU 只出字节）——若走侧车路线照此办
- 验收：冷 TTFT ≤15s；热 PLE <1% 步长

### P3 内核升级（4-6 周，**两项降级**）
1. **IQ 离线 LUT 重打包**：FLUTE 证实"任意码本在概念范围内"（int2/3/4、非均匀、组 32-256），但**它是 sm80+ CUDA/CUTLASS（非 Triton）、Apache-2.0、AOT 形状特化调优**——只能引其技术，内核必须自写 sm70 版（CUTLASS 2.x Volta 式寄存器双缓冲 + smem LUT）。预算不变（4-6 周），天花板引用降为"技术参照 2-4×"
2. **GDN 内核**：主线仍是自家 gated_delta_net.cu X4；FLA 上游修复 cherry-pick **降级为🟡**——上游无 Volta 支持且明确拒收 fp16（数值范围，默认 bf16 而 V100 无 bf16）→ 对齐算法可以，数值必须 fp32 重验；FlashQLA/TileLang 后端是 SM90+，不要
3. CUTLASS sm70 打底（examples/07_volta_tensorop_gemm）
4. DeepGEMM 思想：masked grouped GEMM 去 permutation
- 验收：单流 TG 150→180+，4 流聚合 ≥200

### P4 调度智能（2-3 周）
- overlap 调度（SGLang **v0.4** 2024-12-04，1.1×；POWER9 launch 开销放大收益）
- 混合 radix 缓存（QSA KV 块 + GDN 状态 108MiB/流快照按前缀键控）
- PLE 兼做草稿缓存 + goodput 自适应深度
- roofline 模型选工作点（2-3 天，P0 就该做）
- 验收：共享前缀 TTFT 减半；8 流每流 ≥25

## 6. 许可证表（全部修正）
可直接抄：FLA(MIT) CUTLASS(BSD-3) **FLUTE(Apache-2.0)** Marlin(MIT) DeepGEMM(MIT) llama.cpp(MIT) llamafile(Apache-2.0) **KTransformers(Apache-2.0)** **ExLlamaV2(MIT)** **PowerInfer(MIT)** —— v1 把后三个标成 AGPL/研究许可是错的，全部可用

## 7. 判死清单（+2）
原 12 条保留（GPU-PLE token71 偏差、LMCache 式分层、int8 主 GEMM、活跃专家走盘、cache-reuse、B@8K、部分驻留、QuaRot 后处理、NCCL 调参、PowerInfer 热神经元、UB1024@132K、MonoMoE）
+13. #29887 式专家 GPU LRU 缓存在低命中场景（<75% 反向收益，全驻留下无意义）
+14. 假设"FLA/FLUTE 现成跑 sm70"——都不行，sm70 内核必须自写

## 8. 目标数字（不变，依据更新）
单流 1350/150 保底；4 流每流 ≥50-70；8 流 ≥25-40；最差流 ≥0.8×均值；冷 TTFT ≤15s
谱系勘误：920.56/125.06（8K）与 1003.40/108.81（64K）是同一 winner 两个长度，非两代；1003→1350 的跳变无本地归因报告（X-bus 解锁是合理假设，P0 实测补证据）

## 9. 审核遗留（3 个待办）
1. 跨岛带宽实测（P0）
2. FLA chunk 算法 fp32@sm70 数值验证（P3 前置）
3. MTP draft 专家流量实测（决定去重调度器优先级）
