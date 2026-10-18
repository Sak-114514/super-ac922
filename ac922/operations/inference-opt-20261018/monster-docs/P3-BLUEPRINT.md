# P3 施工蓝图：IQ 家族离线重打包 + V100 张量核化（monster 线预填加速）

> 2026-10-13 汇编（四路调研全部交付后合成）。调研源：FLUTE 深研 / 开源 TC 内核扫描 / monster 代码测绘 / 打包器设计（v1 + 混合族 v2）+ 服务器 GGUF 实测字节账。
> 姊妹文档：`P3-REPACKER-DESIGN.md`（**混合族 v2 终版设计**，类型描述表/FAM 族/内核/roofline 权威来源）、`P3-REPACKER-DESIGN-iq3xxs-v1.md`（格式骨架/验收细节）、`p3-flute/`（FLUTE 源码副本 19 文件）。

## 0. 一页判决

| 问题 | 判决 |
|---|---|
| llama.cpp 预填慢是不是因为 IQ 小块没用上 tensor core？ | **是**。预填 M≥64 时现路径 = 全量反量化写回 fp16 → cuBLAS，专家矩阵流量膨胀 ~4.3×（42.46G 量化字节 → ~190G fp16 往返）；M<64 走 dp4a（纯 CUDA core，TC 闲置） |
| 技术路线 | **Marlin 式"smem/LUT 反量化 → fp16 → wmma(m8n8k4)" + FLUTE 式离线重排消灭 unpack 位操作**。one-hot 路线三重独立否决（§2.2） |
| 打包代价 | **纯字节重排，+0.03%**。不重编码码字 → 不存在"符号并入码本 4.06bpw"问题，85G 显存红线自动满足 |
| 收益主体 | **专家 FFN 42.46G（grouped GEMM）**，其次稠密投影 ~2.4G；PLE 嵌入表 26.86G 是查表对象、与 TC 无关，排除 |
| 预期 | 内核级 3-8×，端到端预填验收门 ≥1.8×（1350 → 2400+ tok/s）；decode 逐位不变（GEMV-on-pack） |
| 硬约束 | 张量数据指针唯一且被 CUDA graph 固化 → **decode GEMV 第一天就必须能读打包布局**（可做到与旧内核逐位一致）；新内核零 host-sync 否则全 graph 退化 eager |

## 1. 真实作战地图（服务器 GGUF 实测，2026-10-13）

文件：`/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf`（73.22 GiB，1256 张量，49 层=48 主干+1 MTP）。
fork 枚举（本地 `ggml/include/ggml.h` 钉死）：16=IQ2_XXS 17=IQ2_XS 18=IQ3_XXS 20=IQ4_NL 21=IQ3_S 22=IQ2_S 23=IQ4_XS；8=Q8_0 12=Q4_K 13=Q5_K 14=Q6_K 30=BF16；41=Q1_0/42=Q2_0 为 fork 复活旧类型。

### 1.1 专家张量（*_exps，147 个 = 49 层 × gate+up+down，合计 **42.46 GiB**）

| 类型 | 张量数 | 字节 | 位置 |
|---|---|---|---|
| IQ3_S | 26 | 8.73 GiB | gate/up（13 层） |
| IQ4_NL | 18 | 7.91 GiB | down（9 层） |
| Q2_0 | 30 | 6.59 GiB | down（15 层） |
| IQ2_S | 20 | 5.00 GiB | gate/up（10 层） |
| IQ2_XS | 20 | 4.52 GiB | gate/up（10 层） |
| IQ2_XXS | 18 | 3.63 GiB | gate/up（9 层） |
| IQ3_XXS | 12 | 3.59 GiB | gate/up（6 层：35/36/41/44/45/47） |
| Q8_0 | 3 | 2.49 GiB | MTP 层专家 |

形状：gate/up = [2560(K), 640(N), 512(E)]，down = [640(K), 2560(N), 512(E)]；512 专家 top-10，专家维 640。
预填 ubatch=512 → 路由 5120 行摊 512 专家 → **每专家 M̄≈10、泊松式不均**（0 行与 30+ 行并存）；2048 批时 M̄≈40。MTP decode chain rows≤32 → ne2≤320。

### 1.2 非专家（合计 30.75 GiB）

- **PLE 逐层嵌入表 26.86 GiB**：`per_layer_token_embd.weight` [160, 320001536] IQ4_NL——纯 token gather，**不打包不 TC 化，透传**
- 稠密量化 GEMM ~2.4G：Q6_K attn_gate 0.81 + Q5_K output 0.54 + IQ4_XS attn_qkv 0.36 + 非专家 IQ3_S 0.38 + Q4_K shexp 0.28（M=全 ubatch，算力区）
- BF16 1.43G（output_hc 系 486 张量）——`mmf.cu` 现成 F16-TC 路径已覆盖，不动

### 1.3 之前数字的勘误

- ~~"58.1G 专家张量"~~：把非专家 type20 混入。真实 42.46G。
- ~~"IQ4_NL 34.77G 是收益主体"~~：其中 26.86G 是 PLE。专家侧 IQ4_NL 仅 7.91G，最大单项是 IQ3_S 8.73G。
- ~~"符号并入码本 4.06bpw → 98G 超 96G 红线"~~：那是 FLUTE W4G256 重编码路线的成本；字节中性重排路线不存在此问题。

## 2. 路线判决

### 2.1 V100 硬件底牌

fp16 TC 112-125 TFLOPS（m8n8k4 / wmma 16×16×16，fork `mma.cuh:1427-1465` 已封装 Volta tile）；**无 int8 TC**（s8 mma 需 SM75+，llama.cpp 新 MMQ 的 TC 路径锁 Turing+）、无 ldmatrix、无 cp.async；HBM 900GB/s；96KB smem/SM、64K reg/SM。dp4a 31-62 TOPS → **纯算力天花板 2-4×**，但预填真正的收益来自消灭反量化写回的流量膨胀（~4.3×），两者叠加。

### 2.2 one-hot / LUT-GEMM 路线：三重独立否决（v1 设计 §2.1）

1. K 维膨胀下界 8×（幅值字母表实测恰 8 档 + 符号）→ 有效算力 112/8 = 14 TFLOPS 硬顶；
2. 二级 scale（d×ss[hi4]，依赖行与 k/32）不可因子化，one-hot 侧无法吸收；
3. smem 账：BLOCK_M=64×K=256 的 A 片膨胀后 256KB > 96KB。
QServe 式乘法 LUT 另需激活聚类 ≤16 级（MoE 激活不受控）且依赖 int8 TC（V100 没有）。**全灭。**

### 2.3 采用路线：字节中性重排 + 装配期反量化

- 离线：98B 超块拆 d | grid 字节 | sign/scale 字节 三类，按内核线程遍历序重排（tile = 64 超块 = 6272B，逐字节等量）。权重在 k 上数值顺序不变 → GEMV 可逐位等价。
- 在线：grid 表预展开 fp16 进 smem（2KB 常驻）；每 quad 1B 索引 → 一次取 4 个 fp16 幅值；符号 XOR 0x8000（零指令误差）；scale fp32 链乘一次舍入（≤1 ulp）；装配进 wmma fragment。
- R0（加载期整体反量化 fp16+cuBLAS）：需 ~190G×2 显存，弃。

## 3. 打包格式 monpack（骨架继承 v1，混合族扩展见 §9）

- 边车 `<gguf>.monpack` + GGUF 原文件不动；header(gguf_sha128 绑定) + 常量区(各类型码本) + 张量 blob 区 + 定长 index
- tile 序：专家维最外 → n_tile → k_tile 最内（grouped 内核按专家取 tile 不跨段）
- **类型描述表驱动**：IQ2/IQ3/IQ4 家族共享同构反量化框架（LUT smem + XOR 符号 + 两级 scale），per-type 参数 = {码本表, 块结构, scale 档数, 索引位宽}；Q2_0/Q8_0 走更简单的均匀量化条目
- 字节开销 +0.03%；只打包 GEMM 张量（专家 42.46G + 稠密 2.4G ≈ 44.9G blob），PLE/embd 透传 → sidecar 磁盘峰值 ~118G（源 73 + 目标 45）
- 转换：POWER9 numpy 流式重排，SATA 425MB/s → **~7-15 min**（含校验 pass）；.tmp + 原子 rename

## 4. 内核与引擎集成点（fork file:line，代码测绘代理已核实）

| 落点 | 位置 | 动作 |
|---|---|---|
| 加载器 | `llama-model-loader.cpp` init 后 | sidecar mmap + O(1) 校验失败静默回退；`load_tensor_data` 指针改道 + `ggml_cuda_monpack_register` |
| 分派 | `ggml-cuda.cu:2722`（mul_mat）/ `:2821-2824`（mul_mat_id grouped MMQ） | `is_monpack(ptr)` 注册表命中 → M≥16 走 `monpack_tc_grouped`，否则 `monpack_gemv` |
| TC 内核 | 新增（骨架 = v1 设计 §5.2 + `mmf.cu` Volta tile 复用） | 80 持久 block，smem 70-74KB opt-in 96K，8×wmma16 拼 32×32 输出，M̄≈10-40 用 J=8/16 细粒度 tile |
| decode 保护 | `ggml-cuda.cu:2767-2790` SM70 chunk-MMVQ | 不动；被 GEMV-on-pack 原位替换（逐位等价，容差 1e-6） |
| graph 兼容 | `ggml-cuda.cu:3793` 检查 | 新路径零 host-sync 即自动可录；MTP 形状集 M∈1..512×topk 在固定 graph key 下稳定 |
| 开关 | env 三级 | `MONSTER_MONPACK`(总)/`GGML_CUDA_MONPACK_TC_MIN_M`(=16)/`GGML_CUDA_MONPACK_DISABLE`(kill-switch)；删边车=文件级全局回滚 |

A/B 四象限（旧路径 / pack+全 GEMV / pack+TC≥16 / pack+全 TC）沿用 `matrix_runner.py` 512-token 逐位框架，**同槽位对比**（P14 教训：跨槽 logits rel-L2 可达 1.5e-2）。

## 5. roofline 与收益预期

| 对象 | M 形态 | 有效 FLOP/B | 受限于 | TC 化收益来源 |
|---|---|---|---|---|
| 专家 grouped（42.46G 主体） | M̄≈10-40/专家，变长 | 52-209 | 带宽→算力混合区 | ① 消灭反量化写回（流量 /4.3）② dp4a→TC（M≥16 段） |
| 稠密投影（2.4G） | M=512 固定 | ~2700 | 算力 | 纯 dp4a/dequant→TC |
| PLE/embd（27G） | gather | — | 带宽 | 无（透传） |

**M̄≈10 是纯带宽区（设计 §6.1 关键修正）**：有效强度 35.6 FLOP/B ≪ γ=124 → TC 利用率仅 ~0.4%。TC 化的真正收益是：① 消灭反量化写回的 HBM 往返——专家路径每前向流量 478GB→50GB（**~9.5×**，含写 F16+读回的完整往返）；② 消灭 grouped 小 M 的 cuBLAS/launch 风暴。mma/wmma 只是"零写回地算 fp16"的最廉价载体。
端到端（设计 §6.3 权威估计）：ubatch=512 前向墙 380ms → 全族覆盖后 126-219ms，**预填 2340-4060 tok/s（点估计 2900 = 2.15×，置信度中）**；MVP（仅 IQ4_NL，18.6% 专家字节）只带来 1.08-1.13×——**MVP 是管道验证，性能门 ≥1.8×=2430 tok/s 挂在 P3 全族覆盖后**。decode ~150 tok/s 逐位不变。
新瓶颈排序（设计 §6.4）：smem 装配吞吐（56% 预算占用）> 变长 M 泊松波尾（静态轮转 ≤5%）> 残余稠密 dequant+cuBLAS（~12ms）> LUT bank 冲突。

## 6. 显存与部署账

- 6 卡 tensor split：73.2G 全模型 ≈ **12.2G/卡**，字节中性重排不变；每卡余 ~3G 给 KV/graph/激活
- sidecar 落盘 ~45G（只打包 GEMM 张量）+ 源 73G；转换期峰值 ~118G 空闲盘（服务器已清理 275G，充裕）
- 转换 7-15 min，一次性；A/B 与回滚零风险（边车删除即回旧路径）

## 7. 验收协议（继承 v1 T0-T4）

| 层 | 通过标准 |
|---|---|
| T0 往返 | 打包→逆排→原字节 100% 相等（147+稠密张量全覆盖） |
| T1 反量化 | packed 反量化 vs `dequantize_row_*` fp32 参考，逐元素 ≤1 fp16 ulp |
| T2 GEMM 数值 | vs fp64 referee：rel-L2(new) ≤ max(rel-L2(old))×1.5+1e-6，绝对门 ≤1e-3；对齐 P15 fixture 形状复跑 |
| T3 自然分布 | 200 条语料 128-token 贪心同槽位 A/B：top-1 一致率 ≥99.5%、top-20 KL ≤1e-3、3 次重跑逐位一致、PPL 增量 ≤0.002 |
| T4 性能 | 内核级 M=40 ≥3×；端到端预填 ≥1.8×；decode 回退 ≤1%；VRAM 增量 ≤300MB |

## 8. 工作量分级与阶段（人力 = 1 资深 CUDA + 0.5 系统支持，总 6-9 周）

- **P3.0 去风险（0.5 周）**：nsys 实测 FFN 占比收窄收益区间；`mrt-probe` 钉死 Q2_0/IQ3_S/IQ2_S 位域（**Q2_0 反推 bpw≈2.25 与官方 2.5 不符是最大未钉死项**，钉死前 Q2_0 条目降级透传）；从 `mma.cuh:1427-1465` 抄录 lane↔fragment 映射冻结 layout_rev=1
- **P1（MVP，1.5-2 周）**：格式冻结 + 类型描述表 + FAM-A（IQ4_NL）条目 + 转换器（fork 树内构建，枚举同源零分歧）+ T0/T1 全绿
- **P2（存储层换血，1-1.5 周，低风险）**：FAM-B/C 条目 + GEMV-on-pack 全类型 + decode 逐位验证——这步交付后即使 TC 内核失败也已锁定布局红利
- **P3（主战场，2-3 周）**：`monpack_tc_grouped`（持久 80 CTA + 设备侧静态确定性 work list，graph key 稳定 + T3 逐位复现）+ 分派开口 + T2/T4
- **P4（转正，1-2 周）**：稠密 FAM-D 可选 + T3 灰度（`MONSTER_MONPACK_TYPES` 分族白名单上线）+ 回滚演练
- 风险 Top3：smem 装配效率不足（备选寄存器直装+warp shuffle）；FAM-B 位域钉错（T0 位级往返硬门兜底）；MVP 后误判收益不足而弃（MVP 1.1× 是预期内，性能门在 P3）

## 9. 产出索引

| 文档 | 内容 |
|---|---|
| `P3-REPACKER-DESIGN-iq3xxs-v1.md` | 格式逐字段/转换伪代码/内核骨架/T0-T4 细则（按纯 IQ3_XXS，字节账已被本蓝图 §1.3 勘误） |
| `P3-REPACKER-DESIGN.md` | **混合族 v2 终版设计（已交付）**：类型描述表/FAM-A·B·C 族/逐字段格式/转换器/内核/roofline §6 权威数字/工作量分级。数字已与蓝图 §1 勘误对账一致（34.77 = 7.91 专家 + 26.86 PLE） |
| `p3-flute/`（19 文件 744K） | FLUTE 源码副本：packbits_utils / qgemm_kernel / tile_scheduler / CuTe atom 对照 |
| 开源扫描简报 | MMQ/Marlin/exllama/Machete/QServe 选型对比 + V100 TC 天花板校准 |
| 代码测绘简报 | fork 执行路径 file:line 全图（vecdotq/mmvq/mmq/mmf/mma/graph/MTP） |

## 10. 开工请示点

1. ~~MVP 从 IQ4_NL 还是 IQ3_S 起步~~ **已决（设计 §8）**：IQ4_NL 起步（结构最简，两周 T0/T1 绿灯），性能主收益在 P3 阶段 FAM-B 全族覆盖后
2. 施工分支：在 `flashnext-llama-experiment/sources/llama-volta-jusko` 上开 `p3-monpack` 分支，monster 常备引擎不动
3. P3.0 探针先行授权（nsys 一次预填画像 + Q2_0/IQ3_S/IQ2_S 块结构 hexdump）——0.5 周，产出决定描述表定稿
