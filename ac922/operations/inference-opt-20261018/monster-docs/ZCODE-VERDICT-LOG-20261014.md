# ZCode 乱码案实验台账（移交 Codex · 勿重复实验）

> 2026-10-14 21:0x（本地）。作者：ZCode。目的：语义乱码案的全部已完成判决臂记录在此，
> **以下每一臂都不需要重跑**。当前进度推进到 BD2''（见文末"在途"）。
> 判决统一协议：`launch-p3-arm.sh` 配方（ISLAND=3,3 / FAST全开 / P2P_AR=1 / KV q8_0 / FA on / SLOTS=1），
> 语义探针 = POST /v1/completions "The capital of France is" 24 token temp=0 seed=7。
> 干净参照输出：" Paris. The capital of Germany is Berlin. ..."（b3）。
> 乱码参照输出："!! военно! военно! ..."（MTP 开）或 "!!!!!!!..."（MTP 关，token 0 连发）。

## 一、基础事实（勿再验证）

- 干净 = **b3-1afa565**（`/home/debian/ac922-llama-next/sources/llama-v100-v3/build-ac922/bin`，其 libggml-cuda 为 10-11 03:27 构建）
- 乱码 = **全部 monster 树构建**：j12（runtime-before，10-14 11:00）、a1/a2/a3（12:06/12:22/13:46）、runtime-combined 系
- 乱码确定性：同配置逐字节复现（含 draft_n=42 / accepted=11 恒定）
- MoE FMA 修复（__fmul_rn/__fadd_rn）**10-11 10:09 起就在 monster 树**（build-monster 的 moe 对象 10-13 23:11 编译，晚于修复）——所有 monster 二进制都含它，**不是答案**（上游 #29168/#29381 是真实故事但不是我们的病）
- 驱动/工具链全机唯一：550.54.15 + /usr/local/cuda-12.4（三家 runtime ldd 一致）——环境无罪
- 两树 17 个共享脏文件里 9 个 hash 相同（含 qsa-attn/topk-moe/moe-reduction）——这些无罪

## 二、已完成的排除（每条一行证据）

| 嫌疑 | 判决 | 臂/证据 |
|---|---|---|
| tensor-split 切分 | 无罪 | b3 × ISLAND 3,3 × 全配方 = "Paris..." 干净（10-14 14:51 探针） |
| C1 AR 分段核 | 无罪 | ringoff 臂：j12 + RING_MIN=2⁶⁰ 强制 one-shot，仍乱码 |
| C1 FIFO 单岛门 | 无罪 | 门 env=GGML_META_FIFO_PIPELINE；metaoff 臂置 0 仍乱码 |
| MTP/投机解码机制 | 无罪 | mtpoff 臂：AC922_TEST_MTP=0（timings 无 draft 计数证实），输出 "!×24" 仍乱码 |
| 配方 env（FAST 9 项/META 3 项/KV_PAD/六旋钮对齐） | 无罪 | FAST=0/metaoff/codexvals 各臂，乱码照旧 |
| 构建竞态 | 无罪 | j12 与 a3（两套独立构建）同乱码 |
| chat template / server 模板路径 | 无罪 | 两树 server.cpp md5 相同；b3 同裸提示词干净 |
| NCCL/回退 AR 路径 | 无效对照 | 本机 NCCL 永远 init 失败回退 p2p，nccl 臂等于没换 |
| argmax_split NaN→token0（静态审查线索） | 转化器非生产者 | argmax.cu 两树逐字相同（:100 bi=-1），b3 带着它仍干净；只解释输出格式（"!"×24=token 0 连发）不解释坏 logits 来源 |
| QSA pooled multi / MULTI_CELL_SEL | 未激活 | 配方显式 =0（ENGINE-MASTER A2 节 env 清单） |
| GDN 状态寻址（静态审查主线索） | **被 BD1 击毙** | 还原 gated_delta_net.cu+mmvq.cu 至 HEAD（=b3 内容）重编（runtime-built-...122136Z）→ 仍乱码、逐字节同 |
| gated_delta_net.cu / mmvq.cu / lightning-indexer / mmq-moe-grouped | 无罪 | BD1（前两个）+ FAST=0 关 MOE_GROUPED/fused-indexer 仍乱码（后两个） |
| 基础矩阵乘/图执行 | 无罪 | 你的 island-p4 fixture 用 runtime-before 库 MUL_MAT/ADD_ID 逐位精确 exact=1 |
| 跨树 .so 混装定位法 | **方法无效** | 单换任一 lib 两个方向都乱码（内部 meta↔cuda 协议随树演化，混装=伪证）；全部混装臂作废 |

## 三、当前嫌疑收敛（BD2'' 在途）

monster 相对 b3 的剩余差异分三片，逐片还原到 b3 内容（私有副本 incremental-build.py --run，不动共享树）：

1. **BD1（已完成，❌不治）**：gated_delta_net.cu + mmvq.cu → HEAD
2. **BD2''（构建中）**：ple-row-cache.h → HEAD（**当前头号嫌疑**：monster 把 180 行行缓存实现从头文件搬进 ple-row-cache-api.{h,cpp}（P1 重构），b3 用原地实现；PLE=26.8GiB 表 CPU 必经、always-on、确定性、fixture 未覆盖）+ ggml-backend-meta.cpp → v100v3 工作树版
   - 已知坑：还原 ggml-backend.h 或 ggml-cuda.cu 会触发 166/170 CUDA 对象重建被脚本自保拦截，已撤出
3. **BD3（若 BD2'' 不治）**：libllama 侧 → v100v3 内容（llama-context.cpp/graph/qwen4exp/batch/memory-hybrid-idx/loader/ext + server-context/task）
   - 注意 ggml-cuda.cu 的 monster 增量 +1217/-38（含 monpack/structural/path-log 等，多数 env 门控）需单独小步处理

## 四、给你的三条接口约定

1. **GPU 占用**：我的判决臂每次占 GPU 约 4 分钟（加载+探针），间隙查 `nvidia-smi` 再发车；你的 dual-context-runner（30-60 分钟）请在我的臂间隙跑或告诉我暂停
2. **共享树**：我全程只改 `experiments/hardware-next-20261014/src/`（你的私有副本），当前被我还原的文件清单见 §三；**共享树 /home/debian/ac922-monster/src 我一个字节没动**
3. **语义基线**：你的 runtime-combined / dual-context-runner 的语义门（semantic-checks.py）请等本台账结案后再跑——乱码未破前，任何模型级数字都在坏基底上

## 五、在途与下一步

- BD2'' 构建（incremental-build.py --run）后台运行中 → 冻结 runtime → 探针判决
- 干净 → 单文件二分（先 ple-row-cache.h 单独还原）；乱码 → BD3（libllama 侧）
- 结案后顺序：语义门 → 你的六项序列重开（P4 双 ctx 真实模型 → semantic-checks → 性能账）
- 全部臂的原始 JSON 存档：Mac `monster-engine/`（swapb3cuda2/mtpoff/bd1 等 probe 脚本与输出）+ 本文件

---

## 六、终局破案（21:4x 增补，Zcode）

**路径级定罪：`LLAMA_QSA_INDEXER_IMPL=fused` 在 monster 树上数值坏。**

证据链（全部同配方语义探针）：
1. **true-fast0 破局臂**（j12 + 四个真实关闭的 SM70 env，经你审计的 setenv 顺序问题修正）：输出前 21 token 干净（Paris/Berlin/Rome/Spain）——证明毒在这四个里
2. **四臂单开刀**：bf16=0 乱 / moerwk=0 乱 / **idxstd(standard) 干净** / topk=0 乱——唯一干净臂 = indexer 换 standard
3. **64 token 长探**（j12 + 仅 INDEXER=standard）：9 组国家首都全对、MTP 接受率 75%（46/61）、TG 48 tok/s——**standard 路径全净**
4. **BD4 反转臂**（lightning-indexer.cu→HEAD + b3-libllama + fused 开）：仍乱码——**lightning-indexer.cu 本体无罪**，坏点在 fused 的分派/调用侧（候选：ggml-cuda.cu monster 增量 +1217 行，或 mmq-moe-grouped.cu 被 fused 无视 env 直接调用）
5. BD3（libllama 18 文件还原）乱码——libllama 侧整片无罪（PAD_AUTO/DRAFT_CACHE strings 验证还原生效）

**立即可用的生产缓解：所有 monster 系服务/实验加 `LLAMA_QSA_INDEXER_IMPL=standard`**（一个 env，语义干净，indexer 性能略有损失）。你的六项序列、dual-context-runner、全部性能实验现在可以在这个缓解下开跑。

在途：BD5 = mmq-moe-grouped.cu→HEAD 构建（切分 ggml-cuda.cu vs mmq-moe-grouped 的最后一刀）。

---
## §七 BD 复刻终局链（2026-10-14 晚，本轮新增）

**拓扑关键事实（修正全程认知）**：1afa565→1e542c5 四提交**完全没碰 ggml/**。monster↔b3 的全部差异 = **两侧活树脏层**：
- b3 = 1afa565 + v100v3 脏 23 文件（全部在 monster 脏清单交集内）
- b7 = 1e542c5(=1afa565 的 ggml/) + monster 脏层（交集 23 + monster 独有 ~30 文件/头）

**BD 判决臂记录（默认配方 fused 开，探针 "The capital of France is" 24tok temp=0 seed=7）**：
| 臂 | 内容 | 结果 |
|---|---|---|
| BD7b | ggml-backend.cpp→提交版（保 monster backend.h）| 乱码（backend.cpp 无罪；其 +105 活树增量默认配方下全为 env 门控死代码/日志/等价重构）|
| BD8a | allreduce-p2p.cu→提交版（另 4 算子两树一致=空操作）| 乱码（allreduce 无罪；seg 内核唯一门=RING_MIN env，C1 已证关掉仍乱）|
| BD8b | ggml-cuda.cu→提交版 | 乱码（monster 侧 +1156 无罪）|
| BD9a | ggml-cuda.cu→**v3 脏版**（b3 实际状态）| 乱码（v3 侧 +61 修复[冷图 AR proc/topk 越界检查]也不是解药）|
| BD9b | qsa-attn×2/topk-moe/moe-weighted-reduction→v3 脏版 ⇒ **EXP 源码= b3 逐字节复刻** | **乱码！** |

**BD9b 是分水岭：源码内容整体排除。** 剩余差异只能在构建层：
1. monster 独有脏文件仍参与编译：llama-graph.cpp(+84-4)、llama-model-loader.cpp(+51-5)、llama-ext.h(+19)、llama-graph.h(+29-1, allow_reuse 更严格——安全方向)、models.h、llama-memory-hybrid-idx.h、include/llama.h（全为兼容增量，但 llama-graph/model-loader 的 .cpp 改动是行为性的）
2. CMakeLists(monster) 编入 v3 没有的 TU：ple-prefetch.cpp / monpack-loader.cpp(带桩) / ple-row-cache-api.cpp
3. flags 差异已排除：b3 也有 -use_fast_math；MONPACK_AVAILABLE=1 只影响 loader 桩 TU
4. ggml-backend.h monster 版 = 纯声明增量（+12 API 声明），无害

**v3 侧 ggml-cuda.cu 61 行改动存档**（修复候选，BD9a 证明单独不解乱码）：
- 冷图 P2P AR 入口 ggml_backend_cuda_p2p_ar_allreduce(+47) 注册为 proc——暗示捕获/冷图执行分叉
- topk_moe_fusion 5 处 node_idx 越界检查——fused 图形状变化下防御
- GGML_CUDA_NCCL_F32 env（默认关）

**GPU 协调**：Codex 实验期间我方探针暂停；BD10 臂构建（纯 CPU）可预完成，GPU 空闲后补探针。

**下一步 BD10**：monster 独有脏 .cpp/.h（llama-graph/model-loader/ext 等）→提交版，达成"源码+头=b3 完整复刻（除 CMake 编入的死 TU）"→ 构建 → 等 GPU 探针。若仍乱 → 最后战场=CMakeLists TU 集合。

---
## §八 三代理静态审计终局 + REPACK 完整因果链（2026-10-14 深夜）

**用户指令：停止编译循环，转静态审计。** 三并行子代理结论：

### Agent A（monster 独有 8 文件审计）
全部 hunk 逐行读完：**8 文件所有改动在当前复现 env 下全部惰性**（env 门控/默认参数/sidecar 已禁用）。monpack 权重重定向（model-loader.cpp:1647-1790，tensor 字节置换+校验跳过）是唯一"门控翻转即高危"项，但被 MONSTER_MONPACK=0（所有启动脚本显式设置）+ 唯一 sidecar 已于 10-13 22:22 改名 .disabled-by-p3 双重禁用。allow_reuse 更严格=安全方向。

### Agent B（构建体系审计）——破案核心
**TOP1：GGML_CPU_REPACK——b3(build-ac922)=ON / monster(build-monster)=OFF / EXP(build-reuse)=OFF（cache 文本继承）**
- 起源：/home/debian/ac922-monster/BUILD-MINIMAL.sh:8 显式 `-DGGML_CPU_REPACK=OFF`（2026-10-10 19:38 配置）
- v3 用上游默认 ON
- 运行时铁证：docs/p3-arm.log 全部 44 次启动，6 次干净臂（b3）全部打印 `REPACK = 1` feature 行，38 次乱码臂全部没有（monster 构建不注册 repack buffer → 无此行）——**100% 相关**
- 机制：REPACK=OFF 时 ggml_backend_cpu_repack_buffer_type 不注册（nm 对象级验证：v3 的 ggml-cpu.cpp.o 有 U 引用，monster/build-reuse 为 0），CPU MUL_MAT 走非重排通用 vec_dot 路径
- **它同时解释 BD1-BD10 全链失败**：incremental-build.py 按 compile_commands 文本改写复制、拒绝 CMake 再配置（is_build_configuration 改动即 blocked）→ cache 的 REPACK=OFF 被永久冻结 → 任何源码还原都改变不了编译产物行为
- 其余排除：优化/math flags 零差异、链接/ldd/驱动一致、三个 meta 桩零引用无遮蔽、MONPACK_AVAILABLE 对 v3 内容共享文件无效

### Agent C（fused 契约交叉审查）
- 算子契约自洽：ggml.c:6598-6651 形状断言、CUDA 三内核（vec/tile/wmma）逐字节寻址、CPU 实现（ops.cpp:12304-12383）与 qwen4exp 构造逐项吻合；mask=加法语义 0=有效；weights=1.0 时 fused 与 standard 数学等价 → **算术层排除**
- decode(n_tps<8) 走 vec 内核全程 F32 无精度差 → 乱码必为结构性
- **发现 4（缝合 REPACK 的机制）**：llama-context.cpp:704-707 probe 结果写入 cparams.fused_lid，但 qwen4exp.cpp:1692-1695 只读环境变量、无视 probe——若 ggml_cuda_lightning_indexer_supported（neq0==128、neq1∈{4,32,64}、nb[i≥1]%16、k 类型表）失败，LIGHTNING_INDEXER 节点被**静默调度到 CPU**（CPU 无 supports_op 拒绝项）
- 旧树（llama-volta-jusko:874-898）有 ggml_cont() 防御+bias 折叠+min-blocks 门控（decode 永不 fused），v3 重写时全部丢弃；无 cont() 时对齐假设依赖上游 rope/norm 输出的偶然连续性

### 完整因果链（当前最优假设）
1. 两树默认配方均 fused ON。b3 干净的观测 = **fused ON + REPACK ON + 干净**（非"fused 本身坏"——修正 §一-§六 的路径级定罪叙述）
2. fused 图使 LIGHTNING_INDEXER（或其依赖链张量）落到 CPU 执行
3. CPU 侧：REPACK=ON（b3）走 repack 重排内核 → 数值正确；REPACK=OFF（monster）走通用 vec_dot 路径 → 在 IQ3_XXS × 相关形状下确定性算错 → "военно"×N
4. standard 路径全部留在 CUDA → 不碰 CPU → 与 REPACK 无关 → 两树皆干净（idxstd 臂干净的真因）
5. 病灶在 BUILD-MINIMAL.sh 的构建配置选择，与源码内容无关 → BD1-BD10 全部无效的统一解释

### 待终局验证（GPU 空闲后，一次即结案）
在 EXP 下复制新 build 目录、仅改 -DGGML_CPU_REPACK=ON 重配置构建 → 冻结 runtime → 默认配方探针：
- 干净 → 终审定罪 GGML_CPU_REPACK=OFF（文件级：BUILD-MINIMAL.sh:8）
- 乱码 → REPACK 排除，转次级嫌疑（仍是 monster 版 ggml-cuda.cu + MONPACK_AVAILABLE=1 的组合）

### 修复决策（验证后执行）
- **根治**：BUILD-MINIMAL.sh 删除 -DGGML_CPU_REPACK=OFF，monster 树重配置构建
- **备选缓解**（已可用）：LLAMA_QSA_INDEXER_IMPL=standard（全 CUDA，绕开 CPU 路径）
- 附加建议（Agent C 最小修复假设）：fused 门控回归 probe 结果+恢复 ggml_cont() 防御

---
## §九 Codex 并行数据点对 REPACK 假说的修正（2026-10-15 只读调查）

**Codex 夜间工作画像**（experiments/standard-semantic-20261014 + standard-opt-20261014，只读取证）：
- 00:58Z 轮：semantic-native **16/16 pass**（single/concurrent_2/concurrent_4/slot_reuse 全过）
- 01:17Z：**P4 dual-context-runner 跑通**（双 bank×2 lanes、MTP-4、pooled 132096、semantic_pass=true、fallback_lanes=0）——六项计划第 4 项落地
- 01:25Z：candidate-perf-v3 轮（续 perf 配方迭代）+ benchmark-stream（65536 真实 token 池、8192 in/lane）
- 其未解问题：greedy-parity tokens_equal=false（first_diff=1/41/87，配方间贪心不逐字节一致）

**关键反例（修正 §八 REPACK 根因假说）**：
1. Codex 的 profile 用 **LLAMA_QSA_INDEXER_IMPL=fused**（非 standard），16/16 干净，profile 标记 "fused-indexer-semantic-approved"
2. trace 实证：lightning indexer 走 **CUDA vec/tile 内核**（path=vector batch=2 / path=tile batch=35 kv=64 type=f32）——非 CPU 回退
3. 其 runtime-perf-v2/v3 的源码 = 当前 monster 活树（qwen4exp.cpp/ggml-cuda.cu/meta.cpp 三文件 sha256 与活树完全一致），构建体系 build-reuse **REPACK=OFF**
4. → **fused + monster 源码 + REPACK=OFF + Codex env 全家桶 = 干净**。REPACK=OFF 不充分致病，从"根因"降级为"相关因素"（6/6 vs 38/38 相关性仍真实——它 correlat 不 causat）

**乱码必要条件的重新表述**：乱码 = 默认探针配方（launch-p3-arm.sh 基线 env）× fused × EXP/monster 构建的组合性质。区分变量在 Codex profile 与默认配方之差的 env 里（SPARSE_KERNEL 已排除——默认开，两配方同走 sparse）：
候选：LLAMA_QSA_GATHER_MAX_TOKENS=32 / LLAMA_PREFILL_KV_PAD=8192 / LLAMA_QWEN4EXP_HC_FUSION=1 / GGML_CPU_PLE_*（ROW_CACHE/PAGE_BATCH/线程）/ GGML_CUDA_GRAPH_OPT+STRUCTURAL / GGML_FT_GRAPH_CACHE / GGML_CUDA_ALLREDUCE=nccl / GGML_CUDA_VOLTA_GDN_X4 / GGML_CUDA_FATTN_ONECAT=0 等。

**终局验证计划修正**（替代"仅 REPACK=ON 单测"）：
- 臂A：乱码探针配方 + Codex v2-safe env 全家桶（预期干净=复现 Codex）→ 然后逐组减 env 二分出"保护因子"
- 臂B：REPACK=ON 构建 + 默认配方（预期干净=REPACK 因子确认）——两臂交叉后因果网络闭合
- 全部待 GPU/测试许可后执行
