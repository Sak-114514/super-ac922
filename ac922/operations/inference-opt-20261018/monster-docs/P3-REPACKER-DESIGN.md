# P3 设计文档：monpack v2 混合族离线重打包器 + V100 TC 内核消费格式

> 2026-10-13。本文档即 P3-BLUEPRINT.md §9 预留的"混合族扩展设计"。骨架继承 `P3-REPACKER-DESIGN-iq3xxs-v1.md`（monpack 边车、字节中性重排、GEMV-on-pack、T0-T4 验收框架），扩展为**类型描述表驱动**的混合族格式，覆盖 GSQ-RCO 检查点全部 8 种量化类型的专家张量。
> **数字基准声明**：任务书原始输入中的"58.1G 专家 / IQ4_NL 34.77G 收益主体 / type 30·42 自定义类型"已被 P3-BLUEPRINT §1.3 勘误（34.77G 中 26.86G 是 PLE 逐层嵌入表；fork 枚举已在本地 `ggml/include/ggml.h` 钉死：30=BF16、42=Q2_0）。本文全程采用勘误后数字，勘误对照见 §1.2。字节总量与任务书完全对账一致（34.77 = 7.91 专家 + 26.86 PLE；9.10 = 8.73 专家 + 0.38 稠密），无矛盾，只是归属修正。

---

## 0. 结论速览

| 决策点 | 结论 | 章节 |
|---|---|---|
| 格式 | `<gguf>.monpack` 边车（GGUF 不动）+ 文件内定长二进制 index（权威）+ `manifest.json`（仅工具/调试，引擎不读） | §3.1 |
| 族划分 | FAM-A(IQ4_NL, LUT16+d) / FAM-B(IQ2·IQ3 族, grid+sign+两级scale) / FAM-C(Q2_0/Q8_0 均匀) / 透传(PLE、BF16、K-quant 稠密) | §3.2 |
| 字节开销 | 逐张量字节中性，+0.03%（头+index+对齐）；blob 全量 44.87G（专家 42.46 + 稠密 2.4） | §3.8 |
| 显存 | 设备侧权重总量不变 ≈73.2G ≤ **85G 红线**，余 11.75G（≈2G/卡）给 KV/graph/激活 | §3.8 |
| 转换 | fork 树内构建（枚举零分歧）+ 探针模式；SATA 425MB/s 下 4-6 min（纯转换）/ 7-12 min（含校验），磁盘峰值 118G | §4 |
| 内核 | `monpack_tc_grouped`（持久 block + 设备侧 work list，变长 M_e，k-外层/n-内层复用 A）；M≤8 走 `monpack_gemv`（与旧 MMVQ **逐位一致**） | §5.3-5.4 |
| A/B 与回滚 | env 三级（蓝图既定）+ 新增 `MONSTER_MONPACK_TYPES` 灰度白名单；删边车=文件级回滚；graph 指针固化 → 切换需重启，无热切换 | §5.1 |
| roofline | M̄≈10 纯带宽区：专家路径 HBM 478GB→50GB（约 1/9.5）；预填 380ms→126-219ms，**预期 2340-4060 tok/s（点估计 2900，2.15×，置信度中）**；验收门 ≥1.8×=2430 tok/s | §6 |
| 新瓶颈排序 | smem 装配吞吐（56% 预算占用）> 变长 M 波尾（≤5%）> 残余稠密 dequant+cuBLAS（~12ms）> LUT bank 冲突 | §6.4 |
| MVP | IQ4_NL 单族先跑通管道（T0/T1 两周绿灯，蓝图 §10.1 既定）；性能主收益在 FAM-B 覆盖后（P3 阶段） | §8 |

---

## 1. 数字基准与勘误对齐

### 1.1 专家张量字节账（采用 P3-BLUEPRINT §1.1，服务器实测）

文件 `/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf`（73.22 GiB，1256 张量，49 层 = 48 主干 + 1 MTP）。专家张量 `*_exps` 共 147 个（49 层 × gate+up+down），**合计 42.46 GiB**；形状 gate/up = [2560(K), 640(N), 512(E)]，down = [640(K), 2560(N), 512(E)]，512 专家 top-10，专家维 640。

| fork type | 类型 | 张量数 | GiB | bpw（由字节反推） | 本文族归属 |
|---|---|---|---|---|---|
| 21 | IQ3_S | 26 | 8.73 | 3.44 | **FAM-B** |
| 20 | IQ4_NL | 18 | 7.91 | 4.50 | **FAM-A（MVP）** |
| 42 | Q2_0（fork 复活旧型） | 30 | 6.59 | ≈2.25▲ | **FAM-C** |
| 22 | IQ2_S | 20 | 5.00 | 2.50 | FAM-B |
| 17 | IQ2_XS | 20 | 4.52 | 2.125 | FAM-B |
| 16 | IQ2_XXS | 18 | 3.63 | 2.0625 | FAM-B |
| 18 | IQ3_XXS | 12 | 3.59 | 3.0625 | FAM-B |
| 8 | Q8_0（MTP 层专家） | 3 | 2.49 | 8.5 | FAM-C |
| | **专家合计** | 147 | **42.46** | 均≈3.4 | |

▲Q2_0 由 6.59 GiB ÷ (30×838.86M 元素) 反推 ≈2.25 bpw，与旧版 Q2_0（10B/32 元素 = 2.5 bpw）不符，疑 fork 改过块结构（9B/32？）——**P3.0 探针钉死，钉死前 FAM-C 的 Q2_0 条目按描述表参数留空降级为透传**（见 §3.6）。

### 1.2 任务书数字 → 蓝图勘误对照

| 任务书原始表述 | 勘误后事实 | 处理 |
|---|---|---|
| 专家 FFN 58.1 GiB | **42.46 GiB**（58.1 混入了非专家 type20） | 全文按 42.46 |
| type20 34.77G 是收益主体 | 26.86G 是 PLE 嵌入表（纯 gather，透传）；专家侧 IQ4_NL 仅 **7.91G**；最大单项是 IQ3_S 8.73G | MVP 仍从 IQ4_NL 起步（结构最简），但性能主收益在 FAM-B（§6.3） |
| type 42 = 自定义 GSQ/RCO 38 张量 6.60G | **Q2_0**（fork 复活旧型），30 张量 6.59G，down_exps | 给 FAM-C 正式条目；描述表同时保留 PASSTHROUGH 槽位以备探针推翻 |
| type 30 = 自定义 486 张量 1.43G | **BF16**（output_hc 系），`mmf.cu` 现 F16-TC 路径已覆盖 | 透传 |
| 96G 聚合 / 85G 全专家驻留红线 | 不变。6 卡 tensor split 73.2G ≈ 12.2G/卡 | §3.8 |

### 1.3 硬件与负载锚点（沿用任务书钉死事实）

V100 SXM2 ×6（每卡 HBM 900 GB/s、112 TFLOPS fp16-TC、96KB smem/SM、64K reg/SM、80 SM）；仅 mma.sync.m8n8k4 / wmma（fork `mma.cuh:1427-1465` 已封装 Volta tile<32,4>/<8,4>）；无 ldmatrix / cp.async / int8-TC / wgmma。机器平衡点 γ = 112e12/900e9 ≈ **124 FLOP/B**。
预填 ubatch=512 → 路由 ne2=512×10=5120 行摊 512 专家 → **每专家 M̄≈10，泊松式不均**（0 行与 30+ 行并存）；ubatch=2048 → M̄≈40。decode MTP chain rows≤32 → ne2≤320。基线：monster 线预填 ~1350 tok/s（8K prompt，前向 380 ms/512 tok），decode ~150 tok/s。
fork 集成点：预填 M≥64 = 全量反量化写回 F16 + cuBLAS；decode M≤8 = MMVQ dp4a（`ggml-cuda.cu:2767-2790`）；mul_mat 分派 `:2722`、mul_mat_id grouped MMQ `:2821-2824`、cuBLAS fallback 需 host-sync 且 graph 下被禁（`:2833`）；`mmf.cu:133` should_use_mmf 对量化张量返回 false；graph 兼容检查 `:3793`。

---

## 2. 设计总览

三件套，单向数据流：

```
GGUF(不动) ──mrt-repack(离线, fork 树内)──▶ .monpack 边车 + manifest.json
                                              │
llama_model_loader ──mmap 边车+O(1)校验───────┘
        │ 指针改道 + ggml_cuda_monpack_register(指针注册表)
        ▼
ggml-cuda 分派: is_monpack(ptr)?
   ├─ 否 → 旧路径原样（MMVQ/MMQ/dequant+cuBLAS），代码零改动
   └─ 是 → M_total ≥ TC_MIN_M(16) → monpack_tc_grouped（新 TC 内核，含 grouped 调度）
           否                     → monpack_gemv（打包布局上的 GEMV，与旧内核逐位一致）
```

设计不变量（全族共享，继承 v1）：

1. **纯字节重排，不重编码码字** → 逐张量字节中性，精度零损失，85G 红线自动满足；
2. **权重在每行的 k 维数值顺序不变**（只是存储分组变）→ GEMV-on-pack 逐位等价的前提；
3. **tile 网格全族统一**：K-tile = 256（= 64× mma.m8n8k4 的 k4 = 16× wmma k16），N-tile = 64（= 4× wmma n16），专家维最外 → grouped 内核按专家取 tile 不跨段，m8n8k4 倍数自动满足；
4. **类型描述表驱动**：新增量化类型 = 填一张描述表条目 + 一对 pack/unpack 函数，格式与内核框架不动。

---

## 3. monpack v2 格式逐字段

### 3.1 文件总体

```
<gguf路径>.monpack                    # 边车二进制（权威，引擎只读这个）
<gguf路径>.monpack.manifest.json      # 工具/调试元数据（引擎不读、不解析）

┌────────────────────────────────────────────────────────────┐
│ Header            4 KB   定长，offset 0                     │
│ 类型描述表        ≤2 KB  FAM_A..FAM_C × 定长 128B 条目       │
│ 常量区            ≤16 KB grid/ksigns/ss16/码本 + sha 校验    │
│ 张量 blob 区      每 blob 4 KB 对齐，tile 16B 对齐内部        │
│ Index             N × 96B 定长记录（放文件尾，先写 blob 后写）│
└────────────────────────────────────────────────────────────┘
```

**Header 字段（4KB，逐字段）**：

| 偏移 | 字段 | 类型 | 语义 |
|---|---|---|---|
| 0 | magic | char[8] | `"MPK2"` + 版本字节 |
| 8 | version | u32 | 2（混合族）；layout_rev 单列 |
| 12 | endian/flags | u32 | bit0 小端；bit1 含稠密条目；bit2 含透传副本（默认 0，透传不复制字节） |
| 16 | gguf_identity | char[64] | sha256(gguf 头 64KB ∥ 文件长度 ∥ 尾 64KB) hex —— 边车↔GGUF 绑定，防错配 |
| 80 | fork_tag | char[32] | `"monster-gsq-rco"` + ggml commit hash（枚举表来源存证） |
| 112 | layout_rev | u32 | tile/unit 排布公式版本（内核与转换器必须同 rev，不等即拒载） |
| 116 | n_tensors / n_families / n_luts | u32×3 | 条目数 |
| 128 | desc_off / const_off / blob_off / index_off | u64×4 | 各段偏移 |
| 160 | blob_total / index_crc32 | u64/u32 | blob 总长；index 段校验 |
| 176.. | reserved | — | 清零，未来 PASSTHROUGH 副本、FAM-D 等扩展 |

**Index 记录（96B 定长，逐字段）**：

| 字段 | 类型 | 语义 |
|---|---|---|
| name_hash64 | u64 | 张量名 FNV-1a（避免文件内变长字符串） |
| gguf_tensor_idx | u32 | 回指 GGUF 张量表 |
| src_type / family_id | u32×2 | fork 类型号；0=PASSTHROUGH, 1=FAM-A, 2=FAM-B, 3=FAM-C |
| ne0..ne2 (K,N,E) | u32×3 | 逻辑形状 |
| n_tiles_n / n_tiles_k | u32×2 | = ceil(N/64), K/256 |
| blob_off / blob_len | u64×2 | 段位置与长度（=Σ tile 字节 + 对齐，逐张量字节中性） |
| src_len / src_sha128 | u64 / char[16] | 原 GGUF 字节长度与截断 sha256（T0 校验源） |
| blob_sha128 | char[16] | 重排段截断 sha256（加载期懒校验用） |
| pad / reserved | u32 | 对齐到 96B |

**manifest.json（非权威）**：创建环境、fork commit、探针结论、逐张量 {name, type, family, tiles, offsets, 双 sha, w/d 数值范围统计}、转换耗时、内联 T0 结果。供人查与工具对账，引擎永远不读。

### 3.2 类型描述表（本设计的核心扩展）

每族一条 128B 描述符，转换器与内核共用同一份常量（内核侧编译期实例化，加载期比对 const 区 sha 一致）：

```c
struct FamDesc {                    // 128B
  uint32 family_id;                 // 1/2/3
  uint32 block_size;                // 每块权重数：A=32, B=256, C=32
  uint16 d_bytes, code_bytes, aux_bytes;   // 块内三域字节数（见 §3.3-3.5）
  uint16 scale_levels;              // 二级 scale 档数：B=16（ss16），A/C=0
  uint16 sign_src;                  // 0=无(LUT 含符号) 1=ksigns_iq2xs 2=独立sign位
  uint16 grid_entries, grid_dim;    // grid LUT 形状；非 grid 族=0
  uint16 unit_weights;              // 每线程单元覆盖权重数（A=64, B=64, C=64/32）
  uint16 unit_code_bytes, unit_aux_bytes;
  uint32 lut_mask;                  // 引用常量区哪些 LUT（bit0=kvalues_iq4nl…）
  uint32 dequant_rev;               // 反量化代数 rev（T1 用）
  char    code_name[16];            // "IQ4_NL" 等调试用
  uint8  reserved[64];              // FAM-D(K-quant)/未来 GSQ-RCO 自定义槽位
};
```

**统一 tile 字节账**：tile = N64 × K256，字节数 = 64 × (256/block_size) × (d+code+aux 字节) —— 与原 GGUF 逐字节等量，只是三域分离重排。全族 tile 网格同构 → grouped 内核一份调度代码消费所有族。

### 3.3 FAM-A 条目：IQ4_NL（type 20 的 GEMM 张量，MVP）

**源结构**：block 32 权重 = F16 d (2B) + 16B 码（32×4bit 索引，16 项非线性码本 `kvalues_iq4nl`）= 18B → 4.5 bpw。反量化代数：`w = d × kvalues_iq4nl[nib]`（无符号位、无二级 scale，全族最简）。

**tile 布局**（9216B = 64 行 × 144B，字节精确等量）：

```
tile(64n × 256k) = [ d 平面 1024B ][ code 平面 8192B ]
d 平面:   d[row][kb8]  row=0..63, kb8=0..7（每行 256k 含 8 个 32 块）F16 → 64×8×2B
code 平面: 256 个 32B 单元，unit[t]:
             row  = t >> 2        （warp 一轮 = 32 单元 = 8 行 × 全 256k）
             kseg = t & 3         （64k 段）
             32B = 64 个 nibble，单元内 k 升序     ← GEMV 逐位等价的 k-序不变量
```

**内核消费**：warp 一轮 32 线程各取 32B（128B×8 事务全合并）+ 4B d（`d[row][kseg*2..+1]` 恰为该 64k 段的 2 个 scale，一次 ld.32）→ 每权重 1 次 smem LUT gather（fp16）+ 1 次 fp32 乘 → 写 smem W 片（XOR 交错）→ wmma。

**LUT 副本策略**：`kvalues_iq4nl` fp16×16 = 32B，常量区存一份；CTA 启动时向量复制进 smem（ld.global.v4→st.shared.v4），**×16 副本（lane 按 lane&15 私有化）= 512B**，gather 零 bank 冲突。

### 3.4 FAM-B 条目：IQ2_XXS/XS/S、IQ3_XXS/S（type 16/17/18/21/22）

**统一反量化框架**（继承 v1 对 IQ3_XXS 的钉死结构，其余成员按同框架填参）：

```
w[n,k] = d[n,⌊k/256⌋] · ss16[hi4] · grid_mag[code] · (±1)
       = F16 超块 scale × 16 档二级 scale × grid LUT 幅值 × ksigns 符号
符号 = fp16 幅值 XOR 0x8000（零指令误差）；scale = fp32 链乘一次舍入 ≤1 ulp
```

**tile 布局**（以 IQ3_XXS 为例，6272B = 64 行 × 98B；其余成员 tile 字节 = 64×块字节，同构）：

```
tile = [ d 平面 128B ][ code 平面 6144B ]
d 平面:   d[row] F16 ×64（1 超块 = 整个 K-tile）
code 平面: 256 个 24B 单元，unit[t]:
             row = t>>2, ib32 对 = (t&3)*2 与 +1（64 权重 = 2 个 32 权重子块）
             16B = 2×8 grid 索引字节（1B → grid 查 4 幅值）
              8B = 2×4B aux（高 4 位 hi4 二级 scale 档 + 28 位 ksigns 7bit 选择）
```

IQ3_S/IQ2_S 带 12-16B 级附加 scale 数组 → 描述符 `aux_bytes` 增大、d 平面扩展为 d+sub-scale 平面（`d_bytes` 字段吸收），**框架不变**。▲各成员块内精确位域以 fork `ggml-common.h` 结构体 + `dequantize_row_*` 为唯一真值，P3.0 由探针工具钉死填表（v1 已钉 IQ3_XXS：`ggml-common.h L404-411`、`ggml-quants.c L2575-2603`）。

**LUT 副本策略**：grid 预展开 fp16 进 smem（IQ3_XXS 256×4×2B = 2KB；IQ2 族 256×8×2B = 4KB）×2 副本 + ksigns_iq2xs 128B×4 副本 + ss16 32B×4 副本 ≈ **4.3-8.3KB/CTA 常驻**。grid gather 每 4 权重一次 8B 读（uint32→4 fp16），256 行 × 8B 跨满 bank，2 副本 + W 片 XOR 交错后冲突 <1.2 路。

### 3.5 FAM-C 条目：Q2_0（type 42）与 Q8_0（type 8）

- **Q8_0**：block 32 = F16 d + 32×int8 = 34B。纯算术反量化 `w = q × d`，无 LUT。tile = 64×8×34B = 17408B。MTP 层专家（2.49G）。
- **Q2_0**：block 32，块字节 ▲（反推 ≈9B：1B scale + 8B 码？fork 改造版），4 级均匀 LUT（或仿射 `(v-c)×d`，探针定）。tile = 64×8×块字节。down_exps 15 层 6.59G——**探针钉死前此条目 family_id 落 0（透传），格式零改动可切回**。

### 3.6 透传槽位（family_id = 0）

| 对象 | 原因 | 路径 |
|---|---|---|
| PLE `per_layer_token_embd` 26.86G（type 20 按名字排除） | 纯 token gather，非 GEMM | GGUF 原路径 |
| BF16 output_hc 1.43G（type 30） | `mmf.cu` 现 F16-TC 已覆盖 | 不动 |
| 稠密 K-quant：Q6_K attn_gate 0.81 + Q5_K output 0.54 + IQ4_XS attn_qkv 0.36 + 非专家 IQ3_S 0.38 + Q4_K shexp 0.28（合计 ≈2.4G） | v1 透传；FAM-D（仿射 scale-min，无 LUT）为 P4 可选条目 | 现反量化+cuBLAS |
| F32/F16 零头 | — | 不动 |

透传 = index 记存在性与 sha（对账用），**不复制字节**；引擎对这些张量走 GGUF 原路径。预留 `flags.bit2` 支持 v2 "透传副本"（把 2.4G 稠密也复制进边车统一管理），默认关。

### 3.7 对齐、padding 与 m8n8k4 倍数核算

| 量 | 值 | 核算 |
|---|---|---|
| K-tile | 256 | = 64× mma.m8n8k4 的 k4；= 16× wmma 16×16×16 的 k16。K=2560（40 tile）/640（10 tile）整除，**零 padding** |
| N-tile | 64 | = 8× mma n8；= 4× wmma n16。N=640/2560 整除，零 padding |
| 通用形状 | K→ceil 256，N→ceil 64 | 补行 d=0（全零行）、补 k 块 d=0；本检查点所有 GEMM 张量形状恰好整除，实际 padding = 0 |
| 对齐 | header 4K / blob 4K / tile 内单元 16B 段 | 全部 2 的幂，mmap 友好 |

### 3.8 内存开销账（设备 + 磁盘 + 红线）

| 项 | 数值 | 说明 |
|---|---|---|
| blob：FAM-A 专家 | 7.91 GiB | MVP |
| blob：FAM-B 专家 | 25.47 GiB | IQ3_S 8.73 + IQ2_S 5.00 + IQ2_XS 4.52 + IQ2_XXS 3.63 + IQ3_XXS 3.59 |
| blob：FAM-C 专家 | 9.08 GiB | Q2_0 6.59 + Q8_0 2.49 |
| blob：稠密（P4 可选） | 2.40 GiB | FAM-D 未定前透传不计入 |
| **blob 合计（全量）** | **42.46-44.87 GiB** | 逐张量字节中性 |
| 头+描述表+常量+index+对齐 | ≈ 14 MB | **+0.03%** |
| **设备侧权重总量** | **≈73.2 GiB（不变）≤ 85G 红线** | 重排替换原字节，不新增驻留；余 11.75G ≈ 2G/卡（KV/graph/激活/工作区 ~1.7G 见 §5.3） |
| 磁盘峰值 | 73 + 45 + 临时 ≈ **118G** | 服务器已清 275G，充裕 |

加载期工作区（额外显存，计入 11.75G 余量）：monpack 注册表 ≤2MB、work list 缓冲 ≤320KB（§5.5）、graph 池内激活不变——合计 **≤300MB**（T4 验收门）。

---

## 4. 转换工具 `mrt-repack`（fork 树内）

### 4.1 fork 枚举差异的处理（任务书问题 2 的正面回答）

三层防御，**首选方案从根上消灭分歧**：

1. **树内构建（首选）**：工具放 `tools/mrt/`，直接 link fork 的 `ggml` 类型特征表（`ggml_type_size` / `block_size` / `dequantize_row_*`）——枚举、块结构、反量化代数与引擎**编译期同源，零分歧可能**；
2. **探针模式**：`mrt-probe <gguf>` 读每类型 1 个块 hexdump + 反量化统计，人工比对已知格式 → 钉死 ▲ 项（Q2_0 块字节、IQ3_S/IQ2_S 位域），结论写进 manifest.json；
3. **独立兜底**：若必须脱离树构建，内置 vendored fork 枚举表（BOOL=7, ARRAY=9, UINT64=10；16/17/18/20/21/22/23/8/12/13/14/30/41/42 按蓝图 §1 钉死表），加载期 `fork_tag + gguf_identity` 双重绑定，identity 不符拒载。

### 4.2 流程与伪代码

```
1. 读 GGUF 张量表（流式 header，不加载权重）→ 按 §3.2 族表 + 名字规则(*_exps / GEMM 白名单)分类
2. 磁盘/形状预检：空闲 ≥ blob×1.05+1G；K%256==0、N%64==0（本检查点恒真，否则中止）
3. 逐张量逐专家逐 n-tile 流式重排（峰值内存 = 1 tile 缓冲 9-18KB）：
      for t in targets:
        fam = plan[t.type];  fam.assert_layout(t)          # 块字节/形状核对
        for e in E: for n0 in range(0,N,64):
          raw = read_rows(t, e, n0, K)                      # 64 行 × K 原始块
          out.write(fam.pack_tile(raw))                     # 三域分离 + 单元交织，§3.3/3.4
          sha_src.update(raw); sha_blob.update(packed)
          assert fam.unpack_tile(packed) == raw             # T0 内联：位级往返自证
4. 写 index（尾部）+ header + 常量区 sha → fsync → 原子 rename(.tmp → .monpack)
5. emit manifest.json（探针结论、耗时、逐张量 sha、数值范围统计）
```

```python
# FAM-A pack_tile 示意（numpy 向量化；POWER9 远快于 I/O，CPU 非瓶颈）
def pack_tile_iq4nl(rows64xK):                    # (64, 8, 18) 视图：8 个 32 块/行
    d    = rows64xK[:, :, 0:2]                    # → d 平面 64×8×2B（保持行序）
    nib  = rows64xK[:, :, 2:]                     # (64, 8, 16B)，单元内 k 升序拼接
    code = np.empty((64, 4, 32), np.uint8)
    for kseg in range(4):
        code[:, kseg] = nib[:, kseg*8:(kseg+1)*8].reshape(64, 32)   # 8 块×16B → 32B 单元
    return d.tobytes() + code.reshape(256, 32).tobytes()
```

**校验和体系**：逐张量 `src_sha128 + blob_sha128`（截断 sha256，index 内）；index 段 CRC32（header 内）；常量区 sha256（绑定码本/grid 与内核编译期常量一致）；GGUF 绑定 `gguf_identity`。加载期分级：默认 O(1)（magic/identity/长度/layout_rev），`MONSTER_MONPACK_VERIFY=full` 时后台线程懒校验 blob_sha128。

### 4.3 耗时账（SATA 425 MB/s）

| 阶段 | 数据量 | 时间 |
|---|---|---|
| 读目标张量（流式） | 42.5-44.9 GB | 100-106 s |
| 写边车 | 42.5-44.9 GB | 100-106 s（与读流水线重叠；同盘读写竞争 1.5× ≈ 150-160 s） |
| CPU 重排 + 内联 T0 | 内存带宽级 | 与 I/O 重叠，非瓶颈 |
| 校验 pass（可选/验收期） | 再读 45 GB | +106 s |
| **合计** | | **纯转换 4-6 min；含校验 7-12 min** |

断点续转：按张量粒度记 `.tmp.progress`，重跑跳过已完成张量（可选，默认整文件重转更简单——反正 6 分钟）。

---

## 5. 内核消费侧

### 5.1 加载器认边车 + A/B 开关与回滚

```cpp
// llama-model-loader.cpp，init 后、create_tensor 前：
if (env MONSTER_MONPACK != "0" && exists(sidecar)) {
    mp = monpack_mmap(sidecar);
    if (!monpack_validate(mp, gguf_ctx))          // magic/identity/layout_rev/逐张量 ne/len/sha(O(1)
        mp = nullptr, log WARN "monpack ignored"; // 校验失败 = 静默回退旧路径
}
// load_tensor_data：data_src = mp ? monpack_blob(mp, t) : gguf_data(t);
//                   if (mp) ggml_cuda_monpack_register(data_src, fam_desc);
```

| 开关 | 语义 | 缺省 | 级别 |
|---|---|---|---|
| 边车文件存在性 | 文件级回滚：删除/改名即全局回滚 | — | 部署级 |
| `MONSTER_MONPACK` | 加载器认不认边车 | 1 | 进程级（重启生效） |
| `MONSTER_MONPACK_TYPES` | **新增**：灰度白名单（如 `"20"` 只激活 IQ4_NL；`"20,21,18"` 分族上线） | 全部 | 进程级 |
| `GGML_CUDA_MONPACK_TC_MIN_M` | TC 路径 M 阈值 | 16 | 运行期 |
| `GGML_CUDA_MONPACK_DISABLE` | kill-switch（回 GEMV-on-pack） | 0 | 运行期 |

**无热切换**：张量数据指针被 CUDA graph 固化，A/B 四象限（旧路径 / pack+全 GEMV / pack+TC≥16 / pack+全 TC）各需独立进程跑，env 切换 + 重启。回滚零风险：边车删掉 = 100% 旧字节旧路径，logits 逐位复现（T3 演练项）。

### 5.2 分派开口（file:line 落点）

| 落点 | 位置 | 动作 |
|---|---|---|
| mul_mat 分派 | `ggml-cuda.cu:2722` | `is_monpack(src0->data)` 注册表命中（O(1) hash）→ §5.1 表分支 |
| mul_mat_id 分派 | `ggml-cuda.cu:2821-2824`（grouped MMQ 前） | 同上；命中即 `monpack_tc_grouped`（M_total≥16）或 `monpack_gemv` |
| decode MMVQ | `ggml-cuda.cu:2767-2790` | 代码不动；monpack 激活时被 GEMV-on-pack 原位替换（逐位，§5.4） |
| `mmf.cu:133` should_use_mmf | 不改 | BF16/F16 继续走 mmf；新内核独立成文件 `monpack-kernels.cu`，复用 `mma.cuh:1427-1465` Volta tile 封装 |
| cuBLAS fallback | `:2833` | 不触碰；monpack 路径全程零 host-sync，graph 下永不落入 fallback |

### 5.3 `monpack_tc_grouped`：变长 M 的 grouped TC 内核

**形状现实**：每专家 M_e ∈ {0,1..30+}（泊松，M̄=10）→ 这不是稠密大 M GEMM，是**变长 grouped GEMM**。设计要点：

- **持久 block**：grid 固定 = 80 CTA（1/SM，256 线程 8 warp）——launch 配置与 M_e 无关 → CUDA graph key 稳定；
- **设备侧 work list**（graph 内前置节点构建，零 host sync）：

```
kernel build_worklist(ids, counts[E]):        # 路由结果已在设备上
  for e with counts[e]>0: for nt in n_tiles_n:
      emit {e, nt, m=counts[e], rows_off=prefix[e]}   # 静态确定性划分（T3 逐位复现要）
  # 缓冲按最坏 512×40 项 ×16B = 320KB 预分配，graph 池内
```

- **循环结构 k-外层 / n-内层**（关键：A 片复用，§6.2 的 A 流量账成立）：

```
__global__ persistent monpack_tc_grouped(work_list wl, params p):
  smem: W 片 64n×64k fp16 ×2buf(16KB) | A 片 16m×256k fp16(8KB) | LUT 常驻 §3.3/3.4 | ≈ 30-40KB
  for wi = blockIdx.x; wi < wl.n; wi += gridDim.x:        # 静态轮转，确定性
    {e, n0, m, rows} = wl[wi]
    acc[16m][64n] fp32 寄存器
    for kt in 0..K/256:
      阶段1 装配: 256 线程各取 1 个 24B/32B 单元（128B×6 全合并事务）
        FAM-A: w16 = lut16[nib] × d            （1 smem gather + 1 mul / 权重）
        FAM-B: mag4 = grid_fp16[c]; w16 = mag ^ sign(ksigns) 后 ×fp16(d×ss16[hi4])
        → 写 W 片 smem（XOR 交错消 bank 冲突），双缓冲预取下一 kt
      阶段2 A 装载: m 行 gather（行 id 查 CSR，行内 K 段连续全合并）→ smem A 片
      阶段3 计算: 4 warp × 4×(wmma 16×16×16)（= 64×m8n8k4 阵列）累加 fp32
    epilogue: fp32→fp16 写 C[e] 的 m 行 × 64 列（m<16 掩码，M_e 逐专家天然变长）
```

- **tile 参数**（M̄≈10 场景）：M-tile = 16（wmma m16；M_e≤8 为主的 decode-ish 分布可切 M-tile=8 走裸 m8n8k4 路径，tuning 项）；**J（warp 级 n 粒度）= 16 默认**（N-tile 64 = 4 warp × J16），J=8 为极限细粒度备选（需裸 mma 路径，寄存器换并行度）；K-tile 256 装载 / 64 分级装配。浪费率：M_e<16 的 mma 掩码浪费 ≈ (16−M_e)/16 ≈ 37%（带宽受限区无关紧要，§6.1）；n 方向零浪费（64|640/2560）。
- **负载不均**：work item 粒度 = (专家 × n-tile) ≈ 512×10~40 = 5-20K 项 ≫ 80 CTA → 静态轮转波尾 ≤ max(单专家 n-tail)/总波数 ≤ 5%；不采用 atomic 窃取（保 T3 逐位复现）。

### 5.4 `monpack_gemv`：decode 逐位保护

张量指针唯一且被 graph 固化 → monpack 激活后 decode（M≤8，ne2≤320 MTP chain）必须能读打包布局。因重排保持每行 k 数值序（§3.3/3.4 单元内 k 升序 + (row,kseg) 可逆映射），GEMV 内核按同映射取字节、喂旧 MMVQ 的 lane↔k 映射，**与旧 vec_dot 逐位一致**（容差 1e-6 仅作护栏）。decode 带宽不变（同字节流），性能回归≈0。

### 5.5 CUDA graph 兼容清单（`ggml-cuda.cu:3793` 检查项）

1. 零 host-sync：work list 构建为 graph 内前置节点；无 cudaMemcpy D2H、无 cudaDeviceSynchronize；
2. 零动态分配：work list 320KB + 全部 smem 静态，从 graph 私有池预分配；
3. launch 配置固定：持久 grid 80×256，与 M_e/ne2 数据解耦；MTP draft 行 fold 进 ubatch → M_e 分布漂移只改设备端 work list 内容，graph key 不变（形状集 M∈1..512×topk 在固定 key 下稳定的既定结论）；
4. 混合分派安全：未打包专家张量（灰度白名单外）走旧 grouped MMQ，同 graph 内并存，无 cuBLAS fallback 触发（`:2833`）；
5. 验收：全 graph key 集（含 MTP）capture 成功率 100% 为硬门（T4）。

---

## 6. Roofline（按真实负载形状）

### 6.1 定位：M̄≈10 是纯带宽区

有效强度 = 3.56×M FLOP/B（16/4.5 档），M̄=10 → **35.6 FLOP/B ≪ γ=124 → 带宽受限**；M̄=40 → 147 → 计算受限边缘（工程 55-75 TF）。**TC 化的真正收益不是 FLOPs（TC 利用率在 M̄=10 下仅 ~0.4%），而是：① 消灭反量化写回的 HBM 往返（流量 ~10×），② 消灭 grouped 小 M 的 cuBLAS/launch 开销**。mma/wmma 只是"零写回地算 fp16"的最廉价载体。

### 6.2 每前向 HBM 流量账（专家路径，ubatch=512，6 卡 tensor split）

| 路径 | 读量化 | 写 F16 | 读 F16 | 激活/C | 合计 |
|---|---|---|---|---|---|
| 现状（dequant 写回 + cuBLAS） | 45.6 GB | ≈214 GB | ≈214 GB | 4.7 GB | **≈478 GB** |
| 新路径（FAM 全量 + GEMV/TC） | 45.6 GB | 0 | 0 | 4.7 GB（A 经 k-外/n-内复用，读一次） | **≈50.3 GB** |

F16 字节数 = 123.3e9 权重 × 2B ≈ 214 GB（avg 3.4 bpw）。每卡（÷6）：现状 79.7 GB → 理想 88.5 ms；新路径 8.4 GB → 理想 9.3 ms。smem 装配吞吐核算：900 GB/s 权重流 → 20e9 权重/s/SM × ~5B smem（LUT gather + W 片写读）= 100 GB/s < 179 GB/s 可用 → **56% 占用，4.5× 内核吞吐裕量是也上限所在**（见 §6.4）。

### 6.3 端到端预测与置信度

前向墙 380 ms = 专家 FFN（现状）+ 残差（attn/PLE/稠密/开销）。现状专家 FFN 实际耗时 ∈ [177, 266] ms（理想 88.5 ms × cuBLAS 小 M 与 launch 风暴的 2-3× 实效率）；残差 ∈ [114, 203] ms。新路径专家 FFN = 12-16 ms（理想 9.3 × 0.6-0.75 装配效率）。

| 覆盖 | 新前向墙 | 预填 tok/s | 相对 1350 |
|---|---|---|---|
| MVP（仅 FAM-A，18.6% 专家字节） | 334-350 ms | 1460-1530 | 1.08-1.13× |
| 全族（FAM-A+B+C，100%） | **126-219 ms** | **2340-4060**（点估计 2900） | **1.7-3.0×** |

置信度：**中**（区间宽因"现状 FFN 实效率"未实测——P3.0 用 nsys 钉死 FFN 占比后区间收窄到 ±20%）。MVP 数字是覆盖字节占比的算术推论，置信度高。decode：GEMV-on-pack 逐位 → ~150 tok/s 不变（回归 ≤1% 为验收门）。

**验收门（任务书"比 1350 快多少才算成"）**：端到端预填 **≥1.8× = 2430 tok/s**（蓝图 T4 既定）；MVP 阶段门 = 不回退 + 内核级 M=40 fixture ≥3×；拉伸目标 2.5× = 3400。

### 6.4 新瓶颈排序（任务书要求明确指出）

| 排序 | 瓶颈 | 量级 | 缓解 |
|---|---|---|---|
| 1 | smem 装配吞吐（LUT gather + W 片写/读 ≈5B/权重） | 56% smem 带宽预算；实现劣化时 TC 段掉到 40-50 TF | 双缓冲、W 片寄存器直装变体（v1 风险表既定备选）、J=8 细粒度 |
| 2 | 变长 M 泊松波尾 | 静态轮转下 ≤5% | work item 细到 (专家×n-tile)；必要时 J 降 8 |
| 3 | 残余稠密 2.4G 仍走 dequant+cuBLAS（M=512 算力区） | ~12 ms/前向 | P4 FAM-D 条目 |
| 4 | LUT bank 冲突 | 副本策略后 <1.2 路 | §3.3/3.4 副本数调参 |
| 5 | A 侧 gather / 路由 scatter | 行内 K 连续，全合并 | 复用 fork 既有 permute 节点 |
| 6 | attention @8K fp16（残差地板） | 不在本方案范围 | 独立赛道 |

---

## 7. 验收协议（继承 v1 T0-T4，混合族矩阵化）

| 层 | 内容 | 通过标准 |
|---|---|---|
| **T0 位级往返** | 全部打包张量：pack→unpack→与 GGUF 原字节比对（转换器内联 + 独立复检工具）；**逐族 × 逐形状类**（K∈{640,2560}×N∈{640,2560}×E∈{1,512}） | 逐字节 100% 相等 |
| **T1 反量化等价** | packed 布局上反量化 vs fork `dequantize_row_*`（fp32 参考）；随机码 + 真实权重码各 1024 块/类型 | 逐元素 ≤1 fp16 ulp |
| **T2 GEMM 数值** | 固定种子 N(0,1) + 自然激活分布；M∈{1,8,10,31,40,128,512}；**泊松生成的变长 M_e 向量**（seed 固定复现）；referee = fp64 反量化 GEMM；同口径测旧路径；对齐 P15 fixture 形状 | rel-L2(new) ≤ max(rel-L2(old))×1.5 + 1e-6；绝对门 ≤1e-3（fp16 输入 fp32 累加，只有累加序差异） |
| **T3 自然分布** | 200 条真实语料 128-token 贪心，**同槽位 A/B**（P14 教训：跨槽 rel-L2 可达 1.5e-2）；另加 8K prompt 预填 logits 比对 | top-1 一致率 ≥99.5%；top-20 KL ≤1e-3；3 次重跑逐位一致（静态 work list 保证）；PPL 增量 ≤0.002；logits cos ≥0.9999 |
| **T4 性能与工程** | nsys + 既有锚点；graph 全 key 集（含 MTP）capture | 内核级 M=40 ≥3×；端到端预填 ≥1.8×（2430 tok/s）；decode 回退 ≤1%；VRAM 增量 ≤300MB；**capture 成功率 100%**；`MONSTER_MONPACK=0` 重启后 logits hash 与部署前逐位一致（回滚演练） |

A/B 四象限（旧路径 / pack+全 GEMV / pack+TC≥16 / pack+全 TC）用 `matrix_runner.py` 512-token 逐位框架，同槽位对比。

---

## 8. 工作量分级与风险

### 8.1 阶段（对齐蓝图 §8，人力 = 1 资深 CUDA + 0.5 系统支持）

| 阶段 | 内容 | 工期 | 出口 |
|---|---|---|---|
| **P3.0 去风险** | nsys 实测 FFN 占比（收窄 §6.3 区间）；`mrt-probe` 钉死 Q2_0/IQ3_S/IQ2_S 位域；从 `mma.cuh:1427-1465` 抄录 lane↔fragment 映射冻结 layout_rev=1；确认 6 卡分片边界与 64 行 tile 对齐 | 0.5 周 | 探针报告 + 描述表定稿 |
| **P1 = MVP** | FAM-A（IQ4_NL）条目 + 转换器 + manifest + 加载器/env + T0/T1 | 1.5-2 周 | T0/T1 全绿（蓝图 §10.1：两周内可见） |
| **P2 存储层换血** | FAM-B/C 条目 + `monpack_gemv` 全类型 + decode 逐位验证 | 1-1.5 周 | decode ≤1% 回归；即使 TC 内核失败也已锁定布局红利 |
| **P3 主战场** | `monpack_tc_grouped` + 分派开口 + work list + T2/T4 | 2-3 周 | 端到端 ≥1.8× |
| **P4 转正** | 稠密 FAM-D 可选 + T3 灰度（`MONSTER_MONPACK_TYPES` 分族上线）+ 回滚演练 | 1-2 周 | 灰度转正 |

### 8.2 风险清单

| # | 风险 | 概率/影响 | 缓解 |
|---|---|---|---|
| 1 | smem 装配效率不足，TC 段仅 40-50 TF | 中/中 | 带宽受限区仍保 ≥2× 内核收益；寄存器直装+shuffle 变体备选 |
| 2 | FAM-B 位域钉错（IQ3_S/IQ2_S/Q2_0 结构与官方有 fork 改动） | 中/高 | 探针先行；树内构建同源；T0 位级往返是硬门，错即拒 |
| 3 | lane↔fragment 映射抄错 → 单元交织全错 | 低/高 | layout_rev 绑定 + 转换器/内核 init 时 exhaustive 双射自检 |
| 4 | graph capture 破坏（隐藏 host-sync / 动态分配） | 低/高 | §5.5 清单 + T4 capture 100% 硬门；持久内核结构天然免疫 |
| 5 | 变长 M 负载不均波尾 >10%（小批 decode-ish） | 中/低 | work item 细化 + J=8；decode 本来走 GEMV |
| 6 | 边车与 GGUF 不同步（模型文件更新） | 低/中 | gguf_identity 绑定，不符静默回退 + WARN |
| 7 | 转换中断电/半文件 | 低/低 | .tmp + 原子 rename（既定） |
| 8 | 灰度期混布（部分专家打包部分不打包）分派遗漏 | 中/中 | 注册表白名单严格按张量；混布场景进 T2 矩阵 |
| 9 | 6 卡分片切点不在 64 行倍数 | 低/中 | P3.0 确认；必要时 loader 按切点拼 tile（仅加载期成本） |
| 10 | MVP 后误判收益不足而弃（E2E 仅 1.1×） | 中/高 | §6.3 已明示 MVP 是管道验证；性能门在 P3 全族覆盖 |

---

## 9. 任务书条款对照（traceability）

| 任务书要求 | 章节 |
|---|---|
| 1. sidecar 逐字段（头/段表/K 维交织到 m8n8k4 倍数/LUT 副本/scale 布局/padding/内存开销%与红线核算；IQ4_NL 与 IQ2/IQ3 分条目；30/42 槽位） | §3.1-3.8（30/42 按勘误分别处理：BF16 透传 §3.6、Q2_0 正式条目 §3.5） |
| 2. 转换流程（fork 枚举、耗时、校验和） | §4 |
| 3. 内核消费接口（加载器认边车、A/B+回滚 env、mmf 开口、M̄≈10 的 J=8/16 tile） | §5 |
| 4. roofline（M≈10 grouped、tok/s 区间与置信度、新瓶颈） | §6 |
| 5. 验收协议（固定逐位/自然 A/B/性能门） | §7 |
| 6. 工作量分级（MVP/完整/风险） | §8 |
