# super-ac922 中文说明(AC922 POWER9 / V100 实验记录)

> 本分支(`ac922-ppc64le`)是 [1Cat-vLLM](README.md) 在 IBM AC922
> (POWER9 / Tesla V100-SM70)上的**构建快照与实验记录仓库**。英文总览见
> [ac922/README.md](ac922/README.md),上游项目介绍见 [README.md](README.md)。

## 这个仓库里有什么

| 内容 | 入口 |
|---|---|
| **TP3 + DFlash2 服务战役(2026-09-29)** | [ac922/tp3-dflash2-trial/README.md](ac922/tp3-dflash2-trial/README.md) |
| ppc64le 构建快照、依赖 wheel 清单 | [ac922/README.md](ac922/README.md) |
| 家用温控踩坑(风扇策略断电事故复盘) | [ac922/PITFALLS-FAN-THERMAL.md](ac922/PITFALLS-FAN-THERMAL.md) |
| 低内存/内存管理踩坑(幽灵 DIMM、swap、图捕获头寸) | [ac922/PITFALLS-MEMORY-LOWMEM.md](ac922/PITFALLS-MEMORY-LOWMEM.md) |

## 核心成绩(2026-09-29 实测,Qwen3.8-27B AWQ)

在一台 AC922(POWER9,6×V100-SXM2-16GB,取 NVLink 三角岛 0/1/2)上:

- **TP3 无损部署**:混合注意力模型(16 全注意力 + 48 GDN)的 Q/KV/GDN 头
  加载期无损复制,贪心输出与现役 TP2 **20/20 逐 token 一致**
- **单流有效解码 110 tok/s**(DFlash2 深度 7,接受率 82.2%)——裸解码的
  2.9 倍,验收线(30-38)的 2.9 倍
- **四并发 ~180 tok/s 持续 30 分钟,零错误**(1,265 请求,32 万 token)
- **199k 预填充 167s(TP2 基线 375s,2.2×)**;KV 池 398,857 tokens @ fp8
- 16GB V100 ×3,GPU 全程 43–57 °C,零热事件

方法:加载期权重手术(源列/复制列各乘 ½ 的无损头复制 + 词表/MLP 内部
补齐)+ 有界 KV 公共页方案(解决上游 `NotImplementedError`),checkpoint
文件零改动。完整数学、数据与复现步骤见试验目录 README。

## 快速上手

```bash
git clone https://github.com/Sak-114514/super-ac922.git
cd super-ac922
git am ac922/tp3-dflash2-trial/patches/0001-*.patch   # 应用 TP3+DFlash2 变更
# 按 ac922/tp3-dflash2-trial/README.md 配置权重与环境变量后启动
```

## 许可

Apache-2.0(与上游 vLLM / 1Cat-vLLM 一致)。不包含模型权重、凭据与任何
第三方二进制。
