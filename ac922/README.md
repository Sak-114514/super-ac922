# IBM AC922 / POWER9 ppc64le build snapshot

This directory records an **experimental** CUDA 12.4 build of 1Cat-vLLM and
its Python dependencies for an IBM Power System AC922 with Tesla V100 (SM70)
GPUs. It is based on upstream [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)
commit `db292f9a49318459f064075bfdbbed438b4e77a3`. The build host ran
Debian 12.11, Python 3.11, POWER9, glibc 2.36, CUDA 12.4 and NVIDIA driver
550.54.15. The host also uses a custom kernel; the wheels have **not** been
validated as portable across other Power systems or distributions.

> 中文：这是 AC922 上的实验性 ppc64le 构建记录。已验证 wheel 完整性和部分
> 导入／小算子；截至本快照，完整 Qwen3.8 模型服务与并发基准尚未验收。

## What is available

The [release manifest](release/WHEELS.sha256) names eight locally compiled
ppc64le wheels. Release binaries are kept outside Git history and should be
attached to a GitHub Release. Verify them with `sha256sum -c WHEELS.sha256`
after downloading all eight assets.

| Build | Verified so far | Important pairing |
| --- | --- | --- |
| PyTorch 2.10.0 CUDA 12.4, from PyTorch commit `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0` | Wheel integrity, Python import, CPU tensor sum | Standalone pinned-Torch candidate; V100 runtime check remains pending |
| 1Cat-vLLM `0.1.dev1+gdb292f9a4.cu124` | Wheel integrity and native extension imports; an isolated SM70 GDN kernel ran on V100 | Built **against** [`power-torch-cuda124` 2.12.0a1](https://github.com/RomanMoz/power-torch-cuda124/releases/tag/v2.12.0a1), **not** the PyTorch 2.10 wheel above |
| safetensors, tokenizers, tiktoken, msgspec, uvloop, psutil | Wheel hashes and archive integrity | Built locally for Python 3.11/ppc64le; ABI tags differ by package |

The 1Cat wheel's metadata requests `torch==2.10.0`, whereas the experimental
wheel above was compiled against the separate `power-torch-cuda124` 2.12.0a1
distribution. Do not combine the two release rows into one environment or
infer PyTorch C++ ABI compatibility from a successful wheel build. The
experimental 1Cat installation used `--no-deps` in an isolated environment;
full model loading, serving, MTP/DFlash2, and 4/8-request benchmarks remain
unverified at this snapshot.

## Reproduction material

- [`scripts/`](scripts/) contains the host-specific build, smoke-test, and
  benchmark commands. The default paths reflect the original AC922 layout;
  `AC922_WORK_ROOT` and `AC922_BUILD_ROOT` are available where noted.
- [`dependencies/WHEEL_CACHE.sha256`](dependencies/WHEEL_CACHE.sha256) records
  the 98 downloaded or compiled toolchain/cache files in the build snapshot.
  It is an **artifact inventory**, not a resolver lock or a claim that every
  listed package is needed by the Qwen serving path.
- [`dependencies/README.md`](dependencies/README.md) separates locally built
  wheels, vendor wheels, source archives, system packages, and excluded assets.
- [`release/RELEASE_NOTES.md`](release/RELEASE_NOTES.md) gives binary provenance,
  compatibility limits, and checks performed.

No model weights, Hugging Face tokens, subscription URLs, SSH material, BMC
configuration, host logs, NVIDIA proprietary archives, or third-party
PyTorch 2.12 binary are included in this fork or its proposed release assets.
Get those from their respective owners under their own terms.

The upstream source and this AC922 build material retain the repository's
Apache-2.0 license. Each rebuilt dependency keeps its own upstream license;
see [`release/THIRD_PARTY_NOTICES.md`](release/THIRD_PARTY_NOTICES.md).

## TP3 + DFlash2 serving trial (2026-09-29)

[`tp3-dflash2-trial/`](tp3-dflash2-trial/README.md) records a complete serving
campaign on this tree: **lossless TP3 head replication** for the hybrid
Qwen3.8-27B AWQ model (greedy parity 20/20 vs TP2), **DFlash2 speculative
decoding at 110 tok/s single stream (82.2% acceptance)**, and a **30-minute
4-way concurrency run at ~180 tok/s aggregate with zero errors**. The patch,
harnesses, per-request records, and the full report are included.

## TP3 + native MTP4 trial and production switch (2026-09-29/30)

[`tp3-mtp4-trial/`](tp3-mtp4-trial/README.md) is the sequel: the same fork
drives the model's **native single-layer MTP head** instead of the DFlash2
drafter. Three config-level blockers (the draft re-reading the raw
checkpoint, a 5120-wide `fc` that cannot split by 3, `--hf-overrides` not
reaching the draft) were fixed in four files with checkpoints untouched.
The bounded trial measured **242,119 KV tokens, 69.69 tok/s at 64k,
83.9% acceptance**; a three-tier utilization scan picked 0.91; and after a
production-only CUDA-Graph surprise (1.13 GiB vs 0.37 in the trial) the
service went to production at **0.88 with a 227,555-token KV pool** (2.3×
the DFlash2 production pool), **69.74 tok/s** single-stream 64k decode and
**116.30 tok/s** all-live decode at 4×50k. DFlash remains installed as a
one-command rollback. Patches, harnesses, per-request records, and the full
report are included.

Operational pitfall records from the same campaign and earlier:

- [`PITFALLS-FAN-THERMAL.md`](PITFALLS-FAN-THERMAL.md) — home-datacenter fan
  policy: hard RPM boundaries, the OCC activation window, and the incident
  where a fan-policy change cut power mid-compile.
- [`PITFALLS-MEMORY-LOWMEM.md`](PITFALLS-MEMORY-LOWMEM.md) — "ghost DIMM"
  HBM apertures, swap exhaustion, RAM-image build hygiene, and CUDA-graph
  headroom on 16GB cards.
