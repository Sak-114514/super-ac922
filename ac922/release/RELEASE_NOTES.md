# AC922 ppc64le binary candidate — 2026-09-28

This is a **candidate** artifact set for Python 3.11, Linux ppc64le, CUDA
12.4, and V100 SM70. The eight wheels and their hashes are listed in
[`WHEELS.sha256`](WHEELS.sha256); binaries are distributed as release assets,
not committed to the 1Cat source tree.

## Build provenance

- 1Cat source: [`1CatAI/1Cat-vLLM` at `db292f9a4`](https://github.com/1CatAI/1Cat-vLLM/commit/db292f9a49318459f064075bfdbbed438b4e77a3).
- PyTorch 2.10 source: [`pytorch/pytorch` at `449b1768`](https://github.com/pytorch/pytorch/commit/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0).
- Experimental 1Cat build dependency: `power-torch-cuda124` 2.12.0a1,
  original distribution by RomanMoz; **not included** in these assets.
- Target CUDA architecture: `sm_70`; CUDA toolkit 12.4.131; host compiler
  GCC 12.2; Python 3.11; Debian 12.11 ppc64le.

## Verification boundary

All eight staged wheels match [`WHEELS.sha256`](WHEELS.sha256) and pass ZIP
integrity checks. PyTorch 2.10 imported and computed a CPU tensor sum of
`6.0`. The isolated 1Cat wheel built against PowerTorch 2.12 loaded the
selected native extensions (`vllm._C`, SM70 W4A16, FlashAttention-V100,
FlashAttention 2, and FlashQLA GDN). A small GDN V100 kernel smoke passed
before the full wheel was built.

The release candidate **does not yet establish** successful Qwen3.8-27B
model loading, API serving, 4/8-concurrency throughput, speculative decoding,
PyTorch 2.10 GPU runtime, or general portability. Results from other machines
and upstream model benchmarks must not be attributed to these wheels.

The standalone PyTorch 2.10 wheel and the experimental 1Cat wheel target
**different Torch ABIs**. They are two separate investigation lanes, not a
single ready-to-install stack.
