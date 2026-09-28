# Third-party wheel notices

These binaries are rebuilds of upstream open-source packages. A rebuilt wheel
does not transfer upstream ownership to this fork. Preserve the license files
inside each wheel and consult the linked source release before redistribution.

| Wheel | Source | License evidence |
| --- | --- | --- |
| `torch-2.10.0` | [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause in wheel `LICENSE` and `NOTICE` |
| `1cat_vllm-0.1.dev1+gdb292f9a4.cu124` | [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) | Apache-2.0 in wheel; bundled FlashQLA license also present |
| `safetensors-0.8.0` | [Hugging Face safetensors](https://github.com/huggingface/safetensors) | Upstream license file inside wheel |
| `tokenizers-0.22.2` | [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) | Apache-2.0 license in the source archive; [supplemental copy](licenses/tokenizers-LICENSE) because the locally built wheel lacks it |
| `tiktoken-0.14.0` | [OpenAI tiktoken](https://github.com/openai/tiktoken) | MIT license inside wheel |
| `msgspec-0.21.1` | [msgspec](https://github.com/jcrist/msgspec) | BSD-3-Clause inside wheel |
| `uvloop-0.22.1` | [uvloop](https://github.com/MagicStack/uvloop) | MIT and Apache license files inside wheel |
| `psutil-7.2.2` | [psutil](https://github.com/giampaolo/psutil) | BSD-3-Clause inside wheel |

The release set intentionally excludes NVIDIA cuDNN/CUPTI archives, model
weights, the downloaded PowerTorch 2.12 wheel, and IBM/PyPI dependency wheels.
The dependency inventory links to their original distributors instead.
