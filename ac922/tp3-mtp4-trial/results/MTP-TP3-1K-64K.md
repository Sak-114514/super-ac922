# AC922 native TP3 + MTP4 bounded trial (2026-09-29)

An isolated RAM overlay of the existing Torch 2.12 / CUDA 12.4 runtime used
the local Qwen3.8-27B AWQ checkpoint and `model-mtp.safetensors`. The target
and native MTP layer were configured for 36 Q / 6 KV heads, with BF16 MTP
weights expanded in memory and the MLP padded to 17664. The actual checkpoint
passed all seven tensor-shape and half-output checks before model startup.

Trial settings: TP3 on GPU0/1/2, native MTP depth 4, target and MTP KV
`fp8_e4m3`, 65536 maximum model length, 4096 maximum batched tokens, four
maximum sequences, 0.90 GPU memory utilization. The API bound only to
`127.0.0.1:8003`. Production DFlash on port 8000 was stopped for the trial.

Startup succeeded. Model loading used 7.79 GiB per GPU; available KV memory
was 5.26 GiB per GPU and the server reported 242119 KV tokens. CUDA Graph
capture used 1.13 GiB per GPU. The two cold 128-token streamed completions
used unique cache salts and returned all requested tokens:

| Prompt | TTFT | Prefill | Decode | Total |
|---|---:|---:|---:|---:|
| 1k | 0.653 s | 1531.94 tok/s | 61.52 tok/s | 2.717 s |
| 64k | 35.693 s | 1793.05 tok/s | 69.69 tok/s | 37.516 s |

The MTP draft counters changed by 236 proposed and 198 accepted tokens
(83.9%). No CUDA OOM, fatal engine error, or NCCL timeout appeared in the
trial log, and the isolated API remained HTTP 200 after both requests.

Four additional one-wave concurrency checks used the same 0.90 utilization.
Each client submitted exactly one request; no sustained load test was run:

| Wave | Completed | Output tokens | Elapsed | Max Running / Waiting | Max GPU |
|---|---:|---:|---:|---:|---:|
| 2×1k, 128 output each | 2 | 256 | 6.4 s | 2 / 0 | 41°C |
| 4×1k, 128 output each | 4 | 512 | 4.2 s | 4 / 0 | 41°C |
| 2×64k, 32 output each | 2 | 64 | 67.4 s | 2 / 0 | 59°C |
| 4×60k, 32 output each | 4 | 128 | 125.6 s | 4 / 0 | 61°C |

There were zero request or monitor errors. Four 60k requests were running in
101 one-second samples. At that wave's sampled GPU-memory peak, `nvidia-smi`
showed only 31–32 MiB free per card. Later streaming waves had different
sampled minima, so one wave cannot establish a durable free-memory margin.

The run did not check token-by-token quality against DFlash or TP2, 200k
requests, or 30-minute stability; none of those is implied by these waves.

The first MTP process was stopped and DFlash production was temporarily
restored before the utilization comparisons below. `mtp4-1k64k.jsonl`,
`mtp4-before.prom`,
`mtp4-after.prom`, `mtp-c2-1k.*`, `mtp-c4-1k.*`, `mtp-c2-64k.*`,
`mtp-c4-60k.*`, and `serve-mtp-r{1,2}.log` retain the raw evidence.

## Longer output and utilization comparison

Each row was one concurrent streaming wave with unique prefixes: 512 output
tokens per 1k request or 256 per long request. “All-live decode” counts
returned token IDs only during the interval when every client was decoding;
end-to-end output rate includes prefill.

| Utilization | Wave | All-live decode | End-to-end output | Running / Waiting | Min GPU free |
|---:|---|---:|---:|---:|---:|
| 0.90 | 2×1k | 153.38 tok/s | 129.75 tok/s | 2 / 0 | 985 MiB |
| 0.90 | 4×1k | 367.47 tok/s | 232.93 tok/s | 4 / 0 | 577 MiB |
| 0.90 | 2×64k | 84.73 tok/s | 7.13 tok/s | 2 / 0 | 275 MiB |
| 0.90 | 4×60k | no overlap | 6.24 tok/s | 4 / 0 | 249 MiB |
| 0.92 | 2×1k | 179.95 tok/s | 141.33 tok/s | 2 / 0 | 645 MiB |
| 0.92 | 4×1k | 352.48 tok/s | 237.50 tok/s | 4 / 0 | 237 MiB |
| 0.91 | 2×1k | 181.50 tok/s | 145.57 tok/s | 2 / 0 | 183 MiB |
| 0.91 | 4×1k | 370.35 tok/s | 286.36 tok/s | 4 / 0 | 183 MiB |
| 0.91 | 2×64k | 84.67 tok/s | 7.07 tok/s | 2 / 0 | 215 MiB |
| 0.91 | 4×60k | 96.22 tok/s | 7.65 tok/s | 4 / 0 | 183 MiB |
| 0.91 | 4×50k | 123.56 tok/s | 9.69 tok/s | 4 / 0 | 91 MiB |

All requests completed without HTTP or monitor errors. In the 0.90 four-long
wave, no interval had all four requests decoding simultaneously: one client's
first token arrived after earlier clients finished. The 0.92 long-input
cases were not run after short-input headroom fell to 237 MiB. Although
0.91 passed bounded 4×50k and 4×60k trials, its 91 MiB sampled minimum
free memory was too narrow to transfer to a cold production process without
checking that process's own GPU allocations.

## Production MTP API

An isolated persistent TP3+MTP runtime was installed without modifying the
earlier DFlash runtime. The DFlash user unit remains installed but disabled;
`ac922-coding-api-tp3-mtp.service` is enabled on the original LAN port 8000
with the same API key, coding parser, and middleware. Its first 0.91
production start captured 1.13 GiB of CUDA Graph memory per GPU and left
only 37–38 MiB free at idle despite a 249400-token KV pool. No request was
sent in that state. The unit was backed up, changed to utilization 0.88,
and restarted.

At 0.88, production started with 7.79 GiB model loading per GPU, 0.37 GiB
CUDA Graph capture, and **227555 KV tokens**. Unauthorized model listing
returned HTTP 401; authorized OpenAI chat and Anthropic messages both
returned HTTP 200. A first 64k/128 cold-prefix completion decoded at
41.10 tok/s; a second cold-prefix request with a new cache salt decoded at
**69.74 tok/s** (TTFT 35.735 s), consistent with the isolated run.

The authenticated production 4×50k/256 wave completed all four requests,
max Running=4 / Waiting=0, in 107.734 s. All-live decode was **116.30
tok/s**, end-to-end output rate **9.50 tok/s**, per-client decode median
**10.45 tok/s**, maximum GPU temperature 60°C, and minimum sampled free GPU
memory **359 MiB/card**. The service remained HTTP 200. No production
4×60k wave, long stability run, or token-by-token quality comparison was
performed. The 0.88 production measurements must not be replaced with the
0.91 isolated trial's numbers.

Persistent production files are under
`/home/debian/ac922-vllm/runtime-tp3-mtp-20260929/`, with
`start_coding_api_tp3_mtp_gpu012.sh`, `serve_tp3_mtp.sh`, and the user unit
`ac922-coding-api-tp3-mtp.service`. DFlash's unit and runtime remain as a
rollback. Raw production results are `mtp088-prod-64k*.jsonl`,
`mtp088-prod-c4-50k-o256.json`, and `coding-api-tp3-mtp.log`.
