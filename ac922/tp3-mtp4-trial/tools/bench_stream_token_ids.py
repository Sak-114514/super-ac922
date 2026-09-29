#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure streamed token IDs, including multi-token speculative chunks."""

import argparse
import json
import random
import time
from pathlib import Path
from urllib.request import Request, urlopen


def run(
    url: str, model: str, length: int, max_tokens: int, key: str | None = None
) -> dict:
    generator = random.Random(length)
    prompt = [generator.randrange(5000, 150000) for _ in range(length)]
    cache_salt = f"ac922-tp3-{time.time_ns()}"
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "return_token_ids": True,
        "cache_salt": cache_salt,
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(
        url,
        data=json.dumps(body).encode(),
        headers=headers,
    )
    started = time.monotonic()
    ttft = None
    first_chunk_tokens = 0
    generated_tokens = 0
    stream_events = 0
    with urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            event = json.loads(data)
            choices = event.get("choices") or []
            if not choices:
                continue
            token_ids = choices[0].get("token_ids") or []
            if not token_ids:
                continue
            if ttft is None:
                ttft = time.monotonic() - started
                first_chunk_tokens = len(token_ids)
            generated_tokens += len(token_ids)
            stream_events += 1
    total = time.monotonic() - started
    if ttft is None or generated_tokens != max_tokens:
        raise ValueError(f"incomplete stream: ttft={ttft}, tokens={generated_tokens}")
    decode_seconds = total - ttft
    return {
        "prompt_tokens": length,
        "cache_salt": cache_salt,
        "generated_tokens": generated_tokens,
        "stream_events": stream_events,
        "first_chunk_tokens": first_chunk_tokens,
        "ttft_s": round(ttft, 3),
        "prefill_tps": round(length / ttft, 2),
        "decode_s": round(decode_seconds, 3),
        "decode_tps_after_first_token": round(
            (generated_tokens - 1) / decode_seconds, 2
        ),
        "decode_tps_after_first_chunk": round(
            (generated_tokens - first_chunk_tokens) / decode_seconds, 2
        ),
        "total_s": round(total, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("lengths", help="Comma-separated prompt lengths")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--model", default="qwen3.8-27b-ac922")
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    key = args.key_file.read_text().strip() if args.key_file else None
    for length in (int(value) for value in args.lengths.split(",")):
        record = run(args.url, args.model, length, args.max_tokens, key)
        print(json.dumps(record), flush=True)
        with args.output.open("a") as output:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
