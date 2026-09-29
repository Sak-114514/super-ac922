#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure one concurrent streaming wave using returned token IDs."""

import argparse
import json
import random
import re
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

METRIC = re.compile(
    r"^vllm:(num_requests_running|num_requests_waiting)\{[^}]*\} ([0-9.]+)$",
    re.M,
)


def request_once(
    worker_id: int,
    barrier: threading.Barrier,
    url: str,
    model: str,
    length: int,
    max_tokens: int,
    key: str | None,
) -> dict:
    rng = random.Random(20260929 + worker_id * 1000000 + length)
    body = {
        "model": model,
        "prompt": [rng.randrange(5000, 150000) for _ in range(length)],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "return_token_ids": True,
        "cache_salt": f"mtp-concurrent-{length}-{worker_id}-{time.time_ns()}",
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(
        url,
        data=json.dumps(body).encode(),
        headers=headers,
    )
    barrier.wait()
    started = time.monotonic()
    events: list[tuple[float, int]] = []
    try:
        with urlopen(request, timeout=300) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                choices = json.loads(payload).get("choices") or []
                token_ids = choices[0].get("token_ids") or [] if choices else []
                if token_ids:
                    events.append((time.monotonic(), len(token_ids)))
        finished = time.monotonic()
        generated = sum(count for _, count in events)
        if generated != max_tokens:
            raise ValueError(f"generated {generated} of {max_tokens} tokens")
        first = events[0][0]
        decode_seconds = finished - first
        return {
            "worker": worker_id,
            "started": started,
            "first": first,
            "finished": finished,
            "events": events,
            "generated_tokens": generated,
            "ttft_s": round(first - started, 3),
            "decode_tps": round((generated - events[0][1]) / decode_seconds, 2),
            "total_s": round(finished - started, 3),
        }
    except Exception as exc:
        return {"worker": worker_id, "error": repr(exc), "started": started}


def watch(base: str, stop: threading.Event, samples: list[dict]) -> None:
    last_gpu = 0.0
    gpu_c: list[int] = []
    gpu_free_mib: list[int] = []
    while not stop.is_set():
        now = time.monotonic()
        try:
            with urlopen(f"{base}/metrics", timeout=5) as response:
                metrics = {
                    key: int(float(value))
                    for key, value in METRIC.findall(response.read().decode())
                }
            if now - last_gpu >= 5:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=temperature.gpu,memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )
                values = [line.split(",") for line in result.stdout.splitlines()[:3]]
                gpu_c = [int(value[0]) for value in values]
                gpu_free_mib = [int(value[1]) for value in values]
                last_gpu = now
            samples.append(
                {"time": now, **metrics, "gpu_c": gpu_c, "gpu_free_mib": gpu_free_mib}
            )
        except Exception as exc:
            samples.append({"time": now, "monitor_error": repr(exc)})
        stop.wait(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8003")
    parser.add_argument("--model", default="qwen3.8-27b-ac922")
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.clients <= 8 or args.prompt_length < 1 or args.max_tokens < 2:
        parser.error("clients must be 1..8; lengths must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    key = args.key_file.read_text().strip() if args.key_file else None

    barrier = threading.Barrier(args.clients)
    stop = threading.Event()
    samples: list[dict] = []
    monitor = threading.Thread(
        target=watch, args=(args.base, stop, samples), daemon=True
    )
    monitor.start()
    with ThreadPoolExecutor(max_workers=args.clients) as pool:
        futures = [
            pool.submit(
                request_once,
                worker_id,
                barrier,
                f"{args.base}/v1/completions",
                args.model,
                args.prompt_length,
                args.max_tokens,
                key,
            )
            for worker_id in range(args.clients)
        ]
        results = [future.result() for future in futures]
    stop.set()
    monitor.join(timeout=5)

    good = [record for record in results if "error" not in record]
    first_all = max((record["first"] for record in good), default=0.0)
    finish_first = min((record["finished"] for record in good), default=0.0)
    all_live_tokens = sum(
        count
        for record in good
        for event_time, count in record["events"]
        if first_all < event_time <= finish_first
    )
    all_live_seconds = max(0.0, finish_first - first_all)
    total_seconds = max((record["finished"] for record in good), default=0.0) - min(
        (record["started"] for record in results), default=0.0
    )
    summary = {
        "clients": args.clients,
        "prompt_tokens_each": args.prompt_length,
        "completed": len(good),
        "errors": [record for record in results if "error" in record],
        "generated_tokens": sum(record["generated_tokens"] for record in good),
        "elapsed_s": round(total_seconds, 3),
        "end_to_end_output_tps": round(
            sum(record["generated_tokens"] for record in good) / total_seconds, 2
        )
        if total_seconds > 0
        else None,
        "median_client_ttft_s": round(
            statistics.median(record["ttft_s"] for record in good), 3
        )
        if good
        else None,
        "median_client_decode_tps": round(
            statistics.median(record["decode_tps"] for record in good), 2
        )
        if good
        else None,
        "all_live_decode_tps": round(all_live_tokens / all_live_seconds, 2)
        if all_live_seconds > 0
        else None,
        "all_live_tokens": all_live_tokens,
        "max_running": max(
            (sample.get("num_requests_running", -1) for sample in samples),
            default=-1,
        ),
        "max_waiting": max(
            (sample.get("num_requests_waiting", -1) for sample in samples),
            default=-1,
        ),
        "max_gpu_c": max(
            (
                temperature
                for sample in samples
                for temperature in sample.get("gpu_c", [])
            ),
            default=None,
        ),
        "min_gpu_free_mib": min(
            (free for sample in samples for free in sample.get("gpu_free_mib", [])),
            default=None,
        ),
        "requests": results,
        "samples": samples,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in ("requests", "samples")
            }
        )
    )
    if len(good) != args.clients:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
