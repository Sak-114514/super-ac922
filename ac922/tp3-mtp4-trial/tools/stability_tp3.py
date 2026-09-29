#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run concurrent requests while recording scheduler and thermal evidence."""

import argparse
import json
import random
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

METRIC = re.compile(
    r"^vllm:(num_requests_running|num_requests_waiting)\{[^}]*\} ([0-9.]+)$", re.M
)


def write_jsonl(path: Path, record: dict, lock: threading.Lock) -> None:
    with lock, path.open("a") as output:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def gpu_temperatures() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return [int(value) for value in result.stdout.splitlines()[:3]]


def monitor(
    metrics_url: str,
    path: Path,
    stop: threading.Event,
    finished: threading.Event,
    lock: threading.Lock,
) -> None:
    last_thermal = 0.0
    temperatures: list[int] = []
    while not finished.is_set():
        now = time.monotonic()
        try:
            with urlopen(metrics_url, timeout=5) as response:
                values = {
                    key: int(float(value))
                    for key, value in METRIC.findall(response.read().decode())
                }
            if now - last_thermal >= 10:
                temperatures = gpu_temperatures()
                last_thermal = now
            record = {
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "running": values.get("num_requests_running", -1),
                "waiting": values.get("num_requests_waiting", -1),
                "gpu_c": temperatures,
            }
            write_jsonl(path, record, lock)
            if temperatures and max(temperatures) >= 70:
                stop.set()
        except Exception as exc:
            write_jsonl(path, {"monitor_error": repr(exc)}, lock)
        finished.wait(1)


def worker(
    worker_id: int,
    barrier: threading.Barrier,
    deadline: float,
    url: str,
    model: str,
    prompt_length: int,
    max_tokens: int,
    requests_per_client: int,
    path: Path,
    stop: threading.Event,
    lock: threading.Lock,
) -> None:
    barrier.wait()
    iteration = 0
    while (
        time.monotonic() < deadline
        and not stop.is_set()
        and (requests_per_client == 0 or iteration < requests_per_client)
    ):
        generator = random.Random(20260929 + worker_id * 1000000 + iteration)
        prompt = [generator.randrange(5000, 150000) for _ in range(prompt_length)]
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "return_token_ids": True,
            "cache_salt": f"tp3-stability-{worker_id}-{iteration}-{time.time_ns()}",
        }
        request = Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        record: dict[str, int | float | str] = {
            "worker": worker_id,
            "iteration": iteration,
        }
        try:
            with urlopen(request, timeout=300) as response:
                result = json.load(response)
            token_ids = result["choices"][0]["token_ids"]
            if len(token_ids) != max_tokens:
                raise ValueError(f"generated {len(token_ids)} of {max_tokens}")
            record["generated_tokens"] = len(token_ids)
            record["elapsed_s"] = round(time.monotonic() - started, 3)
        except Exception as exc:
            record["error"] = repr(exc)
            stop.set()
        write_jsonl(path, record, lock)
        iteration += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8003")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=1800)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--requests-per-client", type=int, default=0)
    parser.add_argument("--prompt-length", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--model", default="qwen3.8-27b-ac922")
    args = parser.parse_args()
    if not 1 <= args.clients <= 8 or args.requests_per_client < 0:
        parser.error("clients must be 1..8 and requests-per-client must be >= 0")
    requests_path = args.output_prefix.with_suffix(".requests.jsonl")
    metrics_path = args.output_prefix.with_suffix(".metrics.jsonl")
    summary_path = args.output_prefix.with_suffix(".summary.json")
    if requests_path.exists() or metrics_path.exists() or summary_path.exists():
        raise FileExistsError(args.output_prefix)
    lock = threading.Lock()
    stop = threading.Event()
    finished = threading.Event()
    barrier = threading.Barrier(args.clients)
    start = time.monotonic()
    watcher = threading.Thread(
        target=monitor,
        args=(f"{args.base}/metrics", metrics_path, stop, finished, lock),
        daemon=True,
    )
    watcher.start()
    with ThreadPoolExecutor(max_workers=args.clients) as pool:
        futures = [
            pool.submit(
                worker,
                worker_id,
                barrier,
                start + args.duration,
                f"{args.base}/v1/completions",
                args.model,
                args.prompt_length,
                args.max_tokens,
                args.requests_per_client,
                requests_path,
                stop,
                lock,
            )
            for worker_id in range(args.clients)
        ]
        for future in futures:
            future.result()
    finished.set()
    watcher.join(timeout=5)
    requests = [json.loads(line) for line in requests_path.read_text().splitlines()]
    samples = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    errors = [record for record in requests if "error" in record]
    temperatures = [max(s["gpu_c"]) for s in samples if s.get("gpu_c")]
    summary = {
        "elapsed_s": round(time.monotonic() - start, 1),
        "completed_requests": len(requests) - len(errors),
        "generated_tokens": sum(r.get("generated_tokens", 0) for r in requests),
        "errors": errors,
        "max_running": max((s.get("running", -1) for s in samples), default=-1),
        "max_waiting": max((s.get("waiting", -1) for s in samples), default=-1),
        "samples_running_four": sum(s.get("running") == 4 for s in samples),
        "samples_running_clients": sum(
            s.get("running") == args.clients for s in samples
        ),
        "max_gpu_c": max(temperatures, default=None),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if errors or stop.is_set() or summary["max_running"] < args.clients:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
