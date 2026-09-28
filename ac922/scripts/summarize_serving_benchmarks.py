#!/usr/bin/env python3
"""Summarize 1Cat-vLLM 4/8-concurrency serving benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

EXPECTED_REQUESTS = 32
EXPECTED_OUTPUT_TOKENS = 128


def validate_result(result: dict, path: Path) -> None:
    """Reject partial runs before comparing their throughput numbers."""
    lengths = result.get("output_lens", [])
    if (
        result.get("completed") != EXPECTED_REQUESTS
        or result.get("failed") != 0
        or len(lengths) != EXPECTED_REQUESTS
        or any(length != EXPECTED_OUTPUT_TOKENS for length in lengths)
        or len(result.get("itls", [])) != EXPECTED_REQUESTS
        or len(result.get("errors", [])) != EXPECTED_REQUESTS
    ):
        raise ValueError(
            f"{path}: expected {EXPECTED_REQUESTS} successful requests with "
            f"{EXPECTED_OUTPUT_TOKENS} output tokens and detailed timing each"
        )


def decode_rates(result: dict) -> list[float]:
    rates = []
    counts = result.get("output_lens", [])
    errors = result.get("errors") or [None] * len(counts)
    for count, intervals, error in zip(
        counts,
        result.get("itls", []),
        errors,
    ):
        if error or count < 2 or not intervals:
            continue
        seconds = sum(intervals)
        if seconds > 0:
            rates.append(len(intervals) / seconds)
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()

    print("| Run | Concurrency | Completed/Failed | Aggregate output tok/s | Median request decode tok/s | Spec acceptance |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for path in args.results:
        result = json.loads(path.read_text())
        validate_result(result, path)
        rates = decode_rates(result)
        acceptance = result.get("spec_decode_acceptance_rate")
        median_rate = f"{statistics.median(rates):.2f}" if rates else "n/a"
        acceptance_text = f"{acceptance:.3f}" if acceptance is not None else "n/a"
        print(
            f"| {path.stem} | {result.get('max_concurrency', '?')} | "
            f"{result.get('completed', '?')}/{result.get('failed', '?')} | "
            f"{result.get('output_throughput', 0):.2f} | "
            f"{median_rate} | {acceptance_text} |"
        )


if __name__ == "__main__":
    main()
