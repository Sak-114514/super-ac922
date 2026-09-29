#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture deterministic TP2 outputs and compare them with a TP3 trial."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from urllib.request import Request, urlopen

CASES = (
    [(1000, seed) for seed in range(7)]
    + [(8000, seed) for seed in range(7)]
    + [(64000, seed) for seed in range(6)]
)


def prompt_tokens(length: int, seed: int) -> list[int]:
    generator = random.Random(20260929 + seed * 100000 + length)
    return [generator.randrange(5000, 150000) for _ in range(length)]


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Local /v1/completions endpoint")
    parser.add_argument("output", type=Path, help="JSONL result file")
    parser.add_argument("--reference", type=Path, help="Previously captured TP2 JSONL")
    parser.add_argument("--key-file", type=Path, help="Read API key without logging it")
    parser.add_argument("--model", default="qwen3.8-27b-ac922")
    parser.add_argument("--case", action="append", help="Run only this case ID")
    args = parser.parse_args()

    reference = {}
    if args.reference is not None:
        for line in args.reference.read_text().splitlines():
            record = json.loads(line)
            reference[record["case"]] = record
    existing = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            record = json.loads(line)
            existing[record["case"]] = record
    key = args.key_file.read_text().strip() if args.key_file else None

    mismatches = sum(record.get("matches_tp2") is False for record in existing.values())
    for length, seed in CASES:
        case = f"p{length}-s{seed}"
        if args.case and case not in args.case:
            continue
        prompt = prompt_tokens(length, seed)
        prompt_hash = token_digest(prompt)
        if case in existing:
            if existing[case]["prompt_sha256"] != prompt_hash:
                raise ValueError(f"saved prompt does not match {case}")
            if args.reference is not None:
                prior_digest = token_digest(reference[case]["output_token_ids"])
                if existing[case].get("reference_token_sha256") != prior_digest:
                    raise ValueError(f"saved TP3 comparison uses another TP2 {case}")
                if existing[case].get("matches_tp2") not in (True, False):
                    raise ValueError(f"saved TP3 comparison is incomplete for {case}")
            print(f"SKIP {case}: already captured", flush=True)
            continue
        body = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": 32,
            "min_tokens": 32,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
            "logprobs": 1,
        }
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = Request(args.url, data=json.dumps(body).encode(), headers=headers)
        started = time.monotonic()
        with urlopen(request, timeout=900) as response:
            result = json.load(response)
        elapsed = time.monotonic() - started
        choice = result["choices"][0]
        output_ids = choice["token_ids"]
        if not isinstance(output_ids, list) or len(output_ids) != 32:
            raise ValueError(f"{case}: expected 32 generated token IDs")
        record = {
            "case": case,
            "prompt_length": length,
            "prompt_sha256": prompt_hash,
            "output_token_ids": output_ids,
            "token_logprobs": (choice.get("logprobs") or {}).get("token_logprobs"),
            "elapsed_s": round(elapsed, 3),
        }
        if args.reference is not None:
            prior = reference[case]
            if prior["prompt_sha256"] != prompt_hash:
                raise ValueError(f"TP2 reference prompt does not match {case}")
            record["reference_token_sha256"] = token_digest(prior["output_token_ids"])
            record["matches_tp2"] = output_ids == prior["output_token_ids"]
            if not record["matches_tp2"]:
                mismatches += 1
                record["first_mismatch"] = next(
                    i
                    for i, (old, new) in enumerate(
                        zip(prior["output_token_ids"], output_ids)
                    )
                    if old != new
                )
        with args.output.open("a") as output:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
        status = "MATCH" if record.get("matches_tp2") else "DONE"
        if record.get("matches_tp2") is False:
            status = "DIFF"
        print(f"{status} {case} {elapsed:.1f}s", flush=True)
    if args.reference is not None and mismatches:
        raise SystemExit(f"TP3 differs from TP2 in {mismatches} cases")


if __name__ == "__main__":
    main()
