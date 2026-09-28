"""Stop the isolated 1Cat build before it starves AC922's RAM-only Torch build."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import time

import psutil

ROOT = Path("/home/debian/ac922-vllm/ram-build/fs")
RAM_MOUNT = ROOT.parent
ALT_PID_FILE = ROOT / "third-party-test/1cat-torch212-build.pid"
GIB = 1024**3


def log(message: str) -> None:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    print(f"{now} {message}", flush=True)


def available_ram() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing")


def available_tmpfs() -> int:
    fs = os.statvfs(RAM_MOUNT)
    return fs.f_bavail * fs.f_frsize


def stop_alternate(process: psutil.Process, reason: str) -> None:
    children = process.children(recursive=True)
    log(f"stopping alternate build PID {process.pid}: {reason}; {len(children)} descendants")
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        process.terminate()
    except psutil.NoSuchProcess:
        pass


def main() -> None:
    low_samples = 0
    while True:
        try:
            pid = int(ALT_PID_FILE.read_text().strip())
            process = psutil.Process(pid)
            command = " ".join(process.cmdline())
        except (FileNotFoundError, ValueError, psutil.NoSuchProcess):
            log("alternate build is no longer running; monitor exits")
            return
        if "venv-third-party/bin/python" not in command or "pip wheel" not in command:
            log(f"PID {pid} is not the expected alternate build; monitor exits")
            return

        ram = available_ram()
        tmpfs = available_tmpfs()
        if ram < 5 * GIB or tmpfs < 3 * GIB:
            stop_alternate(process, f"hard threshold: RAM {ram / GIB:.1f} GiB, tmpfs {tmpfs / GIB:.1f} GiB")
            return
        low_samples = low_samples + 1 if ram < 7 * GIB else 0
        if low_samples >= 2:
            stop_alternate(process, f"RAM below 7 GiB twice: {ram / GIB:.1f} GiB")
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
