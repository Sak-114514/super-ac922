#!/usr/bin/env python3
"""Keep CPU memory bounded while letting NVIDIA manage verified V100 HBM."""

import datetime
import os
from pathlib import Path
import socket
import sys
import time


ROOT = Path('/sys/devices/system/memory')
NODES = Path('/sys/devices/system/node')
DEVICE_TREE = Path('/sys/firmware/devicetree/base')
LIMIT = 32 * 1024**3
V100_HBM_BYTES = 16 * 1024**3
GPU_LOCATIONS = {'GPU%d' % number for number in range(6)}
POLICY = ROOT / 'auto_online_blocks'


def be32(path):
    data = path.read_bytes()
    if len(data) != 4:
        raise RuntimeError('invalid device-tree cell: %s' % path)
    return int.from_bytes(data, 'big')


def gpu_memory_ranges(block_size):
    """Return GPU location -> (NUMA node, first block, end block).

    POWER9 firmware describes each NVLink GPU's HBM as a memory node.  The
    memory-region phandle on each GPU PCI node identifies its backing range.
    Only the first 16 GiB of each 128 GiB aperture can be GPU HBM on this
    six-V100-16GB machine; blocks elsewhere still receive the safety sweep.
    """
    refs = {}
    for pci in DEVICE_TREE.glob('pciex*/pci@*'):
        location = pci / 'ibm,loc-code'
        region = pci / 'memory-region'
        if not location.exists() or not region.exists():
            continue
        name = location.read_bytes().rstrip(b'\0').decode('ascii')
        if name not in GPU_LOCATIONS:
            continue
        ref = be32(region)
        if name in refs and refs[name] != ref:
            raise RuntimeError('conflicting memory-region for %s' % name)
        refs[name] = ref
    if set(refs) != GPU_LOCATIONS or len(set(refs.values())) != 6:
        raise RuntimeError('incomplete six-GPU device-tree memory mapping: %r' % refs)

    by_ref = {ref: name for name, ref in refs.items()}
    ranges = {}
    for memory in DEVICE_TREE.glob('memory@*'):
        phandle = memory / 'phandle'
        if not phandle.exists():
            continue
        ref = be32(phandle)
        if ref not in by_ref:
            continue
        reg = (memory / 'reg').read_bytes()
        if len(reg) != 16:
            raise RuntimeError('invalid GPU memory reg: %s' % memory)
        address = int.from_bytes(reg[:8], 'big')
        aperture = int.from_bytes(reg[8:], 'big')
        node = be32(memory / 'ibm,chip-id')
        if (address % block_size or V100_HBM_BYTES % block_size or
                aperture < V100_HBM_BYTES or not 250 <= node <= 255):
            raise RuntimeError('invalid GPU HBM range or node: %s' % memory)
        ranges[by_ref[ref]] = (node, address // block_size,
                               (address + V100_HBM_BYTES) // block_size)
    if set(ranges) != GPU_LOCATIONS or len({r[0] for r in ranges.values()}) != 6:
        raise RuntimeError('incomplete six-GPU HBM range mapping: %r' % ranges)
    return ranges


def is_gpu_block(path, number, ranges):
    return any(first <= number < end and
               (path / ('node%d' % node)).is_symlink()
               for node, first, end in ranges.values())


def cpu_memtotal_kb():
    total = 0
    for node in (0, 8):
        path = NODES / ('node%d' % node) / 'meminfo'
        if not path.exists():
            raise RuntimeError('CPU NUMA memory information missing: %s' % path)
        line = next((line for line in path.read_text().splitlines()
                     if 'MemTotal:' in line), None)
        if line is None:
            raise RuntimeError('CPU NUMA MemTotal missing: %s' % path)
        total += int(line.split('MemTotal:', 1)[1].split()[0])
    return total


def notify():
    address = os.environ.get('NOTIFY_SOCKET')
    if not address:
        return
    if address.startswith('@'):
        address = '\0' + address[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.sendto(b'WATCHDOG=1', address)


def sweep():
    if POLICY.read_text().strip() != 'offline':
        POLICY.write_text('offline\n')
    size = int((ROOT / 'block_size_bytes').read_text().strip(), 16)
    if size <= 0 or LIMIT % size:
        raise RuntimeError('unexpected memory block size %d' % size)
    boundary = LIMIT // size
    # A changed firmware map needs operator review.  The initramfs has already
    # offlined high-address blocks and the policy above remains offline; do
    # not guess that an unclassified block is safe to remove from a live GPU.
    gpu_ranges = gpu_memory_ranges(size)
    changed = 0
    failed = []
    examined = 0
    for path in ROOT.glob('memory[0-9]*'):
        number = int(path.name[6:])
        if number < boundary:
            continue
        if is_gpu_block(path, number, gpu_ranges):
            # NVIDIA explicitly onlines POWER9 HBM into ZONE_MOVABLE.  A
            # generic udev/kernel online into Normal is not driver-managed.
            state = path / 'state'
            if (state.read_text().strip() == 'online' and
                    (path / 'valid_zones').read_text().strip() != 'Movable'):
                try:
                    state.write_text('offline\n')
                    changed += 1
                except OSError:
                    failed.append(number)
                if state.read_text().strip() != 'offline' and number not in failed:
                    failed.append(number)
            continue
        examined += 1
        state = path / 'state'
        if state.read_text().strip() != 'offline':
            try:
                state.write_text('offline\n')
                changed += 1
            except OSError:
                failed.append(number)
        if state.read_text().strip() != 'offline' and number not in failed:
            failed.append(number)
    if POLICY.read_text().strip() != 'offline':
        raise RuntimeError('auto_online_blocks changed during sweep')
    memtotal_kb = cpu_memtotal_kb()
    if memtotal_kb > 34 * 1024 * 1024:
        raise RuntimeError('CPU NUMA MemTotal exceeds physical DIMMs: %d kB' % memtotal_kb)
    if failed:
        raise RuntimeError('non-GPU blocks still online: %r' % failed[:20])
    return examined, changed, memtotal_kb


def main():
    while True:
        examined, changed, memtotal_kb = sweep()
        if changed:
            print('%s ac922-memory-guard offlined=%d examined=%d MemTotal=%d kB' %
                  (datetime.datetime.utcnow().isoformat() + 'Z', changed, examined, memtotal_kb),
                  flush=True)
        notify()
        time.sleep(2)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('%s ac922-memory-guard FAILED: %s' %
              (datetime.datetime.utcnow().isoformat() + 'Z', exc),
              file=sys.stderr, flush=True)
        sys.exit(1)
