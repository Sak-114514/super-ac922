# AC922 SATA32 kernel source patch

`88se9235-dma-mask32.patch` is the source-only AHCI adjustment used for the
AC922 Marvell 88SE9235 controller. It limits that PCI device's DMA mask to 32
bits while preserving AHCI's high-register initialization. The Debian Linux
source file declares `SPDX-License-Identifier: GPL-2.0-or-later`; the patch and
config snapshot remain under that upstream license. See `COPYING`.

The patch and source-provenance notes are published. Compiled kernels, initrds,
modules, NVIDIA driver archives, boot UUIDs, and root-disk backups are not.
Those are device- and kernel-specific build outputs and should be regenerated
locally from the matching Debian source package and headers.

The associated GRUB script is deliberately a template. Replace every
`@...@` value after checking the target machine's filesystem UUIDs. Do not
copy the AC922 UUIDs from a live `grub.cfg` or the private recovery archive.

The current AC922 validation covered low-risk SATA reads and GPU/CUDA
initialization on the specific installed stack. It did not prove long-term
storage reliability or model-serving readiness. The older broad `AHCI 32-bit`
experiment is not included as a recommended patch.
