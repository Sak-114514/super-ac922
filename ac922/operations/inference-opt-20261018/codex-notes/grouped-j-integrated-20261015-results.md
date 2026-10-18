# Integrated adaptive grouped MMQ fixture results (2026-10-15)

## Verdict

The real GPU prefix helper plus two original grouped MMQ launches was faster than legacy J64 prefix plus one original J64 grouped launch for all 21 packed-format/route cases. The full outputs were bitwise identical in every case, and both prefix implementations produced the expected offsets.

Timing includes the GPU prefix scan and grouped MMQ kernel(s), including the extra adaptive J64 kernel launch even when the hot queue is empty. Timing excludes host input generation, weight expansion, Q8 activation generation, and H2D copies. There are no cold/hot activation, ids, or bounds queue copies: both paths use the same global ids/bounds/Y buffers.

## Per-case timings

CUDA event average per graph replay; each graph had 8 warmups and 120 timed replays. Legacy graph has 2 nodes (legacy J64 prefix + J64 MMQ); adaptive graph has 3 nodes (actual dual-prefix + J32 cold MMQ + J64 hot MMQ).

| Route | Type | Legacy J64 prefix + MMQ (µs) | Adaptive prefix + cold J32/hot J64 (µs) | Speedup |
|---|---|---:|---:|---:|
| skew | IQ2_XXS | 1390.2421 | 939.6736 | 1.4795× |
| skew | IQ2_XS | 1738.4448 | 1050.3510 | 1.6551× |
| skew | IQ2_S | 1744.5546 | 1054.5835 | 1.6543× |
| skew | IQ3_XXS | 1300.8555 | 957.1157 | 1.3591× |
| skew | IQ3_S | 1355.3664 | 987.4091 | 1.3726× |
| skew | IQ4_NL | 1123.3621 | 817.7493 | 1.3737× |
| skew | Q2_0 | 1322.4790 | 807.3899 | 1.6380× |
| uniform20 | IQ2_XXS | 1266.2017 | 814.1312 | 1.5553× |
| uniform20 | IQ2_XS | 1619.8655 | 915.4730 | 1.7694× |
| uniform20 | IQ2_S | 1638.0160 | 917.9050 | 1.7845× |
| uniform20 | IQ3_XXS | 1311.0186 | 871.4752 | 1.5044× |
| uniform20 | IQ3_S | 1343.5649 | 903.6203 | 1.4869× |
| uniform20 | IQ4_NL | 1095.9189 | 735.1808 | 1.4907× |
| uniform20 | Q2_0 | 1297.5616 | 678.5792 | 1.9122× |
| zeros | IQ2_XXS | 918.5280 | 820.0619 | 1.1201× |
| zeros | IQ2_XS | 1253.1115 | 985.8133 | 1.2711× |
| zeros | IQ2_S | 1255.7142 | 984.6614 | 1.2753× |
| zeros | IQ3_XXS | 937.7878 | 847.0699 | 1.1071× |
| zeros | IQ3_S | 975.6928 | 873.5915 | 1.1169× |
| zeros | IQ4_NL | 834.1590 | 679.8336 | 1.2270× |
| zeros | Q2_0 | 976.1536 | 699.5883 | 1.3953× |

Unweighted mean of per-type speedups: skew 1.5046×, uniform20 1.6433×, zeros 1.2161×. These are fixture-only averages, not model-level throughput estimates.

## Correctness and prefix checks

All 21 cases passed exact comparison: `bitdiff=0`, `max_abs=0`, `nonfinite=0`. All baseline and adaptive offset arrays matched the host-derived expected prefix arrays. Raw stdout reports per-case tile totals; for example, skew counts gave baseline 528 tiles, adaptive cold 448 and hot 80; uniform20 had baseline 512 and adaptive cold 512/hot 0; zeros route had 128 experts with count 0 and gave baseline 384, cold 256, hot 128 tiles.

## Fixture boundary

- Device: GPU2 UUID `GPU-e6529eb8-aa1f-1d02-5d63-dfb7995bf1eb`, Tesla V100-SXM2-16GB, SM70.
- Routes are synthetic: skew (16×128, 48×64, 112×24, 160×12, 160×3, 16×2), uniform20 (512×20), and zeros (128×0, 256×20, 128×40), each with 10,240 assignments and shuffled global destination IDs.
- Weights use 16 actual packed rows from expert 0 in `fixtures/volta-tc-real/manifest.json`, repeated over synthetic experts/output rows. Activations use Q8_1_MMQ blocks with random FP32 scale mantissas and the same global-token input bytes for both paths.
- This is packed-real-weights/synthetic-routing kernel evidence, not production routing, not a full model, and not end-to-end throughput.
- GPU2 returned to 1 MiB usage, 0% utilization, and no compute applications after the runner exited.

Artifacts:

- Exact raw stdout: `notes/grouped-j-integrated-20261015-results.log`
- Structured output: `notes/grouped-j-integrated-20261015-results.json`
- Fixture source/binary: `fixtures/grouped-j-integrated-20261015/grouped-j-integrated-fixture.cu` and `fixtures/grouped-j-integrated-20261015/grouped-j-integrated-fixture`
- Build log and plan/prep notes: `notes/grouped-j-integrated-20261015-build.log`, `notes/grouped-j-integrated-20261015-plan.md`, `notes/grouped-j-integrated-20261015-prep.md`
