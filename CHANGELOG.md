# Changelog

## Batched decode: up to 8 requests in one cycle

Two RTX 3090s, 8 parallel requests, total tok/s: **Q4_K_M 298 to 624 (2.1x)**, Q8_0 561 to 595. One request: Q4_K_M 186 tok/s, Q8_0 141.

Every step below was found and checked with the per-phase profiler (`MK_PROFILE=batch`): time per phase per cycle, block arrival at each barrier, a "min-composed" estimate that filters out other GPU users, and a per-step record of block 0 (activation loads, compute, reduce, send, peer wait). Each build ran the same sweep (Q4_K_M and Q8_0, 1 to 8 parallel requests, coding prompt, 512 tokens each, greedy, `-d 4`).

### Build by build

| build | change | Q4_K_M 1 req | Q4_K_M 8 req | Q8_0 1 req | Q8_0 8 req |
|---|---|---|---|---|---|
| h | Q8_0 tensor-core rows up to 32 (baseline) | | | 142 | 561 |
| i | K-quants on dp4a, capped at 10 rows per cycle | 174 | 298 | | 560 |
| k | K-quant tensor-core rows (16/32 by slot count), swizzled activation layout | 174 | 433 | | 500 |
| l | K-quant weights in 16-row tiles | 163 | 506 | | 518 |
| m | Q6_K whole-block path; FFN gate/up 4 rows per step | 180 | 513 | | 506 |
| n | Q4_K/Q5_K whole-block path | 179 | 542 | | 509 |
| o | bf16 exchange between the cards, only rows in use cross PCIe, one fence per block | 183 | 585 | 146 | 561 |
| p | one GEMV instance per weight type (type checked once per phase) | 183 | 583 | 146 | 570 |
| q | int8 exchange with a scale per 32 values | 182 | 603 | 139 | 587 |
| r | 3 K-quant groups in flight in the residual GEMV: slower, reverted | 182 | 585 | 139 | 584 |
| s/t | profiler step record; faster code generation, confirmed by a back-to-back drift check | **186** | **624** | 141 | **595** |

Time per cycle at 8 requests, Q4_K_M: 62.5 ms (k) to 43.7 ms (t). Build i was fast per cycle (33 ms) but carried at most 10 rows, so it could not batch more than about 2 requests' drafts.

### What moved the numbers

- **Tensor cores past 10 rows** (k to n): the dp4a GEMVs keep every row in registers and spill above 10 rows. int8 `mma.m16n8k32` straight from the GGUF blocks lifts the cap to 32 rows; tiling the weights and decoding whole Q4_K/Q5_K/Q6_K superblocks at once took Q4_K_M from 433 to 542.
- **Cheaper exchange** (o, p, q): the three phases that sum across the cards (attention output, DeltaNet output, FFN down) went from 24.8 ms (m) to 15.8 ms (t) per cycle on Q4_K_M. Partials now cross PCIe as int8 with a scale per 32 values, only for rows in use, with one system fence per block instead of one per item. Quality is unchanged: KL divergence against llama.cpp 0.00424 (int8) vs 0.00410 (fp32).
- **One code path per weight type** (p): the GEMV no longer switches on the type inside its inner loop. Net speed was flat at 8 requests because the residual phases gained what the projections lost; the projection loss is still open.
- **Measured, not guessed** (r): keeping three weight groups in flight looked like a latency fix but made the residual phases 10 to 14 µs slower per call, so it was reverted.
- **Drift check** (s): q, s and t were rerun back to back to separate a real gain from GPU clock drift (q 603.6, s 622.5, t 623.2 tok/s).

### Per phase, Q4_K_M, 8 requests, ms per cycle

| phase | m | o | q | t |
|---|---|---|---|---|
| FFN down (sum across cards) | 14.54 | 12.46 | 10.66 | 10.09 |
| FFN gate/up | 8.92 | 8.14 | 8.37 | 8.18 |
| DeltaNet projections | 8.04 | 7.51 | 8.93 | 8.68 |
| DeltaNet output (sum across cards) | 7.81 | 5.69 | 4.52 | 4.41 |
| DeltaNet core | 3.81 | 3.83 | 3.91 | 3.92 |
| attention projections | 2.15 | 1.93 | 2.53 | 2.49 |
| attention output (sum across cards) | 2.43 | 1.82 | 1.39 | 1.34 |
| everything else | 4.80 | 4.61 | 4.59 | 4.54 |
| **cycle** | 52.49 | 45.99 | 44.89 | 43.64 |

### Next

The step record shows where the remaining time goes: the system fence after sending the partials costs 24 to 28 µs per residual call (about 3.4 ms per cycle), while waiting for the other card costs only 1 to 2 µs. Next up: a fence-free send in the style of NCCL's LL protocol (data and flag in one 8-byte store), then the projection loss from build p.

### Also in this release

- `mk_server.py --slots N`: scheduler thread, request queue, slot assignment by longest cached prefix, prefill between cycles, per-slot streams and `/status`.
- CLI `--slots N` with a per-slot comparison against a solo run; `bench/batch_bench.py` (server or llama-server `--parallel`); `bench/kq_mma_bench.cu`, `bench/q6_dp4a_bench.cu`.
- `MK_OPT`, `MK_BATCH_ROWS`, `MK_PROFILE=batch` (see the README's [Options](README.md#options)).
