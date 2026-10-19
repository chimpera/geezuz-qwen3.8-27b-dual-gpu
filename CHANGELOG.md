# Changelog

## v0.2.0: batched decode, up to 8 requests in one cycle

Two RTX 3090s, 8 parallel requests, total tok/s: **Q4_K_M 298 to 625 (2.1x)**, Q8_0 561 to 610. One request: Q4_K_M 190 tok/s, Q8_0 144.

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
| s/t | profiler step record; faster code generation, confirmed by a back-to-back drift check | 186 | 624 | 141 | 595 |
| w | release store for the exchange flags without a full system fence | 190 | 627 | 144 | 599 |
| y | projections: the two warps of a pair split each weight slice | | 621 | | 606 |
| za | w + y | **190** | **625** | **144** | **610** |

Time per cycle at 8 requests, Q4_K_M: 62.5 ms (k) to 42.7 ms (za). Build i was fast per cycle (33 ms) but carried at most 10 rows, so it could not batch more than about 2 requests' drafts.

### What moved the numbers

- **Tensor cores past 10 rows** (k to n): the dp4a GEMVs keep every row in registers and spill above 10 rows. int8 `mma.m16n8k32` straight from the GGUF blocks lifts the cap to 32 rows; tiling the weights and decoding whole Q4_K/Q5_K/Q6_K superblocks at once took Q4_K_M from 433 to 542.
- **Cheaper exchange** (o, p, q): the three phases that sum across the cards (attention output, DeltaNet output, FFN down) went from 24.8 ms (m) to 15.8 ms (t) per cycle on Q4_K_M. Partials now cross PCIe as int8 with a scale per 32 values, only for rows in use, with one system fence per block instead of one per item. Quality is unchanged: KL divergence against llama.cpp 0.00424 (int8) vs 0.00410 (fp32).
- **One code path per weight type** (p): the GEMV no longer switches on the type inside its inner loop. Net speed was flat at 8 requests because the residual phases gained what the projections lost; the projection loss is still open.
- **Measured, not guessed** (r): keeping three weight groups in flight looked like a latency fix but made the residual phases 10 to 14 µs slower per call, so it was reverted.
- **Lighter fence** (w): the flag that tells the other card a partial has arrived is now a release store (`st.release.sys`) without a sequentially consistent system fence before it. Time from written to sent per residual call: 28 to 24 µs.
- **Pairs split the projections** (y): the projections have fewer 16-row tiles than the grid has warps, so the two warps of a pair now split each slice and add through shared memory. Attention projections 2.49 to 2.34 ms per cycle, DeltaNet projections 8.68 to 8.30.
- **Tried and dropped**: a fence-free send in the style of NCCL's LL protocol (data and flag in one 8-byte store) was slower from 2 requests up; moving the DeltaNet alpha/beta rows into the projection loop cost more in register pressure than it saved.
- **Drift check** (s): q, s and t were rerun back to back to separate a real gain from GPU clock drift (q 603.6, s 622.5, t 623.2 tok/s).

### Per phase, Q4_K_M, 8 requests, ms per cycle

| phase | m | o | q | t | za |
|---|---|---|---|---|---|
| FFN down (sum across cards) | 14.54 | 12.46 | 10.66 | 10.09 | 9.94 |
| FFN gate/up | 8.92 | 8.14 | 8.37 | 8.18 | 8.15 |
| DeltaNet projections | 8.04 | 7.51 | 8.93 | 8.68 | 8.30 |
| DeltaNet output (sum across cards) | 7.81 | 5.69 | 4.52 | 4.41 | 4.24 |
| DeltaNet core | 3.81 | 3.83 | 3.91 | 3.92 | 3.83 |
| attention projections | 2.15 | 1.93 | 2.53 | 2.49 | 2.34 |
| attention output (sum across cards) | 2.43 | 1.82 | 1.39 | 1.34 | 1.28 |
| everything else | 4.80 | 4.61 | 4.59 | 4.54 | 4.46 |
| **cycle** | 52.49 | 45.99 | 44.89 | 43.64 | 42.52 |

### Next

Sending the partials still takes about 24 µs per residual call from written to sent, against a transfer floor of about 14 µs (184 KB per card per call at 13.2 GB/s), while waiting for the other card costs 1 to 2 µs. The projections still give work to fewer than half of the warps. Both are open.

### Also in this release

- Project name Geezuz; binaries `geezuz`, `geezuz_tp2`, `libgeezuz.so`, `libgeezuz_tp2.so`; prebuilt sm_86 tarball (`make release`).
- `mk_server.py --slots N`: scheduler thread, request queue, slot assignment by longest cached prefix, prefill between cycles, per-slot streams and `/status`.
- CLI `--slots N` with a per-slot comparison against a solo run; `bench/batch_bench.py` (server or llama-server `--parallel`); `bench/kq_mma_bench.cu`, `bench/q6_dp4a_bench.cu`.
- `MK_OPT`, `MK_BATCH_ROWS`, `MK_PROFILE=batch` (see the README's [Options](README.md#options)).
