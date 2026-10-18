// Q8_0 weight layout shared by the decode GEMVs and the prefill GEMM.
#pragma once
#include <cstddef>

// Q8_0 rows are stored in 16-row tiles so a warp's tensor-core load of one 64-value group of a tile is
// 1 KB contiguous: group q of row r at q8_wo(r, K) + 1024 q (repack unit order within the 64 bytes), its
// two block scales at q8_do(r, K / 32) + 32 q.  Allocations round N up to whole tiles.
__host__ __device__ __forceinline__ size_t q8_wo(int r, int K) { return (size_t) (r >> 4) * 16 * K + 64 * (r & 15); }
__host__ __device__ __forceinline__ size_t q8_do(int r, int nd) { return (size_t) (r >> 4) * 16 * nd + 2 * (r & 15); }
constexpr int Q8_GW = 1024, Q8_GD = 32; // bytes / scales between consecutive 64-value groups of a row
inline int q8_rows(int n) { return (n + 15) & ~15; }

// K-quant rows (Q4_K / Q5_K / repacked Q6_K: bb bytes per 256-value block) in the same 16-row tiles: block b of
// row r at kq_wo(r, rb, bb) + KQ_T * bb * b, so one block of a tile's 16 rows is 16 bb contiguous bytes.
constexpr int KQ_T = 16;
__host__ __device__ __forceinline__ size_t kq_wo(int r, int rb, int bb) { return (size_t) (r >> 4) * 16 * rb + (size_t) (r & 15) * bb; }
// block bytes of a k-quant ggml type (Q4_K 12, Q5_K 13, Q6_K 14 repacked), 0 otherwise
__host__ __device__ constexpr int kq_bb(int type) { return type == 12 ? 144 : type == 13 ? 176 : type == 14 ? 224 : 0; }
