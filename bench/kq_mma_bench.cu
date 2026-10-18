// Tensor-core tile loop of the batched decode GEMVs (mma.cuh) on an ffn_gu-shaped phase: 2 x 9216 rows, K = 5120,
// TR = 16 activation rows resident in shared memory, one 16-row tile per warp.  Compares weight layouts:
//   q4n  Q4_K, GGUF row-major blocks          q4t  Q4_K, 16-row tiles (block b of a tile's rows contiguous)
//   q80  Q8_0, the decoder's 16-row tiles (reference)    q6n / q6t  repacked Q6_K, row-major / tiled
//   q4r / q6r  tiled, each lane's bytes of a block contiguous (fewer, wider loads per block)
//   q4i  q4t with int32 accumulation per block (timing only)
//   q6s  q6t's layout, one block per step: wide nibble loads + quad transpose, high bits / meta once per block
//   nvcc -O3 -arch=sm_86 -std=c++17 kq_mma_bench.cu -o kq_mma_bench && ./kq_mma_bench
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

#ifndef BK_K
#define BK_K 5120
#endif
#ifndef BK_TR
#define BK_TR 16
#endif
constexpr int NROW = 2 * 9216, K = BK_K, TR = BK_TR, NC = TR / 8, NT = 512, NCOPY = 4;
constexpr int M4 = 0x0F0F0F0F;
constexpr int SMB = TR * K + K / 32 * TR * 4;

__device__ __forceinline__ unsigned long long l2_evict_first() {
    unsigned long long pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    return pol;
}
__device__ __forceinline__ int4 ldw4(const void * p) {
    int4 r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.s32 {%0,%1,%2,%3}, [%4], %5;"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(l2_evict_first()));
    return r;
}
__device__ __forceinline__ int2 ldw2(const void * p) {
    int2 r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v2.s32 {%0,%1}, [%2], %3;" : "=r"(r.x), "=r"(r.y) : "l"(p), "l"(l2_evict_first()));
    return r;
}
__device__ __forceinline__ int ldw1(const void * p) {
    int r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.s32 %0, [%1], %2;" : "=r"(r) : "l"(p), "l"(l2_evict_first()));
    return r;
}
__device__ __forceinline__ void mma_s8(int (&c)[4], int a0, int a1, int a2, int a3, int b0, int b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}
__device__ __forceinline__ int xc_swz(int t) { return 16 * (((t & 3) << 1) | ((t >> 2) & 1)); }

__device__ __forceinline__ void q4k_scales(const int4 & h, int p, float & fA, float & gA, float & fB, float & gB) {
    const float2 dm = __half22float2(*reinterpret_cast<const __half2 *>(&h.x));
    const int sh = (p & 1) * 16;
    const unsigned y = (unsigned) h.y >> sh, z = (unsigned) h.z >> sh, w = (unsigned) h.w >> sh;
    unsigned sc0, sc1, m0, m1;
    if (p < 2) {
        sc0 = y & 63; sc1 = (y >> 8) & 63; m0 = z & 63; m1 = (z >> 8) & 63;
    } else {
        sc0 = (w & 0xF) | ((y >> 2) & 0x30); sc1 = ((w >> 8) & 0xF) | ((y >> 10) & 0x30);
        m0 = ((w >> 4) & 0xF) | ((z >> 2) & 0x30); m1 = ((w >> 12) & 0xF) | ((z >> 10) & 0x30);
    }
    fA = dm.x * (float) sc0; fB = dm.x * (float) sc1; gA = dm.y * (float) m0; gB = dm.y * (float) m1;
}

// block b of row r: q4n row-major, q4t 16-row tiles
template <bool TILED>
__device__ __forceinline__ const uint8_t * q4_blk(const uint8_t * w, int r, int b) {
    constexpr int RB = K / 256 * 144;
    return TILED ? w + (size_t) (r >> 4) * 16 * RB + (size_t) b * 16 * 144 + (r & 15) * 144 : w + (size_t) r * RB + b * 144;
}

template <bool TILED, int PD, bool HD1 = false>
__device__ __forceinline__ void tile_q4(const uint8_t * w, int ra, int rb, const int8_t * xs, const __half * xdd, const short * xqq,
                                        float (&acc)[NC][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, sw = xc_swz(g);
    const int8_t * xb = xs + g * K;
    const __half * xd = xdd + 2 * tig;
    const short * xq = xqq + 2 * tig;
    const int rr[2] = {ra, rb};
    for (int q = 0; q < K / 64; q += PD) {
        int4 hd[PD][2];
        int2 wv[PD][2];
#pragma unroll
        for (int p = 0; p < PD; p++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const uint8_t * blk = q4_blk<TILED>(w, rr[h], (q + p) >> 2);
                if (!HD1 || p == 0) hd[p][h] = HD1 ? ldw4(blk) : __ldg(reinterpret_cast<const int4 *>(blk));
                else hd[p][h] = hd[0][h];
                wv[p][h] = ldw2(blk + 16 + 32 * ((q + p) & 3) + 8 * tig);
            }
#pragma unroll
        for (int p = 0; p < PD; p++) {
            const int gp = (q + p) & 3;
            int2 V[2][2];
            float f[2][2], gm[2][2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                q4k_scales(hd[p][h], gp, f[0][h], gm[0][h], f[1][h], gm[1][h]);
                V[0][h] = make_int2(wv[p][h].x & M4, wv[p][h].y & M4);
                V[1][h] = make_int2((wv[p][h].x >> 4) & M4, (wv[p][h].y >> 4) & M4);
            }
#pragma unroll
            for (int bb = 0; bb < 2; bb++) {
                const int jl = 2 * (q + p) + bb;
#pragma unroll
                for (int nt = 0; nt < NC; nt++) {
                    const int2 bx = *reinterpret_cast<const int2 *>(xb + nt * 8 * K + ((32 * jl + 8 * tig) ^ sw));
                    const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(xd + jl * TR + 8 * nt));
                    const short2 qv = *reinterpret_cast<const short2 *>(xq + jl * TR + 8 * nt);
                    const float sx0 = dx.x * (float) qv.x, sx1 = dx.y * (float) qv.y;
                    int c[4];
                    mma_s8(c, V[bb][0].x, V[bb][1].x, V[bb][0].y, V[bb][1].y, bx.x, bx.y);
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int h = e >> 1;
                        acc[nt][e] += (float) c[e] * (f[bb][h] * (e & 1 ? dx.y : dx.x)) - gm[bb][h] * (e & 1 ? sx1 : sx0);
                    }
                }
            }
        }
    }
}

// Q8_0 16-row tiles: group q of row r at (r >> 4) * 16 * K + 64 (r & 15) + 1024 q; scales (r >> 4) * 16 * nd + 2 (r & 15) + 32 q
template <int PD>
__device__ __forceinline__ void tile_q8(const uint8_t * w, const __half * wd, int ra, int rb, const int8_t * xs, const __half * xdd,
                                        float (&acc)[NC][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, odd = tig & 1, sw = xc_swz(g);
    const int8_t * xb = xs + g * K;
    const __half * xd = xdd + 2 * tig;
    const int rr[2] = {ra, rb};
    const uint8_t * wp[2];
    const __half * dp[2];
    for (int h = 0; h < 2; h++) {
        wp[h] = w + (size_t) (rr[h] >> 4) * 16 * K + 64 * (rr[h] & 15);
        dp[h] = wd + (size_t) (rr[h] >> 4) * 16 * (K / 32) + 2 * (rr[h] & 15);
    }
    for (int q = 0; q < K / 64; q += PD) {
        int4 wv[PD][2];
        int s[PD][2];
#pragma unroll
        for (int p = 0; p < PD; p++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                wv[p][h] = ldw4(wp[h] + 1024 * (q + p) + 16 * tig);
                s[p][h] = ldw1(dp[h] + 32 * (q + p));
            }
#pragma unroll
        for (int p = 0; p < PD; p++) {
            int2 E[2], O[2];
            float2 sc[2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int4 v = wv[p][h];
                const int r0 = __shfl_xor_sync(0xffffffffu, odd ? v.x : v.z, 1), r1 = __shfl_xor_sync(0xffffffffu, odd ? v.y : v.w, 1);
                E[h] = odd ? make_int2(r0, r1) : make_int2(v.x, v.y);
                O[h] = odd ? make_int2(v.z, v.w) : make_int2(r0, r1);
                sc[h] = __half22float2(*reinterpret_cast<const __half2 *>(&s[p][h]));
            }
#pragma unroll
            for (int bb = 0; bb < 2; bb++) {
                const int jl = 2 * (q + p) + bb;
#pragma unroll
                for (int nt = 0; nt < NC; nt++) {
                    const int2 bx = *reinterpret_cast<const int2 *>(xb + nt * 8 * K + ((32 * jl + 8 * tig) ^ sw));
                    const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(xd + jl * TR + 8 * nt));
                    const int2 a0 = bb ? O[0] : E[0], a8 = bb ? O[1] : E[1];
                    int c[4];
                    mma_s8(c, a0.x, a8.x, a0.y, a8.y, bx.x, bx.y);
                    const float w0 = bb ? sc[0].y : sc[0].x, w8 = bb ? sc[1].y : sc[1].x;
                    acc[nt][0] += (float) c[0] * (w0 * dx.x);
                    acc[nt][1] += (float) c[1] * (w0 * dx.y);
                    acc[nt][2] += (float) c[2] * (w8 * dx.x);
                    acc[nt][3] += (float) c[3] * (w8 * dx.y);
                }
            }
        }
    }
}


__device__ __forceinline__ void mma_s8_16(int (&c)[4], int a0, int a1, int b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3]) : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

// repacked Q6_K (224 B blocks), same tiling choice as tile_q4
template <bool TILED, int PD>
__device__ __forceinline__ void tile_q6(const uint8_t * w, int ra, int rb, const int8_t * xs, const __half * xdd, float (&acc)[NC][4]) {
    constexpr int RB = K / 256 * 224;
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, sw = xc_swz(g);
    const int8_t * xb = xs + g * K;
    const __half * xd = xdd + 2 * tig;
    const int rr[2] = {ra, rb};
    for (int q = 0; q < K / 64; q += PD) {
        int2 wv[PD][2], mt[PD][2];
        int4 h6[PD][2];
#pragma unroll
        for (int p = 0; p < PD; p++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int b = (q + p) >> 2, gp = (q + p) & 3, r = rr[h];
                const uint8_t * blk = TILED ? w + (size_t) (r >> 4) * 16 * RB + (size_t) b * 16 * 224 + (r & 15) * 224 : w + (size_t) r * RB + b * 224;
                wv[p][h] = make_int2(ldw1(blk + 32 * gp + 4 * tig), ldw1(blk + 32 * gp + 16 + 4 * tig));
                h6[p][h] = ldw4(blk + 128 + 16 * gp);
                mt[p][h] = ldw2(blk + 192 + 8 * gp);
            }
#pragma unroll
        for (int p = 0; p < PD; p++) {
            int V[2][2][2], sc[2][2][2];
            float dw[2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                dw[h] = __half2float(__ushort_as_half((unsigned short) (mt[p][h].y & 0xffff)));
                const unsigned hb[2][2] = {{(unsigned) h6[p][h].x, (unsigned) h6[p][h].y}, {(unsigned) h6[p][h].z, (unsigned) h6[p][h].w}};
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    const int v = c ? wv[p][h].y : wv[p][h].x;
                    V[0][c][h] = __vsub4((v & M4) | (int) (((hb[c][0] >> (2 * tig)) & 0x03030303u) << 4), 0x20202020);
                    V[1][c][h] = __vsub4(((v >> 4) & M4) | (int) (((hb[c][1] >> (2 * tig)) & 0x03030303u) << 4), 0x20202020);
#pragma unroll
                    for (int bb = 0; bb < 2; bb++) sc[bb][c][h] = (int) (int8_t) (mt[p][h].x >> (8 * (2 * bb + c)));
                }
            }
#pragma unroll
            for (int bb = 0; bb < 2; bb++) {
                const int jl = 2 * (q + p) + bb;
#pragma unroll
                for (int nt = 0; nt < NC; nt++) {
                    const int8_t * xr = xb + nt * 8 * K;
                    const int b0 = *reinterpret_cast<const int *>(xr + ((32 * jl + 4 * tig) ^ sw));
                    const int b1 = *reinterpret_cast<const int *>(xr + ((32 * jl + 16 + 4 * tig) ^ sw));
                    const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(xd + jl * TR + 8 * nt));
                    int c0[4], c1[4];
                    mma_s8_16(c0, V[bb][0][0], V[bb][0][1], b0);
                    mma_s8_16(c1, V[bb][1][0], V[bb][1][1], b1);
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int h = e >> 1;
                        acc[nt][e] += (float) (c0[e] * sc[bb][0][h] + c1[e] * sc[bb][1][h]) * (dw[h] * (e & 1 ? dx.y : dx.x));
                    }
                }
            }
        }
    }
}

// "r" layouts: one 256-value block of a row per loop step, each lane's bytes for the whole block contiguous in 16 B
// chunks (chunk k of lane tig at 64 k + 16 tig, so one load instruction covers 64 contiguous bytes of each row) and
// the block header read once as a broadcast.  Q4_K r: [0,64) groups 0-1 qs, [64,128) groups 2-3, [128,144) header.
__device__ __forceinline__ const uint8_t * kt_blk(const uint8_t * w, int r, int b, int bb) {
    return w + (size_t) (r >> 4) * 16 * (K / 256) * bb + (size_t) b * 16 * bb + (r & 15) * bb;
}

__device__ __forceinline__ void tile_q4r(const uint8_t * w, int ra, int rb, const int8_t * xs, const __half * xdd, const short * xqq,
                                         float (&acc)[NC][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, sw = xc_swz(g);
    const int8_t * xb = xs + g * K;
    const __half * xd = xdd + 2 * tig;
    const short * xq = xqq + 2 * tig;
    const int rr[2] = {ra, rb};
    for (int b = 0; b < K / 256; b++) {
        int4 hd[2], wq[2][2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint8_t * blk = kt_blk(w, rr[h], b, 144);
            wq[h][0] = ldw4(blk + 16 * tig);
            wq[h][1] = ldw4(blk + 64 + 16 * tig);
            hd[h] = ldw4(blk + 128);
        }
#pragma unroll
        for (int p = 0; p < 4; p++) {
            int2 V[2][2];
            float f[2][2], gm[2][2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                q4k_scales(hd[h], p, f[0][h], gm[0][h], f[1][h], gm[1][h]);
                const int4 c = wq[h][p >> 1];
                const int2 wv = (p & 1) ? make_int2(c.z, c.w) : make_int2(c.x, c.y);
                V[0][h] = make_int2(wv.x & M4, wv.y & M4);
                V[1][h] = make_int2((wv.x >> 4) & M4, (wv.y >> 4) & M4);
            }
#pragma unroll
            for (int bb = 0; bb < 2; bb++) {
                const int jl = 8 * b + 2 * p + bb;
#pragma unroll
                for (int nt = 0; nt < NC; nt++) {
                    const int2 bx = *reinterpret_cast<const int2 *>(xb + nt * 8 * K + ((32 * jl + 8 * tig) ^ sw));
                    const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(xd + jl * TR + 8 * nt));
                    const short2 qv = *reinterpret_cast<const short2 *>(xq + jl * TR + 8 * nt);
                    const float sx0 = dx.x * (float) qv.x, sx1 = dx.y * (float) qv.y;
                    int c[4];
                    mma_s8(c, V[bb][0].x, V[bb][1].x, V[bb][0].y, V[bb][1].y, bx.x, bx.y);
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int h = e >> 1;
                        acc[nt][e] += (float) c[e] * (f[bb][h] * (e & 1 ? dx.y : dx.x)) - gm[bb][h] * (e & 1 ? sx1 : sx0);
                    }
                }
            }
        }
    }
}

// Q6_K r (224 B): [0,64) groups 0-1 low nibbles {g c0, g c1} per group, [64,128) groups 2-3, [128,192) one word of
// high bits per group (byte j: 2 bits each for (bb, c) at shift 2 (2 bb + c)), [192,208) the 16 int8 scales
// (4 gp + 2 bb + c), [208,210) d.
__device__ __forceinline__ void tile_q6r(const uint8_t * w, int ra, int rb, const int8_t * xs, const __half * xdd, float (&acc)[NC][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, sw = xc_swz(g);
    const int8_t * xb = xs + g * K;
    const __half * xd = xdd + 2 * tig;
    const int rr[2] = {ra, rb};
    for (int b = 0; b < K / 256; b++) {
        int4 wl[2][2], wh[2], sc4[2];
        float dw[2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint8_t * blk = kt_blk(w, rr[h], b, 224);
            wl[h][0] = ldw4(blk + 16 * tig);
            wl[h][1] = ldw4(blk + 64 + 16 * tig);
            wh[h] = ldw4(blk + 128 + 16 * tig);
            sc4[h] = ldw4(blk + 192);
            dw[h] = __half2float(__ushort_as_half((unsigned short) (ldw1(blk + 208) & 0xffff)));
        }
#pragma unroll
        for (int p = 0; p < 4; p++) {
            int V[2][2][2], sc[2][2][2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int4 c4 = wl[h][p >> 1];
                const int lo[2] = {(p & 1) ? c4.z : c4.x, (p & 1) ? c4.w : c4.y};
                const unsigned hi = (unsigned) (p == 0 ? wh[h].x : p == 1 ? wh[h].y : p == 2 ? wh[h].z : wh[h].w);
                const unsigned sw4 = (unsigned) (p == 0 ? sc4[h].x : p == 1 ? sc4[h].y : p == 2 ? sc4[h].z : sc4[h].w);
#pragma unroll
                for (int c = 0; c < 2; c++)
#pragma unroll
                    for (int bb = 0; bb < 2; bb++) {
                        const int nib = bb ? (lo[c] >> 4) & M4 : lo[c] & M4;
                        V[bb][c][h] = __vsub4(nib | (int) (((hi >> (2 * (2 * bb + c))) & 0x03030303u) << 4), 0x20202020);
                        sc[bb][c][h] = (int) (int8_t) (sw4 >> (8 * (2 * bb + c)));
                    }
            }
#pragma unroll
            for (int bb = 0; bb < 2; bb++) {
                const int jl = 8 * b + 2 * p + bb;
#pragma unroll
                for (int nt = 0; nt < NC; nt++) {
                    const int8_t * xr = xb + nt * 8 * K;
                    const int b0 = *reinterpret_cast<const int *>(xr + ((32 * jl + 4 * tig) ^ sw));
                    const int b1 = *reinterpret_cast<const int *>(xr + ((32 * jl + 16 + 4 * tig) ^ sw));
                    const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(xd + jl * TR + 8 * nt));
                    int c0[4], c1[4];
                    mma_s8_16(c0, V[bb][0][0], V[bb][0][1], b0);
                    mma_s8_16(c1, V[bb][1][0], V[bb][1][1], b1);
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int h = e >> 1;
                        acc[nt][e] += (float) (c0[e] * sc[bb][0][h] + c1[e] * sc[bb][1][h]) * (dw[h] * (e & 1 ? dx.y : dx.x));
                    }
                }
            }
        }
    }
}

// 4 x 4 word transpose across the lanes of a quad: lane t gets word t of every quad lane's x (y[j] = x of lane j).
__device__ __forceinline__ int4 quad_tr(const int4 x, int t) {
    const bool b1 = t & 2, b0 = t & 1;
    const int k0 = b1 ? x.z : x.x, k1 = b1 ? x.w : x.y;
    const int r0 = __shfl_xor_sync(0xffffffffu, b1 ? x.x : x.z, 2), r1 = __shfl_xor_sync(0xffffffffu, b1 ? x.y : x.w, 2);
    // E[row bit 1][col bit 0]
    const int e00 = b1 ? r0 : k0, e01 = b1 ? r1 : k1, e10 = b1 ? k0 : r0, e11 = b1 ? k1 : r1;
    const int a = __shfl_xor_sync(0xffffffffu, b0 ? e00 : e01, 1), c = __shfl_xor_sync(0xffffffffu, b0 ? e10 : e11, 1);
    const int m0 = b0 ? e01 : e00, m1 = b0 ? e11 : e10;
    return b0 ? make_int4(a, m0, c, m1) : make_int4(m0, a, m1, c);
}

// current Q6_K layout, one block per step: each quad lane loads two 16 B nibble chunks (lane t: chunks t, 4 + t)
// and a quad transpose hands lane t its 4 bytes of every chunk; high bits and meta are read once per block.
__device__ __forceinline__ void tile_q6s(const uint8_t * w, int ra, int rb, const int8_t * xs, const __half * xdd, float (&acc)[NC][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, sw = xc_swz(g);
    const int8_t * xb = xs + g * K;
    const __half * xd = xdd + 2 * tig;
    const int rr[2] = {ra, rb};
    for (int b = 0; b < K / 256; b++) {
        int4 lo[2][2], hb[2][4], mt[2][2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint8_t * blk = kt_blk(w, rr[h], b, 224);
            lo[h][0] = ldw4(blk + 16 * tig);
            lo[h][1] = ldw4(blk + 64 + 16 * tig);
#pragma unroll
            for (int gp = 0; gp < 4; gp++) hb[h][gp] = ldw4(blk + 128 + 16 * gp);
            mt[h][0] = ldw4(blk + 192);
            mt[h][1] = ldw4(blk + 208);
        }
#pragma unroll
        for (int h = 0; h < 2; h++) {
            lo[h][0] = quad_tr(lo[h][0], tig);
            lo[h][1] = quad_tr(lo[h][1], tig);
        }
#pragma unroll
        for (int p = 0; p < 4; p++) {
            int V[2][2][2], sc[2][2][2];
            float dw[2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int4 l4 = lo[h][p >> 1], m4 = mt[h][p >> 1];
                const int lw[2] = {(p & 1) ? l4.z : l4.x, (p & 1) ? l4.w : l4.y};
                const int scw = (p & 1) ? m4.z : m4.x;
                dw[h] = __half2float(__ushort_as_half((unsigned short) (((p & 1) ? m4.w : m4.y) & 0xffff)));
                const unsigned hw[2][2] = {{(unsigned) hb[h][p].x, (unsigned) hb[h][p].y}, {(unsigned) hb[h][p].z, (unsigned) hb[h][p].w}};
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    V[0][c][h] = __vsub4((lw[c] & M4) | (int) (((hw[c][0] >> (2 * tig)) & 0x03030303u) << 4), 0x20202020);
                    V[1][c][h] = __vsub4(((lw[c] >> 4) & M4) | (int) (((hw[c][1] >> (2 * tig)) & 0x03030303u) << 4), 0x20202020);
#pragma unroll
                    for (int bb = 0; bb < 2; bb++) sc[bb][c][h] = (int) (int8_t) (scw >> (8 * (2 * bb + c)));
                }
            }
#pragma unroll
            for (int bb = 0; bb < 2; bb++) {
                const int jl = 8 * b + 2 * p + bb;
#pragma unroll
                for (int nt = 0; nt < NC; nt++) {
                    const int8_t * xr = xb + nt * 8 * K;
                    const int b0 = *reinterpret_cast<const int *>(xr + ((32 * jl + 4 * tig) ^ sw));
                    const int b1 = *reinterpret_cast<const int *>(xr + ((32 * jl + 16 + 4 * tig) ^ sw));
                    const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(xd + jl * TR + 8 * nt));
                    int c0[4], c1[4];
                    mma_s8_16(c0, V[bb][0][0], V[bb][0][1], b0);
                    mma_s8_16(c1, V[bb][1][0], V[bb][1][1], b1);
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int h = e >> 1;
                        acc[nt][e] += (float) (c0[e] * sc[bb][0][h] + c1[e] * sc[bb][1][h]) * (dw[h] * (e & 1 ? dx.y : dx.x));
                    }
                }
            }
        }
    }
}

// tile_q6s, high bits too: lane t loads group t's 16 B, packs each quad lane's 2-bit fields into one word
// (q6r's high-bit word) and a second quad transpose hands them out.  current Q6_K layout, one block per step: each quad lane loads two 16 B nibble chunks (lane t: chunks t, 4 + t)
// and a quad transpose hands lane t its 4 bytes of every chunk; high bits and meta are read once per block.
template <bool QH1>
__device__ __forceinline__ void tile_q6s2(const uint8_t * w, int ra, int rb, const int8_t * xs, const __half * xdd, float (&acc)[NC][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, sw = xc_swz(g);
    const int8_t * xb = xs + g * K;
    const __half * xd = xdd + 2 * tig;
    const int rr[2] = {ra, rb};
    for (int b = 0; b < K / 256; b++) {
        int4 lo[2][2], hb[2][2], mt[2][2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint8_t * blk = kt_blk(w, rr[h], b, 224);
            lo[h][0] = ldw4(blk + 16 * tig);
            lo[h][1] = ldw4(blk + 64 + 16 * tig);
            hb[h][0] = ldw4(blk + 128 + 16 * tig);
            mt[h][0] = ldw4(blk + 192);
            mt[h][1] = ldw4(blk + 208);
        }
#pragma unroll
        for (int h = 0; h < 2; h++) {
            lo[h][0] = quad_tr(lo[h][0], tig);
            lo[h][1] = quad_tr(lo[h][1], tig);
            const unsigned wv[4] = {(unsigned) hb[h][0].x, (unsigned) hb[h][0].y, (unsigned) hb[h][0].z, (unsigned) hb[h][0].w};
            int pk[4]; // pk[t]: lane t's fields of this lane's group; word 2 c + bb at shift 2 (2 bb + c)
#pragma unroll
            for (int t = 0; t < 4; t++) {
                unsigned v = 0;
#pragma unroll
                for (int c = 0; c < 2; c++)
#pragma unroll
                    for (int bb = 0; bb < 2; bb++) v |= ((wv[2 * c + bb] >> (2 * t)) & 0x03030303u) << (2 * (2 * bb + c));
                pk[t] = (int) v;
            }
            hb[h][1] = quad_tr(make_int4(pk[0], pk[1], pk[2], pk[3]), tig); // word gp: group gp
        }
#pragma unroll
        for (int p = 0; p < 4; p++) {
            int V[2][2][2], sc[2][2][2];
            float dw[2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int4 l4 = lo[h][p >> 1], m4 = mt[h][p >> 1];
                const int lw[2] = {(p & 1) ? l4.z : l4.x, (p & 1) ? l4.w : l4.y};
                const int scw = (p & 1) ? m4.z : m4.x;
                dw[h] = __half2float(__ushort_as_half((unsigned short) (((p & 1) ? m4.w : m4.y) & 0xffff)));
                const unsigned hi = (unsigned) (p == 0 ? hb[h][1].x : p == 1 ? hb[h][1].y : p == 2 ? hb[h][1].z : hb[h][1].w);
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    V[0][c][h] = __vsub4((lw[c] & M4) | (int) (((hi >> (2 * c)) & 0x03030303u) << 4), 0x20202020);
                    V[1][c][h] = __vsub4(((lw[c] >> 4) & M4) | (int) (((hi >> (2 * (2 + c))) & 0x03030303u) << 4), 0x20202020);
#pragma unroll
                    for (int bb = 0; bb < 2; bb++) sc[bb][c][h] = (int) (int8_t) (scw >> (8 * (2 * bb + c)));
                }
            }
#pragma unroll
            for (int bb = 0; bb < 2; bb++) {
                const int jl = 8 * b + 2 * p + bb;
#pragma unroll
                for (int nt = 0; nt < NC; nt++) {
                    const int8_t * xr = xb + nt * 8 * K;
                    const int b0 = *reinterpret_cast<const int *>(xr + ((32 * jl + 4 * tig) ^ sw));
                    const int b1 = *reinterpret_cast<const int *>(xr + ((32 * jl + 16 + 4 * tig) ^ sw));
                    const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(xd + jl * TR + 8 * nt));
                    int c0[4], c1[4];
                    mma_s8_16(c0, V[bb][0][0], V[bb][0][1], b0);
                    mma_s8_16(c1, V[bb][1][0], V[bb][1][1], b1);
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int h = e >> 1;
                        acc[nt][e] += (float) (c0[e] * sc[bb][0][h] + c1[e] * sc[bb][1][h]) * (dw[h] * (e & 1 ? dx.y : dx.x));
                    }
                }
            }
        }
    }
}

// q4t with integer accumulation per block: the 6-bit sub-block scales and minimums multiply the MMA sums and the
// activation block sums in int32, one float update per block (activations at one scale per 256 values, as q8_K).
// Timing only (the stand-in activation scale is the first sub-block's).
__device__ __forceinline__ void tile_q4i(const uint8_t * w, int ra, int rb, const int8_t * xs, const __half * xdd, const short * xqq,
                                         float (&acc)[NC][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, sw = xc_swz(g);
    const int8_t * xb = xs + g * K;
    const __half * xd = xdd + 2 * tig;
    const short * xq = xqq + 2 * tig;
    const int rr[2] = {ra, rb};
    for (int b = 0; b < K / 256; b++) {
        int4 hd[2];
        int2 wv[4][2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint8_t * blk = q4_blk<true>(w, rr[h], b);
            hd[h] = ldw4(blk);
#pragma unroll
            for (int p = 0; p < 4; p++) wv[p][h] = ldw2(blk + 16 + 32 * p + 8 * tig);
        }
        int ci[NC][4] = {}, mi[NC][4] = {};
#pragma unroll
        for (int p = 0; p < 4; p++) {
            int sc[2][2], mn[2][2]; // [sub-block][row]
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int sh = (p & 1) * 16;
                const unsigned y = (unsigned) hd[h].y >> sh, z = (unsigned) hd[h].z >> sh, ww = (unsigned) hd[h].w >> sh;
                if (p < 2) {
                    sc[0][h] = y & 63; sc[1][h] = (y >> 8) & 63; mn[0][h] = z & 63; mn[1][h] = (z >> 8) & 63;
                } else {
                    sc[0][h] = (ww & 0xF) | ((y >> 2) & 0x30); sc[1][h] = ((ww >> 8) & 0xF) | ((y >> 10) & 0x30);
                    mn[0][h] = ((ww >> 4) & 0xF) | ((z >> 2) & 0x30); mn[1][h] = ((ww >> 12) & 0xF) | ((z >> 10) & 0x30);
                }
            }
#pragma unroll
            for (int bb = 0; bb < 2; bb++) {
                const int jl = 8 * b + 2 * p + bb;
                const int2 V0 = bb ? make_int2((wv[p][0].x >> 4) & M4, (wv[p][0].y >> 4) & M4) : make_int2(wv[p][0].x & M4, wv[p][0].y & M4);
                const int2 V1 = bb ? make_int2((wv[p][1].x >> 4) & M4, (wv[p][1].y >> 4) & M4) : make_int2(wv[p][1].x & M4, wv[p][1].y & M4);
#pragma unroll
                for (int nt = 0; nt < NC; nt++) {
                    const int2 bx = *reinterpret_cast<const int2 *>(xb + nt * 8 * K + ((32 * jl + 8 * tig) ^ sw));
                    const short2 qv = *reinterpret_cast<const short2 *>(xq + jl * TR + 8 * nt);
                    int c[4];
                    mma_s8(c, V0.x, V1.x, V0.y, V1.y, bx.x, bx.y);
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int h = e >> 1;
                        ci[nt][e] += c[e] * sc[bb][h];
                        mi[nt][e] += mn[bb][h] * (e & 1 ? qv.y : qv.x);
                    }
                }
            }
        }
        float dm[2][2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&hd[h].x));
            dm[h][0] = f.x; dm[h][1] = f.y;
        }
#pragma unroll
        for (int nt = 0; nt < NC; nt++) {
            const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(xd + 8 * b * TR + 8 * nt));
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int h = e >> 1;
                acc[nt][e] += (e & 1 ? dx.y : dx.x) * (dm[h][0] * (float) ci[nt][e] - dm[h][1] * (float) mi[nt][e]);
            }
        }
    }
}

template <int V>
__global__ void __launch_bounds__(NT, 1) phase(const uint8_t * w, const __half * wd, const int8_t * gx, float * out) {
    extern __shared__ __align__(16) uint8_t sm[];
    int8_t * xs = reinterpret_cast<int8_t *>(sm);
    __half * xd = reinterpret_cast<__half *>(sm + TR * K);
    short * xq = reinterpret_cast<short *>(xd + K / 32 * TR);
    for (int i = threadIdx.x; i < TR * K / 16; i += NT) reinterpret_cast<int4 *>(xs)[i] = reinterpret_cast<const int4 *>(gx)[i];
    for (int i = threadIdx.x; i < K / 32 * TR; i += NT) {
        xd[i] = __float2half(0.01f * (1 + (i & 7)));
        xq[i] = (short) ((i * 37) % 200 - 100);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
    const int gw = (threadIdx.x >> 5) * gridDim.x + blockIdx.x, W = (NT / 32) * gridDim.x;
    for (int tile = gw; tile < NROW / 16; tile += W) {
        float acc[NC][4] = {};
        const int ra = 16 * tile + g, rb = ra + 8;
        if constexpr (V == 0) tile_q4<false, 4>(w, ra, rb, xs, xd, xq, acc);
        else if constexpr (V == 1) tile_q4<true, 4>(w, ra, rb, xs, xd, xq, acc);
        else if constexpr (V == 2) tile_q8<4>(w, wd, ra, rb, xs, xd, acc);
        else if constexpr (V == 3) tile_q6<false, 4>(w, ra, rb, xs, xd, acc);
        else if constexpr (V == 4) tile_q6<true, 4>(w, ra, rb, xs, xd, acc);
        else if constexpr (V == 5) tile_q4r(w, ra, rb, xs, xd, xq, acc);
        else if constexpr (V == 6) tile_q6r(w, ra, rb, xs, xd, acc);
        else if constexpr (V == 7) tile_q6s(w, ra, rb, xs, xd, acc);
        else if constexpr (V == 8) tile_q6s2<true>(w, ra, rb, xs, xd, acc);
        else tile_q4i(w, ra, rb, xs, xd, xq, acc);
#pragma unroll
        for (int nt = 0; nt < NC; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) out[(size_t) (8 * nt + 2 * tig + (e & 1)) * NROW + 16 * tile + g + 8 * (e >> 1)] = acc[nt][e];
    }
}

int main() {
    std::mt19937 rng(1);
    const size_t q6b = (size_t) NROW * (K / 256) * 224, q4b = (size_t) NROW * (K / 256) * 144, q8b = (size_t) NROW * K, q8d = (size_t) NROW * (K / 32);
    // random Q4_K blocks with small positive scales; the tiled copy holds the same rows re-laid
    std::vector<uint8_t> h4(q4b), h4t(q4b), h8(q8b);
    std::vector<__half> h8d(q8d);
    for (size_t i = 0; i < q4b; i++) h4[i] = (uint8_t) rng();
    for (int r = 0; r < NROW; r++)
        for (int b = 0; b < K / 256; b++) {
            uint8_t * blk = h4.data() + ((size_t) r * (K / 256) + b) * 144;
            const __half d = __float2half(0.001f), m = __float2half(0.0005f);
            memcpy(blk, &d, 2);
            memcpy(blk + 2, &m, 2);
            memcpy(h4t.data() + (size_t) (r >> 4) * 16 * (K / 256) * 144 + (size_t) b * 16 * 144 + (r & 15) * 144, blk, 144);
        }
    for (size_t i = 0; i < q8b; i++) h8[i] = (uint8_t) rng();
    std::vector<uint8_t> h6(q6b), h6t(q6b);
    for (size_t i = 0; i < q6b; i++) h6[i] = (uint8_t) rng();
    for (int r = 0; r < NROW; r++)
        for (int b = 0; b < K / 256; b++) {
            uint8_t * blk = h6.data() + ((size_t) r * (K / 256) + b) * 224;
            const __half d = __float2half(0.001f);
            for (int p = 0; p < 4; p++) memcpy(blk + 192 + 8 * p + 4, &d, 2);
            memcpy(h6t.data() + (size_t) (r >> 4) * 16 * (K / 256) * 224 + (size_t) b * 16 * 224 + (r & 15) * 224, blk, 224);
        }
    // r layouts from the tiled copies, block by block (same positions, bytes reordered per lane)
    std::vector<uint8_t> h4r(q4b), h6r(q6b);
    for (size_t o = 0; o < q4b; o += 144) {
        const uint8_t * s4 = h4t.data() + o;
        uint8_t * d4 = h4r.data() + o;
        for (int t = 0; t < 4; t++)
            for (int gp = 0; gp < 4; gp++) memcpy(d4 + 64 * (gp >> 1) + 16 * t + 8 * (gp & 1), s4 + 16 + 32 * gp + 8 * t, 8);
        memcpy(d4 + 128, s4, 16);
    }
    for (size_t o = 0; o < q6b; o += 224) {
        const uint8_t * s6 = h6t.data() + o;
        uint8_t * d6 = h6r.data() + o;
        memset(d6, 0, 224);
        for (int t = 0; t < 4; t++)
            for (int gp = 0; gp < 4; gp++) {
                for (int c = 0; c < 2; c++) memcpy(d6 + 64 * (gp >> 1) + 16 * t + 8 * (gp & 1) + 4 * c, s6 + 32 * gp + 16 * c + 4 * t, 4);
                uint32_t hw = 0;
                for (int c = 0; c < 2; c++)
                    for (int bb = 0; bb < 2; bb++) {
                        uint32_t ow;
                        memcpy(&ow, s6 + 128 + 16 * gp + 4 * (2 * c + bb), 4);
                        hw |= ((ow >> (2 * t)) & 0x03030303u) << (2 * (2 * bb + c));
                    }
                memcpy(d6 + 128 + 16 * t + 4 * gp, &hw, 4);
            }
        for (int gp = 0; gp < 4; gp++) memcpy(d6 + 192 + 4 * gp, s6 + 192 + 8 * gp, 4);
        memcpy(d6 + 208, s6 + 192 + 4, 2);
    }
    for (size_t i = 0; i < q8d; i++) h8d[i] = __float2half(0.001f);
    std::vector<int8_t> hx(TR * K);
    for (auto & v : hx) v = (int8_t) (rng() % 255 - 127);
    uint8_t *d4[NCOPY], *d4t[NCOPY], *d8[NCOPY], *d6[NCOPY], *d6t[NCOPY], *d4r[NCOPY], *d6r[NCOPY];
    __half * d8d[NCOPY];
    for (int c = 0; c < NCOPY; c++) {
        CK(cudaMalloc(&d4[c], q4b)); CK(cudaMalloc(&d4t[c], q4b)); CK(cudaMalloc(&d8[c], q8b)); CK(cudaMalloc(&d8d[c], q8d * 2));
        CK(cudaMemcpy(d4[c], h4.data(), q4b, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d4t[c], h4t.data(), q4b, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d8[c], h8.data(), q8b, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d8d[c], h8d.data(), q8d * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d6[c], q6b)); CK(cudaMalloc(&d6t[c], q6b));
        CK(cudaMemcpy(d6[c], h6.data(), q6b, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d6t[c], h6t.data(), q6b, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d4r[c], q4b)); CK(cudaMalloc(&d6r[c], q6b));
        CK(cudaMemcpy(d4r[c], h4r.data(), q4b, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d6r[c], h6r.data(), q6b, cudaMemcpyHostToDevice));
    }
    int8_t * dx;
    float * dout;
    CK(cudaMalloc(&dx, TR * K));
    CK(cudaMemcpy(dx, hx.data(), TR * K, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dout, (size_t) TR * NROW * 4));
    CK(cudaFuncSetAttribute(phase<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB));
    CK(cudaFuncSetAttribute(phase<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB));
    CK(cudaFuncSetAttribute(phase<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB));
    CK(cudaFuncSetAttribute(phase<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB));
    CK(cudaFuncSetAttribute(phase<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB));
    CK(cudaFuncSetAttribute(phase<5>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB));
    CK(cudaFuncSetAttribute(phase<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB));
    CK(cudaFuncSetAttribute(phase<7>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB));
    CK(cudaFuncSetAttribute(phase<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB));
    CK(cudaFuncSetAttribute(phase<9>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMB));
    int nsm;
    CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    std::vector<float> ref4((size_t) TR * NROW), ref6(ref4.size()), got(ref4.size());
    const char * names[10] = {"q4n", "q4t", "q80", "q6n", "q6t", "q4r", "q6r", "q6s", "q6s2", "q4i"};
    for (int v = 0; v < 10; v++) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        auto run = [&](int c) {
            if (v == 0) phase<0><<<nsm, NT, SMB>>>(d4[c], nullptr, dx, dout);
            else if (v == 1) phase<1><<<nsm, NT, SMB>>>(d4t[c], nullptr, dx, dout);
            else if (v == 2) phase<2><<<nsm, NT, SMB>>>(d8[c], d8d[c], dx, dout);
            else if (v == 3) phase<3><<<nsm, NT, SMB>>>(d6[c], nullptr, dx, dout);
            else if (v == 4) phase<4><<<nsm, NT, SMB>>>(d6t[c], nullptr, dx, dout);
            else if (v == 5) phase<5><<<nsm, NT, SMB>>>(d4r[c], nullptr, dx, dout);
            else if (v == 6) phase<6><<<nsm, NT, SMB>>>(d6r[c], nullptr, dx, dout);
            else if (v == 7) phase<7><<<nsm, NT, SMB>>>(d6t[c], nullptr, dx, dout);
            else if (v == 8) phase<8><<<nsm, NT, SMB>>>(d6t[c], nullptr, dx, dout);
            else phase<9><<<nsm, NT, SMB>>>(d4t[c], nullptr, dx, dout);
        };
        for (int i = 0; i < 4; i++) run(i % NCOPY);
        CK(cudaDeviceSynchronize());
        const int it = 40;
        cudaEventRecord(e0);
        for (int i = 0; i < it; i++) run(i % NCOPY);
        cudaEventRecord(e1);
        CK(cudaEventSynchronize(e1));
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        const bool is6 = v == 3 || v == 4 || (v >= 6 && v <= 8);
        const double us = 1000.0 * ms / it, bytes = v == 2 ? (double) (q8b + q8d * 2) : is6 ? (double) q6b : (double) q4b;
        // every re-laid variant must reproduce the row-major results bit for bit
        std::vector<float> & ref = is6 ? ref6 : ref4;
        CK(cudaMemcpy(v == 0 || v == 3 ? ref.data() : got.data(), dout, ref.size() * 4, cudaMemcpyDeviceToHost));
        const bool cmp = v == 1 || (v >= 4 && v <= 8);
        double md = 0;
        if (cmp)
            for (size_t i = 0; i < ref.size(); i++) md = fmax(md, fabs(ref[i] - got[i]));
        printf("%s  %8.1f us  %6.0f GB/s", names[v], us, bytes / us / 1e3);
        if (cmp) printf("   max |vs row-major| %.3g", md);
        printf("\n");
    }
    return 0;
}
