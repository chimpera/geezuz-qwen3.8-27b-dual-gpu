// dp4a row loop of the single-sequence GEMVs (rows_q6k + accR in qwen38_mk.cu) on Q6_K, 2 x 9216 rows, K = 5120,
// R = 2 rows per warp, T activation rows.  q6 = the current repacked block (per 32-value unit: 16 B low nibbles,
// 8 B high bits, 8 B group meta); q6r = the tensor-core friendly block of kq_mma_bench.cu (lane-contiguous 16 B
// chunks), read here one chunk per unit with signed values instead of the chunk sums.
//   nvcc -O3 -arch=sm_86 -std=c++17 q6_dp4a_bench.cu -o q6_dp4a_bench && ./q6_dp4a_bench
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

constexpr int NROW = 2 * 9216, K = 5120, U = K / 32, IT = U / 32, NT = 512, NCOPY = 4, RB = K / 256 * 224;
constexpr int M4 = 0x0F0F0F0F;

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
__host__ __device__ __forceinline__ size_t wo(int r) { return (size_t) (r >> 4) * 16 * RB + (size_t) (r & 15) * 224; }

struct SX { const int4 * xa; const int4 * xb; const uint2 * xm; };

template <int T, int R>
__device__ __forceinline__ void accR(const SX & s, int u, const int (&A)[R][4], const int (&B)[R][4], const float (&fA)[R], const float (&gA)[R],
                                     const float (&fB)[R], const float (&gB)[R], float (&acc)[R][T]) {
#pragma unroll
    for (int t = 0; t < T; t++) {
        const int ix = t * U + u;
        const int4 va = s.xa[ix], vb = s.xb[ix];
        const uint2 mm = s.xm[ix];
        const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(&mm.x));
        const int qq = (int) mm.y;
        const float sqA = dx.x * (float) (short) (qq & 0xffff), sqB = dx.y * (float) (short) (qq >> 16);
#pragma unroll
        for (int k = 0; k < R; k++) {
            int sa = __dp4a(A[k][0], va.x, 0);
            sa = __dp4a(A[k][1], va.y, sa);
            sa = __dp4a(A[k][2], va.z, sa);
            sa = __dp4a(A[k][3], va.w, sa);
            int sb = __dp4a(B[k][0], vb.x, 0);
            sb = __dp4a(B[k][1], vb.y, sb);
            sb = __dp4a(B[k][2], vb.z, sb);
            sb = __dp4a(B[k][3], vb.w, sb);
            acc[k][t] += fA[k] * (dx.x * (float) sa) + fB[k] * (dx.y * (float) sb) - gA[k] * sqA - gB[k] * sqB;
        }
    }
}

// verbatim rows_q6k (current layout)
template <int T, int R, int BS = 16 * 224>
__device__ __forceinline__ void rows_q6(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int4 q[R];
        int2 hb[R], meta[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * BS;
            q[k] = ldw4(blk + 32 * p + 16 * hh);
            hb[k] = ldw2(blk + 128 + 8 * (u & 7));
            meta[k] = ldw2(blk + 192 + 8 * p);
        }
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const float d = __half2float(__ushort_as_half((unsigned short) (meta[k].y & 0xffff)));
            fA[k] = d * (float) (int8_t) (meta[k].x >> (8 * hh));
            fB[k] = d * (float) (int8_t) (meta[k].x >> (16 + 8 * hh));
            gA[k] = 32.0f * fA[k];
            gB[k] = 32.0f * fB[k];
            const unsigned hx = (unsigned) hb[k].x, hy = (unsigned) hb[k].y;
            A[k][0] = (q[k].x & M4) | (int) (((hx >> 0) & 0x03030303u) << 4);
            A[k][1] = (q[k].y & M4) | (int) (((hx >> 2) & 0x03030303u) << 4);
            A[k][2] = (q[k].z & M4) | (int) (((hx >> 4) & 0x03030303u) << 4);
            A[k][3] = (q[k].w & M4) | (int) (((hx >> 6) & 0x03030303u) << 4);
            B[k][0] = ((q[k].x >> 4) & M4) | (int) (((hy >> 0) & 0x03030303u) << 4);
            B[k][1] = ((q[k].y >> 4) & M4) | (int) (((hy >> 2) & 0x03030303u) << 4);
            B[k][2] = ((q[k].z >> 4) & M4) | (int) (((hy >> 4) & 0x03030303u) << 4);
            B[k][3] = ((q[k].w >> 4) & M4) | (int) (((hy >> 6) & 0x03030303u) << 4);
        }
        accR<T, R>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// q6r: unit u = (block b, chunk k, lane slot tg) holds 4 values of each of the 8 16-value sub-blocks (gg, c, bb) of
// groups 2k, 2k+1; its activations are word tg of units 8b + 4k + 2gg + c (xa for bb 0, xb for bb 1).
template <int T, int R>
__device__ __forceinline__ void rows_q6r(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, kk = (u >> 2) & 1, tg = u & 3;
        int V[R][2][2][2]; // [row][gg][c][bb]
        int scw[R][2];
        float d[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * (16 * 224);
            const int4 ql = ldw4(blk + 64 * kk + 16 * tg);
            const int2 hq = ldw2(blk + 128 + 16 * tg + 8 * kk);
            const int2 sc = ldw2(blk + 192 + 8 * kk);
            d[k] = __half2float(__ushort_as_half((unsigned short) (ldw1(blk + 208) & 0xffff)));
            scw[k][0] = sc.x;
            scw[k][1] = sc.y;
            const int lo[2][2] = {{ql.x, ql.y}, {ql.z, ql.w}};
            const unsigned hi[2] = {(unsigned) hq.x, (unsigned) hq.y};
#pragma unroll
            for (int gg = 0; gg < 2; gg++)
#pragma unroll
                for (int c = 0; c < 2; c++)
#pragma unroll
                    for (int bb = 0; bb < 2; bb++) {
                        const int nib = bb ? (lo[gg][c] >> 4) & M4 : lo[gg][c] & M4;
                        V[k][gg][c][bb] = __vsub4(nib | (int) (((hi[gg] >> (2 * (2 * bb + c))) & 0x03030303u) << 4), 0x20202020);
                    }
        }
#pragma unroll
        for (int t = 0; t < T; t++) {
            float part[R] = {};
#pragma unroll
            for (int gg = 0; gg < 2; gg++)
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    const int ix = t * U + 8 * b + 4 * kk + 2 * gg + c;
                    const int xa = reinterpret_cast<const int *>(s.xa + ix)[tg], xb = reinterpret_cast<const int *>(s.xb + ix)[tg];
                    const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(&s.xm[ix].x));
#pragma unroll
                    for (int k = 0; k < R; k++) {
                        const int s0 = (int) (int8_t) (scw[k][gg] >> (8 * c)), s1 = (int) (int8_t) (scw[k][gg] >> (8 * (2 + c)));
                        part[k] += (float) (s0 * __dp4a(V[k][gg][c][0], xa, 0)) * dx.x + (float) (s1 * __dp4a(V[k][gg][c][1], xb, 0)) * dx.y;
                    }
                }
#pragma unroll
            for (int k = 0; k < R; k++) acc[k][t] += d[k] * part[k];
        }
    }
}

template <int T, int V>
__global__ void __launch_bounds__(NT, 1) phase(const uint8_t * w, const int4 * gxa, const int4 * gxb, const uint2 * gxm, float * out) {
    extern __shared__ __align__(16) uint8_t sm[];
    int4 * xa = reinterpret_cast<int4 *>(sm);
    int4 * xb = xa + T * U;
    uint2 * xm = reinterpret_cast<uint2 *>(xb + T * U);
    for (int i = threadIdx.x; i < T * U; i += NT) { xa[i] = gxa[i]; xb[i] = gxb[i]; xm[i] = gxm[i]; }
    __syncthreads();
    const SX s{xa, xb, xm};
    const int gw = (threadIdx.x >> 5) * gridDim.x + blockIdx.x, W = (NT / 32) * gridDim.x, lane = threadIdx.x & 31;
    for (int it = gw; it < NROW / 2; it += W) {
        float acc[2][T] = {};
        const uint8_t * rp[2] = {w + (V == 2 ? (size_t) (2 * it) * RB : wo(2 * it)), w + (V == 2 ? (size_t) (2 * it + 1) * RB : wo(2 * it + 1))};
        if constexpr (V == 1) rows_q6r<T, 2>(rp, s, acc);
        else if constexpr (V == 2) rows_q6<T, 2, 224>(rp, s, acc);
        else rows_q6<T, 2>(rp, s, acc);
#pragma unroll
        for (int k = 0; k < 2; k++)
#pragma unroll
            for (int t = 0; t < T; t++) {
                float v = acc[k][t];
                for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if (lane == 0) out[(size_t) t * NROW + 2 * it + k] = v;
            }
    }
}

template <int T>
void bench(const std::vector<uint8_t * > & dold, const std::vector<uint8_t *> & dnew, const std::vector<uint8_t *> & drm, const int4 * xa, const int4 * xb, const uint2 * xm, float * dout,
           int nsm) {
    const int smb = T * U * 40;
    CK(cudaFuncSetAttribute(phase<T, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, smb));
    CK(cudaFuncSetAttribute(phase<T, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smb));
    CK(cudaFuncSetAttribute(phase<T, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smb));
    std::vector<float> ref((size_t) T * NROW), got(ref.size());
    for (int v = 0; v < 3; v++) {
        auto run = [&](int c) {
            if (v == 1) phase<T, 1><<<nsm, NT, smb>>>(dnew[c], xa, xb, xm, dout);
            else if (v == 2) phase<T, 2><<<nsm, NT, smb>>>(drm[c], xa, xb, xm, dout);
            else phase<T, 0><<<nsm, NT, smb>>>(dold[c], xa, xb, xm, dout);
        };
        for (int i = 0; i < 4; i++) run(i % NCOPY);
        CK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        const int it = 40;
        cudaEventRecord(e0);
        for (int i = 0; i < it; i++) run(i % NCOPY);
        cudaEventRecord(e1);
        CK(cudaEventSynchronize(e1));
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        const double us = 1000.0 * ms / it;
        CK(cudaMemcpy(v ? got.data() : ref.data(), dout, ref.size() * 4, cudaMemcpyDeviceToHost));
        printf("T=%-2d %s  %7.1f us  %5.0f GB/s", T, v == 1 ? "q6r     " : v == 2 ? "q6 rowmj" : "q6 tiled", us, (double) NROW * RB / us / 1e3);
        if (v) {
            double md = 0, mx = 0;
            for (size_t i = 0; i < ref.size(); i++) { md = fmax(md, fabs(ref[i] - got[i])); mx = fmax(mx, fabs(ref[i])); }
            printf("   max diff %.3g (max |ref| %.3g)", md, mx);
        }
        printf("\n");
    }
}

int main() {
    std::mt19937 rng(1);
    const size_t nb = (size_t) NROW * RB;
    std::vector<uint8_t> h6(nb), h6r(nb);
    for (auto & v : h6) v = (uint8_t) rng();
    for (size_t o = 0; o < nb; o += 224) {
        const __half d = __float2half(0.001f);
        for (int p = 0; p < 4; p++) memcpy(h6.data() + o + 192 + 8 * p + 4, &d, 2);
    }
    // same repack as kq_mma_bench.cu
    for (size_t o = 0; o < nb; o += 224) {
        const uint8_t * s6 = h6.data() + o;
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
    // activations: T_MAX rows of U units, chunk sums consistent with the values
    constexpr int TM = 10;
    std::vector<int8_t> ha((size_t) TM * U * 16), hb(ha.size());
    std::vector<uint2> hm((size_t) TM * U);
    for (auto & v : ha) v = (int8_t) (rng() % 255 - 127);
    for (auto & v : hb) v = (int8_t) (rng() % 255 - 127);
    for (size_t i = 0; i < hm.size(); i++) {
        int sa = 0, sb = 0;
        for (int j = 0; j < 16; j++) { sa += ha[16 * i + j]; sb += hb[16 * i + j]; }
        const __half2 dx = __floats2half2_rn(0.01f * (1 + i % 7), 0.01f * (1 + i % 5));
        memcpy(&hm[i].x, &dx, 4);
        hm[i].y = (uint32_t) (uint16_t) (short) sa | ((uint32_t) (uint16_t) (short) sb << 16);
    }
    // row-major copy of the current layout (rows not tiled)
    std::vector<uint8_t> hrm(nb);
    for (int r = 0; r < NROW; r++)
        for (int b = 0; b < K / 256; b++) memcpy(hrm.data() + (size_t) r * RB + b * 224, h6.data() + wo(r) + (size_t) b * 16 * 224, 224);
    std::vector<uint8_t *> dold(NCOPY), dnew(NCOPY), drm(NCOPY);
    for (int c = 0; c < NCOPY; c++) {
        CK(cudaMalloc(&dold[c], nb)); CK(cudaMalloc(&dnew[c], nb));
        CK(cudaMemcpy(dold[c], h6.data(), nb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dnew[c], h6r.data(), nb, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&drm[c], nb));
        CK(cudaMemcpy(drm[c], hrm.data(), nb, cudaMemcpyHostToDevice));
    }
    int4 *xa, *xb;
    uint2 * xm;
    float * dout;
    CK(cudaMalloc(&xa, ha.size())); CK(cudaMalloc(&xb, hb.size())); CK(cudaMalloc(&xm, hm.size() * 8));
    CK(cudaMemcpy(xa, ha.data(), ha.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(xb, hb.data(), hb.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(xm, hm.data(), hm.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dout, (size_t) TM * NROW * 4));
    int nsm;
    CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    bench<1>(dold, dnew, drm, xa, xb, xm, dout, nsm);
    bench<5>(dold, dnew, drm, xa, xb, xm, dout, nsm);
    bench<10>(dold, dnew, drm, xa, xb, xm, dout, nsm);
    return 0;
}
