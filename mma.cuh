// Tensor-core GEMV rows (int8 mma.m16n8k32 / m16n8k16) for batched cycles of TR = MR, MRM or MAXR rows, weights
// in Q8_0, Q4_K, Q5_K or Q6_K.  The dp4a GEMVs keep all rows in registers and stop at MAXR_DP rows; here a warp
// computes 16 weight rows x TR activation rows per instruction, with the activations in shared memory in natural
// order.  Included by qwen38_mk.cu inside its namespace.

constexpr int MR = 16, MRM = 24; // tensor-core widths of two and three 8-column MMA tiles; MAXR takes four

// Activation K-slice in shared memory: token t's int8 values at x + t * tsb in natural order with its 16-byte
// chunks swizzled by xc_swz(t) (conflict-free fragment loads without a pad), per natural 32-block j and token t
// the fp16 scale d[j * TR + t] and the int sum of the block's values qs[j * TR + t] (k-quant minimums).
struct XC {
    const int8_t * x;
    const __half * d;
    const short * qs;
    int tsb;
};

__host__ __device__ constexpr int xc_bytes(int tr, int ks) { return tr * ks + ks / 32 * tr * 4; }

// The 8 rows of one fragment load (t & 7) hit distinct 16-byte bank groups for 4- and 8-byte loads.
__device__ __forceinline__ int xc_swz(int t) { return 16 * (((t & 3) << 1) | ((t >> 2) & 1)); }

// K slices of a K-long input at TR rows: the fewest whose slice is a multiple of 512 (resid_mma's 8 K-parts of
// 4 block pairs) and fits below misc
template <int TR, int K>
__host__ __device__ constexpr int mma_ns() {
    int ns = 1;
    while (ns <= K / 512 && (K % ns != 0 || (K / ns) % 512 != 0 || xc_bytes(TR, K / ns) > SM_MISC)) ns++;
    return ns;
}

// s_rs[t] = 1 / rms of row t0 + t (from the per-unit sums of squares beside the quantised row)
template <int TR>
__device__ __forceinline__ void row_rs(const QBuf & q, int t0, float * s_rs) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int t = warp; t < TR; t += NW) {
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < U_D / 32; k++) acc += __ldcg(q.ssq + (t0 + t) * U_D + lane + 32 * k);
        acc = wsum(acc);
        if (lane == 0) s_rs[t] = rsqrtf(acc / (float) D + EPS);
    }
}

// Rows t0 .. t0 + TR - 1, units [u0, u0 + KS / 32) of a quantised activation buffer (per-32 scales:
// q32_store or ffn_gu_mma output) -> XC.  norm: s_rs[t] as in fill_xs.
template <int TR, int KS>
__device__ XC fill_xc(const QBuf & q, int t0, int u0, bool norm, float * s_rs) {
    static_assert(KS % 128 == 0 && xc_bytes(TR, KS) <= SM_MISC, "activation slice fits below misc");
    constexpr int US = KS / 32, TSB = KS;
    const int tid = threadIdx.x;
    int8_t * x = reinterpret_cast<int8_t *>(smem);
    __half * d = reinterpret_cast<__half *>(smem + TR * TSB);
    short * qs = reinterpret_cast<short *>(d + US * TR);
    __syncthreads(); // the previous user of the region may still read it
    fill_begin();
    // unit u holds the 16-chunk u & 1 of natural blocks 2 (u >> 1) (xa) and 2 (u >> 1) + 1 (xb)
    for (int i = tid; i < TR * US * 2; i += NT) {
        const int t = i / (2 * US), r = i - t * 2 * US, u = r >> 1, part = r & 1;
        const size_t gi = (size_t) (t0 + t) * q.U + u0 + u;
        cp_async16(x + t * TSB + ((32 * (2 * (u >> 1) + part) + 16 * (u & 1)) ^ xc_swz(t)), (part ? q.xb : q.xa) + gi * 16);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    // units 2p and 2p + 1 carry the two 16-chunks of blocks 2p (A) and 2p + 1 (B) under one scale per block
    for (int i = tid; i < TR * US / 2; i += NT) {
        const int t = i / (US / 2), p = i - t * (US / 2);
        const uint4 mt = __ldcg(reinterpret_cast<const uint4 *>(q.meta + (size_t) (t0 + t) * q.U + u0 + 2 * p));
        d[2 * p * TR + t] = reinterpret_cast<const __half *>(&mt.x)[0];
        d[(2 * p + 1) * TR + t] = reinterpret_cast<const __half *>(&mt.x)[1];
        qs[2 * p * TR + t] = (short) ((short) (mt.y & 0xffff) + (short) (mt.w & 0xffff));
        qs[(2 * p + 1) * TR + t] = (short) ((short) (mt.y >> 16) + (short) (mt.w >> 16));
    }
    if (norm) row_rs<TR>(q, t0, s_rs);
    else
        for (int t = tid; t < TR; t += NT) s_rs[t] = 1.0f;
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    mark_fill();
    XC c;
    c.x = x;
    c.d = d;
    c.qs = qs;
    c.tsb = TSB;
    return c;
}

__device__ __forceinline__ void mma_s8(int (&c)[4], int a0, int a1, int a2, int a3, int b0, int b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}

// A lane's 16 weight bytes at 16 tig of the row's group q are one 16-chunk of natural block 2q (tig even) or
// 2q + 1 (odd); trading 8 bytes with the partner lane leaves each lane natural values 8 tig .. 8 tig + 7
// of both blocks (e: block 2q, o: 2q + 1), the k order the fragments of both operands use.
__device__ __forceinline__ void w_split(const int4 w, int odd, int2 & e, int2 & o) {
    const int r0 = __shfl_xor_sync(0xffffffffu, odd ? w.x : w.z, 1);
    const int r1 = __shfl_xor_sync(0xffffffffu, odd ? w.y : w.w, 1);
    e = odd ? make_int2(r0, r1) : make_int2(w.x, w.y);
    o = odd ? make_int2(w.z, w.w) : make_int2(r0, r1);
}

// NTL 16-row tiles of repacked Q8_0 rows over block pairs [q0, q1) of the slice in xc (TR activation rows).
// wp[k][h] / dp[k][h]: weight bytes / scales of row g + 8h of tile k at the slice start.  acc[k][nt][i]: row
// g + 8 (i >> 1), activation row 8 nt + 2 tig + (i & 1).  PD block pairs are loaded before any is used;
// q1 - q0 must be a multiple of PD (no tail: guarded loads blow up the compile).
template <int NTL, int PD, int TR>
__device__ __forceinline__ void mma_rows(const uint8_t * const (&wp)[NTL][2], const __half * const (&dp)[NTL][2], const XC & xc,
                                         int q0, int q1, float (&acc)[NTL][TR / 8][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, odd = tig & 1, swz = xc_swz(g);
    const int8_t * xb = xc.x + g * xc.tsb;
    const __half * xd = xc.d + 2 * tig;
    for (int q = q0; q < q1; q += PD) {
        int4 w[PD][NTL][2];
        int s[PD][NTL][2];
#pragma unroll
        for (int p = 0; p < PD; p++)
#pragma unroll
            for (int k = 0; k < NTL; k++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    w[p][k][h] = ldw4(wp[k][h] + Q8_GW * (q + p) + 16 * tig);
                    s[p][k][h] = ldw1(dp[k][h] + Q8_GD * (q + p));
                }
#pragma unroll
        for (int p = 0; p < PD; p++) {
            int2 E[NTL][2], O[NTL][2];
            float2 sw[NTL][2];
#pragma unroll
            for (int k = 0; k < NTL; k++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    w_split(w[p][k][h], odd, E[k][h], O[k][h]);
                    sw[k][h] = __half22float2(*reinterpret_cast<const __half2 *>(&s[p][k][h]));
                }
#pragma unroll
            for (int bb = 0; bb < 2; bb++) {
                const int jl = 2 * (q + p) + bb;
#pragma unroll
                for (int nt = 0; nt < TR / 8; nt++) {
                    const int2 bx = *reinterpret_cast<const int2 *>(xb + nt * 8 * xc.tsb + ((32 * jl + 8 * tig) ^ swz));
                    const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(xd + jl * TR + 8 * nt));
#pragma unroll
                    for (int k = 0; k < NTL; k++) {
                        const int2 a0 = bb ? O[k][0] : E[k][0], a8 = bb ? O[k][1] : E[k][1];
                        int c[4];
                        mma_s8(c, a0.x, a8.x, a0.y, a8.y, bx.x, bx.y);
                        const float w0 = bb ? sw[k][0].y : sw[k][0].x, w8 = bb ? sw[k][1].y : sw[k][1].x;
                        acc[k][nt][0] += (float) c[0] * (w0 * dx.x);
                        acc[k][nt][1] += (float) c[1] * (w0 * dx.y);
                        acc[k][nt][2] += (float) c[2] * (w8 * dx.x);
                        acc[k][nt][3] += (float) c[3] * (w8 * dx.y);
                    }
                }
            }
        }
    }
}

template <int NTL, int NC>
__device__ __forceinline__ void zero_mma(float (&acc)[NTL][NC][4]) {
#pragma unroll
    for (int k = 0; k < NTL; k++)
#pragma unroll
        for (int nt = 0; nt < NC; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc[k][nt][i] = 0.f;
}

// row r of m from the slice starting at element k0: weights / scales for mma_rows
__device__ __forceinline__ const uint8_t * mrow_w(const Mat & m, int r, int k0) { return m.w + q8_wo(r, m.rb) + Q8_GW / 64 * k0; }
__device__ __forceinline__ const __half * mrow_d(const Mat & m, int r, int k0) { return m.d + q8_do(r, m.nd) + Q8_GD * k0 / 64; }

__device__ __forceinline__ void mma_s8_16(int (&c)[4], int a0, int a1, int b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

// mma_rows for the k-quants in 16-row tiles (kq_wo; Q6_K repacked, Q6_BLK), group by group (partial blocks; whole
// blocks take mma_rows_q45b / mma_rows_q6b): rows ra (fragment row g) and
// rb (g + 8), 64-value groups [q0, q1) of the slice starting at element k0.  A lane's 8 bytes at 8 tig of a
// group's nibbles are values 8 tig .. 8 tig + 7 of both its blocks, the k order of mma_rows.  Q4_K / Q5_K: one
// scale and minimum per block (the minimum meets the activation block sums); Q6_K: a scale per 16 values, so
// one m16n8k16 per 16-chunk with values 16 c + 4 tig .. + 3, signed by subtracting 32.  Headers re-read by
// each group of a block go through L1; the value bytes stream.
template <int TYPE, int PD, int TR>
__device__ __forceinline__ void mma_rows_kq(const Mat & m, int ra, int rb, int k0, const XC & xc, int q0, int q1,
                                            float (&acc)[TR / 8][4]) {
    constexpr int BB = TYPE == T_Q4K ? 144 : TYPE == T_Q5K ? 176 : Q6_BLK;
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, sw = xc_swz(g);
    const uint8_t * rw[2] = {m.w + kq_wo(ra, m.rb, BB), m.w + kq_wo(rb, m.rb, BB)};
    const int8_t * xb = xc.x + g * xc.tsb;
    const __half * xd = xc.d + 2 * tig;
    const short * xq = xc.qs + 2 * tig;
    const int G0 = k0 / 64;
    for (int q = q0; q < q1; q += PD) {
        int4 hd[PD][2];
        int2 w[PD][2], qh[PD][2];
        int4 h6[PD][2];
#pragma unroll
        for (int p = 0; p < PD; p++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int G = G0 + q + p;
                const uint8_t * blk = rw[h] + (size_t) (G >> 2) * KQ_T * BB;
                const int gp = G & 3;
                if constexpr (TYPE == T_Q6K) {
                    w[p][h] = make_int2(ldw1(blk + 32 * gp + 4 * tig), ldw1(blk + 32 * gp + 16 + 4 * tig));
                    h6[p][h] = ldw4(blk + 128 + 16 * gp);
                    qh[p][h] = ldw2(blk + 192 + 8 * gp);
                } else {
                    hd[p][h] = __ldg(reinterpret_cast<const int4 *>(blk));
                    if constexpr (TYPE == T_Q5K) qh[p][h] = __ldg(reinterpret_cast<const int2 *>(blk + 16 + 8 * tig));
                    w[p][h] = ldw2(blk + (TYPE == T_Q5K ? 48 : 16) + 32 * gp + 8 * tig);
                }
            }
#pragma unroll
        for (int p = 0; p < PD; p++) {
            const int gp = (G0 + q + p) & 3;
            if constexpr (TYPE == T_Q6K) {
                int V[2][2][2], sc[2][2][2]; // [block][16-chunk][row]
                float dw[2];
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    dw[h] = __half2float(__ushort_as_half((unsigned short) (qh[p][h].y & 0xffff)));
                    const unsigned hb[2][2] = {{(unsigned) h6[p][h].x, (unsigned) h6[p][h].y}, {(unsigned) h6[p][h].z, (unsigned) h6[p][h].w}};
#pragma unroll
                    for (int c = 0; c < 2; c++) {
                        const int v = c ? w[p][h].y : w[p][h].x;
                        V[0][c][h] = __vsub4((v & M4) | (int) (((hb[c][0] >> (2 * tig)) & 0x03030303u) << 4), 0x20202020);
                        V[1][c][h] = __vsub4(((v >> 4) & M4) | (int) (((hb[c][1] >> (2 * tig)) & 0x03030303u) << 4), 0x20202020);
#pragma unroll
                        for (int bb = 0; bb < 2; bb++) sc[bb][c][h] = (int) (int8_t) (qh[p][h].x >> (8 * (2 * bb + c)));
                    }
                }
#pragma unroll
                for (int bb = 0; bb < 2; bb++) {
                    const int jl = 2 * (q + p) + bb;
#pragma unroll
                    for (int nt = 0; nt < TR / 8; nt++) {
                        const int8_t * xr = xb + nt * 8 * xc.tsb;
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
            } else {
                int2 V[2][2]; // [block][row]
                float f[2][2], gm[2][2];
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    q4k_scales(hd[p][h], gp, f[0][h], gm[0][h], f[1][h], gm[1][h]);
                    const int2 v = w[p][h];
                    V[0][h] = make_int2(v.x & M4, v.y & M4);
                    V[1][h] = make_int2((v.x >> 4) & M4, (v.y >> 4) & M4);
                    if constexpr (TYPE == T_Q5K) {
                        const int s0 = 2 * gp, s1 = 2 * gp + 1;
                        V[0][h].x |= ((qh[p][h].x >> s0) & 0x01010101) << 4;
                        V[0][h].y |= ((qh[p][h].y >> s0) & 0x01010101) << 4;
                        V[1][h].x |= ((qh[p][h].x >> s1) & 0x01010101) << 4;
                        V[1][h].y |= ((qh[p][h].y >> s1) & 0x01010101) << 4;
                    }
                }
#pragma unroll
                for (int bb = 0; bb < 2; bb++) {
                    const int jl = 2 * (q + p) + bb;
#pragma unroll
                    for (int nt = 0; nt < TR / 8; nt++) {
                        const int2 bx = *reinterpret_cast<const int2 *>(xb + nt * 8 * xc.tsb + ((32 * jl + 8 * tig) ^ sw));
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
}

// 4 x 4 word transpose across a quad: lane t gets word t of every quad lane's x (y[j] = x of lane j).
__device__ __forceinline__ int4 quad_tr(const int4 x, int t) {
    const bool b1 = t & 2, b0 = t & 1;
    const int k0 = b1 ? x.z : x.x, k1 = b1 ? x.w : x.y;
    const int r0 = __shfl_xor_sync(0xffffffffu, b1 ? x.x : x.z, 2), r1 = __shfl_xor_sync(0xffffffffu, b1 ? x.y : x.w, 2);
    const int e00 = b1 ? r0 : k0, e01 = b1 ? r1 : k1, e10 = b1 ? k0 : r0, e11 = b1 ? k1 : r1; // [row bit 1][col bit 0]
    const int a = __shfl_xor_sync(0xffffffffu, b0 ? e00 : e01, 1), c = __shfl_xor_sync(0xffffffffu, b0 ? e10 : e11, 1);
    const int m0 = b0 ? e01 : e00, m1 = b0 ? e11 : e10;
    return b0 ? make_int4(a, m0, c, m1) : make_int4(m0, a, m1, c);
}

// Q6_K tile over whole blocks [q0, q1) (groups, multiples of 4 from a block start): per block and row a quad loads
// the nibbles as two 16 B chunks per lane and group tig's high bits as one, packs every lane's 2-bit fields into one
// word per group, and two quad transposes hand each lane its bytes; meta is read once.  Same values and k order as
// mma_rows_kq, a third of its loads.
template <int TR>
__device__ __forceinline__ void mma_rows_q6b(const Mat & m, int ra, int rb, int k0, const XC & xc, int q0, int q1,
                                             float (&acc)[TR / 8][4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, sw = xc_swz(g);
    const uint8_t * rw[2] = {m.w + kq_wo(ra, m.rb, Q6_BLK), m.w + kq_wo(rb, m.rb, Q6_BLK)};
    const int8_t * xb = xc.x + g * xc.tsb;
    const __half * xd = xc.d + 2 * tig;
    const int G0 = k0 / 64;
    for (int q = q0; q < q1; q += 4) {
        int4 lo[2][2], hb[2], mt[2][2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint8_t * blk = rw[h] + (size_t) ((G0 + q) >> 2) * KQ_T * Q6_BLK;
            lo[h][0] = ldw4(blk + 16 * tig);
            lo[h][1] = ldw4(blk + 64 + 16 * tig);
            hb[h] = ldw4(blk + 128 + 16 * tig);
            mt[h][0] = ldw4(blk + 192);
            mt[h][1] = ldw4(blk + 208);
        }
#pragma unroll
        for (int h = 0; h < 2; h++) {
            lo[h][0] = quad_tr(lo[h][0], tig);
            lo[h][1] = quad_tr(lo[h][1], tig);
            const unsigned wv[4] = {(unsigned) hb[h].x, (unsigned) hb[h].y, (unsigned) hb[h].z, (unsigned) hb[h].w};
            int pk[4]; // pk[t]: lane t's fields of group tig, (block bb, 16-chunk c) at shift 2 (2 bb + c)
#pragma unroll
            for (int t = 0; t < 4; t++) {
                unsigned v = 0;
#pragma unroll
                for (int c = 0; c < 2; c++)
#pragma unroll
                    for (int bb = 0; bb < 2; bb++) v |= ((wv[2 * c + bb] >> (2 * t)) & 0x03030303u) << (2 * (2 * bb + c));
                pk[t] = (int) v;
            }
            hb[h] = quad_tr(make_int4(pk[0], pk[1], pk[2], pk[3]), tig);
        }
#pragma unroll
        for (int p = 0; p < 4; p++) {
            int V[2][2][2], sc[2][2][2]; // [block][16-chunk][row]
            float dw[2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int4 l4 = lo[h][p >> 1], m4 = mt[h][p >> 1];
                const int lw[2] = {(p & 1) ? l4.z : l4.x, (p & 1) ? l4.w : l4.y};
                const int scw = (p & 1) ? m4.z : m4.x;
                dw[h] = __half2float(__ushort_as_half((unsigned short) (((p & 1) ? m4.w : m4.y) & 0xffff)));
                const unsigned hi = (unsigned) (p == 0 ? hb[h].x : p == 1 ? hb[h].y : p == 2 ? hb[h].z : hb[h].w);
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
                const int jl = 2 * (q + p) + bb;
#pragma unroll
                for (int nt = 0; nt < TR / 8; nt++) {
                    const int8_t * xr = xb + nt * 8 * xc.tsb;
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

// Q4_K / Q5_K tile over whole blocks [q0, q1) (as mma_rows_q6b): the header and Q5_K's high bits once per block,
// the 4 groups' nibbles in flight together; a quarter of mma_rows_kq's header registers.
template <int TYPE, int TR>
__device__ __forceinline__ void mma_rows_q45b(const Mat & m, int ra, int rb, int k0, const XC & xc, int q0, int q1,
                                              float (&acc)[TR / 8][4]) {
    constexpr int BB = TYPE == T_Q4K ? 144 : 176, QO = TYPE == T_Q4K ? 16 : 48;
    const int lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3, sw = xc_swz(g);
    const uint8_t * rw[2] = {m.w + kq_wo(ra, m.rb, BB), m.w + kq_wo(rb, m.rb, BB)};
    const int8_t * xb = xc.x + g * xc.tsb;
    const __half * xd = xc.d + 2 * tig;
    const short * xq = xc.qs + 2 * tig;
    const int G0 = k0 / 64;
    for (int q = q0; q < q1; q += 4) {
        int4 hd[2];
        int2 w[4][2], qh[2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint8_t * blk = rw[h] + (size_t) ((G0 + q) >> 2) * KQ_T * BB;
            hd[h] = ldw4(blk);
            if constexpr (TYPE == T_Q5K) qh[h] = ldw2(blk + 16 + 8 * tig);
#pragma unroll
            for (int p = 0; p < 4; p++) w[p][h] = ldw2(blk + QO + 32 * p + 8 * tig);
        }
#pragma unroll
        for (int p = 0; p < 4; p++) {
            int2 V[2][2]; // [block][row]
            float f[2][2], gm[2][2];
#pragma unroll
            for (int h = 0; h < 2; h++) {
                q4k_scales(hd[h], p, f[0][h], gm[0][h], f[1][h], gm[1][h]);
                const int2 v = w[p][h];
                V[0][h] = make_int2(v.x & M4, v.y & M4);
                V[1][h] = make_int2((v.x >> 4) & M4, (v.y >> 4) & M4);
                if constexpr (TYPE == T_Q5K) {
                    V[0][h].x |= ((qh[h].x >> (2 * p)) & 0x01010101) << 4;
                    V[0][h].y |= ((qh[h].y >> (2 * p)) & 0x01010101) << 4;
                    V[1][h].x |= ((qh[h].x >> (2 * p + 1)) & 0x01010101) << 4;
                    V[1][h].y |= ((qh[h].y >> (2 * p + 1)) & 0x01010101) << 4;
                }
            }
#pragma unroll
            for (int bb = 0; bb < 2; bb++) {
                const int jl = 2 * (q + p) + bb;
#pragma unroll
                for (int nt = 0; nt < TR / 8; nt++) {
                    const int2 bx = *reinterpret_cast<const int2 *>(xb + nt * 8 * xc.tsb + ((32 * jl + 8 * tig) ^ sw));
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

// k-quants: whole blocks of [q0, q1) through the block paths, the partial blocks at either end group by group.
template <int TYPE, int TR>
__device__ __forceinline__ void mma_rows_kqb(const Mat & m, int ra, int rb, int k0, const XC & xc, int q0, int q1,
                                             float (&acc)[TR / 8][4]) {
    const int G0 = k0 / 64;
    const int qs = min(q1, q0 + ((4 - ((G0 + q0) & 3)) & 3)), qe = qs + ((q1 - qs) & ~3);
    if (q0 < qs) mma_rows_kq<TYPE, 1, TR>(m, ra, rb, k0, xc, q0, qs, acc);
    if constexpr (TYPE == T_Q6K) mma_rows_q6b<TR>(m, ra, rb, k0, xc, qs, qe, acc);
    else mma_rows_q45b<TYPE, TR>(m, ra, rb, k0, xc, qs, qe, acc);
    if (qe < q1) mma_rows_kq<TYPE, 1, TR>(m, ra, rb, k0, xc, qe, q1, acc);
}

// One 16-row tile (rows ra = r0 + g, rb = r0 + g + 8 of the calling lane) of m over groups [q0, q1) of the slice at k0.
// TYPE >= 0 fixes m's type at compile time; -1 switches per call, which sizes registers for every decoder at once.
template <int PD, int TR, int TYPE = -1>
__device__ __forceinline__ void mma_tile(const Mat & m, int ra, int rb, int k0, const XC & xc, int q0, int q1,
                                         float (&acc)[1][TR / 8][4]) {
    if constexpr (TYPE == T_Q4K || TYPE == T_Q5K || TYPE == T_Q6K) {
        mma_rows_kqb<TYPE, TR>(m, ra, rb, k0, xc, q0, q1, acc[0]);
    } else if constexpr (TYPE == T_Q8_0) {
        const uint8_t * wp[1][2] = {{mrow_w(m, ra, k0), mrow_w(m, rb, k0)}};
        const __half * dp[1][2] = {{mrow_d(m, ra, k0), mrow_d(m, rb, k0)}};
        mma_rows<1, PD, TR>(wp, dp, xc, q0, q1, acc);
    } else {
        switch (m.type) {
        case T_Q4K: mma_tile<PD, TR, T_Q4K>(m, ra, rb, k0, xc, q0, q1, acc); break;
        case T_Q5K: mma_tile<PD, TR, T_Q5K>(m, ra, rb, k0, xc, q0, q1, acc); break;
        case T_Q6K: mma_tile<PD, TR, T_Q6K>(m, ra, rb, k0, xc, q0, q1, acc); break;
        default: mma_tile<PD, TR, T_Q8_0>(m, ra, rb, k0, xc, q0, q1, acc); break;
        }
    }
}

// One matrix of a phase's chain of warp items (numbered across the phase as gemv_pairs numbers them):
// 16-row tiles [lo, hi) of m -> base[t * stride + row] = dot * s_rs[t].
struct MSeg {
    Mat m;
    int lo, hi;
    float * base;
    int stride;
};
struct MChain {
    MSeg s[3];
    int n;
};

// The chain over the RMS-normed rows of cB.qn (K = D) in K slices.  A projection has fewer 16-row tiles than
// the grid has warps, so the two warps of a pair (same block) split each slice's K groups and reduce through
// shared memory.  Rounds are uniform across the block (every warp takes part in each slice fill and reduce);
// consecutive rounds walk the slices in opposite order.
template <int TR, int TYPE>
__device__ __noinline__ void gemv_mma(const MChain c) {
    constexpr int NS = mma_ns<TR, D>(), KS = D / NS, NC = TR / 8, QH = KS / 64 / 2;
    static_assert(NW % 2 == 0 && NW / 2 * NC * 4 * 32 * 4 <= SM_MISC, "pair partials fit below misc");
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tig = lane & 3, h = warp & 1;
    float * s_rs = misc(smem);
    float * red = reinterpret_cast<float *>(smem) + (warp >> 1) * NC * 4 * 32;
    const int P = nwarps() / 2, lo = c.s[0].lo, hi = c.s[c.n - 1].hi;
    // tiles start lo / 2 pairs in, past the pairs whose warps ran the phase's leading items (gwarp() < lo)
    const int base = lo - lo / 2, p = (warp >> 1) * gridDim.x + blockIdx.x, rounds = (hi - base + P - 1) / P;
    XC xc{};
    int cur = -1; // slice in shared memory
    for (int r = 0; r < rounds; r++) {
        const int it = base + p + r * P;
        const bool on = it >= lo && it < hi;
        int k = 0;
        while (k + 1 < c.n && it >= c.s[k].hi) k++;
        const MSeg & sg = c.s[k];
        const int r0 = 16 * (it - sg.lo);
        float acc[1][NC][4];
        zero_mma<1, NC>(acc);
        for (int i = 0; i < NS; i++) {
            const int s = (r & 1) ? NS - 1 - i : i;
            if (s != cur) {
                xc = fill_xc<TR, KS>(cB.qn, 0, s * KS / 32, true, s_rs);
                cur = s;
                if (i == 1) mark_step(PS_WRITTEN); // step record: second slice in (every warp done with the first)
            }
            if (on) mma_tile<4, TR, TYPE>(sg.m, r0 + g, r0 + g + 8, s * KS, xc, h * QH, (h + 1) * QH, acc);
            if (i == 0) mark_step(PS_RED); // step record: warp 0 done with the first slice
        }
        __syncthreads(); // the slice is dead: the pair partials go in its place
        mark_step(PS_SENT); // step record: every warp done multiplying
        if (on && h)
#pragma unroll
            for (int nt = 0; nt < NC; nt++)
#pragma unroll
                for (int i = 0; i < 4; i++) red[(nt * 4 + i) * 32 + lane] = acc[0][nt][i];
        __syncthreads();
        cur = -1;
        if (!on || h) continue;
#pragma unroll
        for (int nt = 0; nt < NC; nt++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int t = 8 * nt + 2 * tig + (i & 1);
                sg.base[t * sg.stride + r0 + g + 8 * (i >> 1)] = (acc[0][nt][i] + red[(nt * 4 + i) * 32 + lane]) * s_rs[t];
            }
    }
    __syncthreads(); // the partials are read until here; the caller's next shared-memory user may follow at once
    mark_step(PS_COMPUTE);
}

// Per-type instance when every matrix of the chain shares one type (a mixed chain takes the switching one).
template <int TR>
__device__ __forceinline__ void gemv_mma_t(const MChain & c) {
    int type = (cB.opt & OPT_GEMV_SW) ? -1 : c.s[0].m.type;
    for (int i = 1; i < c.n; i++)
        if (c.s[i].m.type != type) type = -1;
    switch (type) {
    case T_Q8_0: gemv_mma<TR, T_Q8_0>(c); break;
    case T_Q4K: gemv_mma<TR, T_Q4K>(c); break;
    case T_Q5K: gemv_mma<TR, T_Q5K>(c); break;
    case T_Q6K: gemv_mma<TR, T_Q6K>(c); break;
    default: gemv_mma<TR, -1>(c); break;
    }
}

// FFN gate/up: item = 32 consecutive rows of both matrices on 4 warps of one block, one 16-row tile each
// (gate 0, gate 1, up 0, up 1) over the full K in slices; the tiles meet in shared memory over the spent
// activations, then silu(g)*u is quantised per natural 32-block into cB.qa (both 16-chunk units carry the
// block's scale, as q32_store writes it).
template <int TR, int TYPE>
__device__ __noinline__ void ffn_gu_mma(const LayerW & W) {
    constexpr int NS = mma_ns<TR, D>(), KS = D / NS, NC = TR / 8, GPB = NW / 4, NI = FF_L / 32;
    static_assert(NW % 4 == 0 && GPB * 4 * TR * 16 * 4 <= xc_bytes(TR, KS), "ffn_gu_mma exchange over the activations");
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tig = lane & 3;
    const int wg = warp >> 2, tl = warp & 3;
    float * s_rs = misc(smem);
    float * ex = reinterpret_cast<float *>(smem) + wg * 4 * TR * 16; // [tile][TR][16 rows]
    const QBuf & q = cB.qa;
    const Mat & m = tl < 2 ? W.fg : W.fu;
    pf_ffn_gu(W);
    XC xc{};
    int cur = -1;
    // block-uniform rounds: the fills and the exchange need every warp of the block
    for (int j0 = GPB * blockIdx.x, r = 0; j0 < NI; j0 += GPB * gridDim.x, r++) {
        const int j = j0 + wg, row = 32 * j + 16 * (tl & 1) + g;
        float acc[1][NC][4];
        zero_mma<1, NC>(acc);
        for (int i = 0; i < NS; i++) {
            const int s = (r & 1) ? NS - 1 - i : i;
            if (s != cur) {
                xc = fill_xc<TR, KS>(cB.qn, 0, s * KS / 32, true, s_rs);
                cur = s;
                if (i == 1) mark_step(PS_WRITTEN);
            }
            if (j < NI) {
                mma_tile<4, TR, TYPE>(m, row, row + 8, s * KS, xc, 0, KS / 64, acc);
            }
            if (i == 0) mark_step(PS_RED);
        }
        __syncthreads(); // every warp is done with the activations
        mark_step(PS_SENT);
        cur = -1;
        if (j < NI) {
#pragma unroll
            for (int nt = 0; nt < NC; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int t = 8 * nt + 2 * tig + (e & 1);
                    ex[(tl * TR + t) * 16 + g + 8 * (e >> 1)] = acc[0][nt][e] * s_rs[t];
                }
        }
        __syncthreads();
        if (j >= NI) continue;
        // lane = row of the item's natural 32-block; unit u carries its 16-chunk lane >> 4
        const int part = j & 1, u = 2 * (j >> 1) + (lane >> 4), rr = lane & 15;
        for (int t = tl; t < TR; t += 4) {
            const float a = silu(ex[((lane >> 4) * TR + t) * 16 + rr]) * ex[((2 + (lane >> 4)) * TR + t) * 16 + rr];
            const float am = wmax(fabsf(a));
            const float d = am / 127.0f;
            const int qi = am == 0.0f ? 0 : (int) roundf(a / d);
            const int gi = t * q.U + u;
            (part ? q.xb : q.xa)[gi * 16 + rr] = (int8_t) qi;
            int qs = qi;
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) qs += __shfl_xor_sync(0xffffffffu, qs, o);
            if (rr == 0) {
                reinterpret_cast<__half *>(q.meta + gi)[part] = __float2half_rn(d);
                reinterpret_cast<short *>(q.meta + gi)[2 + part] = (short) qs;
            }
        }
    }
    mark_step(PS_COMPUTE);
}

template <int TR>
__device__ __forceinline__ void ffn_gu_mma_t(const LayerW & W) {
    switch (W.fg.type == W.fu.type ? W.fg.type : -1) {
    case T_Q8_0: ffn_gu_mma<TR, T_Q8_0>(W); break;
    case T_Q4K: ffn_gu_mma<TR, T_Q4K>(W); break;
    case T_Q5K: ffn_gu_mma<TR, T_Q5K>(W); break;
    case T_Q6K: ffn_gu_mma<TR, T_Q6K>(W); break;
    default: ffn_gu_mma<TR, -1>(W); break;
    }
}

#if MK_TP > 1
// K-sharded residual GEMV over a K = KS * NS input in NS slices: block item = 32 output rows (D / 32
// items, at most 2 per block: the host requires >= 80 blocks); warp w computes row tile w & 1 of every
// item over K-part w >> 1 of each slice.  The 8 parts reduce in shared memory in a fixed order, then the
// TP all-reduce and epilogue of resid32_ks.
template <int TR, int KS, int NS, int TYPE>
__device__ __noinline__ void resid_mma(const Mat W, const QBuf in, float * x, bool add, const float * wnext, int k, int nr) {
    constexpr int QP = KS / 64 / 8, NC = TR / 8; // block pairs per K-part
    constexpr int PD = QP % 4 == 0 ? 4 : QP % 3 == 0 ? 3 : QP % 5 == 0 ? 5 : QP % 2 == 0 ? 2 : 1;
    static_assert(KS % 512 == 0 && NW == 16 && 18 * TR * 32 * 4 <= SM_MISC, "resid_mma layout");
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tig = lane & 3;
    const int tile = warp & 1, kp = warp >> 1;
    const int n_items = (D / 32 - blockIdx.x + gridDim.x - 1) / gridDim.x;
    float * s_rs = misc(smem);
    float * red = reinterpret_cast<float *>(smem); // [8 parts][2 items][TR * 32]
    float * park = red + 16 * TR * 32;             // [n_items][TR * 32]
    pf_resid(W, KS * NS / 1024);
    float acc[2][1][NC][4];
    zero_mma<1, NC>(acc[0]);
    zero_mma<1, NC>(acc[1]);
    for (int s = 0; s < NS; s++) {
        const XC xc = fill_xc<TR, KS>(in, 0, s * KS / 32, false, s_rs);
#pragma unroll
        for (int i = 0; i < 2; i++)
            if (i < n_items) {
                const int r = 32 * (blockIdx.x + i * gridDim.x) + 16 * tile + g;
                mma_tile<PD, TR, TYPE>(W, r, r + 8, s * KS, xc, kp * QP, (kp + 1) * QP, acc[i]);
            }
    }
    __syncthreads(); // the activations are dead: the parts reduce in their place
    mark_step(PS_COMPUTE);
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int nt = 0; nt < NC; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++)
                red[(kp * 2 + i) * TR * 32 + (8 * nt + 2 * tig + (e & 1)) * 32 + 16 * tile + g + 8 * (e >> 1)] = acc[i][0][nt][e];
    __syncthreads();
    for (int i = 0; i < n_items; i++)
        for (int e = threadIdx.x; e < TR * 32; e += NT) {
            float v = 0.f;
#pragma unroll
            for (int p = 0; p < 8; p++) v += red[(p * 2 + i) * TR * 32 + e];
            park[i * TR * 32 + e] = v;
        }
    __syncthreads();
    mark_step(PS_RED);
    // rows past nr are width padding nobody reads: they skip the PCIe exchange
    for (int i = 0; i < n_items; i++) xr_send(blockIdx.x + i * gridDim.x, k, park + i * TR * 32, nr * 32);
    xr_release(n_items, k);
    for (int i = 0; i < n_items; i++) {
        const int j = blockIdx.x + i * gridDim.x;
        const float * pk = park + i * TR * 32;
        const __nv_bfloat16 * peer = cB.xr[k & 1] + j * MAXR * 32;
        xr_wait(j, k);
        for (int t = warp; t < TR; t += NW) {
            const int row = 32 * j + lane;
            float xv = t < nr ? xr_sum(pk[t * 32 + lane], peer, t * 32 + lane) : pk[t * 32 + lane];
            if (add) xv += __ldcg(x + t * D + row);
            x[t * D + row] = xv;
            q32_store(cB.qn, t, j, xv * wnext[row], xv * xv, true);
        }
    }
}
#endif
