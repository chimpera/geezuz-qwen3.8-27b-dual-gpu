// Batched decode cycle (TP=2): one launch verifies and drafts for several sequence slots.  The
// slots' rows share every weight read of the trunk GEMVs (packed slot after slot, c[b] rows each, so
// slots with fewer drafts leave their rows to the others); attention, delta-net and sampling run per
// slot over R = max c rows.  A slot's rows past its c[b] repeat its last row at the following
// positions: they write no row output, and their cache writes land on positions that are rewritten
// before they are ever read (each later cycle writes its own rows before attending, and masks
// positions past them).  Included by qwen38_mk.cu inside its namespace.

struct BLaunch {
    int nb, Tb, nr;   // slots in this cycle, max main rows per slot, main rows in all (<= maxr)
    int maxr;         // row cap: MAXR when every GEMV matrix has a tensor-core decoder (rows past MAXR_DP), else MAXR_DP
    int slot[MAXB];   // their slot indices
    int mtp_on[MAXB]; // run the slot's MTP catch-up (keeps its MTP cache current even without drafts)
    int r0[MAXB];     // first main row of each slot (rows packed: r0[b + 1] = r0[b] + sp[b].T)
    Launch sp[MAXB];  // per slot: T = its verified rows (<= Tb), ndraft, sampling
};

// One batched pass: slot b's rows are r0[b] .. r0[b] + c[b] - 1 (c[b] <= R, nr rows in all), the first at
// position n[b].
struct BRows {
    int nb, R, nr;
    int s[MAXB], n[MAXB], dnc[MAXB], act[MAXB], r0[MAXB], c[MAXB];
};

// slot of packed row r (-1: a width padding row past the last slot's rows)
__device__ __forceinline__ int row_slot(const int * r0, int nb, int nr, int r) {
    if (r >= nr) return -1;
    int b = nb - 1;
    while (r0[b] > r) b--;
    return b;
}

// row count of a pass rounded up to an instantiated GEMV width (the extra rows are never read)
__host__ __device__ constexpr int brows(int r) { return r <= 6 ? (r < 2 ? 2 : r) : r <= 8 ? 8 : r <= MAXR_DP ? MAXR_DP : r <= MR ? MR : r <= MRM ? MRM : MAXR; }

template <int TR>
__device__ __noinline__ void embed_b(const BLaunch & P) {
    const int r = blockIdx.x;
    if (r >= TR) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int b = row_slot(P.r0, P.nb, P.nr, r), t = r - (b < 0 ? 0 : P.r0[b]);
    if (b < 0) b = t = 0; // width padding: a copy of row 0
    const int tok = ((volatile Ctl *) cB.sq[P.slot[b]].ctl)->tok[min(t, P.sp[b].T - 1)];
    const uint8_t * row = stage_embd_row(tok, smem);
    const float * w0 = cW.L[0].attn_norm;
    for (int j = warp; j < U_D; j += NW) {
        const int e = 32 * j + lane;
        const float v = dequant_elem(row, cW.embd_type, e);
        cB.x[r * D + e] = v;
        q32_store(cB.qn, r, j, v * w0[e], v * v, true);
    }
}

// K-split residual GEMV: T activation rows do not fit in shared memory, so K runs in two slices whose
// warp-reduced partial sums accumulate per item in the park region; then the TP all-reduce and epilogue.
template <int T, int ITc, int TYPE>
__device__ __forceinline__ void ks_slice(const Mat & W, const QBuf & in, int u0, bool first, float * park, int n_items) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const SX sx = make_sx(smem, 32 * ITc);
    fill_xs_part<T, 32 * ITc>(in, u0, sx);
    for (int i = 0; i < n_items; i++) {
        const int j = blockIdx.x + i * gridDim.x;
        float a[2][T];
        zero_acc<T, 2>(a);
        mrows<T, ITc, 2, TYPE>(W, 32 * j + 2 * warp, sx, a, u0);
        wsum_acc<T, 2>(a);
        float * pk = park + i * T * 32;
        if (lane == 0) {
#pragma unroll
            for (int t = 0; t < T; t++) {
                pk[t * 32 + 2 * warp] = first ? a[0][t] : pk[t * 32 + 2 * warp] + a[0][t];
                pk[t * 32 + 2 * warp + 1] = first ? a[1][t] : pk[t * 32 + 2 * warp + 1] + a[1][t];
            }
        }
    }
}

template <int T, int IT, int TYPE>
__device__ __noinline__ void resid32_ks(const Mat W, const QBuf in, float * x, bool add, const float * wnext, int k, int nr) {
    constexpr int ITS = (IT + 1) / 2;
    static_assert(T * 32 * ITS <= SM_XU && 3 * T * 32 * 4 <= 4096, "two K slices, <= 3 items per block parked");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n_items = (D / 32 - blockIdx.x + gridDim.x - 1) / gridDim.x;
    float * park = reinterpret_cast<float *>(smem + SM_PARK); // [n_items][T * 32]
    pf_resid(W, IT);
    ks_slice<T, ITS, TYPE>(W, in, 0, true, park, n_items);
    ks_slice<T, IT - ITS, TYPE>(W, in, 32 * ITS, false, park, n_items);
    __syncthreads();
    mark_step(PS_COMPUTE);
    // rows past nr are width padding nobody reads: they skip the PCIe exchange
    for (int i = 0; i < n_items; i++) xr_send(blockIdx.x + i * gridDim.x, k, park + i * T * 32, nr * 32);
    xr_release(n_items, k);
    for (int i = 0; i < n_items; i++) {
        const int j = blockIdx.x + i * gridDim.x;
        const float * pk = park + i * T * 32;
        const __nv_bfloat16 * peer = cB.xr[k & 1] + j * MAXR * 32;
        xr_wait(j, k);
        for (int t = warp; t < T; t += NW) {
            const int row = 32 * j + lane;
            float xv = t < nr ? xr_sum(pk[t * 32 + lane], peer, t * 32 + lane) : pk[t * 32 + lane];
            if (add) xv += __ldcg(x + t * D + row);
            x[t * D + row] = xv;
            q32_store(cB.qn, t, j, xv * wnext[row], xv * xv, true);
        }
    }
}

template <int T, int IT>
__device__ __noinline__ void resid32_ks_t(int type, const Mat W, const QBuf in, float * x, bool add, const float * wnext, int k, int nr) {
    switch (type) {
    case T_Q4K: resid32_ks<T, IT, T_Q4K>(W, in, x, add, wnext, k, nr); break;
    case T_Q5K: resid32_ks<T, IT, T_Q5K>(W, in, x, add, wnext, k, nr); break;
    case T_Q6K: resid32_ks<T, IT, T_Q6K>(W, in, x, add, wnext, k, nr); break;
    default: resid32_ks<T, IT, T_Q8_0>(W, in, x, add, wnext, k, nr); break;
    }
}

template <int T, int IT>
__device__ __forceinline__ void resid_b(int type, const Mat & W, const QBuf & in, float * x, bool add, const float * wnext, int k, int nr) {
    constexpr int K = 1024 * IT;
    if constexpr (T > MAXR_DP) {
        constexpr int NS = mma_ns<T, K>();
        switch (type) {
        case T_Q4K: resid_mma<T, K / NS, NS, T_Q4K>(W, in, x, add, wnext, k, nr); break;
        case T_Q5K: resid_mma<T, K / NS, NS, T_Q5K>(W, in, x, add, wnext, k, nr); break;
        case T_Q6K: resid_mma<T, K / NS, NS, T_Q6K>(W, in, x, add, wnext, k, nr); break;
        default: resid_mma<T, K / NS, NS, T_Q8_0>(W, in, x, add, wnext, k, nr); break;
        }
    } else if constexpr (T * 32 * IT <= SM_XU) resid32_t<T, IT>(type, W, in, x, add, wnext, smem, k);
    else resid32_ks_t<T, IT>(type, W, in, x, add, wnext, k, nr);
}

// attention items of the active slots, numbered slot after slot: slot b's items start at abase(b)
__device__ __forceinline__ int b_nitems(const BRows & B, int b) {
    int nch, CS;
    attn_dims(B.n[b] + B.R, nch, CS);
    return NKV_L * nch;
}

template <int TR>
__device__ __noinline__ void attn_proj_b(const LayerW & W, int aslot, const BRows & B) {
    pf_attn_proj(W);
    if constexpr (TR > MAXR_DP) {
        constexpr int nq = QG_L / 16, nk = KVD_L / 16;
        gemv_mma_t<TR>(MChain{{{W.q, 0, nq, cB.qbuf, QG_L}, {W.k, nq, nq + nk, cB.kbuf, KVD_L}, {W.v, nq + nk, nq + 2 * nk, cB.vbuf, KVD_L}}, 3});
    } else {
        fill_xs<TR>(cB.qn, 0, true, make_sx(smem, U_D), misc(smem));
        attn_qkv<TR>(W);
    }
    pf_block(W.qn, HD * 4);
    pf_block(W.kn, HD * 4);
    int base = 0;
    for (int b = 0; b < B.nb; b++) {
        if (!B.act[b]) continue;
        pf_block(cB.rope + (size_t) B.n[b] * 32, B.R * 32 * 8);
        pf_attn_kv(cB.sq[B.s[b]], aslot, B.n[b], B.R, blockIdx.x, base);
        base += b_nitems(B, b);
    }
}

template <int R, bool Q8>
__device__ __forceinline__ void attn_core_bt(const LayerW & W, int aslot, const BRows & B) {
    const int G = gridDim.x;
    int base = 0;
    for (int b = 0; b < B.nb; b++) {
        if (!B.act[b]) continue;
        int nch, CS;
        attn_dims(B.n[b] + R, nch, CS);
        for (int item = (((int) blockIdx.x - base) % G + G) % G; item < NKV_L * nch; item += G)
            attn_item<R, Q8>(W, aslot, B.n[b], item / nch, item % nch, CS, base + item, cB.sq[B.s[b]], B.r0[b], B.c[b]);
        base += NKV_L * nch;
    }
}

template <int R>
__device__ __noinline__ void attn_core_b(const LayerW & W, int aslot, const BRows & B) {
    if (cB.kvq8) attn_core_bt<R, true>(W, aslot, B);
    else attn_core_bt<R, false>(W, aslot, B);
}

__device__ __forceinline__ void attn_core_bd(const LayerW & W, int aslot, const BRows & B) {
    switch (B.R) {
    case 1: attn_core_b<1>(W, aslot, B); break;
    case 2: attn_core_b<2>(W, aslot, B); break;
    case 3: attn_core_b<3>(W, aslot, B); break;
    case 4: attn_core_b<4>(W, aslot, B); break;
    default: attn_core_b<5>(W, aslot, B); break;
    }
}

// combine split-K partials, apply the sigmoid output gate, quantise for o_proj (per active slot)
__device__ __noinline__ void attn_combine_b(const BRows & B) {
    const int R = B.R, tid = threadIdx.x, d = tid & 255, wh = (tid & 255) >> 5;
    const int G2 = 2 * gridDim.x, hb = blockIdx.x * 2 + (tid >> 8);
    int abase = 0, ibase = 0;
    for (int b = 0; b < B.nb; b++) {
        if (!B.act[b]) continue;
        int nch, CS;
        attn_dims(B.n[b] + R, nch, CS);
        for (int it = ((hb - ibase) % G2 + G2) % G2; it < NH_L * R; it += G2) {
            const int h = it / R, t = it % R, g = h / 6, r = (h % 6) * R + t;
            if (t >= B.c[b]) continue; // a repeat of the slot's last row
            const float2 * ml0 = cB.aml + (size_t) (abase + g * nch) * AR + r;
            float M = -INFINITY;
            for (int c = 0; c < nch; c++) M = fmaxf(M, __ldcg(&ml0[c * AR].x));
            float num = 0.f, den = 0.f;
            for (int c = 0; c < nch; c++) {
                const float2 ml = __ldcg(&ml0[c * AR]);
                const float w = ml.x == -INFINITY ? 0.f : expf(ml.x - M);
                num += w * __ldcg(cB.apart + ((size_t) (abase + g * nch + c) * AR + r) * 256 + d);
                den += w * ml.y;
            }
            const float gate = __ldcg(cB.qbuf + (B.r0[b] + t) * QG_L + h * 512 + 256 + d);
            const float o = num / den / (1.0f + expf(-gate));
            q32_store(cB.qo, B.r0[b] + t, 8 * h + wh, o, 0.f, false);
        }
        abase += NKV_L * nch;
        ibase += NH_L * R;
    }
}

// delta-net input projections: alpha/beta (fp32), qkv -> cB.qkvs (dn_core moves them into each slot's ring), z
template <int TR>
__device__ __noinline__ void dn_proj_b(const LayerW & W, int dl, const BRows & B) {
    const int lane = threadIdx.x & 31;
    float * s_rs = misc(smem);
    pf_dn_proj(W);
    if constexpr (TR > MAXR_DP) {
        row_rs<TR>(cB.qn, 0, s_rs);
        __syncthreads();
    } else fill_xs<TR>(cB.qn, 0, true, make_sx(smem, U_D), s_rs);
    const int n_ab = AB_SPLIT * 2 * NVH_L, n_qkv = QKV_L / 2, n_z = VD_L / 2;
    for (int it = gwarp(); it < n_ab; it += nwarps()) {
        const int row = it % (2 * NVH_L), ch = it / (2 * NVH_L);
        const float * wr = W.ab + (size_t) row * D;
        float acc[TR];
#pragma unroll
        for (int t = 0; t < TR; t++) acc[t] = 0.f;
        for (int k = ch * (D / AB_SPLIT) + lane * 4; k < (ch + 1) * (D / AB_SPLIT); k += 128) {
            const float4 w = ldw4f(wr + k);
#pragma unroll
            for (int t = 0; t < TR; t++) {
                const float4 xv = __ldcg(reinterpret_cast<const float4 *>(cB.x + t * D + k));
                acc[t] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
            }
        }
#pragma unroll
        for (int t = 0; t < TR; t++) acc[t] = wsum(acc[t]) * s_rs[t];
        store_lane_t<TR>(cB.ab + ch * MAXR * 2 * NVH_L, 2 * NVH_L, row, acc);
    }
    if constexpr (TR > MAXR_DP) {
        constexpr int t_qkv = QKV_L / 16, t_z = VD_L / 16;
        gemv_mma_t<TR>(MChain{{{W.qkv, n_ab, n_ab + t_qkv, cB.qkvs, QKV_L}, {W.gate, n_ab + t_qkv, n_ab + t_qkv + t_z, cB.z, VD_L}}, 2});
    } else {
        const int n_all = n_ab + n_qkv + n_z;
        gemv_pairs_t<TR, 5>(W.qkv.type, W.qkv, n_ab, n_ab + n_qkv, n_all, StorePair<TR>{cB.qkvs, QKV_L});
        gemv_pairs_t<TR, 5>(W.gate.type, W.gate, n_ab + n_qkv, n_all, n_all, StorePair<TR>{cB.z, VD_L});
    }
    // warm L2 with the recurrent state of the (slot, head) this block runs first in dn_core
    const int w = blockIdx.x, b = w / NVH_L;
    if (b < B.nb) pf_l2(cB.sq[B.s[b]].dstate + ((size_t) dl * NVH_L + w % NVH_L) * 128 * 128 + threadIdx.x * 32);
}

template <int R>
__device__ __noinline__ void dn_core_b(const LayerW & W, int dl, const BRows & B) {
    for (int w = blockIdx.x; w < B.nb * NVH_L; w += gridDim.x) {
        const int b = w / NVH_L;
        dn_head<R, 1>(W, dl, B.n[b], B.dnc[b], 1, w % NVH_L, cB.sq[B.s[b]], B.r0[b], B.c[b]);
        __syncthreads(); // the next item reuses the shared buffers
    }
}

__device__ __forceinline__ void dn_core_bd(const LayerW & W, int dl, const BRows & B) {
    switch (B.R) {
    case 1: dn_core_b<1>(W, dl, B); break;
    case 2: dn_core_b<2>(W, dl, B); break;
    case 3: dn_core_b<3>(W, dl, B); break;
    case 4: dn_core_b<4>(W, dl, B); break;
    default: dn_core_b<5>(W, dl, B); break;
    }
}

// final RMS norm rows -> h: every row into the slot's h ring (main), or row last[b] into its hm (MTP)
__device__ __noinline__ void write_h_b(const float * x, const QBuf & q, const float * w, const BRows & B, const int * last) {
    const int b = blockIdx.x;
    if (b >= B.nb || !B.act[b] || (last && last[b] < 0)) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, C = B.c[b], r0 = B.r0[b];
    float * s_rs = misc(smem);
    if (warp < C) {
        float acc = 0.f;
        for (int j = lane; j < U_D; j += 32) acc += __ldcg(q.ssq + (r0 + warp) * U_D + j);
        acc = wsum(acc);
        if (lane == 0) s_rs[warp] = rsqrtf(acc / (float) D + EPS);
    }
    __syncthreads();
    const SeqBufs & S = cB.sq[B.s[b]];
    for (int i = threadIdx.x; i < C * D; i += NT) {
        const int t = i / D, e = i - t * D;
        if (last && t != last[b]) continue;
        float * dst = last ? S.hm : S.h_hist + (size_t) ((B.n[b] + t) % H_RING) * D;
        dst[e] = __ldcg(x + (size_t) r0 * D + i) * s_rs[t] * w[e];
    }
}

template <int TR>
__device__ __forceinline__ void ffn_gu_b(const LayerW & W) {
    if constexpr (TR > MAXR_DP) ffn_gu_mma_t<TR>(W);
    else ffn_gu<TR>(W, smem);
}

template <int TR>
__device__ void bmain(const BLaunch & P, unsigned & epoch) {
    BRows B;
    B.nb = P.nb;
    B.R = P.Tb;
    B.nr = P.nr;
    bool samp = false;
    for (int b = 0; b < P.nb; b++) {
        const volatile Ctl * c = cB.sq[P.slot[b]].ctl;
        B.s[b] = P.slot[b];
        B.n[b] = c->n;
        B.dnc[b] = c->dn_c;
        B.act[b] = 1;
        B.r0[b] = P.r0[b];
        B.c[b] = P.sp[b].T;
        samp |= P.sp[b].temp > 0.0f;
    }
    embed_b<TR>(P);
    gsync(epoch, PH_EMBED);
    int dl = 0;
    for (int L = 0; L < NL; L++) {
        const LayerW & W = cW.L[L];
        const float * wnext = (L + 1 < NL) ? cW.L[L + 1].attn_norm : cW.onorm;
        if (W.attn) {
            attn_proj_b<TR>(W, L / 4, B);
            gsync(epoch, PH_ATTN_PROJ);
            attn_core_bd(W, L / 4, B);
            gsync(epoch, PH_ATTN_CORE);
            attn_combine_b(B);
            if (!(cB.opt & OPT_NO_PF)) pf_resid(W.o, U_VD_L / 32);
            gsync(epoch, PH_ATTN_COMB);
            resid_b<TR, U_VD_L / 32>(W.o.type, W.o, cB.qo, cB.x, true, W.post_norm, 2 * L, B.nr);
            if (!(cB.opt & OPT_NO_PF)) pf_ffn_gu(W);
            gsync(epoch, PH_RESID_O);
        } else {
            dn_proj_b<TR>(W, dl, B);
            gsync(epoch, PH_DN_PROJ);
            dn_core_bd(W, dl, B);
            if (!(cB.opt & OPT_NO_PF)) pf_resid(W.sout, U_VD_L / 32);
            gsync(epoch, PH_DN_CORE);
            resid_b<TR, U_VD_L / 32>(W.sout.type, W.sout, cB.qo, cB.x, true, W.post_norm, 2 * L, B.nr);
            if (!(cB.opt & OPT_NO_PF)) pf_ffn_gu(W);
            gsync(epoch, PH_RESID_SOUT);
            dl++;
        }
        ffn_gu_b<TR>(W);
        if (!(cB.opt & OPT_NO_PF)) pf_resid(W.fd, U_FF_L / 32);
        gsync(epoch, PH_FFN_GU);
        resid_b<TR, U_FF_L / 32>(W.fd.type, W.fd, cB.qa, cB.x, true, wnext, 2 * L + 1, B.nr);
        if (L + 1 < NL) {
            if (cW.L[L + 1].attn) pf_attn_proj(cW.L[L + 1]);
            else pf_dn_proj(cW.L[L + 1]);
        }
        gsync(epoch, PH_FFN_DOWN);
    }
    write_h_b(cB.x, cB.qn, cW.onorm, B, nullptr);
    int * s_tok = reinterpret_cast<int *>(misc(smem) + 400);
    const bool am_last = samp ? lm_head<TR, true>(0, false, 4000, smem, s_tok) : lm_head<TR>(0, false, 4000, smem, s_tok);
    if (am_last && threadIdx.x < P.nb) {
        const int b = threadIdx.x, s = P.slot[b], r0 = P.r0[b];
        const Launch & Q = P.sp[b];
        int tg[MAXT];
        for (int i = 0; i < Q.T; i++) tg[i] = s_tok[r0 + i];
        if (Q.temp > 0.0f) sample_targets<1>(Q, Q.T, Q.T, tg, s, r0);
        Ctl * c = cB.sq[s].ctl;
        for (int i = 0; i < Q.T; i++) c->tgt[i] = tg[i];
        main_control<1>(MODE_DECODE, 0, Q.T, tg, s);
        __threadfence();
    }
    gsync(epoch, PH_LMHEAD);
}

// One MTP pass over B.c[b] rows per slot: catch-up rows lo[b] .. lo[b] + cnt[b] - 1 of the slot's
// ctl list (cnt[b] <= B.c[b]), or (draft) draft step di of each active slot.  want[b]: the slot's last row gives its
// next draft token (logits at the final catch-up pass and every draft step).
template <int TR>
__device__ __noinline__ void bmtp_pass(const BLaunch & P, const BRows & B, const int * lo, const int * cnt, bool draft, int di,
                                       const int * want, int p, unsigned & epoch) {
    uint8_t * const sm = smem;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // M0: concat(enorm(embed(tok)), hnorm(h_prev)) -> qe
    const int rb = row_slot(B.r0, B.nb, B.nr, blockIdx.x);
    if (rb >= 0 && B.act[rb]) {
        const int r = blockIdx.x, b = rb, j = r - B.r0[b], jj = min(j, cnt[b] - 1);
        const SeqBufs & S = cB.sq[B.s[b]];
        const volatile Ctl * vc = S.ctl;
        const int pos = B.n[b] + jj;
        const int tok = draft ? vc->tok[di] : vc->mtp_tok[lo[b] + jj];
        const float * hsrc = draft ? S.hm : (pos >= 1 ? S.h_hist + (size_t) ((pos - 1) % H_RING) * D : nullptr);
        float * se = reinterpret_cast<float *>(sm);
        float * sh = se + D;
        float * red = sh + D;
        const uint8_t * erow = stage_embd_row(tok, reinterpret_cast<uint8_t *>(red + 32));
        float ae = 0.f, ah = 0.f;
        for (int e = tid; e < D; e += NT) {
            const float v = dequant_elem(erow, cW.embd_type, e);
            const float hv = hsrc ? __ldcg(hsrc + e) : 0.f;
            se[e] = v;
            sh[e] = hv;
            ae += v * v;
            ah += hv * hv;
        }
        ae = bsum(ae, red);
        ah = bsum(ah, red + 16);
        const float rse = rsqrtf(ae / (float) D + EPS), rsh = rsqrtf(ah / (float) D + EPS);
        // eh_proj is K-split: rank 0 holds the enorm(embed) columns, rank 1 the hnorm(h) columns
        for (int jj2 = warp; jj2 < U_D; jj2 += NW) {
            const int e = 32 * jj2 + lane;
            const float v = cB.rank == 0 ? se[e] * rse * cW.enorm[e] : sh[e] * rsh * cW.hnorm[e];
            q32_store(cB.qe, r, jj2, v, 0.f, false);
        }
    }
    gsync(epoch, PH_MTP_M0);
    const LayerW & W = cW.L[NL];
    const int k0 = 128 + 3 * p;
    resid_b<TR, U_EH_L / 32>(cW.eh.type, cW.eh, cB.qe, cB.xm, false, W.attn_norm, k0, B.nr);
    gsync(epoch, PH_MTP_EH);
    attn_proj_b<TR>(W, 16, B);
    gsync(epoch, PH_MTP_ATTN_PROJ);
    attn_core_bd(W, 16, B);
    gsync(epoch, PH_MTP_ATTN_CORE);
    attn_combine_b(B);
    gsync(epoch, PH_MTP_ATTN_COMB);
    resid_b<TR, U_VD_L / 32>(W.o.type, W.o, cB.qo, cB.xm, true, W.post_norm, k0 + 1, B.nr);
    gsync(epoch, PH_MTP_RESID_O);
    ffn_gu_b<TR>(W);
    gsync(epoch, PH_MTP_FFN_GU);
    resid_b<TR, U_FF_L / 32>(W.fd.type, W.fd, cB.qa, cB.xm, true, cW.shnorm, k0 + 2, B.nr);
    gsync(epoch, PH_MTP_FFN_DOWN);
    bool any = false, prob = false;
    int last[MAXB];
    for (int b = 0; b < B.nb; b++) {
        last[b] = want[b] ? cnt[b] - 1 : -1;
        any |= want[b] != 0;
        prob |= want[b] && P.sp[b].draft_prob && P.sp[b].temp > 0.0f;
    }
    if (!any) return;
    write_h_b(cB.xm, cB.qn, cW.shnorm, B, last);
    int * s_tok = reinterpret_cast<int *>(misc(sm) + 400);
    const bool am_last = prob ? lm_head<TR, true>(0, true, 4001 + p, sm, s_tok) : lm_head<TR>(0, true, 4001 + p, sm, s_tok);
    if (am_last && tid < B.nb && want[tid]) {
        const int b = tid, row = B.r0[b] + last[b];
        const Launch & Q = P.sp[b];
        Ctl * c = cB.sq[B.s[b]].ctl;
        float2 * out = cB.sq[B.s[b]].dcand + (size_t) di * TOPK_MAX;
        int tok = s_tok[row];
        if (Q.draft_prob && Q.temp > 0.0f) {
            // dv_add pads the draft head with duplicate rows: keep each token once, or q(d) halves
            const float2 * cr = cB.cand + (size_t) row * TOPK_MAX;
            int k = 0;
            for (int j = 0; j < TOPK_MAX; j++) {
                const int id = __float_as_int(cr[j].y);
                bool dup = false;
                for (int i = 0; i < k; i++) dup |= __float_as_int(out[i].y) == id;
                if (!dup) out[k++] = cr[j];
            }
            for (int j = k; j < TOPK_MAX; j++) out[j] = make_float2(-INFINITY, __int_as_float(-1));
            float * qs = reinterpret_cast<float *>(sm + 8192) + b * TOPK_MAX; // past lm_head's merge buffers
            const int K = cand_probs(out, Q, qs);
            tok = cand_draw(out, qs, K, -1, urand(Q.seed ^ 0x85ebca6bULL, 3ull * (c->mtp_P + di + 1)));
            for (int j = 0; j < K; j++) out[j].x = qs[j];
            for (int j = K; j < TOPK_MAX; j++) out[j].x = 0.f;
        } else {
            out[0].x = 0.f; // point-mass draft
        }
        c->tok[1 + di] = tok;
        __threadfence();
    }
    gsync(epoch, PH_MTP_LMHEAD);
}

__device__ __forceinline__ void bmtp_pass_d(const BLaunch & P, const BRows & B, const int * lo, const int * cnt, bool draft, int di,
                                            const int * want, int p, unsigned & epoch) {
    switch (brows(B.nr)) {
    case 2: bmtp_pass<2>(P, B, lo, cnt, draft, di, want, p, epoch); break;
    case 3: bmtp_pass<3>(P, B, lo, cnt, draft, di, want, p, epoch); break;
    case 4: bmtp_pass<4>(P, B, lo, cnt, draft, di, want, p, epoch); break;
    case 5: bmtp_pass<5>(P, B, lo, cnt, draft, di, want, p, epoch); break;
    case 6: bmtp_pass<6>(P, B, lo, cnt, draft, di, want, p, epoch); break;
    case 8: bmtp_pass<8>(P, B, lo, cnt, draft, di, want, p, epoch); break;
    case MAXR_DP: bmtp_pass<MAXR_DP>(P, B, lo, cnt, draft, di, want, p, epoch); break;
    case MR: bmtp_pass<MR>(P, B, lo, cnt, draft, di, want, p, epoch); break;
    case MRM: bmtp_pass<MRM>(P, B, lo, cnt, draft, di, want, p, epoch); break;
    default: bmtp_pass<MAXR>(P, B, lo, cnt, draft, di, want, p, epoch); break;
    }
}

// MTP of all slots: the catch-up rows packed into one pass when they fit the row cap (else passes of Rp
// rows per slot, aligned so every slot's last catch-up row is in the final pass), then the draft steps,
// one row per slot.
__device__ void bmtp(const BLaunch & P, unsigned & epoch) {
    const int nb = P.nb;
    int Rb[MAXB], lg[MAXB], nd[MAXB];
    int Rmax = 0, Rsum = 0, maxnd = 0;
    for (int b = 0; b < nb; b++) {
        const volatile Ctl * c = cB.sq[P.slot[b]].ctl;
        Rb[b] = P.mtp_on[b] ? c->mtp_R : 0;
        nd[b] = P.sp[b].ndraft;
        lg[b] = Rb[b] > 0 && c->mtp_logits && nd[b] > 0;
        Rmax = max(Rmax, Rb[b]);
        Rsum += Rb[b];
        if (lg[b]) maxnd = max(maxnd, nd[b]);
    }
    if (Rmax == 0) return;
    const bool packed = Rsum <= P.maxr && Rmax <= MAXT;
    const int Rp = packed ? Rmax : min(MAXT, P.maxr / nb), npass = packed ? 1 : (Rmax + Rp - 1) / Rp;
    BRows B;
    B.nb = nb;
    int lo[MAXB], cnt[MAXB], want[MAXB];
    int p = 0;
    for (int q = 0; q < npass; q++, p++) {
        B.R = Rp;
        B.nr = 0;
        for (int b = 0; b < nb; b++) {
            const volatile Ctl * c = cB.sq[P.slot[b]].ctl;
            const int k = q - (npass - (Rb[b] + Rp - 1) / Rp);
            B.s[b] = P.slot[b];
            B.act[b] = Rb[b] > 0 && k >= 0;
            lo[b] = k * Rp;
            cnt[b] = B.act[b] ? min(Rp, Rb[b] - lo[b]) : 0;
            B.n[b] = B.act[b] ? c->mtp_pos[lo[b]] : 0;
            B.dnc[b] = 0;
            want[b] = B.act[b] && q == npass - 1 && lg[b];
            B.r0[b] = packed ? B.nr : b * Rp;
            B.c[b] = cnt[b];
            B.nr = packed ? B.nr + cnt[b] : (b + 1) * Rp;
        }
        bmtp_pass_d(P, B, lo, cnt, false, 0, want, p, epoch);
    }
    for (int i = 1; i < maxnd; i++, p++) {
        B.R = 1;
        B.nr = nb;
        for (int b = 0; b < nb; b++) {
            const volatile Ctl * c = cB.sq[P.slot[b]].ctl;
            B.act[b] = lg[b] && i < nd[b];
            lo[b] = 0;
            cnt[b] = B.act[b];
            B.r0[b] = b;
            B.c[b] = 1;
            B.n[b] = c->mtp_P + i;
            want[b] = B.act[b];
        }
        bmtp_pass_d(P, B, lo, cnt, true, i, want, p, epoch);
    }
}

__global__ void __launch_bounds__(NT, 1) mk_bcycle(BLaunch P) {
    unsigned epoch = 0;
    if (cB.prof && blockIdx.x == 0 && threadIdx.x == 0) {
        cB.prof[0] = gtimer() << 8;
        s_bar_k = 0;
    }
    switch (brows(P.nr)) {
    case 2: bmain<2>(P, epoch); break;
    case 3: bmain<3>(P, epoch); break;
    case 4: bmain<4>(P, epoch); break;
    case 5: bmain<5>(P, epoch); break;
    case 6: bmain<6>(P, epoch); break;
    case 8: bmain<8>(P, epoch); break;
    case MAXR_DP: bmain<MAXR_DP>(P, epoch); break;
    case MR: bmain<MR>(P, epoch); break;
    case MRM: bmain<MRM>(P, epoch); break;
    default: bmain<MAXR>(P, epoch); break;
    }
    bmtp(P, epoch);
}
