// Diagnostic-only const-reference copies of hc_persistent.cuh helpers.
// Arithmetic/work ownership are intentionally unchanged. Keep the copies in sync.
// Only the explicit diagnostic API below launches these functions.
__device__ __forceinline__ void hcp_ref_norm(const GrMulti& m, int token) {
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const FusedGrArgs& a = m.a[token];
    float* xn = m.xn + (size_t) token * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float4 r[NQ], g[NQ];
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4;
        r[k] = *reinterpret_cast<const float4*>(a.R + i);
        g[k] = *reinterpret_cast<const float4*>(a.w_norm + i);
    }
    if (a.apply) {
#pragma unroll
        for (int k = 0; k < NQ; ++k) {
            const int i = t * 4 + k * THREADS * 4, c = i / N, d = i - c * N;
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r[k].x = fmaf(b.x, gw[c], r[k].x); r[k].y = fmaf(b.y, gw[c], r[k].y);
            r[k].z = fmaf(b.z, gw[c], r[k].z); r[k].w = fmaf(b.w, gw[c], r[k].w);
        }
    }
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4, c = i / N;
        const float sq = r[k].x * r[k].x + r[k].y * r[k].y + r[k].z * r[k].z + r[k].w * r[k].w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        r[k] = make_float4(r[k].x * g[k].x, r[k].y * g[k].y, r[k].z * g[k].z, r[k].w * g[k].w);
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float v = warp_sum(ss[c]);
        if (lane == 0) part[warp][c] = v;
    }
    __syncthreads();
    if (t < HC) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w][t];
        s_rs[t] = rsqrtf(s / (float) N + a.eps);
        a.rs[t] = s_rs[t];
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < NQ; ++k) {
        const int i = t * 4 + k * THREADS * 4;
        const float sr = s_rs[i / N];
        *reinterpret_cast<float4*>(xn + i) = make_float4(r[k].x * sr, r[k].y * sr, r[k].z * sr, r[k].w * sr);
    }

}
__device__ __forceinline__ void hcp_ref_down(const GrMulti& m, int row_block) {
    constexpr int TILEV = 1280, MAX_T = 4;
    constexpr int TQ = TILEV / 8 / 32;      // uint4 weight chunks per lane per tile
    extern __shared__ __align__(16) float tile[];   // [T][TILEV]
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const bool inject_block = row_block == DOWN_BLOCKS;
    const int row = inject_block ? warp : row_block * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = active ? (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) row * D : m.a[0].w_down;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    float acc[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
    for (int base = 0; base < D; base += TILEV) {
        uint4 wv[TQ];
        if (active) {
#pragma unroll
            for (int q = 0; q < TQ; ++q) wv[q] = __ldg(w4 + base / 8 + lane + 32 * q);
        }
        __syncthreads();                                   // the previous tile is consumed
        const float4* src4 = reinterpret_cast<const float4*>(m.xn);
        float4* tile4 = reinterpret_cast<float4*>(tile);
        for (int i = t; i < T * (TILEV / 4); i += THREADS) {
            const int k = i / (TILEV / 4), off = i - k * (TILEV / 4);
            tile4[i] = src4[((size_t) k * D + base) / 4 + off];
        }
        __syncthreads();
        if (!active) continue;
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            const int j = lane + 32 * q;
#pragma unroll
            for (int k = 0; k < MAX_T; ++k)
                if (k < T) acc[k] += dot8(wv[q], tile + k * TILEV + j * 8);
        }
    }
    if (!active) return;
    float s[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) s[k] = k < T ? warp_sum(acc[k]) : 0.0f;
    // lane k writes token k (every lane holds every sum after the xor reduction)
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (k >= T || lane != k) continue;
        if (inject_block) {
            m.a[k].inject_out[row] = s[k];
        } else {
            const float x = s[k] / (float) HC;
            m.a[k].lo[row] = x / (1.0f + __expf(-x));
        }
    }

}
__device__ __forceinline__ void hcp_ref_up(const GrMulti& m, int column_block) {
    __shared__ __align__(16) float lo[4][LR];
    __shared__ float g[4][HC][UPM_COLS];
    const int t = threadIdx.x, j = t & 7, grp = t >> 3;
    const int T = m.T;
    const int d0 = column_block * UPM_COLS;
    static_assert(LR == 40 * 8 && HC * UPM_COLS == 64 && THREADS == 256, "geometry");
    uint4 w[2][5];
    float rv[2] = {0.0f, 0.0f}, wn[2] = {0.0f, 0.0f}, rsc[2] = {0.0f, 0.0f}, bo[2] = {0.0f, 0.0f}, ip[2] = {0.0f, 0.0f};
    const bool apply = j < T && m.a[j < T ? j : 0].apply;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r = grp + 32 * p, c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
#pragma unroll
        for (int q = 0; q < 4; ++q) w[p][q] = __ldg(w4 + j + 8 * q);
        w[p][4] = __ldg(w4 + 32 + j);
        if (j < T) {
            const FusedGrArgs& a = m.a[j];
            rv[p] = a.R[i];
            wn[p] = a.w_norm[i];
            rsc[p] = a.rs[c];
            if (apply) { bo[p] = a.bo_prev[d0 + dd]; ip[p] = a.inj_prev[c]; }
        }
    }
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r = grp + 32 * p, c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if (k >= T) break;
            const float* l = lo[k];
            const float p0 = dot8(w[p][0], l + j * 8) + dot8(w[p][4], l + (32 + j) * 8);   // old lane j
            const float p1 = dot8(w[p][1], l + (j + 8) * 8);                               // old lane j + 8
            const float p2 = dot8(w[p][2], l + (j + 16) * 8);                              // old lane j + 16
            const float p3 = dot8(w[p][3], l + (j + 24) * 8);                              // old lane j + 24
            const float s = xor8((p0 + p2) + (p1 + p3));   // stage 16: (j, j+16), (j+8, j+24); stage 8; then 4, 2, 1
            if (j == k) mine = s;
        }
        if (j < T) {
            float x0 = rv[p];
            if (apply) {
                x0 = fmaf(bo[p], 2.0f * sigmoidf_(ip[p] / (float) HC), x0);
                m.a[j].R_out[i] = x0;
            }
            const float x = x0 * wn[p] * rsc[p];
            g[j][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) s += g[k][c][col];
        m.a[k].mixed[d0 + col] = s / (float) HC;
    }

}

// A whole 32-column q8_1 block belongs to one block, after every mixed output is
// visible. No last-block counters, initialization kernel or session-global state.
__device__ __forceinline__ void hcp_ref_quantize(const GrMulti& m, int column_block) {
    const int token = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (token >= m.T) return;
    const float xi = m.a[token].mixed[column_block * 32 + lane];
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = q8_1_finite(amax / 127.0f);
    GrQ81* y = reinterpret_cast<GrQ81*>(m.a[token].q8_mixed) + column_block;
    y->qs[lane] = q8_1_quant(xi, d, amax);
    if (lane == 0) y->ds = q8_1_ds(d, sum);
}

// The following BF16/F32 row is the ordered-pair MMVF arithmetic from Strata's
// native_bf16.cu (MIT, ggml authors; license above). N=2560 chooses 256 threads in
// mmvf_block_size, so both the FMAs and the two logical-warp trees stay the same.
__device__ __forceinline__ void hcp_ref_router(const GrMulti& m, FusedGrRouter router, int row) {
    const int t = threadIdx.x;
    __shared__ float partials[4][32];
    if (t < 32)
        for (int k = 0; k < m.T; ++k) partials[k][t] = 0.0f;
    __syncthreads();
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const auto* weights = reinterpret_cast<const uint32_t*>(router.weights + (size_t) row * N);
    for (int pair = t; pair < N / 2; pair += THREADS) {
        const uint32_t weight = __ldg(weights + pair);
        const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
        for (int k = 0; k < 4; ++k) if (k < m.T) {
            const float2 x = __ldg(reinterpret_cast<const float2*>(m.a[k].mixed) + pair);
            acc[k] = __fmaf_rn(w0, x.x, acc[k]);
            acc[k] = __fmaf_rn(w1, x.y, acc[k]);
        }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) if (k < m.T) {
        acc[k] = warp_sum(acc[k]);
        if ((t & 31) == 0) partials[k][t / 32] = acc[k];
    }
    __syncthreads();
    if (t < 32)
#pragma unroll
        for (int k = 0; k < 4; ++k) if (k < m.T) {
            const float sum = warp_sum(partials[k][t]);
            if (t == 0) router.logits[(size_t) k * router.n_expert + row] = sum;
        }
}


template<bool Ref, bool Timed>
__global__ void __launch_bounds__(THREADS) hc_diagnostic_kernel(GrMulti m, FusedGrRouter router,
                                                               unsigned long long* phase_cycles) {
    auto grid = cooperative_groups::this_grid();
    unsigned long long stamp[8];
    if constexpr (Timed) { if (threadIdx.x == 0) stamp[0] = clock64(); }
    for (int token = blockIdx.x; token < m.T; token += gridDim.x) {
        if constexpr (Ref) hcp_ref_norm(m, token); else hcp_norm(m, token);
        __syncthreads();
    }
    if constexpr (Timed) { if (threadIdx.x == 0) stamp[1] = clock64(); }
    grid.sync();
    if constexpr (Timed) { if (threadIdx.x == 0) stamp[2] = clock64(); }
    for (int row = blockIdx.x; row <= DOWN_BLOCKS; row += gridDim.x) {
        if constexpr (Ref) hcp_ref_down(m, row); else hcp_down(m, row);
        __syncthreads();
    }
    if constexpr (Timed) { if (threadIdx.x == 0) stamp[3] = clock64(); }
    grid.sync();
    if constexpr (Timed) { if (threadIdx.x == 0) stamp[4] = clock64(); }
    for (int col = blockIdx.x; col < UPM_BLOCKS; col += gridDim.x) {
        if constexpr (Ref) hcp_ref_up(m, col); else hcp_up(m, col);
        __syncthreads();
    }
    if constexpr (Timed) { if (threadIdx.x == 0) stamp[5] = clock64(); }
    const bool tail = m.a[0].q8_mixed != nullptr || router.weights != nullptr;
    if (tail) grid.sync();
    if constexpr (Timed) { if (threadIdx.x == 0) stamp[6] = clock64(); }
    if (tail) {
        if (m.a[0].q8_mixed != nullptr)
            for (int col = blockIdx.x; col < N / 32; col += gridDim.x) {
                if constexpr (Ref) hcp_ref_quantize(m, col); else hcp_quantize(m, col);
            }
        if (router.weights != nullptr)
            for (int row = blockIdx.x; row < router.n_expert; row += gridDim.x) {
                if constexpr (Ref) hcp_ref_router(m, router, row); else hcp_router(m, router, row);
                __syncthreads();
            }
    }
    if constexpr (Timed) {
        if (threadIdx.x == 0) {
            stamp[7] = clock64();
#pragma unroll
            for (int i = 0; i < 7; ++i) phase_cycles[blockIdx.x * 7 + i] = stamp[i + 1] - stamp[i];
        }
    }
}

const void* hcp_diagnostic_function(int variant, int tokens) {
    switch (variant) {
    case 0: return reinterpret_cast<const void*>(hc_persistent_legacy_kernel);
    case 1: return reinterpret_cast<const void*>(hc_diagnostic_kernel<false, true>);
    case 2: return reinterpret_cast<const void*>(hc_diagnostic_kernel<true, false>);
    case 3: return reinterpret_cast<const void*>(hc_diagnostic_kernel<true, true>);
    case 4: return hcp_production_function(tokens, false);
    case 5: return hcp_production_function(tokens, true);
    default: return nullptr;
    }
}
