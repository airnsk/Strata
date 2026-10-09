// Resource-bounded gfx906 HC candidate. Every helper reads the descriptor by
// reference. Exact T bounds remove inactive token storage; per-output reductions
// retain the checked Strata arithmetic order. No forced register limit.
__device__ __forceinline__ void hcp_opt_norm(const GrMulti& m, int work) {
    __shared__ float part[WARPS];
    __shared__ float s_rs;
    const int token = work / HC;
    const FusedGrArgs& a = m.a[token];
    const int c = work % HC;
    float* xn = m.xn + (size_t) token * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const float gw = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss = 0.0f;
    for (int i = t * 4; i < D; i += THREADS * 4) {
        if (i / N != c) continue;                       // another stream's element: another block's
        const int d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
            r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
        }
        const float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
        ss += sq;
        // Retain the unscaled product in the existing scratch instead of reloading R
        // and recomputing the pending write after the reduction.
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
    }
    const float v = warp_sum(ss);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (t == 0) {
        float s = 0.0f;
        for (int w = 0; w < WARPS; ++w) s += part[w];
        s_rs = rsqrtf(s / (float) N + a.eps);
        a.rs[c] = s_rs;
    }
    __syncthreads();
    const float rs = s_rs;
    for (int d = t; d < N; d += THREADS) xn[c * N + d] *= rs;
}

template<int T>
__device__ __forceinline__ void hcp_opt_down(const GrMulti& m, int row_block) {
    constexpr int TILEV = 1280, CHUNKS = TILEV / 8, TQ = CHUNKS / 32;
    extern __shared__ __align__(16) float4 planes[];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const bool inject_block = row_block == DOWN_BLOCKS;
    const int row = inject_block ? warp : row_block * WARPS + warp;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = active ? (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) row * D : m.a[0].w_down;
    const uint4* w4 = reinterpret_cast<const uint4*>(wrow);
    float acc[T] = {};
    for (int base = 0; base < D; base += TILEV) {
        uint4 wv[TQ];
        if (active) {
#pragma unroll
            for (int q = 0; q < TQ; ++q) wv[q] = __ldg(w4 + base / 8 + lane + 32 * q);
        }
        __syncthreads();
        // Same plane layout as gr_down_staged_kernel, with just one buffer:
        // T=4 keeps 20 KiB dynamic LDS instead of double buffering's 40 KiB.
        for (int i = t; i < T * (TILEV / 4); i += THREADS) {
            const int k = i / (TILEV / 4), off = i % (TILEV / 4);
            const float4 value = *reinterpret_cast<const float4*>(m.xn + (size_t) k * D + base + off * 4);
            planes[k * (TILEV / 4) + (off & 1) * CHUNKS + (off >> 1)] = value;
        }
        __syncthreads();
        if (!active) continue;
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            const int j = lane + 32 * q;
            const Bf16x8 weights = unpack8(wv[q]);
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const float4* x = planes + k * (TILEV / 4);
                acc[k] += dot8u(weights, x[j], x[CHUNKS + j]);
            }
        }
    }
    if (!active) return;
#pragma unroll
    for (int k = 0; k < T; ++k) {
        const float sum = warp_sum(acc[k]);
        if (lane == k) {
            if (inject_block) m.a[k].inject_out[row] = sum;
            else { const float x = sum / (float) HC; m.a[k].lo[row] = x / (1.0f + __expf(-x)); }
        }
    }
}

template<int T>
__device__ __forceinline__ void hcp_opt_up(const GrMulti& m, int column_block) {
    __shared__ __align__(16) float lo[T][LR];
    __shared__ float g[T][HC][UPM_COLS];
    const int t = threadIdx.x, j = t & 7, grp = t >> 3;
    const int d0 = column_block * UPM_COLS;
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.a[i / LR].lo[i % LR];
    __syncthreads();
    // Only one row group and its 20 weight dwords live at once. Do not fully
    // unroll the p loop back into the old two-group preload schedule.
#pragma unroll 1
    for (int p = 0; p < 2; ++p) {
        const int r = grp + 32 * p, c = r / UPM_COLS, dd = r % UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
        uint4 w[5];
#pragma unroll
        for (int q = 0; q < 5; ++q) w[q] = __ldg(w4 + j + 8 * q);
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < T; ++k) {
            const float* l = lo[k];
            const float p0 = dot8(w[0], l + j * 8) + dot8(w[4], l + (32 + j) * 8);
            const float p1 = dot8(w[1], l + (j + 8) * 8);
            const float p2 = dot8(w[2], l + (j + 16) * 8);
            const float p3 = dot8(w[3], l + (j + 24) * 8);
            const float sum = xor8((p0 + p2) + (p1 + p3));
            if (j == k) mine = sum;
        }
        if (j < T) {
            const FusedGrArgs& a = m.a[j];
            float x0 = a.R[i];
            if (a.apply) {
                x0 = fmaf(a.bo_prev[d0 + dd], 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC), x0);
                a.R_out[i] = x0;
            }
            const float x = x0 * a.w_norm[i] * a.rs[c];
            g[j][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i % UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
}

template<int T>
__device__ __forceinline__ void hcp_opt_quantize(const GrMulti& m, int column_block) {
    const int token = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (token >= T) return;
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


template<int T>
__device__ __forceinline__ void hcp_opt_router(const GrMulti& m, FusedGrRouter router, int row) {
    const int t = threadIdx.x;
    __shared__ float partials[T][32];
    if (t < 32)
        for (int k = 0; k < T; ++k) partials[k][t] = 0.0f;
    __syncthreads();
    float acc[T] = {};
    const auto* weights = reinterpret_cast<const uint32_t*>(router.weights + (size_t) row * N);
    for (int pair = t; pair < N / 2; pair += THREADS) {
        const uint32_t weight = __ldg(weights + pair);
        const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
#pragma unroll
        for (int k = 0; k < T; ++k) if (k < T) {
            const float2 x = __ldg(reinterpret_cast<const float2*>(m.a[k].mixed) + pair);
            acc[k] = __fmaf_rn(w0, x.x, acc[k]);
            acc[k] = __fmaf_rn(w1, x.y, acc[k]);
        }
    }
#pragma unroll
    for (int k = 0; k < T; ++k) if (k < T) {
        acc[k] = warp_sum(acc[k]);
        if ((t & 31) == 0) partials[k][t / 32] = acc[k];
    }
    __syncthreads();
    if (t < 32)
#pragma unroll
        for (int k = 0; k < T; ++k) if (k < T) {
            const float sum = warp_sum(partials[k][t]);
            if (t == 0) router.logits[(size_t) k * router.n_expert + row] = sum;
        }
}


template<int T, bool Router>
__global__ void __launch_bounds__(THREADS) hc_persistent_kernel(GrMulti m, FusedGrRouter router) {
    auto grid = cooperative_groups::this_grid();
    for (int work = blockIdx.x; work < T * HC; work += gridDim.x) {
        hcp_opt_norm(m, work);
        __syncthreads();
    }
    grid.sync();
    for (int row = blockIdx.x; row <= DOWN_BLOCKS; row += gridDim.x) {
        hcp_opt_down<T>(m, row);
        __syncthreads();
    }
    grid.sync();
    for (int col = blockIdx.x; col < UPM_BLOCKS; col += gridDim.x) {
        hcp_opt_up<T>(m, col);
        __syncthreads();
    }
    if (m.a[0].q8_mixed != nullptr || (Router && router.weights != nullptr)) {
        grid.sync();
        if (m.a[0].q8_mixed != nullptr)
            for (int col = blockIdx.x; col < N / 32; col += gridDim.x) hcp_opt_quantize<T>(m, col);
        if constexpr (Router) {
            if (router.weights != nullptr)
                for (int row = blockIdx.x; row < router.n_expert; row += gridDim.x) {
                    hcp_opt_router<T>(m, router, row);
                    __syncthreads();
                }
        }
    }
}

const void* hcp_production_function(int tokens, bool router) {
#define HCP_CASE(T) case T: return router ? reinterpret_cast<const void*>(hc_persistent_kernel<T, true>) : reinterpret_cast<const void*>(hc_persistent_kernel<T, false>)
    switch (tokens) { HCP_CASE(1); HCP_CASE(2); HCP_CASE(3); HCP_CASE(4); default: return nullptr; }
#undef HCP_CASE
}
