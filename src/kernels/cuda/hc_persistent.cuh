// Castagna-inspired persistent HC scheduling for gfx906. Included only by fused_gr.cu,
// inside its anonymous namespace, after the existing kernels. Default kernels are untouched.
//
// Design reference: benpeterson40/castagna-veloce, commit
// 3483d462715d61f21ed5877836be35a724446359, ggml/src/ggml-cuda/hc-persist.cu.
// Its persistent phase schedule is adapted to Strata's BF16 arithmetic and caller-owned
// buffers. No donor quantizers, device-global stashes, fixed-CU spin barriers or TP
// exchanges are used. The following notice records the donor's license:
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 Ben Peterson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Per-output arithmetic below follows Strata's existing fast norm/up and tiled down.
// This intentionally duplicates the bodies: building with the feature off must not
// change the original kernels' register allocation, reduction trees or dispatch.
__device__ __forceinline__ void hcp_norm(GrMulti m, int token) {
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
__device__ __forceinline__ void hcp_down(GrMulti m, int row_block) {
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
__device__ __forceinline__ void hcp_up(GrMulti m, int column_block) {
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
__device__ __forceinline__ void hcp_quantize(GrMulti m, int column_block) {
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
__device__ __forceinline__ void hcp_router(GrMulti m, FusedGrRouter router, int row) {
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

__global__ void __launch_bounds__(THREADS) hc_persistent_legacy_kernel(GrMulti m, FusedGrRouter router) {
    auto grid = cooperative_groups::this_grid();
    // Virtual work tiles, rather than a fixed number of CUs. Every thread reaches
    // both grid barriers even when this block had no work in the previous phase.
    for (int token = blockIdx.x; token < m.T; token += gridDim.x) {
        hcp_norm(m, token);
        __syncthreads();
    }
    grid.sync();
    for (int row = blockIdx.x; row <= DOWN_BLOCKS; row += gridDim.x) {
        hcp_down(m, row);
        __syncthreads();
    }
    grid.sync();
    for (int col = blockIdx.x; col < UPM_BLOCKS; col += gridDim.x) {
        hcp_up(m, col);
        __syncthreads();
    }
    if (m.a[0].q8_mixed != nullptr || router.weights != nullptr) { // uniform
        grid.sync();
        if (m.a[0].q8_mixed != nullptr)
            for (int col = blockIdx.x; col < N / 32; col += gridDim.x)
                hcp_quantize(m, col);
        if (router.weights != nullptr)
            for (int row = blockIdx.x; row < router.n_expert; row += gridDim.x) {
                hcp_router(m, router, row);
                __syncthreads();
            }
    }
}

#include "hc_persistent_optimized.cuh"

struct HcpDispatch { bool checked = false; int blocks = 0; };
// Host-thread cache only. Device/session buffers and barrier state are never global.
thread_local HcpDispatch hcp_dispatch[64][4][2];
thread_local unsigned long long hcp_launch_count = 0;
std::atomic<int> hcp_override{-1};

bool hcp_requested() {
    const int override = hcp_override.load(std::memory_order_relaxed);
    if (override >= 0) return override != 0;
    static const bool requested = [] {
        const char* value = std::getenv("STRATA_HC_PERSIST");
        return value && std::strcmp(value, "1") == 0;
    }();
    return requested;
}

int hcp_blocks(int tokens, bool router = false) {
    if (tokens < 1 || tokens > 4) return 0;
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess || device < 0 || device >= 64) return 0;
    HcpDispatch& choice = hcp_dispatch[device][tokens - 1][router ? 1 : 0];
    if (choice.checked) return choice.blocks;
    choice.checked = true;
    hipDeviceProp_t prop{};
    int version = 0, cooperative = 0, active = 0;
    // ROCm 6.4 added capture of the direct cooperative launch API, including its
    // cooperative flag on graph replay. Older runtimes must keep the old path.
    if (hipRuntimeGetVersion(&version) != hipSuccess ||
        hipGetDeviceProperties(&prop, device) != hipSuccess ||
        hipDeviceGetAttribute(&cooperative, hipDeviceAttributeCooperativeLaunch, device) != hipSuccess ||
        !detail::hc_persistent_runtime_eligible(version, prop.gcnArchName, cooperative != 0)) return 0;
    const size_t shared = (size_t) tokens * 1280 * sizeof(float);
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&active, hcp_production_function(tokens, router), THREADS, shared) != hipSuccess ||
        active <= 0) return 0;
    // Graph capture does not validate the cooperative bound on all HIP versions.
    // Account for the actual compiled kernel's registers, static LDS and dynamic
    // LDS. HIP's cooperative scheduler provides residency even beside other streams.
    choice.blocks = detail::hc_persistent_grid_blocks(tokens, active, prop.multiProcessorCount);
    return choice.blocks;
}

bool launch_hcp(GrMulti m, cudaStream_t stream, const FusedGrRouter* requested_router, bool* router_written) {
    if (!hcp_requested() || m.T > 4 || m.a[0].q8_down || m.a[0].q8_up || m.a[0].q8_inject) return false;
    FusedGrRouter router{};
    static const bool fuse_router = [] { const char* value = std::getenv("STRATA_HC_PERSIST_ROUTER"); return !value || std::strcmp(value, "0") != 0; }();
    if (fuse_router && requested_router && requested_router->weights && requested_router->logits &&
        (requested_router->n_expert == 256 || requested_router->n_expert == 512) &&
        ((reinterpret_cast<uintptr_t>(requested_router->weights) & 3u) == 0) &&
        ((reinterpret_cast<uintptr_t>(requested_router->logits) & 3u) == 0)) router = *requested_router;
    const int blocks = hcp_blocks(m.T, router.weights != nullptr);
    if (!blocks) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            std::fprintf(stderr, "strata: STRATA_HC_PERSIST requested but this runtime/device has no supported "
                                 "cooperative launch (requires gfx906, HIP >= 6.4 and occupancy); using the existing HC kernels\n");
        return false;
    }
    void* args[] = {&m, &router};
    const auto error = hipLaunchCooperativeKernel(hcp_production_function(m.T, router.weights != nullptr), dim3(blocks), dim3(THREADS),
                                                  args, (size_t) m.T * 1280 * sizeof(float), stream);
    // A failed capture/launch can invalidate the stream. Never silently launch the
    // fallback into that stream or count a request as successful after this point.
    if (error != hipSuccess) {
        std::fprintf(stderr, "strata: persistent HC cooperative launch failed: %s\n", hipGetErrorString(error));
        std::exit(1);
    }
    if (router_written) *router_written = router.weights != nullptr;
    ++hcp_launch_count;
    static std::atomic<bool> announced{false};
    if (!announced.exchange(true))
        std::fprintf(stderr, "strata: experimental persistent HC enabled (gfx906, T<=4, BF16); "
                             "hardware parity and performance must be validated\n");
    return true;
}
