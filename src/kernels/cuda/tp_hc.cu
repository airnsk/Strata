// Output-row HC adaptation of fused_gr.cu. Preserve each complete dot's
// original FP32 operation order; no partial-dot reduction or requantization.
#include "strata/kernels/tp_hc.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
namespace strata::kernels {
namespace {
constexpr int N=2560, HC=4, D=N*HC, LR=320, HALF_N=N/2, HALF_LR=LR/2;
constexpr int THREADS=256, WARPS=THREADS/32, UPM_COLS=16;
// Four row warps retain the same32-lane arithmetic but expose40 owned-row
// blocks instead20 on60-CU MI50. Norm/up keep their original256 threads.
constexpr int DOWN_THREADS=128, DOWN_WARPS=DOWN_THREADS/32;
constexpr int LOCAL_DOWN_BLOCKS=HALF_LR/DOWN_WARPS;
struct Params {
    FusedGrArgs a[kFusedGrMaxT];
    int T=0, rank=0;
    float* xn=nullptr;
    float* local_lo=nullptr;
    const float* full_lo=nullptr;
    float* local_mixed=nullptr;
};
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// 8 bf16 packed in a uint4, unpacked to two float4 once when reused across tokens.
struct Bf16x8 { float4 a, b; };
__device__ __forceinline__ Bf16x8 unpack8(const uint4 w) {
    return {
        make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u),
                    __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xffff0000u)),
        make_float4(__uint_as_float(w.z << 16), __uint_as_float(w.z & 0xffff0000u),
                    __uint_as_float(w.w << 16), __uint_as_float(w.w & 0xffff0000u))
    };
}
__device__ __forceinline__ float dot8u(const Bf16x8& w, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(w.a.x, x0.x, acc);
    acc = fmaf(w.a.y, x0.y, acc);
    acc = fmaf(w.a.z, x0.z, acc);
    acc = fmaf(w.a.w, x0.w, acc);
    acc = fmaf(w.b.x, x1.x, acc);
    acc = fmaf(w.b.y, x1.y, acc);
    acc = fmaf(w.b.z, x1.z, acc);
    acc = fmaf(w.b.w, x1.w, acc);
    return acc;
}
__device__ __forceinline__ float dot8u_ptr(const Bf16x8& w, const float* x) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    return dot8u(w, x4[0], x4[1]);
}

// 8 bf16 packed in a uint4 against 8 16-byte-aligned floats (same 8-FMA order).
__device__ __forceinline__ float dot8(const uint4 w, const float* x) {
    return dot8u_ptr(unpack8(w), x);
}

__global__ void __launch_bounds__(THREADS) norm_kernel(Params m) {
    __shared__ float part[WARPS][HC];
    __shared__ float s_rs[HC];
    const FusedGrArgs& a = m.a[blockIdx.x];
    float* xn = m.xn + (size_t) blockIdx.x * D;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    float gw[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
    float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = t * 4; i < D; i += THREADS * 4) {
        const int c = i / N, d = i - c * N;
        float4 r = *reinterpret_cast<const float4*>(a.R + i);
        if (a.apply) {
            const float4 b = *reinterpret_cast<const float4*>(a.bo_prev + d);
            r.x = fmaf(b.x, gw[c], r.x); r.y = fmaf(b.y, gw[c], r.y);
            r.z = fmaf(b.z, gw[c], r.z); r.w = fmaf(b.w, gw[c], r.w);
        }
        if (a.apply) *reinterpret_cast<float4*>(a.R_out + i) = r;
        const float4 g = *reinterpret_cast<const float4*>(a.w_norm + i);
        float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) if (cc == c) ss[cc] += sq;
        *reinterpret_cast<float4*>(xn + i) = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
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
    for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
}

template <int TILEV, int MAX_T = kFusedGrMaxT>
__global__ void __launch_bounds__(DOWN_THREADS) down_kernel(Params m) {
    constexpr int TQ = TILEV / 8 / 32;      // uint4 weight chunks per lane per tile
    extern __shared__ __align__(16) float tile[];   // [T][TILEV]
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = m.T;
    const bool inject_block = blockIdx.x == LOCAL_DOWN_BLOCKS;
    const int local_row = inject_block ? warp : blockIdx.x * DOWN_WARPS + warp;
    const int row = inject_block ? warp : m.rank * HALF_LR + local_row;
    const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
    const uint16_t* wrow = (inject_block ? m.a[0].w_inject : m.a[0].w_down) + (size_t) (active ? row : 0) * D;
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
        for (int i = t; i < T * (TILEV / 4); i += DOWN_THREADS) {
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
            m.local_lo[k * HALF_LR + local_row] = x / (1.0f + __expf(-x));
        }
    }
}

__device__ __forceinline__ float xor8(float v) {
#pragma unroll
#if defined(__HIPCC__)
    for (int o = 4; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
#else
    for (int o = 4; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
#endif
    return v;
}
__global__ void __launch_bounds__(THREADS) up_kernel(Params m) {
    __shared__ __align__(16) float lo[kFusedGrMaxT][LR];
    __shared__ float g[kFusedGrMaxT][HC][UPM_COLS];
    const int t = threadIdx.x, j = t & 7, grp = t >> 3;
    const int T = m.T;
    const int d0 = m.rank * HALF_N + blockIdx.x * UPM_COLS;
    static_assert(LR == 40 * 8 && HC * UPM_COLS == 64 && THREADS == 256, "geometry");
    uint4 w[2][5];
    float rv[2] = {0.0f, 0.0f}, wn[2] = {0.0f, 0.0f}, rsc[2] = {0.0f, 0.0f}, bo[2] = {0.0f, 0.0f}, ip[2] = {0.0f, 0.0f};
    const bool apply = false; // pending write materialized by norm_kernel
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r = grp + 32 * p, c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
#pragma unroll
        for (int q = 0; q < 4; ++q) w[p][q] = __ldg(w4 + j + 8 * q);
        w[p][4] = __ldg(w4 + 32 + j);
        if (j < T) {
            const FusedGrArgs& a = m.a[j];
            rv[p] = (a.apply ? a.R_out : a.R)[i];
            wn[p] = a.w_norm[i];
            rsc[p] = a.rs[c];
            if (apply) { bo[p] = a.bo_prev[d0 + dd]; ip[p] = a.inj_prev[c]; }
        }
    }
    for (int i = t; i < T * LR; i += THREADS) lo[i / LR][i % LR] = m.full_lo[i];
    __syncthreads();
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int r = grp + 32 * p, c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        float mine = 0.0f;
#pragma unroll
        for (int k = 0; k < kFusedGrMaxT; ++k) {
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
        m.local_mixed[k * HALF_N + blockIdx.x * UPM_COLS + col] = s / (float) HC;
    }

}
struct Span { const void* p; size_t bytes; bool residual=false; };
void valid(Span s, size_t alignment=4) {
    const auto begin=reinterpret_cast<uintptr_t>(s.p);
    if(!begin || begin%alignment || !s.bytes || s.bytes>std::numeric_limits<uintptr_t>::max()-begin)
        throw std::invalid_argument("TP HC invalid/alignment/wrapping span");
}
bool overlaps(Span a,Span b) {
    const auto x=reinterpret_cast<uintptr_t>(a.p),y=reinterpret_cast<uintptr_t>(b.p);
    return x<y+b.bytes && y<x+a.bytes;
}
struct Access {
    Span read[64]{},write[40]{}; int nr=0,nw=0;
    void r(Span s,size_t alignment=4){valid(s,alignment);read[nr++]=s;}
    void w(Span s,size_t alignment=4){valid(s,alignment);write[nw++]=s;}
    void check()const{
        for(int i=0;i<nw;++i){
            for(int j=0;j<i;++j)if(overlaps(write[i],write[j]))throw std::invalid_argument("TP HC outputs overlap");
            for(int j=0;j<nr;++j)if(overlaps(write[i],read[j]) &&
                !(write[i].residual && read[j].residual && write[i].p==read[j].p && write[i].bytes==read[j].bytes))
                throw std::invalid_argument("TP HC input/output overlap");
        }
    }
};
Params parameters(const FusedGrArgs* args,int T,int rank,void* stream) {
    if(!args || !stream || T<1 || T>8 || (rank!=0&&rank!=1))
        throw std::invalid_argument("TP HC requires explicit stream,T1..8,rank0/1");
    const char* v3=std::getenv("STRATA_GR_V3");
    const char* split=std::getenv("STRATA_GR_SPLIT");
    if((v3 && std::atoi(v3)!=0) || (split && split[0]=='1' && split[1]=='\0'))
        throw std::invalid_argument("TP HC exact BF16 rows do not admit output-changing GR_V3/GR_SPLIT policies");
    Params m;m.T=T;m.rank=rank;
    for(int t=0;t<T;++t){
        const auto& a=args[t];
        if(!std::isfinite(a.eps)||a.eps<=0 || a.q8_down||a.q8_up||a.q8_inject||a.q8_mixed||a.q8_cnt ||
           a.w_down!=args[0].w_down||a.w_up!=args[0].w_up||a.w_inject!=args[0].w_inject||a.w_norm!=args[0].w_norm)
            throw std::invalid_argument("TP HC requires canonical BF16 shared weights and positive epsilon; Q8 overrides refused");
        m.a[t]=a;
    }
    return m;
}
void launched(){const auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(std::string("TP HC launch: ")+cudaGetErrorString(e));}
} // namespace
void tp_hc_down(const FusedGrArgs* args,int T,float* xn,float* local_lo,int rank,void* stream) {
    auto m=parameters(args,T,rank,stream);m.xn=xn;m.local_lo=local_lo;
    Access a;
    a.r({args[0].w_norm,D*sizeof(float)},16);
    a.r({args[0].w_down,LR*D*sizeof(uint16_t)},16);
    if(args[0].w_inject)a.r({args[0].w_inject,HC*D*sizeof(uint16_t)},16);
    a.w({xn,static_cast<size_t>(T)*D*sizeof(float)},16);
    a.w({local_lo,static_cast<size_t>(T)*HALF_LR*sizeof(float)},16);
    for(int t=0;t<T;++t){const auto& x=args[t];
        a.r({x.R,D*sizeof(float),true},16);a.w({x.rs,HC*sizeof(float)});
        if(x.w_inject)a.w({x.inject_out,HC*sizeof(float)});
        if(x.apply){a.r({x.bo_prev,N*sizeof(float)},16);a.r({x.inj_prev,HC*sizeof(float)});a.w({x.R_out,D*sizeof(float),true},16);}
    }
    a.check();
    auto cs=static_cast<cudaStream_t>(stream);
    norm_kernel<<<T,THREADS,0,cs>>>(m);
    // TILE1280 preserves lane chunk order and fits T8 in40KiB LDS on MI50.
    down_kernel<1280><<<LOCAL_DOWN_BLOCKS+1,DOWN_THREADS,static_cast<size_t>(T)*1280*sizeof(float),cs>>>(m);
    launched();
}
void tp_hc_up(const FusedGrArgs* args,int T,const float* full_lo,float* local_mixed,int rank,void* stream) {
    auto m=parameters(args,T,rank,stream);m.full_lo=full_lo;m.local_mixed=local_mixed;
    Access a;
    a.r({args[0].w_norm,D*sizeof(float)},16);a.r({args[0].w_up,D*LR*sizeof(uint16_t)},16);
    a.r({full_lo,static_cast<size_t>(T)*LR*sizeof(float)},16);
    a.w({local_mixed,static_cast<size_t>(T)*HALF_N*sizeof(float)},16);
    for(int t=0;t<T;++t){const auto& x=args[t];a.r({x.apply?x.R_out:x.R,D*sizeof(float)},16);a.r({x.rs,HC*sizeof(float)});}
    a.check();
    up_kernel<<<HALF_N/UPM_COLS,THREADS,0,static_cast<cudaStream_t>(stream)>>>(m);
    launched();
}
} // namespace strata::kernels
