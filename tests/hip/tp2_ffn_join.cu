#include "tp2_ffn_join.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace {
__global__ void join_kernel(const float* a, const float* b, const float* w, float* rows,
                            float* out, int T, int K, int N, int g0, bool tp) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= T * N) return;
    const int t = i / N, col = i % N;
    float sum = 0.0f;
    for (int e = 0; e < K; ++e) {
        float v;
        if (tp) v = a[((t * K + e) * N) + col] + b[((t * K + e) * N) + col];
        else if (e < g0) v = a[((t * g0 + e) * N) + col];
        else v = b[((t * (K - g0) + e - g0) * N) + col];
        rows[((t * K + e) * N) + col] = v;
        // Test-only FP32 combine, identical in every benchmark mode. This is
        // not a claim of bitwise equality to either production combine backend.
        sum = __fadd_rn(sum, __fmul_rn(w[t * K + e], v));
    }
    out[i] = sum;
}
}
void tp2_ffn_join(const float* a, const float* b, const float* w, float* rows, float* out,
                  int T, int K, int N, int g0, bool tp, void* stream) {
    join_kernel<<<(T * N + 255) / 256, 256, 0, (cudaStream_t) stream>>>(a,b,w,rows,out,T,K,N,g0,tp);
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

namespace {
__global__ void input_push_kernel(const float* src, float* dst, int n) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n / 4) {
        // Integer vector transport preserves all input bit patterns without FP interpretation.
        reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
        // Each writer fences its own remote writes before kernel completion.
        // The consumer additionally waits for the producer stream's event.
        __threadfence_system();
    }
}
__global__ void rank_reduce_kernel(const float* rows, const float* w, float* dst,
                                   int T, int K, int N, int count, int first, bool peer) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= T * N) return;
    const int t = i / N, col = i % N;
    float sum = 0;
    for (int e = 0; e < count; ++e)
        sum = __fadd_rn(sum, __fmul_rn(w[t*K+first+e],rows[(t*count+e)*N+col]));
    dst[i] = sum;
    if (peer) __threadfence_system();
}
__global__ void sum_vectors_kernel(const float* a, const float* b, float* out, int n) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = __fadd_rn(a[i],b[i]);
}
void launch_check() {
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
}
void tp2_input_push(const float* src,float* dst,int n,void* stream) {
    if (n <= 0 || n % 4) throw std::invalid_argument("input push shape");
    input_push_kernel<<<(n/4+255)/256,256,0,(cudaStream_t)stream>>>(src,dst,n);
    launch_check();
}
void tp2_rank_reduce(const float* rows,const float* w,float* dst,int T,int K,int N,int count,int first,bool peer,void* stream) {
    rank_reduce_kernel<<<(T*N+255)/256,256,0,(cudaStream_t)stream>>>(rows,w,dst,T,K,N,count,first,peer);
    launch_check();
}
void tp2_sum_vectors(const float* a,const float* b,float* out,int n,void* stream) {
    sum_vectors_kernel<<<(n+255)/256,256,0,(cudaStream_t)stream>>>(a,b,out,n);
    launch_check();
}
