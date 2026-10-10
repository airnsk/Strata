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
