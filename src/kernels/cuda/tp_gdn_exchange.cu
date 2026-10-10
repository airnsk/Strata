#include "strata/kernels/tp_gdn_exchange.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
constexpr int kThreads = 256;
constexpr int kLocalY = 3072, kFullY = 6144, kHeadRun = 8 * 128;
constexpr int kLocalOutput = 1280, kFullOutput = 2560;
constexpr int kRoutes = 10, kHalfWords = 360 / sizeof(uint32_t), kFullWords = 2 * kHalfWords;
static_assert(sizeof(float) == sizeof(uint32_t), "GDN exchange requires 32-bit float storage");
static_assert(360 % sizeof(uint32_t) == 0, "hidden exchange must copy complete q8_1 rows");

struct Span { const void* data; size_t bytes; };
void validate(int tokens, int rank, const Span* spans, size_t count) {
    if (tokens < 1 || tokens > 8 || (rank != 0 && rank != 1))
        throw std::invalid_argument("TP GDN exchange requires T1..8 and rank 0 or 1");
    for (size_t i = 0; i < count; ++i) {
        const auto begin = reinterpret_cast<uintptr_t>(spans[i].data);
        if (!begin || begin % alignof(uint32_t) || !spans[i].bytes ||
            spans[i].bytes > std::numeric_limits<uintptr_t>::max() - begin)
            throw std::invalid_argument("TP GDN exchange null, unaligned or wrapping buffer span");
        const auto end = begin + spans[i].bytes;
        for (size_t j = 0; j < i; ++j) {
            const auto other = reinterpret_cast<uintptr_t>(spans[j].data);
            if (begin < other + spans[j].bytes && other < end)
                throw std::invalid_argument("TP GDN exchange overlapping buffer spans");
        }
    }
}
void launch_check() {
    const auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("TP GDN exchange launch: ") + cudaGetErrorString(status));
}

__global__ void push_y_kernel(const uint32_t* src, uint32_t* local, uint32_t* peer, int tokens, int rank) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= tokens * kLocalY) return;
    const int col = i % kLocalY;
    const int dst = (i / kLocalY) * kFullY + (col / kHeadRun) * (2 * kHeadRun) +
                    rank * kHeadRun + col % kHeadRun;
    const uint32_t value = src[i];
    local[dst] = value;
    peer[dst] = value;
    // A fence in a single leader cannot publish writes made by other threads.
    __threadfence_system();
}

__global__ void push_output_kernel(const uint32_t* src, uint32_t* local, uint32_t* peer, int tokens, int rank) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= tokens * kLocalOutput) return;
    const int dst = (i / kLocalOutput) * kFullOutput + rank * kLocalOutput + i % kLocalOutput;
    const uint32_t value = src[i];
    local[dst] = value;
    peer[dst] = value;
    __threadfence_system();
}

__global__ void push_hidden_kernel(const uint32_t* routed_src, uint32_t* routed_local, uint32_t* routed_peer,
                                    const uint32_t* shared_src, uint32_t* shared_local, uint32_t* shared_peer,
                                    int tokens, int rank) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int routed_words = tokens * kRoutes * kHalfWords;
    if (i >= routed_words + tokens * kHalfWords) return;
    const bool routed = i < routed_words;
    const int src_index = routed ? i : i - routed_words;
    const int dst = (src_index / kHalfWords) * kFullWords + rank * kHalfWords + src_index % kHalfWords;
    const uint32_t value = (routed ? routed_src : shared_src)[src_index];
    (routed ? routed_local : shared_local)[dst] = value;
    (routed ? routed_peer : shared_peer)[dst] = value;
    __threadfence_system();
}
}  // namespace

void tp_gdn_push_y(const float* src, float* local_full, float* peer_full, int T, int rank, void* stream) {
    // Validate T before deriving spans: signed invalid inputs must not wrap into byte extents.
    if (T < 1 || T > 8) throw std::invalid_argument("TP GDN Y exchange requires T1..8");
    const Span spans[] = {{src, size_t(T) * kLocalY * sizeof(float)},
                          {local_full, size_t(T) * kFullY * sizeof(float)},
                          {peer_full, size_t(T) * kFullY * sizeof(float)}};
    validate(T, rank, spans, 3);
    push_y_kernel<<<(T * kLocalY + kThreads - 1) / kThreads, kThreads, 0, (cudaStream_t) stream>>>(
        reinterpret_cast<const uint32_t*>(src), reinterpret_cast<uint32_t*>(local_full),
        reinterpret_cast<uint32_t*>(peer_full), T, rank);
    launch_check();
}

void tp_gdn_push_output(const float* src, float* local_full, float* peer_full, int T, int rank, void* stream) {
    if (T < 1 || T > 8) throw std::invalid_argument("TP GDN output exchange requires T1..8");
    const Span spans[] = {{src, size_t(T) * kLocalOutput * sizeof(float)},
                          {local_full, size_t(T) * kFullOutput * sizeof(float)},
                          {peer_full, size_t(T) * kFullOutput * sizeof(float)}};
    validate(T, rank, spans, 3);
    push_output_kernel<<<(T * kLocalOutput + kThreads - 1) / kThreads, kThreads, 0, (cudaStream_t) stream>>>(
        reinterpret_cast<const uint32_t*>(src), reinterpret_cast<uint32_t*>(local_full),
        reinterpret_cast<uint32_t*>(peer_full), T, rank);
    launch_check();
}

void tp_gdn_push_hidden(const uint8_t* routed_src, uint8_t* routed_local, uint8_t* routed_peer,
                        const uint8_t* shared_src, uint8_t* shared_local, uint8_t* shared_peer,
                        int T, int rank, void* stream) {
    if (T < 1 || T > 8) throw std::invalid_argument("TP GDN hidden exchange requires T1..8");
    const size_t half_bytes = size_t(T) * kHalfWords * sizeof(uint32_t);
    const Span spans[] = {{routed_src, kRoutes * half_bytes}, {routed_local, 2 * kRoutes * half_bytes},
                          {routed_peer, 2 * kRoutes * half_bytes}, {shared_src, half_bytes},
                          {shared_local, 2 * half_bytes}, {shared_peer, 2 * half_bytes}};
    validate(T, rank, spans, 6);
    const int words = T * (kRoutes + 1) * kHalfWords;
    push_hidden_kernel<<<(words + kThreads - 1) / kThreads, kThreads, 0, (cudaStream_t) stream>>>(
        reinterpret_cast<const uint32_t*>(routed_src), reinterpret_cast<uint32_t*>(routed_local),
        reinterpret_cast<uint32_t*>(routed_peer), reinterpret_cast<const uint32_t*>(shared_src),
        reinterpret_cast<uint32_t*>(shared_local), reinterpret_cast<uint32_t*>(shared_peer), T, rank);
    launch_check();
}

}  // namespace strata::kernels
