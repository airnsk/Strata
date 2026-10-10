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

namespace strata::kernels {
namespace {
__global__ void publish_partial_kernel(const uint32_t* src, uint32_t* peer, int values) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < values) { peer[i] = src[i]; __threadfence_system(); }
}
__global__ void reduce_partials_kernel(const float* rank0, const float* rank1, float* full, int values) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < values) full[i] = __fadd_rn(rank0[i], rank1[i]);
}
}
void tp_gdn_publish_partial(const float* src, float* peer_inbox, int values, void* stream) {
    if (values <= 0 || values > 8 * 2560 || values % 2560)
        throw std::invalid_argument("TP GDN partial publication requires T1..8 full N2560 rows");
    const Span spans[] = {{src, size_t(values) * sizeof(float)}, {peer_inbox, size_t(values) * sizeof(float)}};
    validate(values / 2560, 0, spans, 2);
    publish_partial_kernel<<<(values + kThreads - 1) / kThreads, kThreads, 0, (cudaStream_t) stream>>>(
        reinterpret_cast<const uint32_t*>(src), reinterpret_cast<uint32_t*>(peer_inbox), values);
    launch_check();
}
void tp_gdn_reduce_partials(const float* rank0, const float* rank1, float* full, int values, void* stream) {
    if (values <= 0 || values > 8 * 2560 || values % 2560 != 0)
        throw std::invalid_argument("TP GDN reduction requires T1..8 full N2560 rows");
    const size_t bytes = size_t(values) * sizeof(float);
    const Span spans[] = {{rank0, bytes}, {rank1, bytes}, {full, bytes}};
    validate(1, 0, spans, 3);
    reduce_partials_kernel<<<(values + kThreads - 1) / kThreads, kThreads, 0, (cudaStream_t) stream>>>(
        rank0, rank1, full, values);
    launch_check();
}
}  // namespace strata::kernels

// Source contract verified against ROCm/clr rocm-7.2.4 amd_hip_atomic.h and
// amd_device_functions.h, plus ROCm/HIP hip_runtime_api.h. This guard only
// establishes compilation support; the session must pass the hardware preflight.
#if defined(STRATA_HIP_GFX906) && defined(__HIPCC__)
#include <hip/hip_version.h>
#if HIP_VERSION_MAJOR == 7 && HIP_VERSION_MINOR == 2 && \
    __has_builtin(__hip_atomic_load) && __has_builtin(__hip_atomic_store)
#define STRATA_TP_GDN_SYSTEM_PROTOCOL 1
#endif
#endif
#ifndef STRATA_TP_GDN_SYSTEM_PROTOCOL
#define STRATA_TP_GDN_SYSTEM_PROTOCOL 0
#endif

namespace strata::kernels {
namespace {
int packet_words(const TpGdnExchangePacket& packet) {
    if (packet.count < 1 || packet.count > 2)
        throw std::invalid_argument("TP GDN packet needs one or two spans");
    Span spans[6];
    size_t n = 0, words = 0;
    for (int i = 0; i < packet.count; ++i) {
        const auto& s = packet.spans[i];
        if (s.rows <= 0 || s.half_words <= 0 || s.run_words <= 0 || s.half_words % s.run_words != 0)
            throw std::invalid_argument("TP GDN packet row or run dimensions");
        const uint64_t w = uint64_t(s.rows) * uint64_t(s.half_words);
        // Both zero-fill and copy launches use signed int indices; no rounding overflow.
        if (w > uint64_t(std::numeric_limits<int>::max() - kThreads) / 2 ||
            words > size_t(std::numeric_limits<int>::max() - kThreads) / 2 - size_t(w))
            throw std::overflow_error("TP GDN packet index range");
        const size_t bytes = size_t(w) * sizeof(uint32_t);
        spans[n++] = {s.src, bytes};
        spans[n++] = {s.local_full, 2 * bytes};
        words += size_t(w);
    }
    if (packet.inbox_bytes != words * sizeof(uint32_t))
        throw std::invalid_argument("TP GDN packet inbox extent must match all source words");
    spans[n++] = {packet.peer_inbox, packet.inbox_bytes};
    spans[n++] = {packet.own_inbox, packet.inbox_bytes};
    validate(1, 0, spans, n);
    return static_cast<int>(words);
}
void protocol_arguments(const TpGdnProtocolControl* control, int rank, int phase = 0,
                        const TpGdnProtocolStamp* stamp = nullptr) {
    if (rank < 0 || rank > 1 || phase < 0 || phase >= kTpGdnProtocolPhases ||
        !control || reinterpret_cast<uintptr_t>(control) % alignof(TpGdnProtocolControl) ||
        sizeof(*control) > std::numeric_limits<uintptr_t>::max() - reinterpret_cast<uintptr_t>(control))
        throw std::invalid_argument("TP GDN protocol control, rank or phase");
    if (stamp && (reinterpret_cast<uintptr_t>(stamp) % alignof(TpGdnProtocolStamp) ||
        sizeof(*stamp) > std::numeric_limits<uintptr_t>::max() - reinterpret_cast<uintptr_t>(stamp)))
        throw std::invalid_argument("TP GDN protocol stamp alignment or extent");
    if (stamp) {
        const Span meta[] = {{control, sizeof(*control)}, {stamp, sizeof(*stamp)}};
        validate(1, 0, meta, 2);
    }
}
void packet_protocol_disjoint(const TpGdnExchangePacket& packet, const TpGdnProtocolControl* control,
                              const TpGdnProtocolStamp* stamp = nullptr) {
    // packet_words already checked all shapes/overflow; include metadata in the
    // address-range check so payload stores cannot corrupt epoch or status words.
    Span spans[8];
    size_t n = 0;
    for (int i = 0; i < packet.count; ++i) {
        const auto& p = packet.spans[i];
        const size_t bytes = size_t(p.rows) * size_t(p.half_words) * sizeof(uint32_t);
        spans[n++] = {p.src, bytes};
        spans[n++] = {p.local_full, 2 * bytes};
    }
    spans[n++] = {packet.peer_inbox, packet.inbox_bytes};
    spans[n++] = {packet.own_inbox, packet.inbox_bytes};
    spans[n++] = {control, sizeof(*control)};
    if (stamp) spans[n++] = {stamp, sizeof(*stamp)};
    validate(1, 0, spans, n);
}
#if STRATA_TP_GDN_SYSTEM_PROTOCOL
static_assert(__atomic_always_lock_free(sizeof(uint64_t), nullptr), "host protocol uint64 atomics must be lock-free");
constexpr uint64_t kProtocolPollLimit = uint64_t{1} << 24;

__device__ __forceinline__ uint64_t signal_load(const TpGdnSignalSlot& slot) {
    return __hip_atomic_load(&slot.value, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void signal_store(TpGdnSignalSlot& slot, uint64_t value) {
    __hip_atomic_store(&slot.value, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ bool protocol_aborted(const TpGdnProtocolControl* c) {
    return signal_load(c->host_abort) != 0 || signal_load(c->abort[0]) != 0 || signal_load(c->abort[1]) != 0;
}
__device__ __forceinline__ int packet_destination(int source, const TpGdnPacketSpan& s, int rank) {
    const int col = source % s.half_words;
    return (source / s.half_words) * (2 * s.half_words) +
           (col / s.run_words) * (2 * s.run_words) + rank * s.run_words + col % s.run_words;
}

__global__ void packet_push_kernel(TpGdnExchangePacket packet, TpGdnProtocolControl* control, int rank, int words) {
    __shared__ int active;
    if (threadIdx.x == 0) active = signal_load(control->epoch) != 0 && !protocol_aborted(control);
    __syncthreads();
    if (!active) return;
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= words) return;
    int in_span = i, span = 0;
    const int first = packet.spans[0].rows * packet.spans[0].half_words;
    if (in_span >= first) { in_span -= first; span = 1; }
    const auto s = packet.spans[span];
    const uint32_t value = static_cast<const uint32_t*>(s.src)[in_span];
    static_cast<uint32_t*>(s.local_full)[packet_destination(in_span, s, rank)] = value;
    static_cast<uint32_t*>(packet.peer_inbox)[i] = value;
    // ALL producing threads publish their own writes; a later kernel leader's
    // fence alone cannot substitute for these fences.
    __threadfence_system();
}

__global__ void protocol_wait_kernel(TpGdnProtocolControl* control, TpGdnProtocolStamp* stamp,
                                     int rank, int phase, uint64_t timeout_ticks) {
    if (threadIdx.x != 0) return;
    const uint64_t started = static_cast<uint64_t>(wall_clock64());
    const uint64_t epoch = signal_load(control->epoch);
    uint64_t polls = 0;
    TpGdnProtocolStatus status = TpGdnProtocolStatus::Ready;
    if (protocol_aborted(control)) status = TpGdnProtocolStatus::Aborted;
    else if (epoch == 0 || signal_load(control->ready[rank][phase]) >= epoch)
        status = TpGdnProtocolStatus::InvalidEpoch;
    else {
        // Stream ordering completes the preceding fenced push before this release.
        signal_store(control->ready[rank][phase], epoch);
        for (;;) {
            if (protocol_aborted(control)) { status = TpGdnProtocolStatus::Aborted; break; }
            if (signal_load(control->epoch) != epoch) { status = TpGdnProtocolStatus::InvalidEpoch; break; }
            const uint64_t peer = signal_load(control->ready[1 - rank][phase]);
            if (peer == epoch) break;
            if (peer > epoch) { status = TpGdnProtocolStatus::InvalidEpoch; break; }
            ++polls;
            const uint64_t elapsed = static_cast<uint64_t>(wall_clock64()) - started;
            if (elapsed >= timeout_ticks || polls >= kProtocolPollLimit) {
                status = TpGdnProtocolStatus::TimedOut; break;
            }
            __builtin_amdgcn_s_sleep(1);
        }
    }
    if (status != TpGdnProtocolStatus::Ready) signal_store(control->abort[rank], uint64_t{1});
    stamp->epoch = epoch;
    stamp->status = static_cast<uint64_t>(status);
    stamp->polls = polls;
    stamp->elapsed_ticks = static_cast<uint64_t>(wall_clock64()) - started;
}

__global__ void packet_unpack_kernel(TpGdnExchangePacket packet, TpGdnProtocolControl* control,
                                     const TpGdnProtocolStamp* stamp, int rank, int phase, int words) {
    __shared__ int good;
    if (threadIdx.x == 0) {
        const uint64_t epoch = signal_load(control->epoch);
        good = epoch != 0 && stamp->epoch == epoch &&
               stamp->status == static_cast<uint64_t>(TpGdnProtocolStatus::Ready) &&
               signal_load(control->ready[rank][phase]) == epoch &&
               signal_load(control->ready[1 - rank][phase]) == epoch && !protocol_aborted(control);
    }
    __syncthreads();
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (good) {
        if (i >= words) return;
        int in_span = i, span = 0;
        const int first = packet.spans[0].rows * packet.spans[0].half_words;
        if (in_span >= first) { in_span -= first; span = 1; }
        const auto s = packet.spans[span];
        static_cast<uint32_t*>(s.local_full)[packet_destination(in_span, s, 1 - rank)] =
            static_cast<const uint32_t*>(packet.own_inbox)[i];
    } else {
        // Do not consume a partially published or stale inbox on ANY failure.
        if (i >= 2 * words) return;
        int in_span = i, span = 0;
        const int first = 2 * packet.spans[0].rows * packet.spans[0].half_words;
        if (in_span >= first) { in_span -= first; span = 1; }
        static_cast<uint32_t*>(packet.spans[span].local_full)[in_span] = 0;
    }
}
#endif  // STRATA_TP_GDN_SYSTEM_PROTOCOL
}  // namespace

bool tp_gdn_protocol_supported() noexcept { return STRATA_TP_GDN_SYSTEM_PROTOCOL != 0; }

void tp_gdn_packet_push(const TpGdnExchangePacket& packet, TpGdnProtocolControl* control, int rank, void* stream) {
    protocol_arguments(control, rank);
    const int words = packet_words(packet);
    packet_protocol_disjoint(packet, control);
#if STRATA_TP_GDN_SYSTEM_PROTOCOL
    packet_push_kernel<<<(words + kThreads - 1) / kThreads, kThreads, 0, (cudaStream_t) stream>>>(packet, control, rank, words);
    launch_check();
#else
    (void)words; (void)stream;
    throw std::runtime_error("TP GDN flat protocol requires gfx906 HIP 7.2 with SYSTEM atomic builtins");
#endif
}

void tp_gdn_protocol_wait(TpGdnProtocolControl* control, TpGdnProtocolStamp* stamp, int rank, int phase,
                          uint64_t timeout_ticks, void* stream) {
    protocol_arguments(control, rank, phase, stamp);
    if (!stamp || !timeout_ticks || timeout_ticks > uint64_t(std::numeric_limits<int64_t>::max()))
        throw std::invalid_argument("TP GDN protocol requires stamp and positive bounded wall-clock timeout");
#if STRATA_TP_GDN_SYSTEM_PROTOCOL
    protocol_wait_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(control, stamp, rank, phase, timeout_ticks);
    launch_check();
#else
    (void)stream;
    throw std::runtime_error("TP GDN flat protocol requires gfx906 HIP 7.2 with SYSTEM atomic builtins");
#endif
}

void tp_gdn_packet_unpack(const TpGdnExchangePacket& packet, TpGdnProtocolControl* control,
                          const TpGdnProtocolStamp* stamp, int rank, int phase, void* stream) {
    protocol_arguments(control, rank, phase, stamp);
    if (!stamp) throw std::invalid_argument("TP GDN unpack requires a protocol stamp");
    const int words = packet_words(packet);
    packet_protocol_disjoint(packet, control, stamp);
#if STRATA_TP_GDN_SYSTEM_PROTOCOL
    packet_unpack_kernel<<<(2 * words + kThreads - 1) / kThreads, kThreads, 0, (cudaStream_t) stream>>>(
        packet, control, stamp, rank, phase, words);
    launch_check();
#else
    (void)words; (void)stream;
    throw std::runtime_error("TP GDN flat protocol requires gfx906 HIP 7.2 with SYSTEM atomic builtins");
#endif
}
}  // namespace strata::kernels
