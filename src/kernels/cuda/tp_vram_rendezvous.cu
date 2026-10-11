#include "strata/kernels/tp_vram_rendezvous.hpp"

#include <limits>
#include <stdexcept>
#include <string>

// Do not infer CUDA fence semantics from a compatibility macro. This experiment
// deliberately names the AMD system scope and compiler fence at every boundary.
#if defined(STRATA_HIP_GFX906) && defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#if HIP_VERSION_MAJOR == 7 && HIP_VERSION_MINOR == 2 && \
    __has_builtin(__hip_atomic_load) && __has_builtin(__hip_atomic_store)
#define STRATA_TP_VRAM_RENDEZVOUS_SUPPORTED 1
// Target-only builtin queries can be false in HIP's host pass. Keep the launcher
// guard identical in both passes; verify the known HIP 7.2 fence on the GPU pass.
#if defined(__HIP_DEVICE_COMPILE__)
#if !defined(__gfx906__) || !__has_builtin(__builtin_amdgcn_fence)
#error "TP VRAM rendezvous requires a gfx906 device target with AMD fence support"
#endif
#endif
#endif
#endif
#ifndef STRATA_TP_VRAM_RENDEZVOUS_SUPPORTED
#define STRATA_TP_VRAM_RENDEZVOUS_SUPPORTED 0
#endif

namespace strata::kernels {
namespace {

struct AddressSpan { uintptr_t begin; size_t bytes; };

void add_span(AddressSpan* spans, int& count, const void* pointer,
              size_t bytes, size_t alignment) {
    const auto begin = reinterpret_cast<uintptr_t>(pointer);
    if (!begin || begin % alignment || !bytes ||
        bytes > std::numeric_limits<uintptr_t>::max() - begin)
        throw std::invalid_argument("TP VRAM rendezvous null, unaligned or wrapping span");
    for (int i = 0; i < count; ++i) {
        if (begin < spans[i].begin + spans[i].bytes &&
            spans[i].begin < begin + bytes)
            throw std::invalid_argument("TP VRAM rendezvous overlapping payload/metadata spans");
    }
    spans[count++] = {begin, bytes};
}

void validate(const TpVramRendezvousPacket& packet,
             const TpVramRendezvousControl* own, const TpVramRendezvousControl* peer,
             const TpVramRendezvousStamp* stamp, int rank, uint64_t timeout_ticks) {
    if ((rank != 0 && rank != 1) || !timeout_ticks ||
        timeout_ticks > uint64_t(std::numeric_limits<int64_t>::max()))
        throw std::invalid_argument("TP VRAM rendezvous rank or wall-clock timeout");
    if (packet.count < 1 || packet.count > 2 ||
        (packet.mode != TpVramRendezvousMode::GatherU32 &&
         packet.mode != TpVramRendezvousMode::SumF32) ||
        (packet.mode == TpVramRendezvousMode::SumF32 && packet.count != 1))
        throw std::invalid_argument("TP VRAM rendezvous mode or span count");
    AddressSpan spans[9]; // Two source/destination pairs, two rings, three metadata objects.
    int count = 0;
    uint64_t words = 0;
    for (int i = 0; i < packet.count; ++i) {
        const auto& span = packet.spans[i];
        if (span.rows <= 0 || span.half_words <= 0 || span.run_words <= 0 ||
            span.half_words % span.run_words ||
            (packet.mode == TpVramRendezvousMode::SumF32 && span.run_words != span.half_words))
            throw std::invalid_argument("TP VRAM rendezvous row/run dimensions");
        const uint64_t n = uint64_t(span.rows) * uint64_t(span.half_words);
        if (n > kTpVramRendezvousMaxWords || words + n > kTpVramRendezvousMaxWords)
            throw std::invalid_argument("TP VRAM rendezvous exceeds bounded model-free packet size");
        words += n;
        const size_t bytes = size_t(n) * sizeof(uint32_t);
        const size_t dst_bytes = packet.mode == TpVramRendezvousMode::GatherU32 ? 2 * bytes : bytes;
        add_span(spans, count, span.src, bytes, alignof(uint32_t));
        add_span(spans, count, span.dst, dst_bytes, alignof(uint32_t));
    }
    if (packet.inbox_words != words)
        throw std::invalid_argument("TP VRAM rendezvous inbox_words must equal the exact source word sum");
    const size_t ring_bytes = kTpVramRendezvousSlots * size_t(kTpVramRendezvousMaxWords) * sizeof(uint32_t);
    add_span(spans, count, packet.own_inbox, ring_bytes, alignof(uint32_t));
    add_span(spans, count, packet.peer_inbox, ring_bytes, alignof(uint32_t));
    add_span(spans, count, own, sizeof(*own), alignof(TpVramRendezvousControl));
    add_span(spans, count, peer, sizeof(*peer), alignof(TpVramRendezvousControl));
    add_span(spans, count, stamp, sizeof(*stamp), alignof(TpVramRendezvousStamp));
}

#if STRATA_TP_VRAM_RENDEZVOUS_SUPPORTED
constexpr int kThreads = 256;

__device__ __forceinline__ uint64_t signal_load(const TpVramSignalSlot& signal) {
    return __hip_atomic_load(&signal.value, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void signal_store(TpVramSignalSlot& signal, uint64_t value) {
    __hip_atomic_store(&signal.value, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void system_release() {
    // LLVM's empty synchronization-scope name is SYSTEM, as used by ROCm
    // 7.2.4 amd_device_functions.h::__threadfence_system. It is not "system".
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
}
__device__ __forceinline__ void system_acquire() {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}
__device__ __forceinline__ bool failed(const TpVramRendezvousControl* own) {
    return signal_load(own->local_failure) != 0 || signal_load(own->peer_failure) != 0;
}
__device__ __forceinline__ uint64_t elapsed(uint64_t started) {
    return static_cast<uint64_t>(wall_clock64()) - started;
}

// All loads, including abort checks, target the executing device's uncached VRAM.
// The shared poll counter/deadline cover both ring-reuse and ready waits together.
__device__ TpVramRendezvousStatus wait_for(const TpVramSignalSlot& signal, uint64_t expected,
                                         const TpVramRendezvousControl* own,
                                         uint64_t started, uint64_t timeout_ticks,
                                         uint64_t& polls) {
    for (;;) {
        if (failed(own)) return TpVramRendezvousStatus::Aborted;
        const uint64_t value = signal_load(signal);
        if (value == expected) return TpVramRendezvousStatus::Ready;
        if (value > expected) return TpVramRendezvousStatus::InvalidEpoch;
        ++polls;
        if (polls >= kTpVramRendezvousPollLimit || elapsed(started) >= timeout_ticks)
            return TpVramRendezvousStatus::TimedOut;
        __builtin_amdgcn_s_sleep(1);
    }
}

__device__ __forceinline__ int destination(int source, const TpVramRendezvousSpan& span, int rank) {
    const int col = source % span.half_words;
    return (source / span.half_words) * (2 * span.half_words) +
           (col / span.run_words) * (2 * span.run_words) + rank * span.run_words + col % span.run_words;
}

__global__ void rendezvous_kernel(TpVramRendezvousPacket packet,
                                  TpVramRendezvousControl* own,
                                  TpVramRendezvousControl* peer,
                                  TpVramRendezvousStamp* stamp, int rank,
                                  uint64_t timeout_ticks) {
    __shared__ uint64_t epoch, started, polls;
    __shared__ int slot;
    __shared__ TpVramRendezvousStatus status;
    if (threadIdx.x == 0) {
        started = static_cast<uint64_t>(wall_clock64());
        polls = 0;
        epoch = signal_load(own->epoch);
        slot = 0;
        status = TpVramRendezvousStatus::Ready;
        if (failed(own)) status = TpVramRendezvousStatus::Aborted;
        else if (epoch == UINT64_MAX) status = TpVramRendezvousStatus::InvalidEpoch;
        else {
            ++epoch; // Single device-local writer; graph replay never captures an epoch value.
            signal_store(own->epoch, epoch);
            slot = static_cast<int>((epoch - 1) % kTpVramRendezvousSlots);
            // An early peer may already have published THIS epoch; that is valid.
            if (signal_load(own->ready[slot]) > epoch)
                status = TpVramRendezvousStatus::InvalidEpoch;
            else {
                const uint64_t reusable = epoch > kTpVramRendezvousSlots ?
                                          epoch - kTpVramRendezvousSlots : 0;
                status = wait_for(own->consumed[slot], reusable, own, started, timeout_ticks, polls);
            }
        }
    }
    __syncthreads();
    if (status == TpVramRendezvousStatus::Ready) {
        const int offset = slot * kTpVramRendezvousMaxWords;
        int packed = 0;
        for (int s = 0; s < packet.count; ++s) {
            const auto span = packet.spans[s];
            const int n = span.rows * span.half_words;
            for (int i = static_cast<int>(threadIdx.x); i < n; i += kThreads)
                packet.peer_inbox[offset + packed + i] = static_cast<const uint32_t*>(span.src)[i];
            packed += n;
        }
        // Every producer orders its OWN peer writes, then all producers join the
        // leader before it publishes the token. A leader-only fence is insufficient.
        system_release();
        __syncthreads();
        if (threadIdx.x == 0) {
            if (failed(own)) status = TpVramRendezvousStatus::Aborted;
            else {
                signal_store(peer->ready[slot], epoch);
                status = wait_for(own->ready[slot], epoch, own, started, timeout_ticks, polls);
            }
        }
        __syncthreads();
        // Uncached inbox and flags are mandatory, not an implicit cache flush
        // assumption. Every reader also names SYSTEM acquire after CTA handoff.
        system_acquire();
        if (status == TpVramRendezvousStatus::Ready) {
            int packed_read = 0;
            for (int s = 0; s < packet.count; ++s) {
                const auto span = packet.spans[s];
                const int n = span.rows * span.half_words;
                const auto* source = static_cast<const uint32_t*>(span.src);
                auto* output = static_cast<uint32_t*>(span.dst);
                for (int i = static_cast<int>(threadIdx.x); i < n; i += kThreads) {
                    const uint32_t local_bits = source[i];
                    const uint32_t peer_bits = packet.own_inbox[offset + packed_read + i];
                    if (packet.mode == TpVramRendezvousMode::GatherU32) {
                        output[destination(i, span, rank)] = local_bits;
                        output[destination(i, span, 1 - rank)] = peer_bits;
                    } else {
                        const float rank0 = __uint_as_float(rank == 0 ? local_bits : peer_bits);
                        const float rank1 = __uint_as_float(rank == 0 ? peer_bits : local_bits);
                        output[i] = __float_as_uint(__fadd_rn(rank0, rank1));
                    }
                }
                packed_read += n;
            }
        }
        // No early thread return: acknowledgement must follow ALL inbox reads.
        // As with publication, each participant orders its own memory accesses
        // before the CTA hands them to the leader's SYSTEM release-store.
        system_release();
        __syncthreads();
        if (threadIdx.x == 0 && status == TpVramRendezvousStatus::Ready) {
            if (failed(own)) status = TpVramRendezvousStatus::Aborted;
            else signal_store(peer->consumed[slot], epoch);
        }
    }
    __syncthreads();
    if (status != TpVramRendezvousStatus::Ready) {
        if (threadIdx.x == 0) {
            // Preserve the first locally observed failure. No other writer uses
            // these fields; even the failure path never needs remote RMW atomics.
            if (signal_load(own->local_failure) == 0)
                signal_store(own->local_failure, static_cast<uint64_t>(status));
            signal_store(peer->peer_failure, signal_load(own->local_failure));
        }
        for (int s = 0; s < packet.count; ++s) {
            const auto span = packet.spans[s];
            const int n = span.rows * span.half_words *
                          (packet.mode == TpVramRendezvousMode::GatherU32 ? 2 : 1);
            for (int i = static_cast<int>(threadIdx.x); i < n; i += kThreads)
                static_cast<uint32_t*>(span.dst)[i] = kTpVramRendezvousPoison;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        stamp->epoch = epoch;
        stamp->status = static_cast<uint64_t>(status);
        stamp->polls = polls;
        stamp->elapsed_ticks = elapsed(started);
    }
}
#endif

} // namespace

bool tp_vram_rendezvous_supported() noexcept { return STRATA_TP_VRAM_RENDEZVOUS_SUPPORTED != 0; }

void tp_vram_rendezvous(const TpVramRendezvousPacket& packet,
                        TpVramRendezvousControl* own, TpVramRendezvousControl* peer,
                        TpVramRendezvousStamp* stamp, int rank,
                        uint64_t timeout_ticks, void* stream) {
    validate(packet, own, peer, stamp, rank, timeout_ticks);
#if STRATA_TP_VRAM_RENDEZVOUS_SUPPORTED
    rendezvous_kernel<<<1, kThreads, 0, static_cast<hipStream_t>(stream)>>>(
        packet, own, peer, stamp, rank, timeout_ticks);
    const auto error = hipGetLastError();
    if (error != hipSuccess)
        throw std::runtime_error(std::string("TP VRAM rendezvous launch: ") + hipGetErrorString(error));
#else
    (void)stream;
    throw std::runtime_error("TP VRAM rendezvous requires gfx906 HIP 7.2 with SYSTEM atomics and fences");
#endif
}

} // namespace strata::kernels
