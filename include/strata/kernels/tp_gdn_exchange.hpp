#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

namespace strata::kernels {

// Two-rank GDN transport for the validated N2560/F640/K10/V6144 geometry.
// CUDA/HIP only. Each call launches one allocation-free, graph-capturable kernel
// on the PRODUCING rank's stream. It writes this rank's disjoint slots in BOTH
// full buffers; the other rank must call the same operation with its own inputs.
// No arithmetic, quantization, padding reads, or truncation of trailing words.
//
// Preconditions checked on the host: T in [1,8], rank 0 or 1, nonnull 4-byte
// aligned pointers, nonwrapping and mutually disjoint buffer address ranges.
// Callers provide allocations covering ALL extents below, a stream on the source
// device, and enabled direct peer access to peer_full. Allocation/device ownership
// cannot be established from raw pointers here and must be checked by the owner.
//
// Ordering belongs to the caller: complete initial uploads/poisoning on BOTH
// devices before their first push, including work on legacy/default streams.
// Every writing thread system-fences its own writes before kernel completion.
// Record each producer's event AFTER its push, then make BOTH consumer streams
// wait for BOTH events before reading the assembled buffers. These inter-rank
// event edges remain outside rank-local graph capture. Do not reuse source or
// destination storage until previous readers/writers are ordered before reuse.
// Invalid descriptors throw before launch; no fallback or partial copy is used.

// src[T,3072] -> local_full[T,6144] and peer_full[T,6144]. Local value heads
// [0..23] map to global heads 16*(h/8) + 8*rank + h%8, with 128 floats/head.
void tp_gdn_push_y(const float* src, float* local_full, float* peer_full,
                   int T, int rank, void* stream);

// src[T,1280] -> rank's contiguous half of BOTH full [T,2560] output rows.
void tp_gdn_push_output(const float* src, float* local_full, float* peer_full,
                        int T, int rank, void* stream);

// Routed src[T*10,360 bytes] and shared src[T,360 bytes] contain native q8_1
// blocks for F/2=320. Copy rank's raw half into corresponding local/peer full
// rows of 720 bytes (F=640), preserving entry order. All six ranges are disjoint.
// The routed and shared transfers share one kernel and one completion event.
void tp_gdn_push_hidden(const uint8_t* routed_src, uint8_t* routed_local, uint8_t* routed_peer,
                        const uint8_t* shared_src, uint8_t* shared_local, uint8_t* shared_peer,
                        int T, int rank, void* stream);

// Ordered rank0 + rank1 reduction of full-width partials. Both devices MUST pass
// pointers in the same rank order, irrespective of which one is local. Source
// delivery must already be ordered by the producer events. No peer reads/spins
// or system fences are hidden here. Output must be distinct from both inputs.
/// Capturable bitwise full-partial peer publication. Each writing thread system-fences;
/// consumer must wait for the producing stream's completion event before reading.
void tp_gdn_publish_partial(const float* src, float* peer_inbox, int values, void* stream);

void tp_gdn_reduce_partials(const float* rank0, const float* rank1, float* full,
                            int values, void* stream);

// Flat graph protocol. These are plain integer objects with an explicit lifetime,
// not std::atomic representations. The owner placement-news a zero-initialized
// control in hipHostMallocMapped | hipHostMallocCoherent memory before device use.
// Host access MUST use lock-free __atomic builtins (acquire/release); GPU access
// uses HIP SYSTEM-scope acquire/release builtins. Every slot has one writer.
// Epoch changes only after both ranks drain; zero/wrap/replay are invalid.
// Abort slots are sticky: host publishes host_abort BEFORE draining a failed graph.
constexpr int kTpGdnProtocolPhases = 8;
struct alignas(64) TpGdnSignalSlot { uint64_t value = 0; };
struct TpGdnProtocolControl {
    TpGdnSignalSlot epoch;
    TpGdnSignalSlot ready[2][kTpGdnProtocolPhases];
    TpGdnSignalSlot abort[2];
    TpGdnSignalSlot host_abort;
};
static_assert(sizeof(TpGdnSignalSlot) == 64 && alignof(TpGdnSignalSlot) == 64);
static_assert(std::is_standard_layout<TpGdnProtocolControl>::value &&
              std::is_trivially_copyable<TpGdnProtocolControl>::value);

enum class TpGdnProtocolStatus : uint64_t { Pending = 0, Ready = 1, Aborted = 2, TimedOut = 3, InvalidEpoch = 4 };
struct TpGdnProtocolStamp {
    uint64_t epoch = 0;
    uint64_t status = 0;
    uint64_t polls = 0;
    uint64_t elapsed_ticks = 0;
};
// Each span contributes rows*half_words raw uint32 words to one packed inbox.
// run_words==half_words: ordinary row halves; run_words==1024 and half_words==3072:
// mapped GDN Y. Hidden uses two spans (routed rows=T*10, shared rows=T, half_words=90).
struct TpGdnPacketSpan {
    const void* src = nullptr;
    void* local_full = nullptr;
    int rows = 0;
    int half_words = 0;
    int run_words = 0;
};
struct TpGdnExchangePacket {
    TpGdnPacketSpan spans[2];
    int count = 0;
    void* peer_inbox = nullptr;
    const void* own_inbox = nullptr;
    size_t inbox_bytes = 0; // EXACT sum of packed source spans, not allocation capacity
};
// Compile-time capability only: gfx906 HIP 7.2.x with required compiler builtins.
// Allocation flags, actual coherent atomics and peer visibility require the
// model-free hardware preflight on the exact runtime before enabling FlatGraph.
bool tp_gdn_protocol_supported() noexcept;
// Inbox allocations MUST be hipDeviceMallocUncached, dedicated to this phase;
// source, full destinations, inboxes, signals and stamps must remain alive through
// replay. All payload, inbox, control and stamp ranges must be mutually disjoint;
// raw address overlaps are rejected. Calls perform no allocation or pointer-attribute/runtime queries.
void tp_gdn_packet_push(const TpGdnExchangePacket&, TpGdnProtocolControl*, int rank, void* stream);
// Stream ordered after ALL producer writes. Publishes ready[rank][phase], waits
// for the exact peer epoch or sticky abort. Deadline uses wall_clock64(), whose
// constant frequency is hipDeviceAttributeWallClockRate (kHz), NEVER core clockRate.
// timeout_ticks is positive and <= INT64_MAX; an independent finite poll cap also
// bounds the loop. On failure publishes this rank's abort before returning.
void tp_gdn_protocol_wait(TpGdnProtocolControl*, TpGdnProtocolStamp*, int rank, int phase,
                          uint64_t timeout_ticks, void* stream);
// Copies peer packed words into this rank's missing mapped half only after a
// successful matching stamp. If either rank/host aborted, or stamp/epoch is bad,
// zeros ALL local_full words (own and peer halves), without reading the inbox.
void tp_gdn_packet_unpack(const TpGdnExchangePacket&, TpGdnProtocolControl*,
                          const TpGdnProtocolStamp*, int rank, int phase, void* stream);

}  // namespace strata::kernels
