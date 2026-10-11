#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace strata::kernels {

// EXPERIMENT ONLY: model-free probe and opt-in isolated hybrid-layer owner.
// Requires the model-free admission on the target pair; not an engine transport.
// One CTA per rank combines peer push, publication, local polling and consumption.
// The conservative two-slot ring includes a consumed acknowledgement; its extra
// peer stores are intentional and are not a claim of optimal performance.
// The fused CTA also rereads its source after publication. Its aim is fewer
// kernel boundaries and no host-control polling, not a proven bandwidth gain.
constexpr int kTpVramRendezvousSlots = 2;
constexpr int kTpVramRendezvousMaxWords = 32768; // Includes every existing T1..8 GDN packet.
constexpr uint32_t kTpVramRendezvousPoison = 0x7fc00000u;
constexpr uint64_t kTpVramRendezvousPollLimit = uint64_t{1} << 24;

enum class TpVramRendezvousMode : uint32_t { GatherU32 = 0, SumF32 = 1 };
enum class TpVramRendezvousStatus : uint64_t {
    Pending = 0, Ready = 1, Aborted = 2, TimedOut = 3, InvalidEpoch = 4
};

// All signals have a single writer. On this rank's allocation, epoch and
// local_failure are written only by this rank; ready, consumed and peer_failure
// are written only by the peer. Only locally resident signals are ever polled.
struct alignas(64) TpVramSignalSlot { uint64_t value = 0; };
struct TpVramRendezvousControl {
    TpVramSignalSlot epoch;
    TpVramSignalSlot ready[kTpVramRendezvousSlots];
    TpVramSignalSlot consumed[kTpVramRendezvousSlots];
    TpVramSignalSlot local_failure;
    TpVramSignalSlot peer_failure;
};
struct TpVramRendezvousStamp {
    uint64_t epoch = 0;
    uint64_t status = 0;
    uint64_t polls = 0;
    uint64_t elapsed_ticks = 0;
};
static_assert(sizeof(TpVramSignalSlot) == 64 && alignof(TpVramSignalSlot) == 64);
static_assert(std::is_standard_layout<TpVramRendezvousControl>::value &&
              std::is_trivially_copyable<TpVramRendezvousControl>::value);

struct TpVramRendezvousSpan {
    const void* src = nullptr;
    void* dst = nullptr;
    int rows = 0;
    int half_words = 0;
    int run_words = 0;
};
struct TpVramRendezvousPacket {
    TpVramRendezvousSpan spans[2];
    int count = 0;
    TpVramRendezvousMode mode = TpVramRendezvousMode::GatherU32;
    uint32_t* own_inbox = nullptr;
    uint32_t* peer_inbox = nullptr;
    // Exact source word sum, not capacity. EACH inbox allocation always contains
    // kTpVramRendezvousSlots * kTpVramRendezvousMaxWords words (256 KiB). The fixed
    // slot stride permits varying packet lengths without overlapping live slots;
    // unused slot words are never copied or read.
    size_t inbox_words = 0;
};

// Compile-time support only: gfx906 build, HIP 7.2.x and required system atomic
// and fence builtins. Does not establish GPU/runtime/peer-memory correctness.
bool tp_vram_rendezvous_supported() noexcept;

// Both ranks MUST use the same operation, span dimensions and ordered call
// sequence. GatherU32 preserves every raw bit: dst[row,2*half_words] interleaves
// rank0/rank1 runs of run_words (half_words % run_words == 0). SumF32 requires one
// span with run_words == half_words; dst has rows*half_words float words and
// always computes exactly rank0 + rank1, irrespective of the executing rank.
//
// Allocation/lifecycle contract, enforced by each diagnostic owner:
// - BOTH controls and BOTH complete inbox rings MUST be independent
//   hipExtMallocWithFlags(..., hipDeviceMallocUncached) VRAM allocations on their
//   owning devices. Mapped host memory and ordinary coarse VRAM are unsupported.
// - Enable direct peer access in both directions. Own source/dst/stamp are local
//   VRAM. Pointers, all payload extents and metadata extents are mutually disjoint.
// - Initialize each control to TpVramRendezvousControl{} once, finish initialization
//   on BOTH GPUs before either launch, and keep all allocations alive through drain.
// - Use exactly one ordered stream/graph sequence per rank and control pair. No
//   concurrent execution on multiple streams or stream reuse by another rank.
//   Queueing successive graph replays on the same ordered stream is allowed.
//   Keep source words unchanged until the fused kernel finishes on that rank.
// - Device-live epochs advance on every call, including captured graph replays.
//   Never reset signals/epochs between replays. UINT64_MAX cannot wrap to zero.
// - Any failure is sticky for the entire pair. Failed calls poison ALL destination
//   words with kTpVramRendezvousPoison and never consume an unready inbox. The owner
//   MUST require BOTH rank stamps/witnesses to pass and both controls' failure
//   fields to remain zero after drain; one output can look valid before its peer
//   discovers failure. Drain both streams before releasing allocations; recovery
//   uses a fresh control pair.
//
// Each writing thread issues an explicit AMD SYSTEM release fence before the
// CTA publishes a SYSTEM release-store to peer ready. A SYSTEM acquire-load of
// own ready and CTA synchronization precede local inbox reads. Consumed ack is
// published only after ALL inbox reads finish; ring reuse waits for that ack.
// No remote atomic RMW, mapped-host polling, host epoch changes, allocation or
// runtime pointer queries occur here. Capture-safe validation uses raw extents.
//
// timeout_ticks is positive and <= INT64_MAX, measured by constant wall_clock64
// (hipDeviceAttributeWallClockRate in kHz, NOT core clockRate). Each wait is bounded
// by the call's deadline plus an independent finite poll cap. These bounds assume
// GPU/memory forward progress; a separate host/process watchdog must cover a hung
// driver/device. No kernel can guarantee recovery from a wedged peer fabric.
void tp_vram_rendezvous(const TpVramRendezvousPacket&,
                        TpVramRendezvousControl* own_control,
                        TpVramRendezvousControl* peer_control,
                        TpVramRendezvousStamp* stamp, int rank,
                        uint64_t timeout_ticks, void* stream);

} // namespace strata::kernels
