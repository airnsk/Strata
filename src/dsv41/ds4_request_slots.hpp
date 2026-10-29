// Single-owner scheduler bookkeeping. GPU workers submit completion messages;
// only the scheduler mutates this table. Not a wire protocol or GPU scheduler.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_set>

namespace strata::ds4 {
struct RequestLease {
    uint32_t slot = 0;
    uint64_t generation = 0;
    bool operator==(const RequestLease& x) const {
        return slot == x.slot && generation == x.generation;
    }
};
struct RequestWork {
    RequestLease lease;
    uint64_t serial = 0;
};
enum class RequestPhase { free, prefill, decode, draining };

template<std::size_t Slots>
class RequestSlots {
public:
    struct Entry {
        uint64_t generation = 0, next_work = 0;
        int64_t capacity = 0, position = 0;
        RequestPhase phase = RequestPhase::free;
        std::unordered_set<uint64_t> pending;
        bool cancel = false;
    };
    const Entry* lookup(RequestLease id) const {
        if (id.slot >= Slots || id.generation == 0) return nullptr;
        const auto& s = slots_[id.slot];
        return s.phase != RequestPhase::free && s.generation == id.generation ? &s : nullptr;
    }
    std::optional<RequestLease> admit(int64_t prompt, int64_t output,
                                      int64_t speculative, int64_t capacity) {
        if (prompt < 0 || output < 0 || speculative < 0 || capacity <= 0 ||
            prompt > capacity || output > capacity-prompt ||
            speculative > capacity-prompt-output) return std::nullopt;
        for (uint32_t i=0; i<Slots; ++i) {
            auto& s=slots_[i];
            if (s.phase!=RequestPhase::free ||
                s.generation==std::numeric_limits<uint64_t>::max()) continue;
            ++s.generation; s.next_work=0; s.capacity=capacity; s.position=0;
            s.cancel=false; s.pending.clear(); s.phase=RequestPhase::prefill;
            return RequestLease{i,s.generation};
        }
        return std::nullopt;
    }
    bool decode_ready(RequestLease id) {
        auto* s=get(id);
        if (!s || s->phase!=RequestPhase::prefill || !s->pending.empty()) return false;
        s->phase=RequestPhase::decode; return true;
    }
    std::optional<RequestWork> begin(RequestLease id) {
        auto* s=get(id);
        if (!s || s->phase==RequestPhase::draining ||
            s->next_work==std::numeric_limits<uint64_t>::max()) return std::nullopt;
        const auto n=++s->next_work; s->pending.insert(n);
        return RequestWork{id,n};
    }
    // Cancellation suppresses delivery, but every submitted GPU/handoff job
    // still needs its own acknowledgement. Duplicate/stale ACKs are rejected.
    bool acknowledge(RequestWork w) {
        auto* s=get(w.lease);
        return s && s->pending.erase(w.serial)==1;
    }
    bool can_deliver(RequestWork w) const {
        const auto* s=lookup(w.lease);
        return s && !s->cancel && s->phase!=RequestPhase::draining &&
               s->pending.count(w.serial)!=0;
    }
    bool commit_position(RequestLease id, int64_t p) {
        auto* s=get(id);
        if (!s || s->phase==RequestPhase::draining || p<s->position || p>s->capacity)
            return false;
        s->position=p; return true;
    }
    bool finish(RequestLease id, bool cancelled) {
        auto* s=get(id); if (!s) return false;
        s->cancel=s->cancel || cancelled; s->phase=RequestPhase::draining; return true;
    }
    bool terminal_ready(RequestLease id) const {
        const auto* s=lookup(id);
        return s && s->phase==RequestPhase::draining && s->pending.empty();
    }
    // Call only after sending the terminal event for this exact generation.
    // New protocol messages must carry the generation on both sides.
    bool release(RequestLease id) {
        if (!terminal_ready(id)) return false;
        auto& s=slots_[id.slot]; s.phase=RequestPhase::free; return true;
    }
private:
    Entry* get(RequestLease id) {
        return lookup(id) ? &slots_[id.slot] : nullptr;
    }
    std::array<Entry,Slots> slots_;
};
}
