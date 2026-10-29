#pragma once
#include "ds4_model.hpp"
#include <mutex>
#include <cstdint>
namespace strata::ds4 {
// CPU mailbox work belongs to the shared executor, not the request currently
// bound by the enqueue thread. Times are aggregate work, never per-request latency.
class BatchExecutionAccounting {
    mutable std::mutex mu_;
    Stats total_;
    uint64_t jobs_ = 0;
public:
    void complete(const Stats& s) {
        std::lock_guard<std::mutex> lock(mu_);
        total_.cpu_ms += s.cpu_ms;
        total_.cpu_experts += s.cpu_experts;
        total_.ahead_picks += s.ahead_picks;
        total_.ahead_hit_k += s.ahead_hit_k;
        total_.ahead_hit += s.ahead_hit;
        total_.ahead_hit2 += s.ahead_hit2;
        total_.ahead_cold += s.ahead_cold;
        total_.ahead_cold_hit += s.ahead_cold_hit;
        total_.ahead_pf += s.ahead_pf;
        total_.store_zc += s.store_zc;
        total_.skipped += s.skipped;
        total_.ram_experts += s.ram_experts;
        for(int i=0;i<16;++i) {
            total_.ahead_rank_n[i] += s.ahead_rank_n[i];
            total_.ahead_rank_hit[i] += s.ahead_rank_hit[i];
        }
        ++jobs_;
    }
    Stats snapshot(uint64_t& jobs) const {
        std::lock_guard<std::mutex> lock(mu_);
        jobs=jobs_;return total_;
    }
};
}
