// Request-owned state, separate from immutable weights and execution scratch.
// No batching capability is advertised by this ownership-only refactor.
#pragma once
#include "ds4_model.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace strata::ds4 {

inline bool request_saved_position_fits(int64_t capacity, int64_t position, uint64_t stored) {
    return position > 0 && position <= capacity && stored == static_cast<uint64_t>(position);
}

// Non-owning device views. An execution lane binds one request's views;
// allocations belong to a resident request arena, not to copied Layer weights.
struct LayerRequestViews {
    uint8_t* win = nullptr;
    float *c_kv = nullptr, *c_score = nullptr;
    uint8_t* snap_win = nullptr;
    float *snap_kv = nullptr, *snap_sc = nullptr;
    uint8_t* ring_bak = nullptr;
    float *c_snap = nullptr, *c_rows = nullptr;
};
struct SourceRequestViews {
    int64_t cap = 0;
    std::vector<uint8_t*> ckv, ik;
};
struct RequestCheckpoint {
    int64_t pos = -1;
    std::vector<std::vector<uint8_t>> win;
    std::vector<std::vector<float>> comp;
};

// SpecList is supplied by the backend. Keeping the host distribution and RNG
// chain with its request prevents another request from changing verify's q.
template<class SpecList, int Window>
struct RequestHostState {
    static constexpr int kChecks = 4;
    using Check = RequestCheckpoint;
    int64_t pos = 0;
    std::vector<int32_t> toks, ctoks;
    Check ck[kChecks];
    bool slot_held = false;
    Stats st;
    int cur_topk_n[Window] = {}, cur_cand_n[Window] = {};
    int n_logits = 0;
    bool logits_all = false;
    std::vector<float> chunk_logits;
    std::function<void(int64_t)> progress;
    std::function<bool()> cancel;
    bool halted = false;
    std::vector<Image> imgs;
    bool spec_win = false;
    std::vector<SpecList> h_specq = std::vector<SpecList>(Window);
    bool drafts_q = false;
    float dq_temp = 0.0f, dq_top_p = 1.0f, dq_min_p = 0.0f;
    int dq_top_k = 0;
    uint64_t dq_seed = 0, dq_counter = 0;
};

// Device logits, taps and d_specq are still execution workspace in the current
// single-request caller. A resident scheduler must preserve/rebind them before
// switching requests; copying RequestHostState alone is NOT state isolation.
}
