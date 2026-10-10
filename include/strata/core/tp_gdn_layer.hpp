#pragma once

#include "strata/core/layout.hpp"
#include "strata/core/tp_gdn_weights.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>

namespace strata::core::tp2 {

// Diagnostic copies only. Token execution and routing never use these host arrays.
// GDN y and recurrence/conv are reconstructed in ORIGINAL global channel order.
// residual is the output AFTER the final FFN HC write; output is the FFN sum.
struct TpGdnLayerSnapshot {
    int tokens = 0;
    uint64_t epoch = 0;
    std::vector<float> residual, attention_input, gdn_output, mixer_output, ffn_input;
    std::vector<float> route_weights, route_logits, output, state, conv;
    std::vector<float> shared_output, shared_gate_logits, expert_parts; // shared output is BEFORE sigmoid scale
    std::vector<int32_t> ids;
    std::vector<uint8_t> routed_hidden_q8, shared_hidden_q8;
};

enum class TpGdnExecution {
    RuntimeCopies, // validated baseline: individual runtime P2P copies
    Consolidated,  // one system-fenced peer push per exchange/rank
    Captured,      // rank-local compute/push graphs; event edges stay outside
    FlatCaptured,  // one graph/rank, bounded device protocol, original HC
    FlatHcCaptured // same graph/protocol, exact HC row-sharded compute
};

struct TpGdnLayerProfile {
    uint64_t epoch = 0;
    int tokens = 0, rank = 0;
    std::vector<std::string> labels;
    std::vector<uint64_t> nanoseconds; // absolute stamps: compare within ONE rank only
    std::array<uint64_t,8> wait_nanoseconds{}, wait_polls{}, wait_status{};
};

// One non-PLE GDN layer, one in-flight proposal, T <= 8. Owns all session,
// stream, exchange and commit storage; immutable weight owners must outlive it.
// This is a layer component, not a full-model generation adapter.
// Calls must be serialized on one host thread; methods complete before return.
// rank=-1 is the unsharded direct-layer reference; two ranks must be 0,1.
// The direct reference aliases complete local outputs/hidden storage in every
// mode, avoiding TP-only self copies. Captured reference uses one proposal graph.
class TpGdnLayer {
public:
    TpGdnLayer(const ModelGeometry&, const TpGdnRankWeights& rank0,
               const TpGdnRankWeights* rank1 = nullptr, int capacity = 8,
               int expert_mode = -1);
    ~TpGdnLayer();
    TpGdnLayer(const TpGdnLayer&) = delete;
    TpGdnLayer& operator=(const TpGdnLayer&) = delete;

    // Explicit startup-only preparation, before the first proposal. Warms the
    // kernels without committing state, then captures both physical state banks.
    // Allocations and graph instantiation happen here, never in propose/commit.
    // Prepare each token count that Captured mode will use. Failure is fatal to
    // this instance, with no silent fallback to another execution mode.
    void prepare_captured(int tokens);
    // Flat modes require the model-free protocol preflight on the target runtime.
    // A separate profiled graph is explicit: timestamp overhead is not benchmark
    // evidence. HC splitting changes compute ownership, not weight allocation.
    void prepare_flat(int tokens, bool shard_hc = false, bool profile = false,
                      uint64_t timeout_us = 100000);
    void set_execution(TpGdnExecution mode, bool profile = false);
    TpGdnLayerProfile profile(int rank = 0) const;
    // Diagnostic fault injection, next flat proposal only. No successful output
    // or commit is permitted after timeout/abort. Missing rank is not launched;
    // delayed rank is launched after host delay. Never use in inference.
    void set_flat_fault_for_test(int missing_rank = -1, int delayed_rank = -1,
                                uint64_t delay_us = 0);
    TpGdnExecution execution() const;

    // Initialization/checkpoint adapter, outside token execution. Canonical
    // recurrence [128,48,128], conv [10240,3]. Clears any outstanding proposal.
    void reset_state(const std::vector<float>& state, const std::vector<float>& conv);

    // Host-input adapter for validation/token-wise ingestion. Copies the same
    // R[T,4,2560] to both ranks. No device allocation occurs in this method.
    void propose(const std::vector<float>& residual, int tokens, uint64_t epoch);
    // Device-input production entry: each source pointer belongs to its rank.
    // Caller must finish input writes before calling; no producer stream is
    // implicitly joined. Copies to owned residuals. Epochs must strictly increase.
    void propose_device(const std::array<const float*, 2>& residual, int tokens, uint64_t epoch);
    // Accept 0..T. Both ranks first compute into private candidate state; only
    // after both complete are candidates published. Any device failure poisons
    // this object; it cannot expose a successful partial commit or be reused.
    void commit(int keep);
    // A rejected window may be replaced only after commit(0).
    const float* residual(int rank = 0) const;
    int device(int rank = 0) const;
    int ranks() const;
    TpGdnLayerSnapshot snapshot(int rank = 0) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core::tp2
