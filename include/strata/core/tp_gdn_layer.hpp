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
    // Input-column diagnostics: one rank's pre-reduction FFN result and
    // independently recomputed expert partials. Empty for output-row owners.
    std::vector<float> local_output_partial, local_shared_partial, local_expert_parts;
    std::vector<int32_t> ids;
    std::vector<uint8_t> routed_hidden_q8, shared_hidden_q8;
};

enum class TpGdnExecution {
    RuntimeCopies, // validated baseline: individual runtime P2P copies
    Consolidated,  // one system-fenced peer push per exchange/rank
    Captured,      // rank-local compute/push graphs; event edges stay outside
    ColumnCaptured,// input-column owners, two event-ordered output reductions
    FlatCaptured,  // one graph/rank, bounded device protocol, original HC
    FlatHcCaptured,// same graph/protocol, exact HC row-sharded compute
    HybridCaptured, // row attention / column FFN, three joins, no FFN hidden gather
    HybridRcclCaptured // opt-in test build: same arithmetic, one RCCL proposal graph/rank
};

struct TpGdnRcclOptions {
    std::string library; // required canonical identity, checked against the loaded DSO
    int timeout_ms = 10000, init_timeout_ms = 60000;
};

struct TpGdnLayerProfile {
    uint64_t epoch = 0;
    int tokens = 0, rank = 0;
    std::vector<std::string> labels;
    std::vector<uint64_t> nanoseconds; // absolute stamps: compare within ONE rank only; zero means inactive
    std::array<uint64_t,8> wait_nanoseconds{}, wait_polls{}, wait_status{};
};

// Test-only frozen-input replay; wall and device-local event spans are separate
// observables, never additive terms of a whole-layer latency decomposition.
struct TpGdnCalibrationSample {
    std::string phase, arm, event_scope;
    int trial = 0, order = 0, calls = 1;
    double wall_ms = 0;
    uint64_t restored_bytes = 0, peer_bytes_per_rank = 0;
    std::array<float,2> device_ms{};
};

// Fine calibration coordinates both full-device references and the hybrid
// candidate so phase inputs and crossover order are actually matched.
struct TpGdnFineCalibrationSample : TpGdnCalibrationSample {
    std::string owner;
    std::array<int,2> devices{{-1,-1}};
    uint64_t input_hash = 0; // FNV-1a of canonical phase inputs, not weight identity
    int groups = 0;
    std::vector<int> group_entries; // canonical resident-plan entry counts, group order
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
    // Prepare each token count that Captured/ColumnCaptured/HybridCaptured uses. Optional
    // profile=true creates separate stamped graphs; timing graphs stay untouched.
    // Hybrid TP uses four segments/three joins, with original row attention.
    // Column TP uses three compute segments and two full-output reductions;
    // row TP retains five segments/four gathers. Failure is fatal to
    // this instance, with no silent fallback to another execution mode.
    void prepare_captured(int tokens, bool profile = false);
    // Isolated benchmark only, HybridRowsColumns owners. Requires the optional
    // RCCL build and an explicitly verified library. Persistent rank workers own
    // communicator calls/capture/replay; no device allocation or host phase joins during
    // a proposal. Captures both physical state banks. Profiling is not supported.
    // A blocked/failed RCCL operation terminates this diagnostic subprocess;
    // possibly live graph/communicator/buffer storage is never freed on failure.
    void prepare_rccl(int tokens, const TpGdnRcclOptions& options);
    // Host admission order, not guaranteed driver execution order. Missing/late
    // rank injection applies to the next proposal only and requires a subprocess.
    void set_rccl_launch_for_test(int first, int missing_rank = -1,
                                 int delayed_rank = -1, uint64_t delay_us = 0);
    // Flat modes require the model-free protocol preflight on the target runtime.
    // They are incompatible with InputColumns/HybridRowsColumns owners.
    // A separate profiled graph is explicit: timestamp overhead is not benchmark
    // evidence. HC splitting changes compute ownership, not weight allocation.
    void prepare_flat(int tokens, bool shard_hc = false, bool profile = false,
                      uint64_t timeout_us = 100000);
    void set_execution(TpGdnExecution mode, bool profile = false);
    TpGdnLayerProfile profile(int rank = 0) const;
    // Opt-in diagnostic only, output-row owners or full reference. No outstanding
    // proposal. Snapshots all mutable rank storage before each valid phase, then
    // restores outside each timed replay. Graphs contain compute OR peer pushes,
    // never stale-input consumers. Exact replay output gates throw on failure.
    // Restores original storage/state; no commit, bank publication, or epoch change.
    std::vector<TpGdnCalibrationSample> calibrate_frozen(
        const std::vector<float>& residual, int tokens, int warmup, int trials);

    // Bounded test-only coefficient probe, T=1 or 8. All layers must be idle,
    // unprofiled, prepared, and reset to the same checkpoint by the caller.
    // full0/full1 are unsharded references on the hybrid ranks' devices.
    // Canonical full0 seams are transplanted before phase snapshots; hidden Q8
    // is sliced by FFN ownership. Each replay restores ALL mutable allocations
    // outside timing and requires byte-identical owner-local results.
    // Down includes combine: full uses grouped Down plus combine, hybrid uses
    // its actual fused local down/combine kernel, without peer publication.
    // Standalone exchange controls are not an additive fused-peer cost estimate.
    // No commits, epochs, bank publication, or production-path changes.
    static std::vector<TpGdnFineCalibrationSample> calibrate_fine_frozen(
        TpGdnLayer& full0, TpGdnLayer& full1, TpGdnLayer& hybrid,
        const std::vector<float>& residual, int tokens, int warmup, int trials);

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
