#pragma once

#include "strata/core/tp_layer_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core::tp2 {
enum class TpGdnPartition { OutputRows, InputColumns };
// Native Q8_1 projection contract: non-MMVQ source types are refused, matching
// the existing Verifier/shared_expert_multi path. BF16 small weights are separate.
struct TpNativeMatrix {
    int type = -1;
    const void* data = nullptr;
    int input = 0, output = 0;
};
struct TpHcWeights {
    const float* norm = nullptr;
    const uint16_t *down = nullptr, *up = nullptr, *inject = nullptr;
};
// Immutable borrowed device views. Only the owner constructs these; its lifetime
// must exceed every stream/graph using them. No WeightRef metadata is rewritten.
struct TpGdnRankWeights {
    int rank = -1, device = -1, layer = -1;
    TpGdnPartition partition = TpGdnPartition::OutputRows;
    TpHcWeights hc[2];
    TpNativeMatrix qkv, z, out, shared_gate, shared_up, shared_down;
    const uint16_t *alpha = nullptr, *beta = nullptr, *router = nullptr, *shared_scalar = nullptr;
    const float *dt = nullptr, *a = nullptr, *conv = nullptr, *norm = nullptr;
    const uint8_t* expert_arena = nullptr;
    size_t expert_bytes = 0; // stride of ONE expert; arena contains 512
    // Independent phase descriptors: GU n_embd=N,n_ff=local_F;
    // OutputRows down: n_embd=local_N,n_ff=full_F; InputColumns down:
    // n_embd=full_N,n_ff=local_F. Both offsets address the same raw blob.
    kernels::NativeExpertLayout gu_layout{}, down_layout{};
};

class TpGdnWeights {
public:
    TpGdnWeights() = default;
    ~TpGdnWeights();
    TpGdnWeights(const TpGdnWeights&) = delete;
    TpGdnWeights& operator=(const TpGdnWeights&) = delete;
    // Canonical small tensors come from the same model's iq_pack output, using
    // WeightTable's exact engine conversions. Native matrices/experts come from
    // validated GGUF spans. Only this non-PLE GDN layer is allocated.
    // rank=-1 is a full single-device reference regardless of partition tag.
    // Ranks 0/1 own identical GU/head shards in either partition. InputColumns
    // packs out along local value_rows and shared/expert down along local F/2;
    // their full-N outputs are partial sums requiring the executor reduction.
    // OutputRows remains the default. HC stays replicated in both partitions.
    // Synchronous initialization; scratch/state belong to the separate executor.
    bool load(const std::vector<std::string>& shards, const std::string& pack_dir,
              const ModelGeometry& geometry, int layer, int rank, int device, std::string& err,
              TpGdnPartition partition = TpGdnPartition::OutputRows);
    const TpGdnRankWeights& weights() const { return weights_; }
    uint64_t weight_bytes() const { return bytes_; }
private:
    void clear() noexcept;
    TpGdnRankWeights weights_{};
    std::vector<void*> allocations_;
    uint64_t bytes_ = 0;
    int device_ = -1;
};
} // namespace strata::core::tp2
