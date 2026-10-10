#pragma once

// CPU-only ownership contract for the first two-rank, full-layer TP executor.
// No allocation of weights/state, GPU dispatch, global mode changes or inference
// integration. Indices address the ORIGINAL tensor's output rows/channels.
// GDN repeats K/Q heads modulo h_k; QSA groups Q heads contiguously by KV head.
#include "strata/core/layout.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace strata::core::tp2 {

struct RowRange {
    int begin = 0;
    int count = 0;
};

struct GdnRankLayout {
    int rank = 0;
    // Local head order -> global head index.
    std::vector<int> qk_heads;
    std::vector<int> value_heads;
    // Source output rows for attn_qkv and source convolution channel order.
    std::vector<int> qkv_rows;
    // Source output rows for attn_gate; also local-y -> original-y reassembly.
    std::vector<int> value_rows;
    // ssm_alpha/beta output rows and ssm_dt/ssm_a elements use value_heads.
    RowRange output_rows;  // literal-row ssm_out ownership, full 6144 input

    // Recurrence is [state_row, value_head, state_col], NOT head-major.
    std::size_t global_state_index(int row, int local_head, int col) const {
        check_state(row, local_head, col);
        return (static_cast<std::size_t>(row) * 48 +
                static_cast<std::size_t>(value_heads[local_head])) * 128 + col;
    }
    std::size_t local_state_index(int row, int local_head, int col) const {
        check_state(row, local_head, col);
        return (static_cast<std::size_t>(row) * 24 + local_head) * 128 + col;
    }
    static constexpr std::size_t state_floats() { return 128u * 24u * 128u; }
    static constexpr int conv_channels() { return 5120; }

private:
    void check_state(int row, int head, int col) const {
        if (row < 0 || row >= 128 || head < 0 || head >= 24 ||
            col < 0 || col >= 128 || value_heads.size() != 24)
            throw std::out_of_range("TP2 GDN state coordinate is outside the rank shard");
    }
};

struct QsaRankLayout {
    int rank = 0;
    std::vector<int> query_heads;
    std::vector<int> kv_heads;
    // attn_q is [query256, gate256] for EACH head, not two giant planes.
    std::vector<int> query_gate_rows;
    std::vector<int> kv_rows;       // attn_k and attn_v output rows
    std::vector<int> output_channels; // local attention -> global 6144 channels
    RowRange output_rows;          // literal-row attn_output, full 6144 input
    // The indexer is replicated, not sharded with attention heads.
    static constexpr int indexer_query_heads() { return 4; }
    static constexpr int indexer_key_channels() { return 128; }
};

namespace layer_detail {
inline void rank_check(int rank) {
    if (rank != 0 && rank != 1)
        throw std::invalid_argument("TP2 layer layout requires rank 0 or 1");
}
inline void base_check(const ModelGeometry& g) {
    if (g.n_embd != 2560 || g.n_layers != 48 || g.qsa_interval != 4 ||
        g.hc != 4 || g.hc_lr != 320 || g.n_expert != 512 || g.n_ff != 640)
        throw std::invalid_argument("TP2 layer layout supports only the validated N2560/F640 48-layer geometry");
}
inline void append_rows(std::vector<int>& rows, int head, int width, int offset = 0) {
    for (int j = 0; j < width; ++j) rows.push_back(offset + head * width + j);
}
}  // namespace layer_detail

inline GdnRankLayout gdn_rank_layout(const ModelGeometry& g, int rank) {
    layer_detail::rank_check(rank);
    layer_detail::base_check(g);
    if (g.ssm_state_size != 128 || g.ssm_k_heads != 16 || g.ssm_v_heads != 48 ||
        g.ssm_d_conv != 4 || g.ssm_conv_channels != 10240 || g.ssm_value_dim != 6144)
        throw std::invalid_argument("TP2 GDN requires S128/K16/V48/conv4 with consistent channel counts");
    GdnRankLayout p;
    p.rank = rank;
    p.output_rows = {rank * 1280, 1280};
    for (int j = 0; j < 8; ++j) p.qk_heads.push_back(8 * rank + j);
    for (int b = 0; b < 3; ++b)
        for (int j = 0; j < 8; ++j) p.value_heads.push_back(16 * b + 8 * rank + j);
    for (const int h : p.qk_heads) layer_detail::append_rows(p.qkv_rows, h, 128);
    for (const int h : p.qk_heads) layer_detail::append_rows(p.qkv_rows, h, 128, 2048);
    for (const int h : p.value_heads) {
        layer_detail::append_rows(p.qkv_rows, h, 128, 4096);
        layer_detail::append_rows(p.value_rows, h, 128);
    }
    return p;
}

inline QsaRankLayout qsa_rank_layout(const ModelGeometry& g, int rank) {
    layer_detail::rank_check(rank);
    layer_detail::base_check(g);
    if (g.n_head != 24 || g.n_head_kv != 2 || g.head_dim != 256 ||
        g.idx_q_heads != 4 || g.idx_key_dim != 128)
        throw std::invalid_argument("TP2 QSA requires Q24/KV2/HD256 and replicated indexer Q4/D128");
    QsaRankLayout p;
    p.rank = rank;
    p.output_rows = {rank * 1280, 1280};
    p.kv_heads.push_back(rank);
    for (int h = rank * 12; h < (rank + 1) * 12; ++h) {
        p.query_heads.push_back(h);
        layer_detail::append_rows(p.query_gate_rows, h, 512);
        layer_detail::append_rows(p.output_channels, h, 256);
    }
    layer_detail::append_rows(p.kv_rows, rank, 256);
    return p;
}

}  // namespace strata::core::tp2
