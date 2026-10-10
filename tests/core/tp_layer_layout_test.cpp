// Pure CPU regression tests. Expected ownership is derived from original
// model semantics independently of the layout builder's packing loops.
#include "strata/core/tp_layer_layout.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

using strata::core::ModelGeometry;
using namespace strata::core::tp2;

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class Fn> static void rejects(Fn fn) {
    bool rejected = false;
    try { fn(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "unsupported geometry/rank was accepted");
}
static void partition(const std::vector<int>& a, const std::vector<int>& b, int size) {
    std::vector<int> seen(size, 0);
    for (const auto* v : {&a, &b}) for (int i : *v) {
        require(i >= 0 && i < size, "source row outside tensor");
        ++seen[i];
    }
    require(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }),
            "ownership is not a bijection");
}
static void reassemble(const std::vector<int>& a, const std::vector<int>& b, int size) {
    // Unequal signed values make swaps, holes and accidental summation visible.
    std::vector<long long> original(size), rebuilt(size, 0);
    for (int i = 0; i < size; ++i) original[i] = (i % 2 ? -1LL : 1LL) * (17LL * i + 5);
    for (const auto* map : {&a, &b}) {
        std::vector<long long> local;
        for (int i : *map) local.push_back(original[i]);
        for (std::size_t j = 0; j < map->size(); ++j) rebuilt[(*map)[j]] = local[j];
    }
    require(original == rebuilt, "output/tensor reassembly failed");
}
int main() {
    try {
        const ModelGeometry g;
        const auto a = gdn_rank_layout(g, 0), b = gdn_rank_layout(g, 1);
        partition(a.qk_heads, b.qk_heads, 16);
        partition(a.value_heads, b.value_heads, 48);
        partition(a.qkv_rows, b.qkv_rows, 10240);
        partition(a.value_rows, b.value_rows, 6144);
        reassemble(a.qkv_rows, b.qkv_rows, 10240);
        reassemble(a.value_rows, b.value_rows, 6144);
        std::vector<int> state_seen(128 * 48 * 128, 0);
        for (const auto* p : {&a, &b}) {
            require(p->qkv_rows.size() == 5120 && p->value_rows.size() == 3072,
                    "wrong GDN local projection dimensions");
            for (int h = 0; h < 8; ++h) for (int d = 0; d < 128; ++d) {
                const int source_q = (p->rank * 8 + h) * 128 + d;
                require(p->qkv_rows[h * 128 + d] == source_q,
                        "GDN Q source row order is wrong");
                require(p->qkv_rows[1024 + h * 128 + d] == 2048 + source_q,
                        "GDN K source row order is wrong");
            }
            for (int v = 0; v < 24; ++v) {
                const int global_v = p->value_heads[v];
                require(global_v % 16 == p->qk_heads[v % 8], "GDN modulo-head contract broken");
                require((global_v % 16) / 8 == p->rank, "wrong independent GDN owner");
                for (int d = 0; d < 128; ++d) {
                    require(p->value_rows[v * 128 + d] == global_v * 128 + d,
                            "GDN output head reassembly is wrong");
                    require(p->qkv_rows[2048 + v * 128 + d] == 4096 + global_v * 128 + d,
                            "GDN V source rows are wrong");
                }
            }
            std::vector<int> local_seen(GdnRankLayout::state_floats(), 0);
            for (int row = 0; row < 128; ++row) for (int h = 0; h < 24; ++h)
                for (int col = 0; col < 128; ++col) {
                    const auto global = p->global_state_index(row, h, col);
                    const auto local = p->local_state_index(row, h, col);
                    require(global == static_cast<std::size_t>(row * 6144 + p->value_heads[h] * 128 + col),
                            "recurrence original state stride is wrong");
                    require(local < local_seen.size() && global < state_seen.size(), "state overflow");
                    ++local_seen[local]; ++state_seen[global];
                }
            require(std::all_of(local_seen.begin(), local_seen.end(), [](int n) { return n == 1; }),
                    "local recurrence state is not bijective");
        }
        require(std::all_of(state_seen.begin(), state_seen.end(), [](int n) { return n == 1; }),
                "global recurrence state is not partitioned exactly");
        bool bad_state = false;
        try { (void)a.global_state_index(128, 0, 0); } catch (const std::out_of_range&) { bad_state = true; }
        require(bad_state, "invalid state coordinate accepted");
        const auto q0 = qsa_rank_layout(g, 0), q1 = qsa_rank_layout(g, 1);
        partition(q0.query_heads, q1.query_heads, 24);
        partition(q0.kv_heads, q1.kv_heads, 2);
        partition(q0.query_gate_rows, q1.query_gate_rows, 12288);
        partition(q0.kv_rows, q1.kv_rows, 512);
        partition(q0.output_channels, q1.output_channels, 6144);
        reassemble(q0.query_gate_rows, q1.query_gate_rows, 12288);
        reassemble(q0.output_channels, q1.output_channels, 6144);
        for (const auto* p : {&q0, &q1}) {
            require(p->indexer_query_heads() == 4 && p->indexer_key_channels() == 128,
                    "indexer must remain replicated");
            for (int h = 0; h < 12; ++h) {
                const int global_h = p->query_heads[h];
                require(global_h / 12 == p->kv_heads[0], "QSA GQA contract broken");
                for (int d = 0; d < 256; ++d) {
                    require(p->query_gate_rows[h * 512 + d] == global_h * 512 + d,
                            "QSA query rows not interleaved per head");
                    require(p->query_gate_rows[h * 512 + 256 + d] == global_h * 512 + 256 + d,
                            "QSA gate rows not interleaved per head");
                }
            }
        }
        require(a.output_rows.begin == 0 && b.output_rows.begin == 1280 &&
                a.output_rows.count == 1280 && b.output_rows.count == 1280 &&
                q0.output_rows.begin == 0 && q1.output_rows.begin == 1280 &&
                q0.output_rows.count == 1280 && q1.output_rows.count == 1280,
                "literal output row ownership wrong");
        rejects([&] { gdn_rank_layout(g, -1); });
        rejects([&] { qsa_rank_layout(g, 2); });
        // Reject every unsupported field used by either contract, not only odd counts.
        for (auto member : {&ModelGeometry::n_embd, &ModelGeometry::n_layers,
                &ModelGeometry::qsa_interval, &ModelGeometry::hc, &ModelGeometry::hc_lr,
                &ModelGeometry::n_expert, &ModelGeometry::n_ff}) {
            auto bad = g; ++(bad.*member);
            rejects([&] { gdn_rank_layout(bad, 0); });
            rejects([&] { qsa_rank_layout(bad, 0); });
        }
        for (auto member : {&ModelGeometry::ssm_state_size, &ModelGeometry::ssm_k_heads,
                &ModelGeometry::ssm_v_heads, &ModelGeometry::ssm_d_conv,
                &ModelGeometry::ssm_conv_channels, &ModelGeometry::ssm_value_dim}) {
            auto bad = g; ++(bad.*member); rejects([&] { gdn_rank_layout(bad, 0); });
        }
        for (auto member : {&ModelGeometry::n_head, &ModelGeometry::n_head_kv,
                &ModelGeometry::head_dim, &ModelGeometry::idx_q_heads, &ModelGeometry::idx_key_dim}) {
            auto bad = g; ++(bad.*member); rejects([&] { qsa_rank_layout(bad, 0); });
        }
        std::cout << "tp_layer_layout: ownership, state, reconstruction and rejection tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "tp_layer_layout: " << e.what() << '\n';
        return 1;
    }
}
