#include "strata/core/tp_gdn_weight_rows.hpp"
#include "strata/core/tp_layer_layout.hpp"
#include "strata/core/tp_gdn_weights.hpp"
#include <stdexcept>
#include <iostream>
#include <functional>
using namespace strata::core;
using namespace strata::core::tp2;
static void require(bool condition) {
    if (!condition) throw std::runtime_error("TP GDN row test failed");
}
static void rejects(const std::function<void()>& f) {
    bool caught = false; try { f(); } catch (const std::exception&) { caught = true; } require(caught);
}
static void check(int width, int count, int type, const std::vector<int>& a, const std::vector<int>& b) {
    int be = 0, bb = 0; require(strata::block_geometry(type, be, bb));
    const size_t pitch = static_cast<size_t>(width / be) * bb;
    std::vector<uint8_t> source(pitch * count), result(source.size(), 0);
    for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<uint8_t>((i * 17 + i / 113) % 251);
    std::vector<int> ownership(static_cast<size_t>(count));
    for (const auto* selected : {&a, &b}) {
        const auto spans = weight_row_spans(type, width, count, source.size(), *selected);
        std::vector<uint8_t> shard(selected->size() * pitch);
        for (const auto& s : spans) std::memcpy(shard.data() + s.destination, source.data() + s.source, s.bytes);
        for (size_t i = 0; i < selected->size(); ++i) {
            const auto r = static_cast<size_t>((*selected)[i]);
            require(++ownership[r] == 1);
            require(std::memcmp(shard.data() + i * pitch, source.data() + r * pitch, pitch) == 0);
            std::memcpy(result.data() + r * pitch, shard.data() + i * pitch, pitch);
        }
    }
    require(result == source);
    for (int n : ownership) require(n == 1);
}
static std::vector<int> range(int first, int count) {
    std::vector<int> v; for (int i = 0; i < count; ++i) v.push_back(first + i); return v;
}
static void check_columns(int width, int count, int type, const std::vector<int>& a, const std::vector<int>& b) {
    int be = 0, bb = 0; require(strata::block_geometry(type, be, bb));
    const size_t pitch = static_cast<size_t>(width / be) * bb;
    std::vector<uint8_t> source(pitch * count), result(source.size());
    std::vector<unsigned char> ownership(source.size());
    for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<uint8_t>((i * 31 + i / 109) % 251);
    for (const auto* selected : {&a, &b}) {
        const auto spans = weight_column_spans(type, width, count, source.size(), *selected);
        const size_t local_pitch = selected->size() / static_cast<size_t>(be) * bb;
        std::vector<uint8_t> shard(local_pitch * count);
        for (const auto& span : spans) {
            require(span.source <= source.size() && span.bytes <= source.size() - span.source);
            require(span.destination <= shard.size() && span.bytes <= shard.size() - span.destination);
            std::memcpy(shard.data() + span.destination, source.data() + span.source, span.bytes);
        }
        // Independent element-to-block oracle, not reversing the span plan.
        for (int row = 0; row < count; ++row) {
            for (size_t j = 0; j < selected->size(); j += static_cast<size_t>(be)) {
                const size_t src = static_cast<size_t>(row) * pitch + static_cast<size_t>((*selected)[j] / be) * bb;
                const size_t dst = static_cast<size_t>(row) * local_pitch + j / static_cast<size_t>(be) * bb;
                for (int k = 0; k < bb; ++k) {
                    require(++ownership[src + k] == 1);
                    require(shard[dst + k] == source[src + k]);
                    result[src + k] = shard[dst + k];
                }
            }
        }
    }
    require(result == source);
    for (auto n : ownership) require(n == 1);
}
static void check_experts() {
    for (int gu : {18,21,22,23}) for (int down : {20,42}) {
        const auto p = plan(gu, down, 2560, 640, DownSplit::Columns);
        std::vector<uint8_t> source(p.original.bytes);
        for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<uint8_t>((i * 19 + i / 137) % 251);
        const auto a = split(p, source, 0), b = split(p, source, 1);
        require(reassemble(p, a, b) == source);
        require(p.shard.down.columns == 320 && p.shard.down.rows == 2560);
        require(p.shard.bytes * 2 == p.original.bytes);
        for (unsigned rank = 0; rank < 2; ++rank) {
            const auto& shard = rank == 0 ? a : b;
            for (size_t row = 0; row < 2560; ++row)
                require(std::memcmp(shard.data() + p.shard.down.offset + row * p.shard.down.row_bytes,
                    source.data() + p.original.down.offset + row * p.original.down.row_bytes + rank * p.shard.down.row_bytes,
                    p.shard.down.row_bytes) == 0);
        }
    }
}
// Exercise the same policy helper used by the loader, not a test-only copy of
// partition dispatch. Matrix input is the contiguous quantization-block axis.
static void check_partition_geometry() {
    for (const auto partition : {TpGdnPartition::OutputRows, TpGdnPartition::InputColumns,
                                 TpGdnPartition::HybridRowsColumns}) {
        for (int rank : {-1, 0, 1}) {
            const auto g = tp_gdn_partition_geometry(partition, rank);
            const bool full = rank == -1;
            require(g.full == full);
            require(g.projection_columns == (!full && partition == TpGdnPartition::InputColumns));
            require(g.ffn_columns == (!full && partition != TpGdnPartition::OutputRows));
            require(g.projection_input == (full || partition != TpGdnPartition::InputColumns ? 6144 : 3072));
            require(g.projection_output == (full || partition == TpGdnPartition::InputColumns ? 2560 : 1280));
            require(g.ffn_input == (full || partition == TpGdnPartition::OutputRows ? 640 : 320));
            require(g.ffn_output == (full || partition != TpGdnPartition::OutputRows ? 2560 : 1280));
        }
        rejects([&]{ (void)tp_gdn_partition_geometry(partition, -2); });
        rejects([&]{ (void)tp_gdn_partition_geometry(partition, 2); });
    }
    for (int rank : {-1, 0, 1}) {
        rejects([&]{ (void)tp_gdn_partition_geometry(static_cast<TpGdnPartition>(-1), rank); });
        rejects([&]{ (void)tp_gdn_partition_geometry(static_cast<TpGdnPartition>(3), rank); });
        rejects([&]{ (void)tp_gdn_partition_geometry(static_cast<TpGdnPartition>(999), rank); });
    }
    // Hybrid attention must retain the accepted row projection, independently
    // of adopting the column FFN policy. Legacy policies keep their dimensions.
    for (int rank : {0, 1}) {
        const auto row = tp_gdn_partition_geometry(TpGdnPartition::OutputRows, rank);
        const auto col = tp_gdn_partition_geometry(TpGdnPartition::InputColumns, rank);
        const auto hybrid = tp_gdn_partition_geometry(TpGdnPartition::HybridRowsColumns, rank);
        require(hybrid.projection_input == row.projection_input && hybrid.projection_output == row.projection_output);
        require(hybrid.ffn_input == col.ffn_input && hybrid.ffn_output == col.ffn_output);
        require(!hybrid.projection_columns && hybrid.ffn_columns);
    }
}

static void check_partition_expert_bytes() {
    constexpr size_t N = 2560, F = 640;
    for (int gu : {18, 21, 22, 23}) for (int down : {20, 42}) {
        int gu_values = 0, gu_bytes = 0, down_values = 0, down_bytes = 0;
        require(strata::block_geometry(gu, gu_values, gu_bytes));
        require(strata::block_geometry(down, down_values, down_bytes));
        const size_t gu_pitch = N / size_t(gu_values) * size_t(gu_bytes);
        const size_t full_down_pitch = F / size_t(down_values) * size_t(down_bytes);
        const size_t half_down_pitch = (F / 2) / size_t(down_values) * size_t(down_bytes);
        for (auto partition : {TpGdnPartition::OutputRows, TpGdnPartition::InputColumns,
                               TpGdnPartition::HybridRowsColumns}) {
            const auto geometry = tp_gdn_partition_geometry(partition, 0);
            const auto p = plan(gu, down, N, F, geometry.ffn_columns ? DownSplit::Columns : DownSplit::OutputRows);
            const auto expected_legacy = plan(gu, down, N, F,
                partition == TpGdnPartition::OutputRows ? DownSplit::OutputRows : DownSplit::Columns);
            require(p == expected_legacy);
            // Full canonical order remains gate, up, down without padding.
            require(p.original.gate.rows == F && p.original.gate.columns == N);
            require(p.original.gate.offset == 0 && p.original.gate.row_bytes == gu_pitch);
            require(p.original.up.offset == F * gu_pitch);
            require(p.original.down.offset == 2 * F * gu_pitch);
            require(p.original.down.bytes == N * full_down_pitch);
            require(p.original.bytes == 2 * F * gu_pitch + N * full_down_pitch);
            // Every TP policy owns F/2 GU rows, even when its down needs full F.
            require(p.shard.gate.rows == F / 2 && p.shard.gate.columns == N);
            require(p.shard.gate.row_bytes == gu_pitch && p.shard.gate.offset == 0);
            require(p.shard.up.offset == (F / 2) * gu_pitch);
            require(p.shard.down.offset == F * gu_pitch);
            require(p.shard.down.rows == size_t(geometry.ffn_output));
            require(p.shard.down.columns == size_t(geometry.ffn_input));
            require(p.shard.down.row_bytes == (geometry.ffn_columns ? half_down_pitch : full_down_pitch));
            require(p.shard.bytes == p.shard.down.offset + p.shard.down.rows * p.shard.down.row_bytes);
            require(2 * p.shard.bytes == p.original.bytes);
            std::vector<uint8_t> source(p.original.bytes);
            for (size_t i = 0; i < source.size(); ++i) source[i] = uint8_t((i * 37 + i / 97) % 251);
            const auto rank0 = split(p, source, 0), rank1 = split(p, source, 1);
            require(reassemble(p, rank0, rank1) == source);
            for (size_t rank = 0; rank < 2; ++rank) {
                const auto& shard = rank ? rank1 : rank0;
                // Independent canonical offsets, without consuming copy spans.
                require(std::memcmp(shard.data(), source.data() + rank * (F / 2) * gu_pitch, (F / 2) * gu_pitch) == 0);
                require(std::memcmp(shard.data() + p.shard.up.offset,
                    source.data() + F * gu_pitch + rank * (F / 2) * gu_pitch, (F / 2) * gu_pitch) == 0);
                const size_t local_rows = geometry.ffn_columns ? N : N / 2;
                const size_t local_pitch = geometry.ffn_columns ? half_down_pitch : full_down_pitch;
                for (size_t row = 0; row < local_rows; ++row) {
                    const size_t source_row = geometry.ffn_columns ? row : rank * (N / 2) + row;
                    const size_t source_column_bytes = geometry.ffn_columns ? rank * half_down_pitch : 0;
                    const size_t source_offset = 2 * F * gu_pitch + source_row * full_down_pitch + source_column_bytes;
                    require(std::memcmp(shard.data() + F * gu_pitch + row * local_pitch,
                                        source.data() + source_offset, local_pitch) == 0);
                }
            }
            rejects([&]{ (void)plan(gu, down, 0, F, p.down_split); });
            rejects([&]{ (void)plan(gu, down, N, 0, p.down_split); });
            rejects([&]{ (void)plan(gu, down, N - 1, F, p.down_split); });
            rejects([&]{ (void)plan(gu, down, N, F - 1, p.down_split); });
            rejects([&]{ (void)plan(gu, down, N, 2, p.down_split); });
        }
    }
    rejects([]{ (void)plan(18, 20, 2560, 640, static_cast<DownSplit>(99)); });
    rejects([]{ (void)plan(999, 20, 2560, 640); });
    rejects([]{ (void)plan(18, 999, 2560, 640); });
}

int main() {
    check_partition_geometry();
    check_partition_expert_bytes();
    const auto a = gdn_rank_layout(ModelGeometry{}, 0), b = gdn_rank_layout(ModelGeometry{}, 1);
    for (int t : {8, 12, 23, 30, 42}) {
        check(2560, 10240, t, a.qkv_rows, b.qkv_rows);
        check(2560, 6144, t, a.value_rows, b.value_rows);
    }
    // Mapped Y consists of three 1024-element runs, never a contiguous 3072 half.
    for (int rank = 0; rank < 2; ++rank) {
        const auto& map = rank == 0 ? a.value_rows : b.value_rows;
        for (int run = 0; run < 3; ++run) for (int i = 0; i < 1024; ++i)
            require(map[run * 1024 + i] == run * 2048 + rank * 1024 + i);
    }
    for (int t : {2,6,7,8,11,12,13,14,16,17,18,20,21,22,23,29,42})
        check_columns(6144, 2560, t, a.value_rows, b.value_rows);
    for (int t : {2,6,7,8,20,42})
        check_columns(640, 2560, t, range(0,320), range(320,320));
    // Hybrid combines precisely these two native-byte ownership operations.
    // Attention projection keeps complete V-wide rows; FFN down keeps F/2
    // columns of every output row. No requantization or mapped-Y column split.
    for (int t : {8, 42}) check(6144, 2560, t, range(0,1280), range(1280,1280));
    for (int t : {20, 42}) check_columns(640, 2560, t, range(0,320), range(320,320));
    check_experts();
    rejects([]{ weight_column_spans(12, 640, 2, 720, range(0,320)); });
    rejects([]{ weight_column_spans(12, 1024, 2, 1152, range(0,320)); });
    rejects([]{ weight_column_spans(42, 640, 2, 360, range(32,320)); });
    rejects([]{ weight_column_spans(42, 640, 2, 360, range(384,320)); });
    rejects([]{ weight_column_spans(42, 640, 2, 359, range(0,320)); });
    rejects([]{ weight_column_spans(42, 640, 2, 360, {}); });
    rejects([]{ weight_column_spans(42, 640, 2, 360, range(-64,320)); });
    rejects([]{ auto c=range(0,320); c[17]=18; weight_column_spans(42,640,2,360,c); });
    rejects([]{ auto c=range(0,320); for(int i=64;i<128;++i)c[i]-=64; weight_column_spans(42,640,2,360,c); });
    rejects([]{ weight_column_spans(999,640,2,360,range(0,320)); });
    rejects([]{ weight_column_spans(0,INT64_MAX,INT64_MAX,360,{0}); });
    check(2560, 48, 30, a.value_heads, b.value_heads);
    check(4, 10240, 0, a.qkv_rows, b.qkv_rows);
    rejects([]{ weight_row_spans(42, 33, 2, 36, {0}); });
    rejects([]{ weight_row_spans(42, 64, 2, 35, {0}); });
    rejects([]{ weight_row_spans(42, 64, 2, 36, {2}); });
    rejects([]{ weight_row_spans(42, 64, 2, 36, {-1}); });
    rejects([]{ weight_row_spans(42, 64, 2, 36, {0,0}); });
    rejects([]{ weight_row_spans(999, 64, 2, 36, {0}); });
    rejects([]{ weight_row_spans(0, INT64_MAX, INT64_MAX, 36, {0}); });
    std::cout << "TP GDN row/column/hybrid geometry, canonical expert offsets and raw reconstruction passed\n";
}
