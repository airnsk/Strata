#include "strata/core/tp_gdn_weight_rows.hpp"
#include "strata/core/tp_layer_layout.hpp"
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
int main() {
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
    std::cout << "TP GDN native row/column ranges and raw reconstruction passed\n";
}
