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
int main() {
    const auto a = gdn_rank_layout(ModelGeometry{}, 0), b = gdn_rank_layout(ModelGeometry{}, 1);
    for (int t : {8, 12, 23, 30, 42}) {
        check(2560, 10240, t, a.qkv_rows, b.qkv_rows);
        check(2560, 6144, t, a.value_rows, b.value_rows);
    }
    check(2560, 48, 30, a.value_heads, b.value_heads);
    check(4, 10240, 0, a.qkv_rows, b.qkv_rows);
    rejects([]{ weight_row_spans(42, 33, 2, 36, {0}); });
    rejects([]{ weight_row_spans(42, 64, 2, 35, {0}); });
    rejects([]{ weight_row_spans(42, 64, 2, 36, {2}); });
    rejects([]{ weight_row_spans(42, 64, 2, 36, {-1}); });
    rejects([]{ weight_row_spans(42, 64, 2, 36, {0,0}); });
    rejects([]{ weight_row_spans(999, 64, 2, 36, {0}); });
    rejects([]{ weight_row_spans(0, INT64_MAX, INT64_MAX, 36, {0}); });
    std::cout << "TP GDN native row ranges and raw reconstruction passed\n";
}
