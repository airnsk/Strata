// CPU-only native-block layout checks. No claim of two-GPU arithmetic parity.
#include "strata/core/tp2_shard.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <utility>

using namespace strata::core::tp2;

// Do not use assert: the checks must also run in Release builds.
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL at line %d: %s\n", __LINE__, #condition); return false; \
} } while (false)

template<class Exception, class Fn>
bool throws(Fn&& fn) {
    try { fn(); }
    catch (const Exception&) { return true; }
    catch (...) { return false; }
    return false;
}

bool check_layout(int gu_type, int down_type, std::int64_t n, std::int64_t f,
                  std::size_t gu_block_bytes, DownSplit split_kind = DownSplit::Columns) {
    const auto p = plan(gu_type, down_type, n, f, split_kind);
    const bool row_tp = split_kind == DownSplit::OutputRows;
    const auto N = static_cast<std::size_t>(n);
    const auto F = static_cast<std::size_t>(f);
    const auto down_block_values = down_type == 20 ? 32U : 64U;
    // Independently calculate expected sizes rather than deriving them from
    // the copy regions that are also used by split/reassemble.
    const auto gu_row = N / 256 * gu_block_bytes;
    const auto down_row = F / down_block_values * 18;
    const auto gu_bytes = F * gu_row;
    const auto down_bytes = N * down_row;
    CHECK(p.gu_format.values == 256 && p.gu_format.bytes == gu_block_bytes);
    CHECK(p.down_format.values == down_block_values && p.down_format.bytes == 18);
    CHECK((p.original.gate == MatrixLayout{F, N, gu_row, 0, gu_bytes}));
    CHECK((p.original.up == MatrixLayout{F, N, gu_row, gu_bytes, gu_bytes}));
    CHECK((p.original.down == MatrixLayout{N, F, down_row, 2 * gu_bytes, down_bytes}));
    CHECK((p.shard.gate == MatrixLayout{F / 2, N, gu_row, 0, gu_bytes / 2}));
    CHECK((p.shard.up == MatrixLayout{F / 2, N, gu_row, gu_bytes / 2, gu_bytes / 2}));
    CHECK((p.shard.down == MatrixLayout{row_tp ? N / 2 : N, row_tp ? F : F / 2,
                                      row_tp ? down_row : down_row / 2, gu_bytes, down_bytes / 2}));
    CHECK(p.original.bytes == 2 * gu_bytes + down_bytes);
    CHECK(p.shard.bytes * 2 == p.original.bytes);

    std::vector<std::uint8_t> blob(p.original.bytes);
    // Non-periodic deterministic bytes exercise scales, metadata and codes
    // equally; the planner intentionally never interprets quantized content.
    std::uint32_t seed = 0x91357aceU;
    for (auto& byte : blob) {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        byte = static_cast<std::uint8_t>(seed);
    }
    const auto saved = blob;
    std::array<std::vector<std::uint8_t>, 2> shards;
    std::vector<unsigned char> coverage(blob.size(), 0);
    for (unsigned rank = 0; rank != 2; ++rank) {
        shards[rank] = split(p, blob, rank);
        const auto& shard = shards[rank];
        CHECK(shard.size() == p.shard.bytes);
        for (std::size_t byte = 0; byte != gu_bytes / 2; ++byte) {
            CHECK(shard[byte] == blob[rank * gu_bytes / 2 + byte]);
            CHECK(shard[gu_bytes / 2 + byte] == blob[gu_bytes + rank * gu_bytes / 2 + byte]);
        }
        for (std::size_t row = 0; row != (row_tp ? N / 2 : N); ++row) {
            for (std::size_t byte = 0; byte != (row_tp ? down_row : down_row / 2); ++byte) {
                CHECK(shard[gu_bytes + row * (row_tp ? down_row : down_row / 2) + byte] ==
                      blob[2 * gu_bytes + row * down_row + rank * (row_tp ? down_bytes / 2 : down_row / 2) + byte]);
            }
        }
        // The source ownership covers the original exactly once, including
        // the strided down matrix. Each destination byte is written once.
        std::vector<unsigned char> destination_coverage(shard.size(), 0);
        for (const auto& region : p.copies[rank]) {
            for (std::size_t row = 0; row != region.rows; ++row) {
                for (std::size_t byte = 0; byte != region.row_bytes; ++byte) {
                    const auto source = region.source_offset + row * region.source_stride + byte;
                    const auto destination = region.destination_offset + row * region.destination_stride + byte;
                    CHECK(source < coverage.size());
                    CHECK(destination < destination_coverage.size());
                    ++coverage[source];
                    ++destination_coverage[destination];
                }
            }
        }
        CHECK(std::all_of(destination_coverage.begin(), destination_coverage.end(),
                          [](unsigned char count) { return count == 1; }));
    }
    CHECK(std::all_of(coverage.begin(), coverage.end(), [](unsigned char count) { return count == 1; }));
    CHECK(blob == saved);
    CHECK(reassemble(p, shards[0], shards[1]) == blob);
    CHECK(throws<std::invalid_argument>([&] { split(p, blob, 2); }));
    CHECK(throws<std::invalid_argument>([&] { split(p, blob, std::numeric_limits<unsigned>::max()); }));
    CHECK(throws<std::invalid_argument>([&] { split(p, std::span(blob).first(blob.size() - 1), 0); }));
    CHECK(throws<std::invalid_argument>([&] { split(p, {}, 0); }));
    CHECK(throws<std::invalid_argument>([&] { reassemble(p, {}, shards[1]); }));
    CHECK(throws<std::invalid_argument>([&] { reassemble(p, shards[0], {}); }));
    auto extra_blob = blob;
    extra_blob.push_back(0);
    CHECK(throws<std::invalid_argument>([&] { split(p, extra_blob, 0); }));
    auto extra_shard = shards[0];
    extra_shard.push_back(0);
    CHECK(throws<std::invalid_argument>([&] { reassemble(p, extra_shard, shards[1]); }));
    auto corrupt = p;
    corrupt.copies[0][2].source_offset = std::numeric_limits<std::size_t>::max();
    CHECK(throws<std::invalid_argument>([&] { split(corrupt, blob, 0); }));
    CHECK(throws<std::invalid_argument>([&] { reassemble(corrupt, shards[0], shards[1]); }));
    return true;
}

bool check_invalid() {
    for (int type : {-1, 0, 1, 16, 17, 19, 20, 42, 99})
        CHECK(throws<std::invalid_argument>([&] { plan(type, 20, 2560, 640); }));
    for (int type : {-1, 0, 1, 18, 21, 22, 23, 99})
        CHECK(throws<std::invalid_argument>([&] { plan(18, type, 2560, 640); }));
    for (auto dim : {std::numeric_limits<std::int64_t>::min(), std::int64_t(-256),
                     std::int64_t(-1), std::int64_t(0)}) {
        CHECK(throws<std::invalid_argument>([&] { plan(18, 20, dim, 640); }));
        CHECK(throws<std::invalid_argument>([&] { plan(18, 20, 2560, dim); }));
    }
    CHECK(throws<std::invalid_argument>([] { plan(18, 20, 2559, 640); }));
    CHECK(throws<std::invalid_argument>([] { plan(18, 20, 2560, 641); }));
    CHECK(throws<std::invalid_argument>([] { plan(18, 20, 2560, 2); }));
    // Full rows can be aligned while half-width rows cut quantization blocks.
    CHECK(throws<std::invalid_argument>([] { plan(18, 20, 2560, 96); }));
    CHECK(throws<std::invalid_argument>([] { plan(18, 42, 2560, 192); }));
    const auto huge = std::numeric_limits<std::int64_t>::max() / 256 * 256;
    CHECK(throws<std::overflow_error>([&] { plan(23, 20, huge, 640); }));
    CHECK(throws<std::overflow_error>([&] { plan(23, 42, 2560, huge); }));
    CHECK(throws<std::overflow_error>([&] { plan(18, 20, huge, huge); }));
    // A case where each matrix fits in size_t, but the combined blob does not.
    if constexpr (sizeof(std::size_t) == 8) {
        constexpr std::int64_t N = 256;
        constexpr auto F = (std::int64_t(1) << 56);
        CHECK(throws<std::overflow_error>([] { plan(23, 20, N, F); }));
    }
    CHECK(throws<std::invalid_argument>([] { plan(18, 20, 2560, 96, DownSplit::OutputRows); }));
    CHECK(throws<std::invalid_argument>([] { plan(18, 20, 2560, 640, static_cast<DownSplit>(99)); }));
    CHECK(throws<std::overflow_error>([&] { plan(23, 42, huge, huge, DownSplit::OutputRows); }));
    CHECK(throws<std::invalid_argument>([] { split(Plan{}, {}, 0); }));
    return true;
}

int main() {
    const std::array<std::pair<int, std::size_t>, 4> types{{{18, 98}, {21, 110}, {22, 82}, {23, 136}}};
    for (const auto& [gu_type, gu_block_bytes] : types) {
        for (int down_type : {20, 42}) {
            if (!check_layout(gu_type, down_type, 2560, 640, gu_block_bytes)) return 1;
            for (auto dims : {std::pair{2560,640}, std::pair{256,192}, std::pair{512,768}})
                if (!check_layout(gu_type, down_type, dims.first, dims.second, gu_block_bytes,
                                  DownSplit::OutputRows)) return 1;
            // The planner is dimension-driven rather than hard-coded to 2560/640.
            if (!check_layout(gu_type, down_type, 256, down_type == 20 ? 64 : 128,
                              gu_block_bytes)) return 1;
            if (!check_layout(gu_type, down_type, 512, 768, gu_block_bytes)) return 1;
        }
    }
    if (!check_invalid()) return 1;
    std::puts("PASS: TP2 native block sizes, rank ownership, raw split/reassembly, alignment and overflow (CPU only)");
    return 0;
}
