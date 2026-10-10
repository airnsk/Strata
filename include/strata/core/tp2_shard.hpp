#pragma once

// CPU-only layout prototype for one native (unchanged GGUF block) expert blob.
// Blob order is gate [F,N], up [F,N], down [N,F]. Each rank owns F/2
// consecutive gate/up rows; down is either matching F/2 columns (default)
// or N/2 complete output rows (test-only output-row TP layout). This is NOT the
// older packed Q2_0 expert layout with interleaved rows and separate planes.
// No GPU/runtime dispatch, quantization, activation packing or output reduction
// is performed here. In particular, byte-exact reassembly is not FFN parity.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace strata::core::tp2 {

struct BlockFormat {
    int type = 0;
    std::size_t values = 0;
    std::size_t bytes = 0;
    bool operator==(const BlockFormat&) const = default;
};

struct MatrixLayout {
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::size_t row_bytes = 0;
    std::size_t offset = 0;
    std::size_t bytes = 0;
    bool operator==(const MatrixLayout&) const = default;
};

struct ExpertLayout {
    MatrixLayout gate;
    MatrixLayout up;
    MatrixLayout down;
    std::size_t bytes = 0;
    bool operator==(const ExpertLayout&) const = default;
};

// Copy rows of raw blocks from the original blob to one compact rank blob.
// Gate/up regions are contiguous; down either gathers half of every row or
// copies a contiguous half of the complete output rows.
struct CopyRegion {
    std::size_t source_offset = 0;
    std::size_t destination_offset = 0;
    std::size_t rows = 0;
    std::size_t row_bytes = 0;
    std::size_t source_stride = 0;
    std::size_t destination_stride = 0;
    bool operator==(const CopyRegion&) const = default;
};

enum class DownSplit { Columns, OutputRows };

struct Plan {
    DownSplit down_split = DownSplit::Columns;
    int gu_type = 0;
    int down_type = 0;
    std::int64_t model_dim = 0;
    std::int64_t ffn_dim = 0;
    BlockFormat gu_format;
    BlockFormat down_format;
    ExpertLayout original;
    ExpertLayout shard;
    // [rank][gate, up, down]; gate/up own low/high F/2 rows.
    // Down owns low/high F/2 columns or low/high N/2 rows, per down_split.
    std::array<std::array<CopyRegion, 3>, 2> copies{};
    bool operator==(const Plan&) const = default;
};

namespace detail {
inline std::size_t add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("TP2 byte offset overflows size_t");
    return a + b;
}
inline std::size_t multiply(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("TP2 byte size overflows size_t");
    return a * b;
}
inline std::size_t dimension(std::int64_t value) {
    if (value <= 0) throw std::invalid_argument("TP2 dimensions must be positive");
    if (static_cast<std::uint64_t>(value) > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("TP2 dimension overflows size_t");
    return static_cast<std::size_t>(value);
}
inline BlockFormat gate_up_format(int type) {
    switch (type) {
        case 18: return {18, 256, 98};   // IQ3_XXS
        case 21: return {21, 256, 110};  // IQ3_S
        case 22: return {22, 256, 82};   // IQ2_S
        case 23: return {23, 256, 136};  // IQ4_XS
        default: throw std::invalid_argument("Unsupported TP2 gate/up GGUF type");
    }
}
inline BlockFormat down_format(int type) {
    switch (type) {
        case 20: return {20, 32, 18};  // IQ4_NL
        case 42: return {42, 64, 18};  // Q2_0, native blocks, not packed planes
        default: throw std::invalid_argument("Unsupported TP2 down GGUF type");
    }
}
inline MatrixLayout matrix(std::size_t rows, std::size_t columns,
                           BlockFormat format, std::size_t offset) {
    if (columns % format.values != 0)
        throw std::invalid_argument("TP2 row or split boundary cuts a quantization block");
    const auto row_bytes = multiply(columns / format.values, format.bytes);
    return {rows, columns, row_bytes, offset, multiply(rows, row_bytes)};
}
inline ExpertLayout expert(std::size_t n, std::size_t f,
                           BlockFormat gu, BlockFormat down) {
    ExpertLayout result;
    result.gate = matrix(f, n, gu, 0);
    result.up = matrix(f, n, gu, result.gate.bytes);
    result.down = matrix(n, f, down, add(result.up.offset, result.up.bytes));
    result.bytes = add(result.down.offset, result.down.bytes);
    return result;
}
}  // namespace detail

// Validate the native blob and its split boundaries. Column TP requires F/2
// aligned to down blocks; output-row TP requires full F down-block alignment
// and F/2 aligned to the 32-value q8_1 hidden activation blocks.
// Throws invalid_argument for unsupported types/dimensions/block alignment and
// overflow_error before any byte size or offset can wrap. Allocates no buffers.
inline Plan plan(int gu_type, int down_type, std::int64_t N, std::int64_t F,
                 DownSplit down_split = DownSplit::Columns) {
    Plan p;
    p.down_split = down_split;
    if (down_split != DownSplit::Columns && down_split != DownSplit::OutputRows)
        throw std::invalid_argument("Unknown TP2 down split");
    p.gu_type = gu_type;
    p.down_type = down_type;
    p.model_dim = N;
    p.ffn_dim = F;
    p.gu_format = detail::gate_up_format(gu_type);
    p.down_format = detail::down_format(down_type);
    const auto n = detail::dimension(N);
    const auto f = detail::dimension(F);
    if (f % 2 != 0) throw std::invalid_argument("TP2 requires an even FFN width");
    if (n % p.gu_format.values != 0 ||
        (down_split == DownSplit::Columns ? f / 2 : f) % p.down_format.values != 0)
        throw std::invalid_argument("TP2 row or split boundary cuts a quantization block");
    p.original = detail::expert(n, f, p.gu_format, p.down_format);
    // Row TP preserves full down rows. Its hidden all-gather boundary must
    // still coincide with a q8_1 activation block (32 values).
    if (down_split == DownSplit::OutputRows && ((f / 2) % 32 != 0 || n % 2 != 0))
        throw std::invalid_argument("TP2 output-row split requires aligned q8 halves and even N");
    p.shard.gate = detail::matrix(f / 2, n, p.gu_format, 0);
    p.shard.up = detail::matrix(f / 2, n, p.gu_format, p.shard.gate.bytes);
    p.shard.down = detail::matrix(down_split == DownSplit::Columns ? n : n / 2,
                                 down_split == DownSplit::Columns ? f / 2 : f,
                                 p.down_format, detail::add(p.shard.up.offset, p.shard.up.bytes));
    p.shard.bytes = detail::add(p.shard.down.offset, p.shard.down.bytes);
    for (std::size_t rank = 0; rank != 2; ++rank) {
        p.copies[rank][0] = {
            detail::add(p.original.gate.offset, rank * p.shard.gate.bytes),
            p.shard.gate.offset, f / 2, p.shard.gate.row_bytes,
            p.original.gate.row_bytes, p.shard.gate.row_bytes};
        p.copies[rank][1] = {
            detail::add(p.original.up.offset, rank * p.shard.up.bytes),
            p.shard.up.offset, f / 2, p.shard.up.row_bytes,
            p.original.up.row_bytes, p.shard.up.row_bytes};
        p.copies[rank][2] = {
            detail::add(p.original.down.offset, rank * (down_split == DownSplit::Columns ? p.shard.down.row_bytes : p.shard.down.bytes)),
            p.shard.down.offset, p.shard.down.rows, p.shard.down.row_bytes,
            p.original.down.row_bytes, p.shard.down.row_bytes};
    }
    return p;
}

namespace detail {
// Plan is inspectable and copyable. Reject an edited/default-constructed plan
// before allowing its offsets to address memory.
inline void validate(const Plan& p) {
    if (!(p == plan(p.gu_type, p.down_type, p.model_dim, p.ffn_dim, p.down_split)))
        throw std::invalid_argument("TP2 plan does not match its dimensions and formats");
}
inline void copy(const CopyRegion& region, const std::uint8_t* source,
                 std::uint8_t* destination, bool reverse) {
    for (std::size_t row = 0; row != region.rows; ++row) {
        const auto original = region.source_offset + row * region.source_stride;
        const auto shard = region.destination_offset + row * region.destination_stride;
        if (reverse) std::memcpy(destination + original, source + shard, region.row_bytes);
        else std::memcpy(destination + shard, source + original, region.row_bytes);
    }
}
}  // namespace detail

// Each returned blob preserves gate/up/down order and owns its raw bytes.
// Sizes must match exactly; trailing bytes (including other experts) are rejected.
inline std::vector<std::uint8_t> split(const Plan& p,
                                     std::span<const std::uint8_t> blob,
                                     unsigned rank) {
    detail::validate(p);
    if (rank >= 2) throw std::invalid_argument("TP2 rank must be zero or one");
    if (blob.size() != p.original.bytes)
        throw std::invalid_argument("TP2 original blob size does not match plan");
    std::vector<std::uint8_t> result(p.shard.bytes);
    for (const auto& region : p.copies[rank])
        detail::copy(region, blob.data(), result.data(), false);
    return result;
}

inline std::vector<std::uint8_t> reassemble(const Plan& p,
                                          std::span<const std::uint8_t> rank0,
                                          std::span<const std::uint8_t> rank1) {
    detail::validate(p);
    if (rank0.size() != p.shard.bytes || rank1.size() != p.shard.bytes)
        throw std::invalid_argument("TP2 shard blob size does not match plan");
    std::vector<std::uint8_t> result(p.original.bytes);
    const std::array<std::span<const std::uint8_t>, 2> shards{rank0, rank1};
    for (std::size_t rank = 0; rank != 2; ++rank)
        for (const auto& region : p.copies[rank])
            detail::copy(region, shards[rank].data(), result.data(), true);
    return result;
}

}  // namespace strata::core::tp2
