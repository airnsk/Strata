#pragma once
// CPU-only raw-row gather plan. It preserves native blocks and rejects partial
// rows, duplicate ownership, overflow and invalid source spans before upload.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/tp2_shard.hpp"
#include <set>
namespace strata::core::tp2 {
struct WeightRowSpan { size_t source, destination, bytes; };
inline std::vector<WeightRowSpan> weight_row_spans(int type, int64_t width, int64_t count,
                                                  size_t source_bytes, const std::vector<int>& rows) {
    int elements = 0, bytes = 0;
    if (width <= 0 || count <= 0 || !strata::block_geometry(type, elements, bytes) || width % elements || rows.empty())
        throw std::invalid_argument("invalid native weight row geometry");
    const size_t pitch = detail::multiply(detail::dimension(width) / static_cast<size_t>(elements), static_cast<size_t>(bytes));
    if (source_bytes != detail::multiply(pitch, detail::dimension(count)))
        throw std::invalid_argument("native weight payload size mismatch");
    std::set<int> seen;
    std::vector<WeightRowSpan> spans;
    size_t destination = 0;
    for (const int r : rows) {
        if (r < 0 || r >= count || !seen.insert(r).second)
            throw std::invalid_argument("native weight row ownership is invalid");
        const size_t source = detail::multiply(static_cast<size_t>(r), pitch);
        if (!spans.empty() && detail::add(spans.back().source, spans.back().bytes) == source)
            spans.back().bytes = detail::add(spans.back().bytes, pitch);
        else spans.push_back({source, destination, pitch});
        destination = detail::add(destination, pitch);
    }
    return spans;
}
} // namespace strata::core::tp2
