#pragma once
#include <stdexcept>
namespace strata::core::tp2 {
inline int hc_down_source_row(int rank, int local) {
    if ((rank != 0 && rank != 1) || local < 0 || local >= 160)
        throw std::invalid_argument("TP HC down coordinate");
    return rank * 160 + local;
}
inline int hc_up_source_row(int rank, int stream, int channel) {
    if ((rank != 0 && rank != 1) || stream < 0 || stream >= 4 || channel < 0 || channel >= 1280)
        throw std::invalid_argument("TP HC up coordinate");
    return stream * 2560 + rank * 1280 + channel;
}
} // namespace strata::core::tp2
