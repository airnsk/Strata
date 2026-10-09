// Host-only eligibility policy for the experimental gfx906 persistent HC read.
#pragma once
#include <cstring>

namespace strata::kernels::detail {

inline bool hc_persistent_runtime_eligible(int runtime_version, const char* arch, bool cooperative) {
    // hipRuntimeGetVersion: major * 10000000 + minor * 100000 + patch.
    // 6.4 added direct cooperative-launch capture and cooperative graph replay.
    return runtime_version >= 60400000 && cooperative && arch != nullptr &&
           std::strncmp(arch, "gfx906", 6) == 0 && (arch[6] == '\0' || arch[6] == ':');
}

constexpr int hc_persistent_grid_blocks(int tokens, int active_per_cu, int cu_count) {
    if (tokens < 1 || tokens > 4 || active_per_cu <= 0 || cu_count <= 0) return 0;
    // At most 160 virtual up tiles. Divide before multiplying to avoid overflow
    // even for malformed capability data; occupancy always bounds a smaller grid.
    return active_per_cu > 160 / cu_count ? 160 : active_per_cu * cu_count;
}

}  // namespace strata::kernels::detail
