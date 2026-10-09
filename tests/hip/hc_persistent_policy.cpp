// CPU-only checks. These validate dispatch policy, not GPU arithmetic or residency.
#include "strata/kernels/hc_persistent_policy.hpp"
#include <climits>
#include <initializer_list>
#include <cstdio>

using namespace strata::kernels::detail;

// Keep checks enabled in Release builds as well.
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL at line %d: %s\n", __LINE__, #condition); return 1; \
} } while (false)

int main() {
    CHECK(!hc_persistent_runtime_eligible(60300000, "gfx906", true));
    CHECK(!hc_persistent_runtime_eligible(60399999, "gfx906", true));
    CHECK(hc_persistent_runtime_eligible(60400000, "gfx906", true));
    CHECK(hc_persistent_runtime_eligible(70200000, "gfx906:sramecc+:xnack-", true));
    CHECK(!hc_persistent_runtime_eligible(70200000, "gfx906", false));
    for (const char* arch : {"", "gfx90", "gfx9060", "gfx906junk", "gfx908", "gfx1100"})
        CHECK(!hc_persistent_runtime_eligible(70200000, arch, true));
    CHECK(!hc_persistent_runtime_eligible(70200000, nullptr, true));
    for (int t = -1; t <= 9; ++t) {
        const bool supported = t >= 1 && t <= 4;
        CHECK(hc_persistent_grid_blocks(t, 1, 60) == (supported ? 60 : 0));
        CHECK(hc_persistent_grid_blocks(t, 2, 60) == (supported ? 120 : 0));
        CHECK(hc_persistent_grid_blocks(t, 3, 60) == (supported ? 160 : 0));
    }
    CHECK(hc_persistent_grid_blocks(4, 0, 60) == 0);
    CHECK(hc_persistent_grid_blocks(4, 1, 0) == 0);
    CHECK(hc_persistent_grid_blocks(4, -1, 60) == 0);
    CHECK(hc_persistent_grid_blocks(4, INT_MAX, INT_MAX) == 160);
    // Every virtual phase tile has exactly one owner for every permitted grid.
    for (int blocks = 1; blocks <= 160; ++blocks) {
        for (int tiles : {4, 41, 80, 160, 256, 512}) {
            int seen[512] = {};
            for (int block = 0; block < blocks; ++block)
                for (int tile = block; tile < tiles; tile += blocks) ++seen[tile];
            for (int tile = 0; tile < tiles; ++tile) CHECK(seen[tile] == 1);
        }
    }
    std::puts("PASS: persistent HC runtime/shape/occupancy policy and virtual tile ownership (CPU only)");
}
