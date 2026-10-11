#pragma once

#include <cstdint>

namespace strata::kernels {

constexpr uint32_t kNativeDownCombinePlanError = uint32_t{1} << 30;

// GPU-owned inverse of a complete K10 grouped plan. Allocate cap_entries i32s,
// cap_entries u64s and n_tokens u32s in distinct spans. The builder overwrites
// every live element each proposal. Consumers must run later on the same stream
// (or after an event) and must not mutate/reuse the plan before down completes.
// Invalid group metadata invalidates all tokens; duplicate or missing destinations
// invalidate only their token, matching the old per-token down/combine blocks.
struct NativeDownRoutePlan {
    int32_t* entry = nullptr;
    unsigned long long* blob = nullptr;
    uint32_t* token_error = nullptr;
};

// Optional standalone builder for frozen-input diagnostics and metadata fault
// gates. Ordinary TP proposals fuse this work into resident_plan_with_inverse.
// Shape is T1..8, K10, 0 < cap_groups <= cap_entries == n_tokens*10. error is
// ORed, not reset. Inputs, inverse outputs and error must be aligned, sized and
// disjoint. This API and the preplanned down consumer are CUDA/HIP only.
void native_expert_down_plan(const unsigned long long* grp_ptr,
                            const int32_t* grp_start, const int32_t* n_groups,
                            const int32_t* ent_dst, int64_t cap_groups, int64_t cap_entries,
                            NativeDownRoutePlan plan, uint32_t* error, int n_tokens, void* stream);

namespace detail {
#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_DOWN_PLAN_FN __device__
#else
#define STRATA_DOWN_PLAN_FN
#endif
// Shared GPU/CPU-test phase logic. The GPU wrapper supplies atomics and block
// barriers between init, visit, missing-entry checks and final status publish.
template<class Atomics>
STRATA_DOWN_PLAN_FN inline void native_down_plan_visit_group(
    int g, const unsigned long long* ptr, const int32_t* start, const int32_t* dst,
    int cap_entries, NativeDownRoutePlan plan, int* global_bad, const Atomics& atomics) {
    const int begin = start[g], end = start[g + 1];
    const unsigned long long blob = ptr[g];
    if (begin < 0 || end <= begin || end > cap_entries || blob == 0) {
        atomics.bit_or(global_bad, 1);
        return;
    }
    for (int e = begin; e < end; ++e) {
        const int d = dst[e];
        if (d < 0 || d >= cap_entries) { atomics.bit_or(global_bad, 1); continue; }
        if (atomics.compare_exchange(plan.entry + d, -1, e) != -1)
            atomics.bit_or(plan.token_error + d / 10, kNativeDownCombinePlanError);
        else plan.blob[d] = blob;
    }
}

template<class Atomics>
STRATA_DOWN_PLAN_FN inline void native_down_plan_check_missing(
    int d, NativeDownRoutePlan plan, const Atomics& atomics) {
    if (plan.entry[d] < 0)
        atomics.bit_or(plan.token_error + d / 10, kNativeDownCombinePlanError);
}
#undef STRATA_DOWN_PLAN_FN
} // namespace detail
} // namespace strata::kernels
