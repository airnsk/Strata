#pragma once
#include "strata/kernels/native_down_plan.hpp"

namespace strata::kernels::detail {
struct NativeDownPlanDeviceAtomics {
    __device__ void bit_or(int* p, int value) const { atomicOr(p, value); }
    __device__ void bit_or(uint32_t* p, uint32_t value) const { atomicOr(p, value); }
    __device__ int compare_exchange(int32_t* p, int before, int after) const {
        return atomicCAS(p, before, after);
    }
};

// Exactly one block, at least max(cap_groups,cap_entries) threads. All input
// writes must precede this call and all block threads must enter together.
__device__ inline void native_down_plan_build_block(
    const unsigned long long* ptr, const int32_t* start, const int32_t* n_groups,
    const int32_t* dst, int cap_groups, int cap_entries, NativeDownRoutePlan plan,
    uint32_t* error, int* global_bad) {
    const int tid = threadIdx.x, tokens = cap_entries / 10;
    const int ng = *n_groups;
    if (tid < cap_entries) { plan.entry[tid] = -1; plan.blob[tid] = 0; }
    if (tid < tokens) plan.token_error[tid] = 0;
    if (tid == 0) {
        *global_bad = ng <= 0 || ng > cap_groups;
        if (!*global_bad) *global_bad = start[0] != 0 || start[ng] != cap_entries;
    }
    __syncthreads();
    const NativeDownPlanDeviceAtomics atomics;
    if (ng > 0 && ng <= cap_groups && tid < ng)
        native_down_plan_visit_group(tid,ptr,start,dst,cap_entries,plan,global_bad,atomics);
    __syncthreads();
    if (tid < cap_entries) native_down_plan_check_missing(tid,plan,atomics);
    __syncthreads();
    if (tid < tokens) {
        if (*global_bad) plan.token_error[tid] = kNativeDownCombinePlanError;
        if (plan.token_error[tid]) atomicOr(error,kNativeDownCombinePlanError);
    }
}
} // namespace strata::kernels::detail

#include <cstddef>
#include <limits>
#include <stdexcept>
namespace strata::kernels::detail {
struct NativeDownPlanSpan { const void* p; size_t bytes, alignment; };
inline void native_down_plan_validate_spans(const NativeDownPlanSpan* spans, size_t count, size_t first_write) {
    for (size_t i=0;i<count;++i) {
        const auto& span=spans[i];
        const uintptr_t begin=reinterpret_cast<uintptr_t>(span.p);
        if (!span.p || begin%span.alignment || span.bytes>std::numeric_limits<uintptr_t>::max()-begin)
            throw std::invalid_argument("native down plan null, unaligned or wrapping span");
    }
    for (size_t w=first_write;w<count;++w) for (size_t other=0;other<w;++other) {
        const auto a=reinterpret_cast<uintptr_t>(spans[w].p),b=reinterpret_cast<uintptr_t>(spans[other].p);
        if (a<b+spans[other].bytes && b<a+spans[w].bytes)
            throw std::invalid_argument("native down plan output/error overlaps an input or each other");
    }
}
}
