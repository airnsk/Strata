#pragma once
#include "strata/kernels/fused_gr.hpp"

namespace strata::kernels {
// Exact BF16 output-row partition of the ordinary fused HC read. T1..8,
// rank0/1, N2560/HC4/LR320. Arguments retain FULL canonical matrix pointers:
// this partitions compute/weight reads, not the immutable allocations yet.
// No Q8 HC overrides or fused-router path; original FP32 activations are used.
// Output-changing STRATA_GR_V3/STRATA_GR_SPLIT policies are refused.
// The caller owns all buffers and must order collectives between these calls.
// Streams explicit, no allocation/synchronization; errors throw before launch.
// Compile with the same floating-point options as fused_gr.cu.
//
// down: replicate original full-stream norm and four inject dots; compute rank's
// 160 complete down rows, then original scale+SiLU -> local_lo[T,160].
// xn_scratch[T,10240] and args[t].rs[4] are outputs. When apply=true, materialize
// ALL R_out[10240] with the original pending-write FMA exactly once. R_out may
// equal R, all other active mutable spans must be disjoint. args.lo/mixed unused.
void tp_hc_down(const FusedGrArgs* args, int T, float* xn_scratch, float* local_lo,
                int rank, void* stream);
// Caller gathers both local_lo halves to full_lo[T,320] before this call.
// up: each owned channel computes ALL four HC up dots using full320 lo, then
// original sigmoid, ordered HC sum and /4 -> local_mixed[T,1280]. Reads R_out
// when apply=true, otherwise R. It NEVER repeats the pending residual write.
// Caller gathers local_mixed into full mixed before consumers. Original args
// remain unchanged and must match the preceding down invocation.
void tp_hc_up(const FusedGrArgs* args, int T, const float* full_lo, float* local_mixed,
              int rank, void* stream);
} // namespace strata::kernels
