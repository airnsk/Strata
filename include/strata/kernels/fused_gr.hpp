// include/strata/kernels/fused_gr.hpp - plan v0.3 P3: the hyper-connection read in TWO kernels, with the
// previous half's write folded in.
//
// The native path spends six kernels per `gr_read` (norm, down MMVF, silu, up MMVF, gate+mean, inject MMVF) and
// one per `gr_write`, 96 + 96 times per token, and its up projection (10240 rows of 320) runs one 160-thread
// block per row: 20.6 us for 6.5 MB.  Here:
//
//   fused_gr_down : R' = R + bo_prev * 2 sigmoid(inj_prev / hc)  (only when `apply`, computed on the fly)
//                   rs[c] = rsqrt(mean(R'[c]^2) + eps),  xn = R' * w_norm * rs
//                   lo[k] = silu((w_down[k] . xn) / hc)          k < hc_lr
//                   inject[c] = w_inject[c] . xn                 when w_inject is given
//   fused_gr_up   : R <- R' in place for this block's columns (when `apply`)
//                   mixed[d] = mean_c  xn[c,d] * sigmoid(w_up[c*n_embd + d] . lo)
//
// FP32 activations and BF16 weights, like the native MMVF contract; the summation order differs from it (G-C
// judges the result).  Geometry is the artifact's: n_embd 2560, hc 4, hc_lr 320.  `inj_prev` and `inject_out`
// must be different buffers (every block reads the former while one block writes the latter).
#pragma once

#include <cstdint>

namespace strata::kernels {

struct FusedGrArgs {
    const float* R = nullptr;          ///< (hc, n_embd), read by `down`; `up` updates it in place when apply
    float* R_out = nullptr;            ///< == R for the in-place update
    bool apply = false;                ///< fold the previous half's gr_write
    const float* bo_prev = nullptr;    ///< that half's block output, n_embd
    const float* inj_prev = nullptr;   ///< that half's injection, hc
    const float* w_norm = nullptr;     ///< (hc * n_embd) f32
    const uint16_t* w_down = nullptr;  ///< bf16 [hc_lr][hc*n_embd]
    const uint16_t* w_up = nullptr;    ///< bf16 [hc*n_embd][hc_lr]
    const uint16_t* w_inject = nullptr;///< bf16 [hc][hc*n_embd], or null (the final mixer)
    float eps = 1e-6f;
    float* lo = nullptr;               ///< workspace, hc_lr floats
    float* rs = nullptr;               ///< workspace, hc floats
    float* inject_out = nullptr;       ///< hc floats (when w_inject)
    float* mixed = nullptr;            ///< n_embd
    /// S23 experiment (STRATA_HC_Q8=1): the GGUF's Q8_0 projections (null: the BF16 ones above); fused_gr_read_multi
    /// only
    const uint8_t* q8_down = nullptr;  ///< Q8_0 [hc_lr][hc*n_embd]
    const uint8_t* q8_up = nullptr;    ///< Q8_0 [hc*n_embd][hc_lr]
    const uint8_t* q8_inject = nullptr;///< Q8_0 [hc][hc*n_embd]
    /// S26 STRATA_QFUSE=1 (fused_gr_read_multi, the default and Q8_0 reads): also write `mixed`'s q8_1 image here (the
    /// bytes native_quantize_q8_1 would write); q8_cnt = n_embd / 32 zeroed counters owned by the caller (token 0's)
    uint8_t* q8_mixed = nullptr;
    unsigned* q8_cnt = nullptr;
};

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr);
void fused_gr_read(const FusedGrArgs& a, void* stream);

/// Plan v0.3 P6: the same read for up to 8 tokens that share the weights (a verify window): the weights are read
/// once for all of them.  `a[t]` is token t's arguments (its own R, pending write, lo, rs, inject, mixed; the four
/// weight pointers and eps must be the same for every t); `xn_scratch` is n_tok * hc * n_embd floats.  Every
/// token's outputs are bitwise `fused_gr_read(a[t])`.
constexpr int kFusedGrMaxT = 8;
/// Returns true when it also wrote the q8_1 images (every token's q8_mixed set and this read supports it).
bool fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream,
                         unsigned long long* stamp_buf = nullptr, int stamp_i0 = 0);

struct FusedGrRouter {
    const uint16_t* weights = nullptr; ///< BF16 [n_expert][2560], FP32 activations
    float* logits = nullptr;           ///< contiguous [n_tok][n_expert]
    int n_expert = 0;                   ///< 256 or 512
};
/// An optional router projection is folded into the persistent read only. With
/// router_written != nullptr, it is reset to false and set true ONLY when the
/// logits were produced; the caller must run its ordinary projection otherwise.
bool fused_gr_read_multi_router(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream,
                         unsigned long long* stamp_buf = nullptr, int stamp_i0 = 0,
                         const FusedGrRouter* router = nullptr, bool* router_written = nullptr);

/// Experimental Castagna-inspired persistent HC, gfx906 only, 1..4 tokens and BF16
/// HC weights. Build with STRATA_HC_PERSIST_BUILD=ON, then STRATA_HC_PERSIST=1
/// enables it; otherwise the original path is unchanged.
/// Unsupported devices, runtimes, token counts or Q8 HC overrides retain the original
/// kernels. Requires ROCm >= 6.4 and a supported cooperative launch. No token-path
/// allocation or device-global scratch. Configure before capture, never concurrently.
/// Test override: -1 = environment (default), 0 = off, 1 = on.
void fused_gr_set_persistent(int on);
/// Runtime/occupancy capability only, independent of the opt-in and weight format.
bool fused_gr_persistent_supported(int n_tok);
/// Test coverage: successful persistent submissions on this host thread. Capturing
/// a graph counts once; replay is submitted by the graph runtime and does not count.
unsigned long long fused_gr_persistent_launches();

#if defined(STRATA_HC_PERSIST_BUILD)
// Explicit test diagnostics only. Legacy variants: 0 original value helpers,
// 1 timed value, 2 untimed reference, 3 timed reference. Current production:
// 4 HC without router, 5 HC with router (both specialized for exact T).
struct FusedGrDiagnosticInfo {
    int registers = 0, max_threads = 0, active_per_cu = 0, blocks = 0;
    uint64_t static_lds_bytes = 0, local_bytes = 0, dynamic_lds_bytes = 0;
};
bool fused_gr_diagnostic_info(int variant, int n_tok, FusedGrDiagnosticInfo* info);
// Test fixture must meet the usual BF16 T<=4 contract. `blocks` must be within
// this variant's reported capacity; timed variants need blocks*7 device uint64s.
bool fused_gr_diagnostic_launch(const FusedGrArgs* a, int n_tok, float* xn, void* stream,
                                const FusedGrRouter* router, int variant, int blocks,
                                unsigned long long* phase_cycles);
#endif

/// The bench only: the AMD latency-hidden kernels on (1) or off (0); -1 = STRATA_GR_FAST.
void fused_gr_set_fast(int on);
/// The multi read's variants (#315; not main's opt-in STRATA_GR_V3 read, which sums in another order), all computing
/// every output with the plain read's operations in its order, so bitwise the plain read's and the single-token
/// read's: plain (0.1.31's default: the norm one block per token, the down projection on 41 blocks), split (the norm
/// one block per token and stream, then the plain down projection) and staged (split's norm, and the down
/// projection's activations staged ahead by cp.async, two half-stream tiles in flight, bank-conflict-free).
/// `fused_gr_check` runs all of them and the single-token read on the current card with random weights and inputs
/// (1..8 tokens, with and without the pending write) and from then on uses there the newest one that agrees with the
/// plain read bit for bit; STRATA_HC_SPLIT=0 keeps the plain read, =1 stops at split.  It runs once per card
/// (Verifier::init calls it) and prints which one runs.  On a card it has not checked, `fused_gr_variant` is the
/// plain read unless STRATA_HC_SPLIT=1 or 2 names a variant.
void fused_gr_check();
int fused_gr_variant();

}  // namespace strata::kernels
