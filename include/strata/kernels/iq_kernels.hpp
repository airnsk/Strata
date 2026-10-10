// include/strata/kernels/iq_kernels.hpp - the i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
// IQ4_NL) and Q2_0 on the GPU for the IQ2_XS / IQ3_XXS model files, and Q4_K / Q5_K / Q5_1 / Q8_0 for Unsloth's
// UD-Q4_K_XL (gate/up Q4_K or Q5_K, down Q5_1 or Q8_0, a Q8_0 embedding).
//
// The block layouts, codebook grids and dot products are llama.cpp's (ggml-common.h, ggml-cuda/vecdotq.cuh,
// ggml-cuda/dequantize.cuh; MIT, see third_party/ggml/LICENSE and VERSION.txt), so a weight means exactly what it
// means in llama.cpp.  Activations are q8_1 (32 values, fp16 scale and fp16 sum), the llama.cpp CUDA contract.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {

/// ggml type ids handled here.
bool iq_supported(int ggml_type) noexcept;
/// The token-embedding types iq_embed_rows and iq_dequant_f32 read: the i-quants above and BF16 (30).
bool embed_type_supported(int ggml_type) noexcept;
/// Bytes of one row of `n` values of `ggml_type` (n a multiple of the type's block).
size_t iq_row_bytes(int ggml_type, int64_t n) noexcept;

/// q8_1 blocks for `n_rows` rows of `n_cols` floats (n_cols a multiple of 32): y is n_rows * n_cols/32 blocks.
void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream);

/// y[c][r] = W[r] . x[c] for `ncols` columns of q8_1 activations (x stride n_in/32 blocks per column).
void iq_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

/// Dequantize `n` contiguous values (n a multiple of 256) to fp16 / fp32.
void iq_dequant_f16(int ggml_type, const void* src, int64_t n, uint16_t* dst, void* stream);
void iq_dequant_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream);
/// Rows `tokens[0..n_tok)` (device ids) of a GGUF embedding table (`row_bytes` per row; the table may be mapped
/// host memory) dequantized to fp32, `n_embd` per row (a multiple of 256).
void iq_embed_rows(int ggml_type, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                   int64_t n_embd, float* out, void* stream);
/// One expert's gate and up matrices (n_ff rows of n_embd each) into the interleaved fp16 layout the prompt path
/// uses: row 2r = gate row r, row 2r+1 = up row r.
void iq_dequant_gu_f16(int ggml_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                       void* stream);

/// The layout of one native expert blob: [gate rows | up rows | down rows], raw GGUF blocks.
struct NativeExpertLayout {
    int gu_type = -1, d_type = -1;
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       // bytes per row
    size_t up_off = 0, down_off = 0;    // byte offsets inside the blob
    size_t bytes = 0;                   // the whole blob
};
NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
/// Whether `native_expert_grouped` has kernels for this gate/up and down type pair at these dimensions, and the
/// prompt path's dequantizer takes both (checked for every layer at startup, before anything is allocated).
bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept;

/// Bytes of scratch `native_expert_grouped` needs for `cap_entries` entries.
size_t native_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff);

/// Grouped experts in the native format: group g's blob at device address grp_ptr[g]; its entries
/// [grp_start[g], grp_start[g+1]) read token ent_tok[e]'s q8_1 activation (n_embd/32 blocks per token in x_q8_1)
/// and write row ent_dst[e] of `out` (n_embd floats).  Counts are read on the device.
/// `grid_groups` (1 .. cap_groups; 0 = cap_groups) groups run side by side, a block row each striding over the rest:
/// a call that usually has few groups or none (the verify window's PCIe share) launches less for the ones it does
/// not have.  The results do not depend on it.
void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           int64_t grid_groups = 0);
/// Call-local phases. GateUp writes quantized SwiGLU rows into scratch; Down reads them there.
enum class NativeExpertPhase { Full = 0, GateUp = 1, Down = 2 };
struct NativeExpertCallOptions {
    NativeExpertPhase phase = NativeExpertPhase::Full;
    int mode = -1;  // environment/device default, ignoring native_expert_set_mode; otherwise an explicit layout
};
/// Syntactic validation, independent of the backend. Mode 1 additionally requires gfx906;
/// HIP targets without experimental layouts accept only -1 and 0. Mode 3 has no implementation.
inline bool native_expert_call_options_valid(const NativeExpertCallOptions& options) noexcept {
    const bool phase = options.phase == NativeExpertPhase::Full || options.phase == NativeExpertPhase::GateUp ||
                       options.phase == NativeExpertPhase::Down;
    const int m = options.mode;
    return phase && (m == -1 || m == 0 || m == 1 || m == 2 || (m >= 4 && m <= 8));
}
/// Byte offset of the entry-major q8_1 hidden rows in native_expert_scratch_bytes(cap_entries, n_ff).
/// Each row holds n_ff/32 blocks of 36 bytes. Separate GU/down views need separate scratch allocations:
/// gather GU hidden rows into the down view's hidden rows before launching Down on the consumer stream.
/// Throws on negative capacity, nonpositive/non-32-aligned width or size_t overflow.
inline size_t native_expert_hidden_q8_offset(int64_t cap_entries, int64_t n_ff) {
    if (cap_entries < 0 || n_ff <= 0 || n_ff % 32 != 0)
        throw std::invalid_argument("native expert hidden scratch dimensions");
    const auto limit = std::numeric_limits<size_t>::max();
    if (static_cast<uint64_t>(cap_entries) > limit || static_cast<uint64_t>(n_ff) > limit)
        throw std::overflow_error("native expert hidden scratch size");
    const size_t cap = static_cast<size_t>(cap_entries), width = static_cast<size_t>(n_ff);
    if (width > limit / sizeof(float) || cap > (limit - 255) / (width * sizeof(float)))
        throw std::overflow_error("native expert hidden scratch size");
    const size_t aligned = (cap * width * sizeof(float) + 255) & ~size_t{255};
    if (aligned > limit / 3) throw std::overflow_error("native expert hidden scratch size");
    return 3 * aligned;
}
/// CUDA/HIP call-local dispatch: never reads or changes the benchmark's phase/mode overrides.
/// Unchanged kernels, scratch format and group metadata contract from native_expert_grouped above.
/// GateUp uses only gu_type, n_embd (input width), n_ff (local hidden width), gu_row and up_off;
/// Down uses only d_type, n_embd (local output width), n_ff (gathered hidden width), d_row and down_off.
/// Active matrix ranges must fit L.bytes; inactive type/row/offset fields are ignored. Thus separate layouts
/// can describe GU N=2560,F=320 and down N=1280,F=640 over the same blob using its actual down_off.
/// GateUp may pass null out/ent_dst; Down may pass null x_q8_1/ent_tok. Shared metadata and scratch are required.
/// Invalid options, backend modes or active layout fields throw before any launch. Explicit split phases
/// never use full-FFN V2/V2K. Full with mode=-1 retains these environment fast paths; explicit modes bypass them.
/// Other kernel tuning switches keep their startup settings; do not mutate benchmark switches concurrently.
/// This entry point has no SYCL implementation; SYCL callers must retain native_expert_grouped.
void native_expert_grouped_explicit(const NativeExpertLayout& L, const unsigned long long* grp_ptr,
                                    const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                                    const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                                    const void* x_q8_1, void* scratch, float* out, void* stream,
                                    const NativeExpertCallOptions& options, int64_t grid_groups = 0);
/// Opt-in column-TP local FFN producer, K=10, T=1..8, N=2560, F=320 (or full F=640).
/// Down types: IQ4_NL(20), Q2_0(42). GU type is immaterial: hidden_q8 is the
/// already computed entry-major q8_1 payload, NOT the scratch allocation base.
/// Group metadata has exactly cap_entries=T*10 entries and ent_dst is a bijection
/// to [token*10+route_slot]. Each block builds that inverse, so grouped expert order
/// never changes routing accumulation order. shared_partial is [T,N] before its
/// scalar sigmoid gate. No expert-parts tensor is materialized.
/// Finish each native 32-lane dot before ordered route weighting; preserve 0+hit,
/// first rounded product, later FMAs and separately rounded shared-gate product.
/// Column splitting and final rank addition still reassociate the full-model dot;
/// this API makes NO full-reference bitwise-equality promise.
/// error must be zeroed by the caller before route planning. Invalid device-side
/// metadata ORs kNativeDownCombinePlanError and writes safe zeros, never indexes
/// missing entries. The owner must reject the proposal when error is nonzero.
/// Inputs, output and error spans must be aligned, sufficiently sized and disjoint
/// from writable outputs. Shape/layout and visible pointer spans are checked on host.
/// Optional peer_output is a distinct full-sized peer inbox. Every writer publishes
/// its local and peer stores with a system fence; consumers require a stream event join.
constexpr uint32_t kNativeDownCombinePlanError = uint32_t{1} << 30;
void native_expert_down_combine(const NativeExpertLayout& L, const unsigned long long* grp_ptr,
                                const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                                int64_t cap_groups, int64_t cap_entries, const void* hidden_q8,
                                const float* route_weights, const float* shared_partial, const float* shared_gate,
                                float* out, uint32_t* error, int n_tokens, void* stream, float* peer_output = nullptr);

/// true: `native_expert_grouped`'s launches before the group stride (STRATA_GROUPED_V1=1 at startup) - a block row
/// per possible group, SwiGLU and the q8_1 quantization as two kernels over all cap_entries.  Bitwise the same results
/// (native_grouped_parity checks it); kept for A/B timing.  Set before graph capture; captured graphs keep theirs.
void native_grouped_set_v1(bool v1);

/// The bench only: the AMD kernel layout (STRATA_EXP_MODE values; -1 = the environment's) and the phase
/// (0 all, 1 gate/up + SwiGLU + quantize, 2 down).
void native_expert_set_mode(int mode, int phase);
/// `iq_mmvq` and `native_expert_grouped` decode each weight part once and apply it to every column / entry;
/// true selects the older kernels that decode it again per column (STRATA_OLD_IQ_MMVQ=1 at startup).  Both give
/// bitwise the same results.  Set before graph capture; captured graphs keep the kernels they captured.
void iq_set_old_kernels(bool old);
bool iq_old_kernels();

}  // namespace strata::kernels
