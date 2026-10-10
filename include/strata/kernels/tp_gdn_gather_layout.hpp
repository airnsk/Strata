#pragma once

// Raw-word destination layout shared by the local gather kernel and its CPU
// policy test. This introduces no arithmetic or quantization of payload words.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_TP_GATHER_HD __host__ __device__
#else
#define STRATA_TP_GATHER_HD
#endif
namespace strata::kernels::tp_gdn_gather_layout {
STRATA_TP_GATHER_HD constexpr int half_width(bool y) { return y ? 3072 : 1280; }
STRATA_TP_GATHER_HD constexpr int column(bool y, int rank, int local_column) {
    return y ? (local_column / 1024) * 2048 + rank * 1024 + local_column % 1024
             : rank * 1280 + local_column;
}
} // namespace strata::kernels::tp_gdn_gather_layout
#undef STRATA_TP_GATHER_HD
