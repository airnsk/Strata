#pragma once

#include <cstdint>

namespace strata::kernels {

// Two-rank GDN transport for the validated N2560/F640/K10/V6144 geometry.
// CUDA/HIP only. Each call launches one allocation-free, graph-capturable kernel
// on the PRODUCING rank's stream. It writes this rank's disjoint slots in BOTH
// full buffers; the other rank must call the same operation with its own inputs.
// No arithmetic, quantization, padding reads, or truncation of trailing words.
//
// Preconditions checked on the host: T in [1,8], rank 0 or 1, nonnull 4-byte
// aligned pointers, nonwrapping and mutually disjoint buffer address ranges.
// Callers provide allocations covering ALL extents below, a stream on the source
// device, and enabled direct peer access to peer_full. Allocation/device ownership
// cannot be established from raw pointers here and must be checked by the owner.
//
// Ordering belongs to the caller: complete initial uploads/poisoning on BOTH
// devices before their first push, including work on legacy/default streams.
// Every writing thread system-fences its own writes before kernel completion.
// Record each producer's event AFTER its push, then make BOTH consumer streams
// wait for BOTH events before reading the assembled buffers. These inter-rank
// event edges remain outside rank-local graph capture. Do not reuse source or
// destination storage until previous readers/writers are ordered before reuse.
// Invalid descriptors throw before launch; no fallback or partial copy is used.

// src[T,3072] -> local_full[T,6144] and peer_full[T,6144]. Local value heads
// [0..23] map to global heads 16*(h/8) + 8*rank + h%8, with 128 floats/head.
void tp_gdn_push_y(const float* src, float* local_full, float* peer_full,
                   int T, int rank, void* stream);

// src[T,1280] -> rank's contiguous half of BOTH full [T,2560] output rows.
void tp_gdn_push_output(const float* src, float* local_full, float* peer_full,
                        int T, int rank, void* stream);

// Routed src[T*10,360 bytes] and shared src[T,360 bytes] contain native q8_1
// blocks for F/2=320. Copy rank's raw half into corresponding local/peer full
// rows of 720 bytes (F=640), preserving entry order. All six ranges are disjoint.
// The routed and shared transfers share one kernel and one completion event.
void tp_gdn_push_hidden(const uint8_t* routed_src, uint8_t* routed_local, uint8_t* routed_peer,
                        const uint8_t* shared_src, uint8_t* shared_local, uint8_t* shared_peer,
                        int T, int rank, void* stream);

}  // namespace strata::kernels
