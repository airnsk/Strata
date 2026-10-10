#pragma once
#include <cstdint>
// Test-only ordered combine. EP: rank0 has experts [0,g0), rank1 [g0,K).
// TP: both ranks have all K partial rows; add rank0+rank1 before router weighting.
void tp2_ffn_join(const float* rank0, const float* rank1, const float* weights,
                  float* expert_rows, float* output, int tokens, int k, int width,
                  int g0, bool tensor_parallel, void* stream);
// Peer-write test transport. Writing threads issue a system fence before kernel
// completion; cross-device events order consumers after those completed writes.
void tp2_input_push(const float* source, float* peer_destination, int values, void* stream);
void tp2_rank_reduce(const float* rows, const float* weights, float* destination,
                      int tokens, int k, int width, int local_count, int first,
                      bool peer_destination, void* stream);
void tp2_sum_vectors(const float* rank0, const float* rank1, float* output,
                      int values, void* stream);
