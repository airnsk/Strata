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
// Copy each entry's raw q8 half into both full-width hidden buffers. No FP decode
// or requantization. Both destinations receive the same low/high half placement.
void tp2_hidden_push(const uint8_t* half, uint8_t* local, uint8_t* peer,
                     int entries, int half_bytes, int rank, void* stream);
// Concatenate low/high output-row shards (also used for route-reduced vectors).
void tp2_concat_rows(const float* low, const float* high, float* out,
                     int rows, int half_width, void* stream);
