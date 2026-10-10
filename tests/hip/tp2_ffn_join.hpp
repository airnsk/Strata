#pragma once
#include <cstdint>
// Test-only ordered combine. EP: rank0 has experts [0,g0), rank1 [g0,K).
// TP: both ranks have all K partial rows; add rank0+rank1 before router weighting.
void tp2_ffn_join(const float* rank0, const float* rank1, const float* weights,
                  float* expert_rows, float* output, int tokens, int k, int width,
                  int g0, bool tensor_parallel, void* stream);
