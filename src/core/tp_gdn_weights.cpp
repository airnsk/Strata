#include "strata/core/tp_gdn_weights.hpp"
#include "strata/core/tp2_shard.hpp"
#include "strata/core/tp_gdn_weight_rows.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <fstream>
#include <array>
#include <filesystem>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace strata::core::tp2 {
namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
std::vector<int> rows(int begin, int count) {
    std::vector<int> v(static_cast<size_t>(count));
    std::iota(v.begin(), v.end(), begin); return v;
}
struct Tensor {
    const strata::TensorInfo* info;
    const uint8_t* data;
};
// Validate every directory span, not merely selected tensors: an ignored
// tensor must not overlap a payload we will read. Unknown formats fail closed.
void validate_directory(const strata::GgufModel& model) {
    for (size_t s = 0; s < model.size(); ++s) {
        const auto& f = model.shard(s);
        std::vector<const strata::TensorInfo*> tensors;
        for (const auto& t : f.tensors()) tensors.push_back(&t);
        std::sort(tensors.begin(), tensors.end(), [](auto a, auto b) {return a->offset < b->offset;});
        uint64_t end = 0;
        for (const auto* t : tensors) {
            if (!model.in_bounds(*t, s) || t->offset < end)
                throw std::runtime_error("invalid/overlapping GGUF span: " + t->name);
            end = t->offset + strata::tensor_payload_bytes(*t);
        }
    }
}
Tensor tensor(const strata::GgufModel& model, const std::string& name,
              std::vector<uint64_t> shape) {
    size_t shard = 0;
    const auto* t = model.find(name, &shard);
    if (!t || t->shape != shape || !model.in_bounds(*t, shard))
        throw std::runtime_error("missing tensor or incompatible shape/span: " + name);
    return {t, model.shard(shard).tensor_data(*t)};
}
}
TpGdnWeights::~TpGdnWeights() { clear(); }
void TpGdnWeights::clear() noexcept {
    if (device_ >= 0) {
        int old = -1; cudaGetDevice(&old);
        if (cudaSetDevice(device_) == cudaSuccess) {
            for (auto* p : allocations_) cudaFree(p);
        }
        if (old >= 0) cudaSetDevice(old);
    }
    allocations_.clear(); bytes_ = 0; device_ = -1; weights_ = {};
}
bool TpGdnWeights::load(const std::vector<std::string>& shards, const std::string& pack_dir,
                       const ModelGeometry& g, int layer, int rank, int device, std::string& err) {
    if (!allocations_.empty()) { err = "TP GDN weights already loaded"; return false; }
    int previous = -1;
    try {
        const auto layout = gdn_rank_layout(g, rank < 0 ? 0 : rank);
        if (rank < -1 || layer < 0 || layer >= g.n_layers || layer == 1 || is_qsa_layer(g, layer))
            throw std::runtime_error("requires a valid non-PLE GDN layer and rank -1, 0 or 1");
        strata::GgufModel model(shards);
        strata::Qwen4ExpGuard guard; guard.experts = 512; guard.experts_used = 10;
        const auto architecture = strata::check_architecture(model.meta(), guard);
        if (!architecture.empty()) throw std::runtime_error(architecture);
        validate_directory(model);
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        // Build a whitelist, then skip EVERY other canonical row, including
        // pack-only tensors. Never size or allocate a whole-model arena.
        std::set<std::string> keep;
        for (const char* suffix : {"hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight",
             "hc_attn_inject.weight", "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight",
             "hc_ffn_inject.weight", "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight",
             "ssm_alpha.weight", "ssm_beta.weight", "ssm_dt.bias", "ssm_a", "ssm_conv1d.weight", "ssm_norm.weight"})
            keep.insert(prefix + suffix);
        std::ifstream index(pack_dir + "/index.txt");
        if (!index) throw std::runtime_error("cannot open canonical pack index");
        std::set<std::string> skip, seen;
        std::string line, name;
        bool valid_header = false;
        while (std::getline(index, line)) {
            std::istringstream in(line);
            if (!(in >> name)) throw std::runtime_error("empty canonical index row");
            if (name[0] == '#') {
                std::string label;
                if (name == "#" && (in >> label) && label == "align") {
                    int64_t alignment = 0;
                    if (!(in >> alignment) || alignment <= 0 || alignment > 4096 || (alignment & (alignment - 1)))
                        throw std::runtime_error("invalid canonical index alignment");
                    valid_header = true;
                }
                continue;
            }
            if (line.size() >= 1000 || name.size() > 255 || !seen.insert(name).second)
                throw std::runtime_error("oversized or duplicate canonical tensor row: " + name);
            std::array<int64_t, 18> fields{};
            for (auto& field : fields) if (!(in >> field)) throw std::runtime_error("malformed canonical index row: " + name);
            std::string extra;
            if (in >> extra) throw std::runtime_error("trailing canonical index fields: " + name);
            // Even skipped rows produce WeightRef metadata in the canonical
            // loader, so guard their signed element multiplication as well.
            if (fields[6] <= 0 || fields[7] < 0 ||
                fields[6] > INT64_MAX / (fields[7] > 0 ? fields[7] : 1))
                throw std::runtime_error("invalid canonical dimensions: " + name);
            if (!keep.count(name)) { skip.insert(name); continue; }
            const auto* src = model.find(name);
            const bool f32 = name.ends_with("norm.weight") || name.ends_with("ssm_a") ||
                             name.ends_with("ssm_dt.bias") || name.ends_with("ssm_conv1d.weight");
            const int64_t count = fields[6] * (fields[7] > 0 ? fields[7] : 1);
            if (!src || src->shape.size() > 2 || src->shape[0] != static_cast<uint64_t>(fields[6]) ||
                (src->shape.size() == 2 ? src->shape[1] : 1) != static_cast<uint64_t>(fields[7] > 0 ? fields[7] : 1) ||
                (f32 ? fields[1] != 2 : (fields[1] != 1 && fields[1] != 4)) ||
                count > (64ll << 20) / 4 || fields[5] != count * (f32 ? 4 : 2) ||
                fields[3] != count * (fields[1] == 4 ? 2 : 4) || fields[2] < 0 || fields[4] < 0 ||
                fields[8] || fields[11] || fields[12] || fields[13] || fields[14] || fields[15] || fields[16] ||
                (fields[0] != 0 && fields[0] != 3))
                throw std::runtime_error("unsupported canonical representation: " + name);
            const auto file_size = std::filesystem::file_size(pack_dir + (fields[0] == 0 ? "/dense.bin" : "/extra.bin"));
            const auto offset = static_cast<uint64_t>(fields[2]);
            if (offset > file_size || static_cast<uint64_t>(fields[3]) > file_size - offset)
                throw std::runtime_error("canonical source span outside file: " + name);
        }
        if (!valid_header) throw std::runtime_error("canonical index has no valid alignment header");
        for (const auto& n : keep) if (!seen.count(n)) throw std::runtime_error("missing canonical tensor: " + n);
        uint64_t canonical_bytes = 0;
        if (!WeightTable::pool_bytes(pack_dir, canonical_bytes, err, &skip)) throw std::runtime_error(err);
        if (!canonical_bytes || canonical_bytes > (64ull << 20))
            throw std::runtime_error("selected canonical layer exceeds 64 MiB safety bound");
        ck(cudaGetDevice(&previous)); ck(cudaSetDevice(device)); device_ = device;
        auto alloc = [&](size_t n) {
            if (!n) throw std::runtime_error("zero-sized allocation");
            void* p = nullptr; ck(cudaMalloc(&p, n));
            try { allocations_.push_back(p); } catch (...) { cudaFree(p); throw; }
            bytes_ += n; return p;
        };
        void* canonical = alloc(static_cast<size_t>(canonical_bytes));
        WeightTable table;
        if (!table.load(pack_dir, canonical, canonical_bytes, err, &skip)) throw std::runtime_error(err);
        auto canonical_view = [&](const char* suffix, int width, int count, WeightKind kind) -> const WeightRef& {
            const auto n = prefix + suffix;
            const auto* r = table.find(n);
            size_t s = 0; const auto* src = model.find(n, &s);
            if (!r || !src || src->shape.empty() || src->shape.size() > 2 ||
                src->shape[0] != static_cast<uint64_t>(width) ||
                (src->shape.size() == 2 ? src->shape[1] : 1) != static_cast<uint64_t>(count) ||
                r->ne0 != width || (r->ne1 > 0 ? r->ne1 : 1) != count ||
                r->elements != static_cast<int64_t>(width) * count || r->kind != kind || r->quantized() ||
                !r->resident || !r->data || r->bytes != static_cast<uint64_t>(width) * count * (kind == WeightKind::F32 ? 4 : 2))
                throw std::runtime_error("incompatible canonical tensor: " + n);
            return *r;
        };
        auto small = [&](const char* suffix, int width, int count, WeightKind kind, const std::vector<int>& selected) {
            const auto& r = canonical_view(suffix, width, count, kind);
            const size_t row_bytes = static_cast<size_t>(width) * (kind == WeightKind::F32 ? 4 : 2);
            auto* dst = static_cast<uint8_t*>(alloc(detail::multiply(selected.size(), row_bytes)));
            for (size_t i = 0; i < selected.size(); ++i) {
                if (selected[i] < 0 || selected[i] >= count) throw std::runtime_error("canonical row out of range");
                ck(cudaMemcpy(dst + i * row_bytes, static_cast<const uint8_t*>(r.data) + selected[i] * row_bytes,
                              row_bytes, cudaMemcpyDeviceToDevice));
            }
            return static_cast<const void*>(dst);
        };
        auto native = [&](const char* suffix, int width, int count, const std::vector<int>& selected) {
            const auto t = tensor(model, prefix + suffix, {static_cast<uint64_t>(width), static_cast<uint64_t>(count)});
            if (!kernels::native_mmvq_supported(t.info->type)) throw std::runtime_error("unsupported native matrix: " + prefix + suffix);
            const size_t payload = static_cast<size_t>(strata::tensor_payload_bytes(*t.info));
            const auto spans = weight_row_spans(t.info->type, width, count, payload, selected);
            const size_t bytes = detail::add(spans.back().destination, spans.back().bytes);
            auto* dst = static_cast<uint8_t*>(alloc(bytes));
            for (const auto& span : spans)
                ck(cudaMemcpy(dst + span.destination, t.data + span.source, span.bytes, cudaMemcpyHostToDevice));
            return TpNativeMatrix{static_cast<int>(t.info->type), dst, width, static_cast<int>(selected.size())};
        };
        TpGdnRankWeights w; w.rank = rank; w.device = device; w.layer = layer;
        const bool full = rank == -1;
        const auto qkv = full ? rows(0, 10240) : layout.qkv_rows;
        const auto hv = full ? rows(0, 48) : layout.value_heads;
        const auto vr = full ? rows(0, 6144) : layout.value_rows;
        const auto output = rows(full ? 0 : rank * 1280, full ? 2560 : 1280);
        const auto hidden = rows(full ? 0 : rank * 320, full ? 640 : 320);
        w.qkv = native("attn_qkv.weight", 2560, 10240, qkv);
        w.z = native("attn_gate.weight", 2560, 6144, vr);
        w.out = native("ssm_out.weight", 6144, 2560, output);
        w.shared_gate = native("ffn_gate_shexp.weight", 2560, 640, hidden);
        w.shared_up = native("ffn_up_shexp.weight", 2560, 640, hidden);
        w.shared_down = native("ffn_down_shexp.weight", 640, 2560, output);
        for (int i = 0; i < 2; ++i) {
            const std::string h = i == 0 ? "hc_attn_" : "hc_ffn_";
            w.hc[i].norm = static_cast<const float*>(canonical_view((h + "norm.weight").c_str(), 10240, 1, WeightKind::F32).data);
            w.hc[i].down = static_cast<const uint16_t*>(canonical_view((h + "down.weight").c_str(), 10240, 320, WeightKind::Bf16InF32).data);
            w.hc[i].up = static_cast<const uint16_t*>(canonical_view((h + "up.weight").c_str(), 320, 10240, WeightKind::Bf16InF32).data);
            w.hc[i].inject = static_cast<const uint16_t*>(canonical_view((h + "inject.weight").c_str(), 10240, 4, WeightKind::Bf16InF32).data);
        }
        w.router = static_cast<const uint16_t*>(canonical_view("ffn_gate_inp.weight", 2560, 512, WeightKind::Bf16InF32).data);
        w.shared_scalar = static_cast<const uint16_t*>(canonical_view("ffn_gate_inp_shexp.weight", 2560, 1, WeightKind::Bf16InF32).data);
        w.norm = static_cast<const float*>(canonical_view("ssm_norm.weight", 128, 1, WeightKind::F32).data);
        w.alpha = static_cast<const uint16_t*>(small("ssm_alpha.weight", 2560, 48, WeightKind::Bf16InF32, hv));
        w.beta = static_cast<const uint16_t*>(small("ssm_beta.weight", 2560, 48, WeightKind::Bf16InF32, hv));
        // Scalars have ne0=48: view as a single row for validation, then gather elements.
        auto scalar_heads = [&](const char* suffix) {
            const auto& r = canonical_view(suffix, 48, 1, WeightKind::F32);
            auto* dst = static_cast<float*>(alloc(hv.size() * sizeof(float)));
            for (size_t i = 0; i < hv.size(); ++i)
                ck(cudaMemcpy(dst + i, static_cast<const float*>(r.data) + hv[i], sizeof(float), cudaMemcpyDeviceToDevice));
            return dst;
        };
        w.dt = scalar_heads("ssm_dt.bias"); w.a = scalar_heads("ssm_a");
        w.conv = static_cast<const float*>(small("ssm_conv1d.weight", 4, 10240, WeightKind::F32, qkv));
        const auto gate = tensor(model, prefix + "ffn_gate_exps.weight", {2560, 640, 512});
        const auto up = tensor(model, prefix + "ffn_up_exps.weight", {2560, 640, 512});
        const auto down = tensor(model, prefix + "ffn_down_exps.weight", {640, 2560, 512});
        if (gate.info->type != up.info->type) throw std::runtime_error("expert gate/up formats differ");
        const auto p = plan(gate.info->type, down.info->type, 2560, 640, DownSplit::OutputRows);
        const auto& blob = full ? p.original : p.shard;
        w.expert_bytes = blob.bytes;
        auto* arena = static_cast<uint8_t*>(alloc(detail::multiply(blob.bytes, 512)));
        w.expert_arena = arena;
        const Tensor source[3] = {gate, up, down};
        const MatrixLayout original[3] = {p.original.gate, p.original.up, p.original.down};
        const MatrixLayout local[3] = {blob.gate, blob.up, blob.down};
        for (size_t expert = 0; expert < 512; ++expert) {
            for (size_t role = 0; role < 3; ++role) {
                const size_t offset = full ? 0 : p.copies[rank][role].source_offset - original[role].offset;
                ck(cudaMemcpy(arena + expert * blob.bytes + local[role].offset,
                              source[role].data + expert * original[role].bytes + offset,
                              local[role].bytes, cudaMemcpyHostToDevice));
            }
        }
        w.gu_layout = {p.gu_type, p.down_type, 2560, full ? 640 : 320,
                       blob.gate.row_bytes, blob.down.row_bytes, blob.up.offset, blob.down.offset, blob.bytes};
        w.down_layout = w.gu_layout; w.down_layout.n_embd = full ? 2560 : 1280; w.down_layout.n_ff = 640;
        ck(cudaDeviceSynchronize());
        weights_ = w;
        ck(cudaSetDevice(previous));
        err.clear(); return true;
    } catch (const std::exception& e) {
        err = std::string("TP GDN weights: ") + e.what(); clear();
        if (previous >= 0) cudaSetDevice(previous);
        return false;
    }
}
} // namespace strata::core::tp2
