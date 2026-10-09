// Opt-in gfx906 persistent hyper-connection read: exact parity and stage-only timing.
// Run: hc_persistent [benchmark_iterations=100] [graph_replays=24]
// Body/resource diagnostic only: hc_persistent --diagnose [iterations=100]
// Exit 77 means the required device/runtime is unavailable; a fallback is a failure.
// This test does not measure decode throughput or claim an end-to-end speedup.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/router_top10.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace K = strata::kernels;

#if defined(STRATA_HIP_GFX906)
namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, MT = 8, PersistentMaxT = 4, TopK = 10;
constexpr size_t Guard = 256, QRow = (N / 32) * 36;
constexpr float Poison = -913.75f;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void require(bool yes, const std::string& what) {
    if (!yes) throw std::runtime_error(what);
}
uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

struct Stream {
    cudaStream_t s = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "create stream"); }
    ~Stream() { if (s) cudaStreamDestroy(s); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

template<class T> struct Buffer {
    uint8_t* base = nullptr;
    T* p = nullptr;
    size_t count;
    explicit Buffer(size_t n) : count(n) {
        check(cudaMalloc(reinterpret_cast<void**>(&base), 2 * Guard + count * sizeof(T)), "allocate buffer");
        p = reinterpret_cast<T*>(base + Guard);
        check(cudaMemset(base, 0xa5, 2 * Guard + count * sizeof(T)), "initialize guard");
    }
    ~Buffer() { if (base) cudaFree(base); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void upload(const std::vector<T>& src) {
        require(src.size() == count, "upload size mismatch");
        check(cudaMemcpy(p, src.data(), count * sizeof(T), cudaMemcpyHostToDevice), "upload buffer");
    }
    std::vector<T> read() const {
        std::vector<T> dst(count);
        check(cudaMemcpy(dst.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost), "read buffer");
        return dst;
    }
    void guards(const char* name) const {
        std::array<uint8_t, 2 * Guard> h{};
        check(cudaMemcpy(h.data(), base, Guard, cudaMemcpyDeviceToHost), "read leading guard");
        check(cudaMemcpy(h.data() + Guard, base + Guard + count * sizeof(T), Guard, cudaMemcpyDeviceToHost),
              "read trailing guard");
        for (uint8_t x : h) require(x == 0xa5, std::string(name) + " guard was overwritten");
    }
};

struct Weights {
    Buffer<float> norm{D};
    Buffer<uint16_t> down{size_t(LR) * D}, up{size_t(D) * LR}, inject{size_t(HC) * D};
    Buffer<uint16_t> router{size_t(512) * N};
    Weights() {
        std::mt19937 rng(781);
        std::normal_distribution<float> normal(0.f, 1.f);
        std::vector<float> n(D);
        for (float& x : n) x = 1.0f + 0.2f * normal(rng);
        norm.upload(n);
        auto fill = [&](Buffer<uint16_t>& b, float scale) {
            std::vector<uint16_t> v(b.count);
            for (uint16_t& x : v) x = bf16(scale * normal(rng));
            b.upload(v);
        };
        fill(down, 0.02f); fill(up, 0.05f); fill(inject, 0.02f); fill(router, 0.025f);
    }
};

struct Inputs {
    std::vector<float> residual, output, injection;
    explicit Inputs(unsigned seed)
        : residual(size_t(MT) * D), output(size_t(MT) * N), injection(size_t(MT) * HC) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.f, 1.f);
        for (float& x : residual) x = normal(rng);
        for (float& x : output) x = 0.125f * normal(rng);
        for (float& x : injection) x = normal(rng);
    }
};

struct Snapshot {
    std::vector<float> residual, xn, lo, rs, inject, mixed, logits;
    std::vector<uint8_t> q8;
    std::vector<unsigned> counters;
    std::vector<int32_t> ids;
    std::vector<float> routing_weights;
};

struct Fixture {
    Stream stream;
    Buffer<float> residual{size_t(MT) * D}, previous{size_t(MT) * N}, injection{size_t(MT) * HC};
    Buffer<float> xn{size_t(MT) * D}, lo{size_t(MT) * LR}, rs{size_t(MT) * HC};
    Buffer<float> inject{size_t(MT) * HC}, mixed{size_t(MT) * N};
    Buffer<float> logits{size_t(MT) * 512};
    Buffer<int32_t> ids{size_t(MT) * TopK};
    Buffer<float> routing_weights{size_t(MT) * TopK};
    Buffer<uint8_t> q8{size_t(MT) * QRow};
    Buffer<unsigned> counters{N / 32};
    std::array<K::FusedGrArgs, MT> args{};
    K::FusedGrRouter router{};
    int tokens = 0, token_offset = 0;
    bool q8_enabled = false, inject_enabled = false, routed = false;

    void reset(const Inputs& input, const Weights& weights, int t, bool apply, bool with_inject, bool with_q8,
               int experts = 0, int offset = 0) {
        check(cudaStreamSynchronize(stream.s), "reset stream");
        require(offset >= 0 && t > 0 && offset + t <= MT, "invalid fixture subgroup");
        tokens = t; q8_enabled = with_q8; inject_enabled = with_inject;
        token_offset = offset;
        routed = false;
        residual.upload(input.residual); previous.upload(input.output); injection.upload(input.injection);
        for (Buffer<float>* b : {&xn, &lo, &rs, &inject, &mixed, &logits, &routing_weights})
            b->upload(std::vector<float>(b->count, Poison));
        ids.upload(std::vector<int32_t>(ids.count, -171));
        router = K::FusedGrRouter{};
        if (experts != 0) {
            router.weights = weights.router.p; router.logits = logits.p + size_t(offset) * 512;
            router.n_expert = experts;
        }
        q8.upload(std::vector<uint8_t>(q8.count, 0x3c));
        counters.upload(std::vector<unsigned>(counters.count, 0u));
        args.fill(K::FusedGrArgs{});
        for (int i = 0; i < t; ++i) {
            const int slot = offset + i;
            K::FusedGrArgs& a = args[i];
            a = K::FusedGrArgs{};
            a.R = residual.p + size_t(slot) * D;
            a.R_out = residual.p + size_t(slot) * D;  // deliberately in-place
            a.apply = apply;
            a.bo_prev = previous.p + size_t(slot) * N;
            a.inj_prev = injection.p + size_t(slot) * HC;
            a.w_norm = weights.norm.p; a.w_down = weights.down.p; a.w_up = weights.up.p;
            a.w_inject = with_inject ? weights.inject.p : nullptr;
            a.eps = 1e-6f;
            a.lo = lo.p + size_t(slot) * LR; a.rs = rs.p + size_t(slot) * HC;
            a.inject_out = inject.p + size_t(slot) * HC;
            a.mixed = mixed.p + size_t(slot) * N;
            if (with_q8) { a.q8_mixed = q8.p + size_t(slot) * QRow; a.q8_cnt = counters.p; }
        }
        // Uploads use the legacy stream; the test's nonblocking stream must observe them.
        check(cudaDeviceSynchronize(), "fixture upload sync");
    }
    void launch(bool persistent, bool expect_persistent) {
        K::fused_gr_set_persistent(persistent ? 1 : 0);
        const auto before = K::fused_gr_persistent_launches();
        bool router_written = true;  // both paths must explicitly report whether they consumed the descriptor
        bool wrote_q8;
        if (router.n_expert != 0) {
            wrote_q8 = K::fused_gr_read_multi_router(args.data(), tokens, xn.p + size_t(token_offset) * D, stream.s, nullptr, 0,
                                                     &router, &router_written);
        } else {
            wrote_q8 = K::fused_gr_read_multi(args.data(), tokens, xn.p + size_t(token_offset) * D, stream.s);
            router_written = false;
        }
        check(cudaGetLastError(), "HC launch");
        require(wrote_q8 == q8_enabled, "incorrect mixed-q8 return value");
        require(router_written == (expect_persistent && router.n_expert != 0), "incorrect router_written flag");
        const auto after = K::fused_gr_persistent_launches();
        require(after - before == (expect_persistent ? 1u : 0u),
                expect_persistent ? "persistent path silently fell back" : "fallback/baseline used the persistent path");
        if (router.n_expert != 0 && !router_written)
            K::bf16_gemv_fp32_mmvf_cols(mixed.p + size_t(token_offset) * N, router.weights, router.logits,
                                       N, router.n_expert, tokens, stream.s);
    }
    void launch(bool persistent) { launch(persistent, persistent); }
    void route() {
        require(router.n_expert == 256 || router.n_expert == 512, "router test geometry");
        int32_t* out_ids = ids.p + size_t(token_offset) * TopK;
        float* out_weights = routing_weights.p + size_t(token_offset) * TopK;
        if (router.n_expert == 512)
            K::native_router_top10_multi(router.logits, out_ids, out_weights, tokens, stream.s);
        else
            K::router_top10(router.logits, tokens, router.n_expert, TopK, out_ids, out_weights, stream.s);
        routed = true;
    }
    Snapshot read() const {
        check(cudaStreamSynchronize(stream.s), "read stream sync");
        residual.guards("R"); previous.guards("bo_prev"); injection.guards("inj_prev");
        xn.guards("xn"); lo.guards("lo"); rs.guards("rs"); inject.guards("inject");
        mixed.guards("mixed"); logits.guards("logits"); q8.guards("q8"); counters.guards("q8 counters");
        ids.guards("routing ids"); routing_weights.guards("routing weights");
        Snapshot s{residual.read(), xn.read(), lo.read(), rs.read(), inject.read(), mixed.read(),
                   logits.read(), q8.read(), counters.read(), ids.read(), routing_weights.read()};
        auto finite = [](const std::vector<float>& v, size_t start, size_t n, const char* label) {
            for (size_t i = start; i < start + n; ++i)
                require(std::isfinite(v[i]), std::string(label) + " is not finite at " + std::to_string(i));
        };
        finite(s.residual, size_t(token_offset) * D, size_t(tokens) * D, "R");
        finite(s.xn, size_t(token_offset) * D, size_t(tokens) * D, "xn");
        finite(s.lo, size_t(token_offset) * LR, size_t(tokens) * LR, "lo");
        finite(s.rs, size_t(token_offset) * HC, size_t(tokens) * HC, "rs");
        finite(s.mixed, size_t(token_offset) * N, size_t(tokens) * N, "mixed");
        if (router.n_expert) finite(s.logits, size_t(token_offset) * 512, size_t(tokens) * router.n_expert, "router logits");
        if (inject_enabled) finite(s.inject, size_t(token_offset) * HC, size_t(tokens) * HC, "inject");
        if (routed) {
            finite(s.routing_weights, size_t(token_offset) * TopK, size_t(tokens) * TopK, "routing weights");
            for (int t = token_offset; t < token_offset + tokens; ++t)
                for (int rank = 0; rank < TopK; ++rank) {
                    const size_t at = size_t(t) * TopK + rank;
                    require(s.ids[at] >= 0 && s.ids[at] < router.n_expert, "top10 id outside expert range");
                    require(s.routing_weights[at] >= 0.0f, "negative routing weight");
                }
        }
        for (unsigned c : s.counters) require(c == 0u, "q8 completion counter was not reset");
        return s;
    }
};

template<class T> void equal(const std::vector<T>& a, const std::vector<T>& b, const std::string& name) {
    require(a.size() == b.size(), name + " size mismatch");
    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0) return;
    size_t first = 0, mismatches = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0) {
            if (mismatches++ == 0) first = i;
        }
    }
    throw std::runtime_error(name + ": " + std::to_string(mismatches) + " values differ bitwise; first at " +
                             std::to_string(first));
}
void equal(const Snapshot& a, const Snapshot& b, const std::string& label) {
    equal(a.residual, b.residual, label + " R"); equal(a.xn, b.xn, label + " xn");
    equal(a.lo, b.lo, label + " lo"); equal(a.rs, b.rs, label + " rs");
    equal(a.inject, b.inject, label + " inject"); equal(a.mixed, b.mixed, label + " mixed");
    equal(a.logits, b.logits, label + " router logits");
    equal(a.q8, b.q8, label + " q8"); equal(a.counters, b.counters, label + " counters");
    equal(a.ids, b.ids, label + " routing ids");
    equal(a.routing_weights, b.routing_weights, label + " routing weights");
}

struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    Graph(Fixture& f, bool persistent) {
        check(cudaStreamBeginCapture(f.stream.s, cudaStreamCaptureModeThreadLocal), "begin graph capture");
        f.launch(persistent);
        check(cudaStreamEndCapture(f.stream.s, &graph), "end graph capture");
        check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate graph");
        size_t nodes = 0;
        check(cudaGraphGetNodes(graph, nullptr, &nodes), "count captured nodes");
        const size_t expected_nodes = persistent ? 1u : (f.router.n_expert ? 4u : 3u);
        require(nodes == expected_nodes,
                std::string(persistent ? "persistent" : "baseline") + " graph has " + std::to_string(nodes) +
                " nodes; expected " + std::to_string(expected_nodes));
    }
    ~Graph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
    void replay(Fixture& f) { check(cudaGraphLaunch(executable, f.stream.s), "graph replay"); }
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
};

void parity(const Weights& w, const Inputs& input, int replays) {
    Fixture reference, candidate;
    int cases = 0;
    for (int t = 1; t <= PersistentMaxT; ++t)
        for (bool apply : {false, true})
            for (bool injection : {false, true})
                for (bool q8 : {false, true}) {
                    const std::string label = "T=" + std::to_string(t) + " apply=" + std::to_string(apply) +
                        " inject=" + std::to_string(injection) + " q8=" + std::to_string(q8);
                    reference.reset(input, w, t, apply, injection, q8);
                    candidate.reset(input, w, t, apply, injection, q8);
                    reference.launch(false); candidate.launch(true);
                    equal(reference.read(), candidate.read(), label + " eager");
                    reference.reset(input, w, t, apply, injection, q8);
                    candidate.reset(input, w, t, apply, injection, q8);
                    Graph baseline(reference, false), persistent(candidate, true);
                    const auto before = K::fused_gr_persistent_launches();
                    // Captured selection must survive changes to the host's switch.
                    for (int i = 0; i < replays; ++i) {
                        K::fused_gr_set_persistent(1); baseline.replay(reference);
                        K::fused_gr_set_persistent(0); persistent.replay(candidate);
                    }
                    equal(reference.read(), candidate.read(), label + " graph");
                    require(K::fused_gr_persistent_launches() == before, "graph replay changed the host launch counter");
                    std::printf("  %s: eager + %d graph replays bitwise equal\n", label.c_str(), replays);
                    ++cases;
                }
    std::printf("Parity: %d configurations, including inactive rows, guards, null injection and q8 bytes.\n", cases);
}

void router_parity(const Weights& w, const Inputs& input, int replays) {
    Fixture reference, candidate;
    for (int experts : {256, 512})
        for (int t = 1; t <= PersistentMaxT; ++t)
            for (bool apply : {false, true})
                for (bool q8 : {false, true}) {
                    const bool injection = (t & 1) != 0;
                    const std::string label = "router=" + std::to_string(experts) + " T=" + std::to_string(t) +
                        " apply=" + std::to_string(apply) + " q8=" + std::to_string(q8);
                    reference.reset(input, w, t, apply, injection, q8, experts, 2);
                    candidate.reset(input, w, t, apply, injection, q8, experts, 2);
                    // The reference keeps the standalone cols MMVF projection after the ordinary HC read.
                    reference.launch(false); candidate.launch(true);
                    reference.route(); candidate.route();
                    equal(reference.read(), candidate.read(), label + " eager");
                    reference.reset(input, w, t, apply, injection, q8, experts, 2);
                    candidate.reset(input, w, t, apply, injection, q8, experts, 2);
                    Graph baseline(reference, false), persistent(candidate, true);
                    K::fused_gr_set_persistent(0);
                    for (int i = 0; i < replays; ++i) {
                        baseline.replay(reference); persistent.replay(candidate);
                    }
                    reference.route(); candidate.route();
                    equal(reference.read(), candidate.read(), label + " graph");
                    std::printf("  %s: offset=2, eager + %d replays; logits, top10 ids and weights bitwise equal\n",
                                label.c_str(), replays);
                }
}

void subgroup_parity(const Weights& w, const Inputs& input, int replays) {
    Fixture reference, candidate;
    for (int t = 1; t <= PersistentMaxT; ++t) {
        reference.reset(input, w, t, true, true, true, 0, 2);
        candidate.reset(input, w, t, true, true, true, 0, 2);
        Graph baseline(reference, false), persistent(candidate, true);
        for (int i = 0; i < replays; ++i) { baseline.replay(reference); persistent.replay(candidate); }
        equal(reference.read(), candidate.read(), "nonzero HC subgroup T=" + std::to_string(t));
    }
    std::puts("Nonzero HC-only subgroups: T=1..4 at token offset 2, q8 and graph replay bitwise equal.");
}

// Real Q8_0 rows: each 34-byte block contains a finite half scale and 32 signed
// bytes. The scale 0x1400 is exactly 2^-10; no native packer or fake pointer is used.
struct Q8Weights {
    Buffer<uint8_t> down{size_t(LR) * (D / 32) * 34};
    Buffer<uint8_t> up{size_t(D) * (LR / 32) * 34};
    Buffer<uint8_t> inject{size_t(HC) * (D / 32) * 34};
    Q8Weights() {
        auto fill = [](Buffer<uint8_t>& buffer, unsigned salt) {
            std::vector<uint8_t> bytes(buffer.count);
            const uint16_t scale = 0x1400u;
            for (size_t block = 0; block < bytes.size() / 34; ++block) {
                std::memcpy(bytes.data() + block * 34, &scale, sizeof(scale));
                for (size_t j = 0; j < 32; ++j) {
                    const int value = int((block * 17 + j * 13 + salt) % 63) - 31;
                    bytes[block * 34 + 2 + j] = static_cast<uint8_t>(static_cast<int8_t>(value));
                }
            }
            buffer.upload(bytes);
        };
        fill(down, 1); fill(up, 29); fill(inject, 47);
        check(cudaDeviceSynchronize(), "Q8 weights upload");
    }
    void bind(Fixture& fixture) const {
        for (int i = 0; i < fixture.tokens; ++i) {
            fixture.args[i].q8_down = down.p;
            fixture.args[i].q8_up = up.p;
            fixture.args[i].q8_inject = inject.p;
        }
    }
};

void fallback_parity(const Weights& w, const Inputs& input) {
    Fixture reference, candidate;
    for (int t = PersistentMaxT + 1; t <= MT; ++t) {
        require(!K::fused_gr_persistent_supported(t), "unsupported token count incorrectly marked supported");
        reference.reset(input, w, t, true, true, true, 512);
        candidate.reset(input, w, t, true, true, true, 512);
        reference.launch(false, false); candidate.launch(true, false);
        reference.route(); candidate.route();
        equal(reference.read(), candidate.read(), "T=" + std::to_string(t) + " fallback");
    }
    const Q8Weights qweights;
    for (int t = 1; t <= PersistentMaxT; ++t) {
        reference.reset(input, w, t, true, true, false, 256);
        candidate.reset(input, w, t, true, true, false, 256);
        qweights.bind(reference); qweights.bind(candidate);
        reference.launch(false, false); candidate.launch(true, false);
        reference.route(); candidate.route();
        equal(reference.read(), candidate.read(), "Q8 HC fallback T=" + std::to_string(t));
    }
    std::puts("Fallback: T=5..8 and real Q8_0 HC weights agree bitwise, with no persistent submissions.");
}

void concurrent(const Weights& w, int replays) {
    const Inputs ia(413), ib(914);
    Fixture ra, rb, ca, cb;
    // Distinct inputs and token counts make any accidental process-global workspace observable.
    ra.reset(ia, w, 3, true, true, true, 256, 1); ca.reset(ia, w, 3, true, true, true, 256, 1);
    rb.reset(ib, w, 4, true, false, true, 512, 3); cb.reset(ib, w, 4, true, false, true, 512, 3);
    for (int i = 0; i < replays; ++i) { ra.launch(false); rb.launch(false); }
    ra.route(); rb.route();
    const Snapshot want_a = ra.read(), want_b = rb.read();
    Graph ga(ca, true), gb(cb, true);
    K::fused_gr_set_persistent(0);
    for (int i = 0; i < replays; ++i) {
        ga.replay(ca); gb.replay(cb);  // no cross-stream events or intervening synchronization
    }
    ca.route(); cb.route();
    equal(want_a, ca.read(), "concurrent stream A");
    equal(want_b, cb.read(), "concurrent stream B");
    std::printf("Two independent streams: %d interleaved graph replays each, bitwise equal.\n", replays);
}

double measure(Fixture& f, bool persistent, int iterations, bool graph_mode) {
    cudaEvent_t start = nullptr, stop = nullptr;
    check(cudaEventCreate(&start), "create start event");
    check(cudaEventCreate(&stop), "create stop event");
    float ms = 0.0f;
    if (graph_mode) {
        Graph graph(f, persistent);
        for (int i = 0; i < 12; ++i) graph.replay(f);
        check(cudaEventRecord(start, f.stream.s), "record start");
        for (int i = 0; i < iterations; ++i) graph.replay(f);
        check(cudaEventRecord(stop, f.stream.s), "record stop");
        check(cudaEventSynchronize(stop), "wait for benchmark");
        check(cudaEventElapsedTime(&ms, start, stop), "elapsed graph time");
    } else {
        for (int i = 0; i < 12; ++i) f.launch(persistent);
        check(cudaEventRecord(start, f.stream.s), "record start");
        for (int i = 0; i < iterations; ++i) f.launch(persistent);
        check(cudaEventRecord(stop, f.stream.s), "record stop");
        check(cudaEventSynchronize(stop), "wait for benchmark");
        check(cudaEventElapsedTime(&ms, start, stop), "elapsed eager time");
    }
    check(cudaEventDestroy(start), "destroy start event");
    check(cudaEventDestroy(stop), "destroy stop event");
    return 1000.0 * double(ms) / iterations;
}


#if defined(STRATA_HC_PERSIST_BUILD)
void diagnose(const Weights& w, const Inputs& input, int iterations) {
    const char* names[] = {"production", "value_timed", "ref_untimed", "ref_timed"};
    const char* phases[] = {"norm", "barrier1", "down", "barrier2", "up", "barrier3", "tail"};
    std::puts("HC BODY DIAGNOSTIC: original production code is unchanged; clones are explicit test launches.\n"
              "Raw cycles are per-block deltas, never cross-CU timestamps or assumed-frequency microseconds.\n"
              "Instrumented attributes/occupancy can differ; compare untimed reference clone to production.\n"
              "Do not sum cross-CU phase medians/maxima; compiler entry/exit work is outside clock samples.\n"
              "Timing uses a multi-node graph to reduce caller submission overhead.");
    Fixture f;
    Buffer<unsigned long long> stamps(160 * 7);
    for (int t : {1, 4}) {
        int common_blocks = 160;
        K::FusedGrDiagnosticInfo info[4];
        for (int v = 0; v < 4; ++v) {
            require(K::fused_gr_diagnostic_info(v, t, &info[v]), "diagnostic resources unavailable");
            common_blocks = std::min(common_blocks, info[v].blocks);
            std::printf("T=%d variant=%s regs=%d local_bytes=%llu static_lds=%llu dynamic_lds=%llu active_per_cu=%d selected_blocks=%d max_threads=%d\n",
                t, names[v], info[v].registers, (unsigned long long)info[v].local_bytes,
                (unsigned long long)info[v].static_lds_bytes, (unsigned long long)info[v].dynamic_lds_bytes,
                info[v].active_per_cu, info[v].blocks, info[v].max_threads);
        }
        std::printf("T=%d common_grid=%d; all common-grid comparisons use this size; native grids are reported separately.\n", t, common_blocks);
        auto launch = [&](int v, int blocks) {
            require(K::fused_gr_diagnostic_launch(f.args.data(), t, f.xn.p, f.stream.s,
                f.router.n_expert ? &f.router : nullptr, v, blocks, stamps.p), "diagnostic launch failed");
            check(cudaGetLastError(), "diagnostic launch status");
        };
        for (int experts : {0, 512}) {
            // Include pending writes and q8 in one-shot bitwise checks, timing
            // apply=false/q8=false below to match the first hardware benchmark.
            for (bool apply : {false, true}) for (bool q8 : {false, true}) {
                f.reset(input, w, t, apply, true, q8, experts);
                f.launch(true);
                if (experts) f.route();
                const Snapshot wanted = f.read();
                for (int v = 0; v < 4; ++v) {
                    f.reset(input, w, t, apply, true, q8, experts);
                    launch(v, common_blocks);
                    if (experts) f.route();
                    equal(wanted, f.read(), std::string("diagnostic ") + names[v]);
                }
            }
            std::printf("T=%d router=%d: all four variants bitwise equal for apply=0/1 q8=0/1, including routing.\n", t, experts);
            f.reset(input, w, t, false, true, false, experts);
            for (int v = 0; v < 4; ++v) {
                // Native occupancy can differ, so report both shared geometry and
                // each variant's own selected grid rather than hiding the change.
                for (int grid_mode = 0; grid_mode < 2; ++grid_mode) {
                    const int blocks = grid_mode ? info[v].blocks : common_blocks;
                    if (grid_mode && blocks == common_blocks) continue;
                    cudaGraph_t graph = nullptr;
                    cudaGraphExec_t exec = nullptr;
                    check(cudaStreamBeginCapture(f.stream.s, cudaStreamCaptureModeThreadLocal), "diagnostic begin capture");
                    for (int n = 0; n < iterations; ++n) launch(v, blocks);
                    check(cudaStreamEndCapture(f.stream.s, &graph), "diagnostic end capture");
                    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "diagnostic instantiate");
                    for (int n = 0; n < 2; ++n) check(cudaGraphLaunch(exec, f.stream.s), "diagnostic warmup");
                    check(cudaStreamSynchronize(f.stream.s), "diagnostic warmup sync");
                    cudaEvent_t start, stop;
                    check(cudaEventCreate(&start), "diagnostic create start");
                    check(cudaEventCreate(&stop), "diagnostic create stop");
                    std::array<float, 3> samples;
                    for (float& ms : samples) {
                        check(cudaEventRecord(start, f.stream.s), "diagnostic start");
                        check(cudaGraphLaunch(exec, f.stream.s), "diagnostic replay");
                        check(cudaEventRecord(stop, f.stream.s), "diagnostic stop");
                        check(cudaEventSynchronize(stop), "diagnostic wait");
                        check(cudaEventElapsedTime(&ms, start, stop), "diagnostic elapsed");
                    }
                    std::sort(samples.begin(), samples.end());
                    std::printf("T=%d router=%d variant=%s grid=%s blocks=%d event_us=%.3f (median3, %d nodes/replay)\n",
                        t, experts, names[v], grid_mode ? "native" : "common", blocks, 1000.0 * samples[1] / iterations, iterations);
                    if (v == 1 || v == 3) {
                        const auto raw = stamps.read();
                        for (int phase = 0; phase < 7; ++phase) {
                            std::vector<unsigned long long> busy, all;
                            for (int b = 0; b < blocks; ++b) {
                                const auto cycles = raw[b * 7 + phase];
                                all.push_back(cycles);
                                bool active = true;
                                if (phase == 0) active = b < t;
                                if (phase == 2) active = b < 41;
                                if (phase == 4) active = b < 160;
                                if (phase == 5 || phase == 6) active = experts != 0;
                                if (active) busy.push_back(cycles);
                            }
                            std::sort(all.begin(), all.end());
                            std::sort(busy.begin(), busy.end());
                            std::printf("  phase=%s raw_cycles_all_min/med/max=%llu/%llu/%llu active_blocks=%zu active_med/max=%llu/%llu\n",
                                phases[phase], all.front(), all[all.size()/2], all.back(), busy.size(),
                                busy.empty() ? 0ull : busy[busy.size()/2], busy.empty() ? 0ull : busy.back());
                        }
                        stamps.guards("diagnostic stamps");
                    }
                    check(cudaEventDestroy(start), "diagnostic destroy start");
                    check(cudaEventDestroy(stop), "diagnostic destroy stop");
                    check(cudaGraphExecDestroy(exec), "diagnostic destroy executable");
                    check(cudaGraphDestroy(graph), "diagnostic destroy graph");
                }
            }
        }
    }
    std::puts("PASS: diagnostic clone parity, resource report and phase timing finished.");
}
#endif

void benchmark(const Weights& w, const Inputs& input, int iterations) {
    Fixture f;
    K::fused_gr_set_fast(1);
    // Restore production's default checked plain/split/staged selection as well;
    // merely turning FAST back on would still leave HC_SPLIT=0's forced plain path.
    unsetenv("STRATA_HC_SPLIT");
    K::fused_gr_check();
    std::printf("HC stage-only timings, default checked variant %d, STRATA_GR_FAST=1; "
                "median of 5 alternating A/B pairs; warm weights; no decode speed claim:\n", K::fused_gr_variant());
    for (int experts : {0, 256, 512})
    for (int t = 1; t <= PersistentMaxT; ++t) {
        // apply=false avoids changing the residual distribution during repeated timing.
        f.reset(input, w, t, false, true, false, experts);
        for (bool graph_mode : {false, true}) {
            std::array<double, 5> times[2]{};
            for (int round = 0; round < 5; ++round)
                for (int order = 0; order < 2; ++order) {
                    const int mode = (round + order) & 1;
                    times[mode][round] = measure(f, mode != 0, iterations, graph_mode);
                }
            for (auto& samples : times) std::sort(samples.begin(), samples.end());
            std::printf("  T=%d router=%d %-5s baseline %.3f us, persistent %.3f us, stage ratio %.3fx\n", t, experts,
                        graph_mode ? "graph" : "eager", times[0][2], times[1][2], times[0][2] / times[1][2]);
        }
    }
}
}  // namespace
#endif

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
#if !defined(STRATA_HIP_GFX906)
    (void) argc; (void) argv;
    std::puts("SKIP: hc_persistent requires the STRATA_HIP_GFX906 backend.");
    return 77;
#else
    try {
        const bool diagnostic = argc > 1 && std::strcmp(argv[1], "--diagnose") == 0;
        const int iterations = diagnostic ? (argc > 2 ? std::atoi(argv[2]) : 100) : (argc > 1 ? std::atoi(argv[1]) : 100);
        const int replays = !diagnostic && argc > 2 ? std::atoi(argv[2]) : 24;
        require(argc <= 3 && iterations > 0 && iterations <= 10000 && replays >= 20,
                "usage: hc_persistent [iterations] [graph_replays >= 20], or hc_persistent --diagnose [iterations]");
        // Keep the reference on its three-launch BF16 path, independent of the user's shell settings.
        setenv("STRATA_GR_V3", "0", 1);
        unsetenv("STRATA_GR_SPLIT");
        setenv("STRATA_HC_SPLIT", "0", 1);
        setenv("STRATA_GR_FAST", "0", 1);
        setenv("STRATA_NO_MULTI_GR", "0", 1);
        K::fused_gr_set_fast(0);
        int devices = 0;
        const cudaError_t status = cudaGetDeviceCount(&devices);
        if (status != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable GPU (%s).\n", cudaGetErrorString(status));
            return 77;
        }
        int device = 0;
        check(cudaGetDevice(&device), "current device");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device), "device properties");
        int runtime = 0;
        check(cudaRuntimeGetVersion(&runtime), "runtime version");
        std::printf("Device %d: %s, HIP runtime %d\n", device, prop.name, runtime);
        for (int t = 1; t <= PersistentMaxT; ++t) {
            if (!K::fused_gr_persistent_supported(t)) {
                std::printf("SKIP: persistent HC unavailable for T=%d; needs gfx906, ROCm >= 6.4 and cooperative capacity.\n", t);
                return 77;
            }
        }
        const Weights weights;
        const Inputs input(73);
#if defined(STRATA_HC_PERSIST_BUILD)
        if (diagnostic) { diagnose(weights, input, iterations); return 0; }
#endif
        parity(weights, input, replays);
        router_parity(weights, input, replays);
        subgroup_parity(weights, input, replays);
        fallback_parity(weights, input);
        concurrent(weights, replays);
        benchmark(weights, input, iterations);
        check(cudaDeviceSynchronize(), "final synchronization");
        K::fused_gr_set_persistent(-1);
        std::puts("PASS: persistent HC parity, graph replay, isolated streams and launch coverage.");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: hc_persistent: %s\n", e.what());
        return 1;
    }
#endif
}
