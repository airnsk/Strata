// Synthetic GPU gate for the opt-in inverse-route planner and down consumer.
// No model files. This is NOT a host substitute for running the HIP/CUDA kernels.
// --peer-device requires a real peer allocation and an event-joined read on it.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace {
constexpr int N = 2560, K = 10, CAP = 80, EXPERTS = 11;
constexpr uint32_t unrelated_error = 8u;
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void need(bool ok, const std::string& what) { if (!ok) throw std::runtime_error(what); }
void on(int device) { ck(cudaSetDevice(device), "set device"); }
template<class T> void exact(const std::vector<T>& a, const std::vector<T>& b, const char* what) {
    need(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0, what);
}
struct Stream {
    cudaStream_t s{};
    Stream() { ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream"); }
    ~Stream() { cudaStreamDestroy(s); }
};
struct Graph {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    Graph() = default;
    Graph(const Graph&) = delete;
    ~Graph() { if (exec) cudaGraphExecDestroy(exec); if (graph) cudaGraphDestroy(graph); }
    template<class F> void capture(cudaStream_t s, F f) {
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture begin");
        f();
        ck(cudaStreamEndCapture(s, &graph), "capture end");
        ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "graph instantiate");
    }
    void launch(cudaStream_t s) { ck(cudaGraphLaunch(exec, s), "graph launch"); }
};
template<class T> struct Buffer {
    T* base = nullptr;
    T* p = nullptr;
    size_t count;
    int device;
    explicit Buffer(size_t n) : count(n) {
        ck(cudaGetDevice(&device), "get device");
        ck(cudaMalloc(reinterpret_cast<void**>(&base), (n + 16) * sizeof(T)), "allocate");
        p = base + 8;
        poison();
    }
    Buffer(const Buffer&) = delete;
    ~Buffer() { cudaSetDevice(device); cudaFree(base); }
    void poison() { ck(cudaMemset(base, 0xa5, (count + 16) * sizeof(T)), "poison"); }
    void put(const std::vector<T>& v) {
        need(v.size() <= count, "upload exceeds fixture capacity");
        ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<T> get(size_t n) const {
        need(n <= count, "download exceeds fixture capacity");
        std::vector<T> v(n);
        ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return v;
    }
    void guards(size_t live, const char* what) const {
        std::vector<uint8_t> bytes((count + 16) * sizeof(T));
        ck(cudaMemcpy(bytes.data(), base, bytes.size(), cudaMemcpyDeviceToHost), "read guards");
        for (size_t i = 0; i < 8 * sizeof(T); ++i) need(bytes[i] == 0xa5, what);
        for (size_t i = (live + 8) * sizeof(T); i < bytes.size(); ++i) need(bytes[i] == 0xa5, what);
    }
};
struct Metadata {
    std::vector<unsigned long long> ptr;
    std::vector<int32_t> start, dst;
    int ng = 0;
};
struct Expected {
    std::vector<int32_t> entry;
    std::vector<unsigned long long> blob;
    std::vector<uint32_t> token_error;
};
// Independent destination-first oracle, deliberately not the shared planner helpers.
Expected inverse(const Metadata& m, int t) {
    const int entries = t * K;
    Expected r{std::vector<int32_t>(entries, -1), std::vector<unsigned long long>(entries),
               std::vector<uint32_t>(t)};
    bool global_bad = m.ng <= 0 || m.ng > entries;
    if (!global_bad) {
        global_bad = m.start[0] != 0 || m.start[m.ng] != entries;
        for (int g = 0; g < m.ng; ++g) {
            const int lo = m.start[g], hi = m.start[g + 1];
            if (lo < 0 || hi <= lo || hi > entries || !m.ptr[g]) { global_bad = true; continue; }
            for (int e = lo; e < hi; ++e) if (m.dst[e] < 0 || m.dst[e] >= entries) global_bad = true;
        }
    }
    for (int d = 0; d < entries; ++d) {
        int found = 0;
        if (m.ng > 0 && m.ng <= entries) for (int g = 0; g < m.ng; ++g) {
            const int lo = m.start[g], hi = m.start[g + 1];
            if (lo < 0 || hi <= lo || hi > entries || !m.ptr[g]) continue;
            for (int e = lo; e < hi; ++e) if (m.dst[e] == d) {
                ++found; r.entry[d] = e; r.blob[d] = m.ptr[g];
            }
        }
        if (found != 1 || global_bad) r.token_error[d / K] = k::kNativeDownCombinePlanError;
    }
    return r;
}
struct Peer {
    int local, remote;
    Buffer<float>* inbox = nullptr;
    Buffer<float>* observed = nullptr;
    Stream* stream = nullptr;
    cudaEvent_t ready{};
    Peer(int device, int peer) : local(device), remote(peer < 0 ? device : peer) {
        if (remote != local) {
            int available = 0;
            ck(cudaDeviceCanAccessPeer(&available, local, remote), "peer accessibility");
            need(available != 0, "requested peer is not accessible");
            const auto e = cudaDeviceEnablePeerAccess(remote, 0);
            if (e != cudaErrorPeerAccessAlreadyEnabled) ck(e, "enable peer");
            else (void)cudaGetLastError();
        }
        ck(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "peer event");
        on(remote);
        inbox = new Buffer<float>(8 * N);
        observed = new Buffer<float>(8 * N);
        stream = new Stream;
        on(local);
    }
    ~Peer() {
        on(remote); delete stream; delete observed; delete inbox;
        on(local); cudaEventDestroy(ready);
    }
    void poison() {
        on(remote); inbox->poison(); observed->poison();
        ck(cudaDeviceSynchronize(), "peer poison drain"); on(local);
    }
    std::vector<float> joined_read(cudaStream_t producer, int t) {
        // No producer stream/device synchronization before the remote reader.
        ck(cudaEventRecord(ready, producer), "record producer completion");
        on(remote);
        ck(cudaStreamWaitEvent(stream->s, ready, 0), "peer event join");
        ck(cudaMemcpyAsync(observed->p, inbox->p, size_t(t) * N * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream->s), "peer-side observation");
        ck(cudaStreamSynchronize(stream->s), "peer consumer drain");
        auto result = observed->get(size_t(t) * N);
        inbox->guards(size_t(t) * N, "peer output wrote past active T");
        observed->guards(size_t(t) * N, "peer reader wrote past active T");
        on(local);
        return result;
    }
};
struct Fixture {
    int f;
    k::NativeExpertLayout layout;
    Stream stream;
    Buffer<uint8_t> arena, hidden, scratch;
    Buffer<unsigned long long> ptr{CAP}, blob{CAP};
    Buffer<int32_t> start{CAP + 1}, dst{CAP}, ng{1}, entry{CAP};
    Buffer<uint32_t> token_error{8}, error{1}, legacy_error{1};
    Buffer<float> x, weights{CAP}, shared{8 * N}, gate{8}, old{8 * N}, out{8 * N}, hits{CAP * N};
    Peer peer;
    Metadata metadata;
    std::vector<float> host_weights, host_shared;
    k::NativeDownRoutePlan plan() { return {entry.p, blob.p, token_error.p}; }
    static size_t blob_bytes(int type, int width) { return 32 + N * k::iq_row_bytes(type, width); }
    Fixture(int type, int width, int device, int remote) : f(width),
        arena(EXPERTS * blob_bytes(type, width)), hidden(CAP * (width / 32) * 36),
        scratch(k::native_expert_scratch_bytes(CAP, width)), x(CAP * width), peer(device, remote) {
        layout.d_type = type; layout.n_embd = N; layout.n_ff = f;
        layout.d_row = k::iq_row_bytes(type, f); layout.down_off = 32; layout.bytes = blob_bytes(type, f);
        std::mt19937 rng(unsigned(631 + type * 19 + f));
        std::vector<uint8_t> bytes(arena.count);
        for (auto& b : bytes) b = uint8_t(rng());
        const size_t block = k::iq_row_bytes(type, type == 20 ? 32 : 64);
        for (int e = 0; e < EXPERTS; ++e) for (size_t off = layout.down_off; off < layout.bytes; off += block) {
            const uint16_t scale = uint16_t(0x2000u | (rng() & 0x3ffu));
            std::memcpy(bytes.data() + size_t(e) * layout.bytes + off, &scale, sizeof(scale));
        }
        arena.put(bytes);
        ck(cudaDeviceSynchronize(), "fixture initialization drain");
    }
    Metadata routes(int t, int variant, bool reorder) {
        const int n = t * K, groups = std::min(n, EXPERTS);
        Metadata m;
        m.ptr.assign(n, 0); m.start.assign(n + 1, 0); m.dst.assign(n, 0);
        std::vector<int> order(groups); std::iota(order.begin(), order.end(), 0);
        if (reorder) std::reverse(order.begin(), order.end());
        int pos = 0;
        for (int expert : order) {
            std::vector<int> destinations;
            for (int d = 0; d < n; ++d) if ((d * 7 + variant) % groups == expert) destinations.push_back(d);
            if (destinations.empty()) continue;
            if (reorder) std::reverse(destinations.begin(), destinations.end());
            m.start[m.ng] = pos;
            m.ptr[m.ng++] = reinterpret_cast<uintptr_t>(arena.p + size_t(expert) * layout.bytes);
            for (int d : destinations) m.dst[pos++] = d;
        }
        m.start[m.ng] = pos;
        need(pos == n, "synthetic routes incomplete");
        return m;
    }
    void upload(int t, const Metadata& m, int values, bool zero_weights = false, bool zero_hidden = false) {
        metadata = m;
        ptr.put(m.ptr); start.put(m.start); dst.put(m.dst); ng.put({m.ng});
        host_weights.resize(t * K); host_shared.resize(size_t(t) * N);
        std::vector<float> values_x(size_t(t) * K * f);
        for (int e = 0; e < t * K; ++e) for (int j = 0; j < f; ++j) {
            const int d = m.dst[e] >= 0 && m.dst[e] < t * K ? m.dst[e] : e;
            values_x[size_t(e) * f + j] = zero_hidden ? 0.0f : float((d * 29 + j * 13 + values * 7) % 127 - 63) / 64.0f;
        }
        for (int d = 0; d < t * K; ++d) host_weights[d] = zero_weights ? (d % 2 ? -0.0f : 0.0f) : float((d * 17 + values * 3) % 23 - 11) / 13.0f;
        for (size_t i = 0; i < host_shared.size(); ++i) host_shared[i] = float((i * 11 + size_t(values) * 19) % 59) / 31.0f - 0.9f;
        weights.put(host_weights); shared.put(host_shared); gate.put(std::vector<float>(t, 0.0f)); x.put(values_x);
        // Uploads use the default stream; this producer is deliberately nonblocking.
        ck(cudaDeviceSynchronize(), "fixture upload drain");
        k::quantize_q8_1_rows(x.p, t * K, f, hidden.p, stream.s);
        ck(cudaStreamSynchronize(stream.s), "quantize hidden");
    }
    void poison() {
        entry.poison(); blob.poison(); token_error.poison(); old.poison(); out.poison(); peer.poison();
        error.put({unrelated_error}); legacy_error.put({unrelated_error});
        ck(cudaDeviceSynchronize(), "fixture poison drain");
    }
    void baseline(int t) {
        k::native_expert_down_combine(layout, ptr.p, start.p, ng.p, dst.p, t * K, t * K,
            hidden.p, weights.p, shared.p, gate.p, old.p, legacy_error.p, t, stream.s);
    }
    void candidate(int t) {
        k::native_expert_down_plan(ptr.p, start.p, ng.p, dst.p, t * K, t * K, plan(), error.p, t, stream.s);
        k::native_expert_down_combine_preplanned(layout, plan(), t * K, hidden.p, weights.p,
            shared.p, gate.p, out.p, error.p, t, stream.s, peer.inbox->p);
    }
    void arithmetic(int t, const std::vector<float>& actual) {
        const size_t live = size_t(t) * K;
        // Force the original 32-lane per-entry dot; the explicit phase ignores
        // environment/device mode selection. The host writes the reduction independently.
        const size_t off = k::native_expert_hidden_q8_offset(live, f);
        ck(cudaMemcpyAsync(scratch.p + off, hidden.p, live * size_t(f / 32) * 36,
                           cudaMemcpyDeviceToDevice, stream.s), "oracle hidden");
        k::native_expert_grouped_explicit(layout, ptr.p, start.p, ng.p, dst.p, nullptr,
            live, live, nullptr, scratch.p, hits.p, stream.s, {k::NativeExpertPhase::Down, 0});
        ck(cudaStreamSynchronize(stream.s), "oracle dot drain");
        const auto h = hits.get(live * N);
        std::vector<float> expected(size_t(t) * N);
        for (int token = 0; token < t; ++token) for (int row = 0; row < N; ++row) {
            volatile float first = (0.0f + h[size_t(token * K) * N + row]) * host_weights[token * K];
            float sum = first;
            for (int slot = 1; slot < K; ++slot) sum = std::fma(0.0f + h[size_t(token * K + slot) * N + row], host_weights[token * K + slot], sum);
            volatile float sh = host_shared[size_t(token) * N + row] * 0.5f;
            expected[size_t(token) * N + row] = sum + sh;
        }
        exact(expected, actual, "independent ordered FMA/shared arithmetic differs");
    }
    std::vector<float> check(int t, Graph* graph, bool oracle) {
        poison(); baseline(t);
        if (graph) graph->launch(stream.s); else candidate(t);
        const auto peer_result = peer.joined_read(stream.s, t);
        const auto actual = out.get(size_t(t) * N), legacy = old.get(size_t(t) * N);
        exact(legacy, actual, "legacy/preplanned output bits differ");
        exact(actual, peer_result, "event-joined peer output differs");
        const auto expected = inverse(metadata, t);
        const auto status = token_error.get(t);
        exact(expected.token_error, status, "token error scope differs");
        const bool bad = std::any_of(status.begin(), status.end(), [](uint32_t v) { return v != 0; });
        const uint32_t expected_error = unrelated_error | (bad ? k::kNativeDownCombinePlanError : 0u);
        need(error.get(1)[0] == expected_error && legacy_error.get(1)[0] == expected_error, "error must OR the plan bit and preserve prior status");
        const auto actual_entry = entry.get(t * K);
        const auto actual_blob = blob.get(t * K);
        for (int token = 0; token < t; ++token) {
            if (status[token]) {
                for (int row = 0; row < N; ++row) {
                    uint32_t bits = 1; std::memcpy(&bits, &actual[size_t(token) * N + row], sizeof(bits));
                    need(bits == 0, "invalid token must write positive zero");
                }
            } else for (int d = token * K; d < (token + 1) * K; ++d) {
                need(actual_entry[d] == expected.entry[d] && actual_blob[d] == expected.blob[d], "inverse differs from destination-first host oracle");
            }
        }
        entry.guards(t * K, "entry capacity overwrite"); blob.guards(t * K, "blob capacity overwrite");
        token_error.guards(t, "token status capacity overwrite"); out.guards(size_t(t) * N, "output capacity overwrite");
        old.guards(size_t(t) * N, "legacy output capacity overwrite");
        if (oracle && !bad) arithmetic(t, actual);
        return actual;
    }
};

int valid_cases(Fixture& f) {
    std::array<Graph, 8> graphs;
    for (int t = 1; t <= 8; ++t) graphs[t - 1].capture(f.stream.s, [&] { f.candidate(t); });
    int cases = 0;
    // Different shape graphs deliberately share every buffer. Return to shorter
    // T after larger T, then revisit an earlier graph with changed metadata.
    for (int t : {1, 8, 2, 7, 3, 6, 4, 5}) {
        f.upload(t, f.routes(t, 0, false), 1);
        const auto first = f.check(t, nullptr, true); ++cases;
        f.upload(t, f.routes(t, 0, true), 1);
        exact(first, f.check(t, &graphs[t - 1], false), "reordered groups/entries changed route arithmetic"); ++cases;
        f.upload(t, f.routes(t, 3, false), 2, true);
        f.check(t, &graphs[t - 1], true); ++cases;
        f.upload(t, f.routes(t, 5, true), 3, false, true);
        f.check(t, &graphs[t - 1], true); ++cases;
        f.upload(t, f.routes(t, 7, false), 4);
        f.check(t, &graphs[t - 1], false); ++cases;
    }
    return cases;
}
int fault_cases(Fixture& f) {
    const int t = 2, n = t * K;
    Graph graph; graph.capture(f.stream.s, [&] { f.candidate(t); });
    const Metadata valid = f.routes(t, 0, false);
    int cases = 0;
    for (int fault = 0; fault < 10; ++fault) {
        Metadata m = valid;
        if (fault == 0) { // duplicate/missing within one token: other token stays valid
            int a = -1, b = -1;
            for (int e = 0; e < n; ++e) if (m.dst[e] < K) { if (a < 0) a = e; else b = e; }
            m.dst[b] = m.dst[a];
        } else if (fault == 1) m.dst[0] = -1;
        else if (fault == 2) m.dst[0] = n;
        else if (fault == 3) m.ng = 0;
        else if (fault == 4) m.ng = n + 1;
        else if (fault == 5) m.start[0] = 1;
        else if (fault == 6) m.start[m.ng] = n - 1;
        else if (fault == 7) m.start[1] = m.start[0];
        else if (fault == 8) m.start[1] = n + 1;
        else m.ptr[0] = 0;
        f.upload(t, m, 2);
        f.check(t, nullptr, false); f.check(t, &graph, false); cases += 2;
        f.upload(t, valid, 3);
        f.check(t, &graph, false); ++cases; // poisoned/malformed plan never survives recovery
    }
    return cases;
}

int resident_cases(Fixture& f) {
    constexpr int ptr_offset = ((4 + (CAP + 1) + 2 * CAP) + 1) & ~1;
    constexpr int words = ptr_offset + 4 * CAP + (CAP + 1) + 1;
    Buffer<int32_t> ids{CAP}, resident{EXPERTS}, original{words}, combined{words};
    Buffer<unsigned long long> offsets{EXPERTS};
    std::vector<int32_t> residency(EXPERTS); std::iota(residency.begin(), residency.end(), 0);
    std::vector<unsigned long long> slot_offsets(EXPERTS);
    for (int i = 0; i < EXPERTS; ++i) slot_offsets[i] = size_t(i) * f.layout.bytes;
    offsets.put(slot_offsets);
    const auto grouped_ptr = [&](int32_t* p) { return reinterpret_cast<unsigned long long*>(p + ptr_offset); };
    const auto launch = [&](int t) {
        k::resident_plan(ids.p, t * K, K, resident.p, EXPERTS, f.arena.p, offsets.p,
            static_cast<long long>(f.layout.bytes), original.p, CAP, nullptr, 0, f.stream.s, f.legacy_error.p);
        k::native_expert_down_combine(f.layout, grouped_ptr(original.p), original.p + 4, original.p,
            original.p + 4 + CAP + 1, t * K, t * K, f.hidden.p, f.weights.p, f.shared.p, f.gate.p,
            f.old.p, f.legacy_error.p, t, f.stream.s);
        k::resident_plan_with_inverse(ids.p, t * K, K, resident.p, EXPERTS, f.arena.p, offsets.p,
            static_cast<long long>(f.layout.bytes), combined.p, CAP, nullptr, 0, f.stream.s, f.error.p, f.plan());
        k::native_expert_down_combine_preplanned(f.layout, f.plan(), t * K, f.hidden.p, f.weights.p,
            f.shared.p, f.gate.p, f.out.p, f.error.p, t, f.stream.s, f.peer.inbox->p);
    };
    std::array<Graph, 8> graphs;
    for (int t = 1; t <= 8; ++t) graphs[t - 1].capture(f.stream.s, [&] { launch(t); });
    int cases = 0;
    for (int pass = 0; pass < 2; ++pass) for (int t : {8, 1, 7, 2, 6, 3, 5, 4}) {
        // Valid, invalid index, nonresident, valid recovery. Graph pointers stay
        // fixed; both active T and routing change across capacity-eight replays.
        for (int fault = 0; fault < 4; ++fault) {
            std::vector<int32_t> input(t * K);
            for (int d = 0; d < t * K; ++d) input[d] = (d * (pass ? 3 : 7) + t + fault) % EXPERTS;
            auto current_residency = residency;
            const bool bad = fault == 1 || fault == 2;
            if (fault == 1) input[t * K - 1] = pass ? -1 : EXPERTS;
            if (fault == 2) current_residency[input[0]] = -1;
            f.upload(t, f.routes(t, pass, false), 3 + pass + fault);
            ids.put(input); resident.put(current_residency);
            f.poison(); original.poison(); combined.poison();
            ck(cudaDeviceSynchronize(), "resident upload/poison drain");
            if (pass == 0 && fault == 0) launch(t); else graphs[t - 1].launch(f.stream.s);
            const auto peer_result = f.peer.joined_read(f.stream.s, t);
            const auto old = f.old.get(size_t(t) * N), actual = f.out.get(size_t(t) * N);
            exact(old, actual, "resident old+down versus fused+preplanned output differs");
            exact(actual, peer_result, "resident peer result differs");
            exact(original.get(words), combined.get(words), "ordinary resident plan bytes changed");
            // Legacy resident failure replaces its error with 1; the down consumer
            // then ORs the plan bit. Only successful residency preserves prior bits.
            const uint32_t want_error = bad ? (1u | k::kNativeDownCombinePlanError) : unrelated_error;
            need(f.error.get(1)[0] == want_error && f.legacy_error.get(1)[0] == want_error, "resident error preservation differs");
            const auto entry = f.entry.get(t * K);
            const auto blob = f.blob.get(t * K);
            const auto status = f.token_error.get(t);
            if (bad) {
                for (int d = 0; d < t * K; ++d) need(entry[d] == -1 && blob[d] == 0, "bad residency reused stale inverse");
                for (auto s : status) need(s == k::kNativeDownCombinePlanError, "bad residency token status");
                exact(std::vector<float>(size_t(t) * N), actual, "bad residency output not safe zero");
            } else {
                const auto p = combined.get(words);
                Metadata m;
                m.ng = p[0];
                m.start.assign(p.begin() + 4, p.begin() + 4 + CAP + 1);
                m.dst.assign(p.begin() + 4 + CAP + 1, p.begin() + 4 + CAP + 1 + t * K);
                m.ptr.resize(CAP);
                std::memcpy(m.ptr.data(), p.data() + ptr_offset, CAP * sizeof(unsigned long long));
                const auto expected = inverse(m, t);
                exact(expected.entry, entry, "resident inverse entries differ");
                exact(expected.blob, blob, "resident inverse blobs differ");
                exact(expected.token_error, status, "resident inverse status differs");
            }
            f.entry.guards(t * K, "resident inverse capacity overwrite");
            f.blob.guards(t * K, "resident blob capacity overwrite");
            f.token_error.guards(t, "resident token capacity overwrite");
            f.out.guards(size_t(t) * N, "resident output capacity overwrite");
            ++cases;
        }
    }
    return cases;
}
int number(const char* value) {
    size_t end = 0; const std::string text(value); const int v = std::stoi(text, &end);
    need(end == text.size() && v >= 0, "device must be a nonnegative integer"); return v;
}
} // namespace

int main(int argc, char** argv) {
    try {
        int device = 0, peer = -1;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if ((arg == "--device" || arg == "--peer-device") && i + 1 < argc) {
                const int v = number(argv[++i]);
                if (arg == "--device") device = v; else peer = v;
            } else throw std::invalid_argument("usage: native_down_plan_parity [--device N] [--peer-device N]");
        }
        int devices = 0;
        const auto available = cudaGetDeviceCount(&devices);
        if (available != cudaSuccess || devices == 0) {
            std::fprintf(stderr, "NATIVE_DOWN_PLAN_PARITY_SKIP no GPU runtime/device; GPU validation was not run\n");
            return 77;
        }
        need(device < devices && (peer < 0 || (peer < devices && peer != device)), "invalid device or distinct peer selection");
        on(device);
        // Calls below use the original complete 32-lane native dot as the
        // arithmetic oracle, independent of environment's tuning switches.
        const bool prior_old = k::iq_old_kernels(); k::iq_set_old_kernels(true);
        int valid = 0, faults = 0, residents = 0;
        for (int type : {20, 42}) for (int f : {320, 640}) {
            Fixture fixture(type, f, device, peer);
            valid += valid_cases(fixture);
            faults += fault_cases(fixture);
            residents += resident_cases(fixture);
            std::printf("NATIVE_DOWN_PLAN_CASE_PASS type=%d F=%d T=1..8 capacity=80 fresh=1 capture=1 replay=1 arithmetic=1 malformed=1 resident=1 peer=%s\n",
                        type, f, peer < 0 ? "local-event-join" : "remote-event-join");
        }
        // The complete matrix above observes writes on the requested peer; a
        // reverse-direction fault/recovery gate checks both peer apertures.
        if (peer >= 0) {
            on(peer);
            Fixture reverse(20, 320, peer, device);
            faults += fault_cases(reverse);
            std::printf("NATIVE_DOWN_PLAN_PEER_PASS source=%d destination=%d reverse=1 event_join=1 safe_zero=1\n", device, peer);
        }
        on(device); ck(cudaDeviceSynchronize(), "final device drain");
        k::iq_set_old_kernels(prior_old);
        std::printf("NATIVE_DOWN_PLAN_COUNTS valid=%d malformed_and_recovery=%d resident=%d real_peer=%d\n", valid, faults, residents, peer >= 0);
        std::puts("NATIVE_DOWN_PLAN_PARITY_PASS");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "NATIVE_DOWN_PLAN_PARITY_FAIL %s\n", e.what()); return 1;
    }
}
