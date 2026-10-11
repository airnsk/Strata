// Standalone, model-free protocol admission. No engine, RCCL or performance claim.
// CPU helper: c++ -std=c++17 -O2 -ffp-contract=off -DSTRATA_VRAM_CPU_SELFTEST
//   tests/hip/tp2_vram_rendezvous.cpp -o /tmp/vram-selftest && /tmp/vram-selftest --selftest
#include <algorithm>
#include <array>
#include <atomic>
#include <cfenv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifndef STRATA_VRAM_CPU_SELFTEST
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include "strata/kernels/tp_vram_rendezvous.hpp"
#define VRAM_HD __host__ __device__
#else
#define VRAM_HD
#endif

namespace {
constexpr int kGuard = 16, kCapacity = 8 * 6144, kWitnessCapacity = 8192;
constexpr uint32_t kPoison = 0x7fc00000u, kCanary = 0xdecab975u;
using Clock = std::chrono::steady_clock;
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
VRAM_HD int width(int kind) { return kind == 0 ? 3072 : kind == 1 ? 1280 : 2560; }
VRAM_HD int run_words(int kind) { return kind == 0 ? 1024 : width(kind); }
VRAM_HD int result_words(int kind, int tokens) { return width(kind) * tokens * (kind == 2 ? 1 : 2); }
VRAM_HD uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
// Epoch changes every word. Raw seams include NaN payloads, signed subnormals and
// ordinary floats. Sum data are finite normal FP32 and explicit signed zeros;
// no subnormal/NaN arithmetic ambiguity in its oracle.
VRAM_HD uint32_t pattern(int kind, int rank, uint64_t epoch, int i) {
    const uint32_t h = mix(uint32_t(i) * 0x85ebca6bu ^ uint32_t(rank + 1) * 0xc2b2ae35u ^ uint32_t(kind + 1) * 0x165667b1u);
    const uint32_t m = h + uint32_t(epoch) * 7919u;
    if (kind != 2) {
        if (i % 17 == 0) return 0x80000000u | (m & 0x007fffffu);
        if (i % 17 == 1) return 0x7fe00000u | (m & 0x001fffffu);
        return (h & 0x80000000u) | 0x3f000000u | (m & 0x007fffffu);
    }
    if (i % 19 == 0) return uint32_t(epoch & 1u) << 31; // -0+-0 and +0++0
    if (i % 19 == 1) return uint32_t((epoch + uint64_t(rank)) & 1u) << 31; // opposite zeros
    if (i % 19 == 2) return (uint32_t(rank) << 31) | 0x3f000000u | ((uint32_t(i) + uint32_t(epoch) * 7919u) & 0x007fffffu); // cancellation
    if (i % 19 == 3) return rank == 0 ? 0x3f800000u | uint32_t(epoch & 1u) : 0x33800000u | (uint32_t(epoch & 1u) << 31); // RNE ties
    return (h & 0x80000000u) | (uint32_t(124 + ((h >> 24) + epoch) % 7) << 23) | (m & 0x007fffffu);
}
uint32_t host_add(uint32_t a, uint32_t b) {
    float af, bf; std::memcpy(&af, &a, 4); std::memcpy(&bf, &b, 4);
    volatile float av = af, bv = bf, sum = av + bv;
    float value = sum; uint32_t bits; std::memcpy(&bits, &value, 4); return bits;
}
uint32_t expected(int kind, int tokens, uint64_t epoch, int i) {
    (void)tokens;
    if (kind == 2) return host_add(pattern(kind, 0, epoch, i), pattern(kind, 1, epoch, i));
    const int w = width(kind), run = run_words(kind), col = i % (2 * w);
    const int rank = (col / run) % 2;
    const int src = (i / (2 * w)) * w + (col / (2 * run)) * run + col % run;
    return pattern(kind, rank, epoch, src);
}
struct Options {
    std::string scenario = "normal";
    std::array<int, 2> devices{0, 1};
    int iterations = 128, timeout_ms = 10000;
    bool selftest = false;
};
int integer(const std::string& s) {
    require(!s.empty() && s[0] >= '1' && s[0] <= '9' && s.size() <= 6, "invalid positive integer");
    require(s.find_first_not_of("0123456789") == std::string::npos, "invalid positive integer");
    return std::stoi(s);
}
Options parse(int argc, char** argv) {
    Options o; std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        require(seen.insert(a).second, "repeated option: " + a);
        if (a == "--selftest") { o.selftest = true; continue; }
        if (a == "--help") {
            std::puts("tp2_vram_rendezvous [--scenario normal|delayed-peer|missing-peer] [--iterations 128..512] [--timeout-ms 1000..60000] [--devices 0,1|1,0] [--selftest]");
            std::exit(0);
        }
        require(++i < argc, "missing option value"); const std::string v = argv[i];
        if (a == "--scenario") o.scenario = v;
        else if (a == "--iterations") o.iterations = integer(v);
        else if (a == "--timeout-ms") o.timeout_ms = integer(v);
        else if (a == "--devices") { require(v == "0,1" || v == "1,0", "exactly devices 0,1 or 1,0 required"); o.devices = v == "0,1" ? std::array<int, 2>{0, 1} : std::array<int, 2>{1, 0}; }
        else throw std::runtime_error("unknown option: " + a);
    }
    require(o.scenario == "normal" || o.scenario == "delayed-peer" || o.scenario == "missing-peer", "invalid scenario");
    require(o.iterations >= 128 && o.iterations <= 512, "iterations must be 128..512");
    require(o.timeout_ms >= 1000 && o.timeout_ms <= 60000, "timeout-ms must be 1000..60000");
    return o;
}
int selftest() {
    require(std::fesetround(FE_TONEAREST) == 0, "host rounding unavailable");
    size_t checked = 0;
    for (int kind = 0; kind < 3; ++kind) for (int t : {1, 8}) for (uint64_t e : {1ull, 2ull, 31ull, 1536ull, 0xfffffffeull, 0xffffffffull, 0x100000000ull, 0x100000001ull}) {
        std::vector<uint32_t> out(result_words(kind, t), kPoison), hits(out.size());
        if (kind < 2) {
            // Independent forward scatter, not the oracle's inverse gather.
            for (int r = 0; r < 2; ++r) for (int i = 0; i < t * width(kind); ++i) {
                int w = width(kind), run = run_words(kind), col = i % w;
                int dst = (i / w) * 2 * w + (col / run) * 2 * run + r * run + col % run;
                out[dst] = pattern(kind, r, e, i); ++hits[dst];
            }
        } else for (size_t i = 0; i < out.size(); ++i) { out[i] = host_add(pattern(kind, 0, e, int(i)), pattern(kind, 1, e, int(i))); hits[i] = 1; }
        bool stale = false;
        for (size_t i = 0; i < out.size(); ++i) {
            require(hits[i] == 1 && out[i] == expected(kind, t, e, int(i)), "host layout oracle mismatch");
            require(out[i] != kPoison, "oracle collides with poison");
            stale |= out[i] != expected(kind, t, e + 1, int(i)); ++checked;
        }
        require(stale, "stale payload mutation missed");
        for (int r = 0; r < 2; ++r) for (int i = 0; i < t * width(kind); ++i)
            require(pattern(kind, r, e, i) != pattern(kind, r, e + 1, i), "producer word did not change");
        out[out.size() / 2] ^= 1u;
        require(out[out.size() / 2] != expected(kind, t, e, int(out.size() / 2)), "bit mutation missed");
    }
    require(host_add(0x80000000u, 0x80000000u) == 0x80000000u && host_add(0x80000000u, 0) == 0, "signed-zero oracle");
    require(host_add(0x3f800000u, 0x33800000u) == 0x3f800000u && host_add(0x3f800001u, 0x33800000u) == 0x3f800002u, "RNE ties oracle");
    std::printf("VRAM_CPU_SELFTEST_PASS checked_words=%zu NOT_GPU_TEST=1\n", checked); return 0;
}

#ifndef STRATA_VRAM_CPU_SELFTEST
using namespace strata::kernels;
void hip_check(hipError_t code, const char* call) {
    if (code != hipSuccess) throw std::runtime_error(std::string(call) + ": " + hipGetErrorString(code));
}
// One persistent host owner per HIP context. Deadlines also cover blocking HIP
// API calls, including allocation/capture/drain. On a timeout we do NOT free live
// buffers or call driver teardown; the separately bounded process is terminated.
class Workers {
public:
    explicit Workers(const Options& o) : options(o) {
        for (int r = 0; r < 2; ++r) threads[r] = std::thread([this, r] { loop(r); });
    }
    ~Workers() { { std::lock_guard<std::mutex> l(mutex); stop = true; } changed.notify_all(); for (auto& t : threads) t.join(); }
    void run(const char* stage, const std::function<void(int)>& fn, int bound = 0) {
        std::unique_lock<std::mutex> l(mutex);
        action = fn; error = nullptr; completed = 0; ++generation; changed.notify_all();
        if (!done.wait_for(l, std::chrono::milliseconds(bound ? bound : options.timeout_ms), [&] { return completed == 2; })) {
            std::fprintf(stderr, "VRAM_WATCHDOG_TIMEOUT scenario=%s stage=%s completed=%d teardown=process_exit_no_release\n", options.scenario.c_str(), stage, completed);
            std::fflush(nullptr); std::_Exit(70);
        }
        if (error) std::rethrow_exception(error);
    }
private:
    void loop(int r) {
        unsigned seen = 0;
        for (;;) {
            std::function<void(int)> fn;
            { std::unique_lock<std::mutex> l(mutex); changed.wait(l, [&] { return stop || generation != seen; }); if (stop) return; seen = generation; fn = action; }
            try { fn(r); } catch (...) { std::lock_guard<std::mutex> l(mutex); if (!error) error = std::current_exception(); }
            { std::lock_guard<std::mutex> l(mutex); ++completed; } done.notify_one();
        }
    }
    const Options& options;
    std::array<std::thread, 2> threads;
    std::mutex mutex; std::condition_variable changed, done;
    std::function<void(int)> action; std::exception_ptr error;
    unsigned generation = 0; int completed = 0; bool stop = false;
};
struct Buffer {
    uint32_t* base = nullptr;
    uint32_t* data() const { return base + kGuard; }
    void allocate() {
        hip_check(hipMalloc(reinterpret_cast<void**>(&base), (kCapacity + 2 * kGuard) * 4), "hipMalloc payload");
        std::vector<uint32_t> init(kCapacity + 2 * kGuard, kCanary);
        std::fill(init.begin() + kGuard, init.end() - kGuard, kPoison);
        hip_check(hipMemcpy(base, init.data(), init.size() * 4, hipMemcpyHostToDevice), "initialize guards");
    }
    void verify(int n, const std::function<uint32_t(int)>& oracle) const {
        std::vector<uint32_t> values(kCapacity + 2 * kGuard);
        hip_check(hipMemcpy(values.data(), base, values.size() * 4, hipMemcpyDeviceToHost), "read payload");
        for (int j = 0; j < int(values.size()); ++j) {
            int i = j - kGuard; uint32_t want = i < 0 || i >= kCapacity ? kCanary : i < n ? oracle(i) : kPoison;
            if (values[j] != want) { char m[160]; std::snprintf(m, sizeof(m), "payload/guard mismatch index=%d want=%08x actual=%08x", i, want, values[j]); throw std::runtime_error(m); }
        }
    }
    void release() { hip_check(hipFree(base), "hipFree payload"); base = nullptr; }
};
struct DeviceState { uint64_t epoch, sequence; };
struct Witness { uint64_t epoch, status, failures, checked_words, polls; };
struct Rank {
    hipStream_t stream = nullptr;
    Buffer src, dst;
    TpVramRendezvousControl* control = nullptr;
    TpVramRendezvousStamp* stamp = nullptr;
    uint32_t* inbox = nullptr;
    DeviceState* state = nullptr;
    Witness* witnesses = nullptr;
    hipGraph_t graph = nullptr; hipGraphExec_t exec = nullptr;
    uint64_t calls = 0, timeout_ticks = 0, epoch_base = 0;
    void graph_release() {
        if (exec) hip_check(hipGraphExecDestroy(exec), "destroy graph exec");
        exec = nullptr;
        if (graph) hip_check(hipGraphDestroy(graph), "destroy graph");
        graph = nullptr;
    }
};
__global__ void produce(uint32_t* src, uint32_t* dst, DeviceState* state, int kind, int tokens, int rank) {
    if (threadIdx.x == 0) { ++state->epoch; ++state->sequence; }
    __syncthreads();
    for (int i = int(threadIdx.x); i < kCapacity; i += int(blockDim.x)) {
        src[i] = i < tokens * width(kind) ? pattern(kind, rank, state->epoch, i) : kPoison;
        dst[i] = kPoison;
    }
}
__device__ uint32_t device_expected(int kind, uint64_t epoch, int i) {
    if (kind == 2) return __float_as_uint(__fadd_rn(__uint_as_float(pattern(kind, 0, epoch, i)), __uint_as_float(pattern(kind, 1, epoch, i))));
    // Inverse mapping is checked against an independent host forward-scatter selftest.
    int w = width(kind), run = run_words(kind), col = i % (2 * w);
    return pattern(kind, (col / run) % 2, epoch, (i / (2 * w)) * w + (col / (2 * run)) * run + col % run);
}
__global__ void witness(const uint32_t* src, const uint32_t* dst, const DeviceState* state,
                        const TpVramRendezvousStamp* stamp, Witness* results, int kind, int tokens, int rank) {
    __shared__ unsigned bad;
    if (!threadIdx.x) bad = 0;
    __syncthreads();
    const uint64_t epoch = state->epoch;
    for (int i = int(threadIdx.x) - kGuard; i < kCapacity + kGuard; i += int(blockDim.x)) {
        bool guard = i < 0 || i >= kCapacity;
        uint32_t want_src = guard ? kCanary : i < tokens * width(kind) ? pattern(kind, rank, epoch, i) : kPoison;
        uint32_t want_dst = guard ? kCanary : i < result_words(kind, tokens) ? device_expected(kind, epoch, i) : kPoison;
        if (src[i] != want_src || dst[i] != want_dst) atomicAdd(&bad, 1u);
    }
    __syncthreads();
    if (!threadIdx.x && state->sequence > 0 && state->sequence <= kWitnessCapacity) {
        if (stamp->epoch != epoch || stamp->status != uint64_t(TpVramRendezvousStatus::Ready)) ++bad;
        results[state->sequence - 1] = {epoch, stamp->status, bad, uint64_t(result_words(kind, tokens)), stamp->polls};
    }
}
TpVramRendezvousPacket packet(std::array<Rank, 2>& ranks, int r, int kind, int tokens) {
    TpVramRendezvousPacket p;
    p.count = 1; p.mode = kind == 2 ? TpVramRendezvousMode::SumF32 : TpVramRendezvousMode::GatherU32;
    p.spans[0] = {ranks[r].src.data(), ranks[r].dst.data(), tokens, width(kind), run_words(kind)};
    p.own_inbox = ranks[r].inbox; p.peer_inbox = ranks[1 - r].inbox; p.inbox_words = size_t(tokens * width(kind));
    return p;
}
void enqueue(std::array<Rank, 2>& ranks, int r, int kind, int tokens, bool checked = true) {
    Rank& x = ranks[r];
    hipLaunchKernelGGL(produce, dim3(1), dim3(256), 0, x.stream, x.src.data(), x.dst.data(), x.state, kind, tokens, r);
    hip_check(hipGetLastError(), "producer launch");
    tp_vram_rendezvous(packet(ranks, r, kind, tokens), x.control, ranks[1 - r].control, x.stamp, r, x.timeout_ticks, x.stream);
    if (checked) {
        hipLaunchKernelGGL(witness, dim3(1), dim3(256), 0, x.stream, x.src.data(), x.dst.data(), x.state, x.stamp, x.witnesses, kind, tokens, r);
        hip_check(hipGetLastError(), "witness launch");
    }
}
void open_lifecycle(Workers& workers, std::array<Rank, 2>& ranks, const Options& o, uint64_t seed = 0) {
    workers.run("fresh-lifecycle", [&](int r) {
        hip_check(hipSetDevice(o.devices[r]), "hipSetDevice");
        auto access = hipDeviceEnablePeerAccess(o.devices[1 - r], 0);
        if (access == hipErrorPeerAccessAlreadyEnabled) (void)hipGetLastError();
        else hip_check(access, "enable peer access");
        Rank& x = ranks[r];
        int wall_khz = 0; hip_check(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, o.devices[r]), "wall clock rate");
        require(wall_khz > 0, "wall clock rate invalid");
        // A device timeout leaves ample room for stamp readback and sticky checks
        // within the independent host deadline. Poll cap may fire earlier.
        x.timeout_ticks = uint64_t(wall_khz) * uint64_t(o.timeout_ms / 4);
        hip_check(hipStreamCreateWithFlags(&x.stream, hipStreamNonBlocking), "create stream");
        x.src.allocate(); x.dst.allocate();
        hip_check(hipExtMallocWithFlags(reinterpret_cast<void**>(&x.control), sizeof(*x.control), hipDeviceMallocUncached), "uncached control allocation");
        hip_check(hipExtMallocWithFlags(reinterpret_cast<void**>(&x.inbox), 2 * kTpVramRendezvousMaxWords * sizeof(uint32_t), hipDeviceMallocUncached), "uncached inbox allocation");
        hip_check(hipMalloc(reinterpret_cast<void**>(&x.stamp), sizeof(*x.stamp)), "stamp allocation");
        hip_check(hipMalloc(reinterpret_cast<void**>(&x.state), sizeof(*x.state)), "producer state allocation");
        hip_check(hipMalloc(reinterpret_cast<void**>(&x.witnesses), kWitnessCapacity * sizeof(Witness)), "witness allocation");
        TpVramRendezvousControl initial{};
        initial.epoch.value = seed;
        if (seed) for (int slot = 0; slot < 2; ++slot) {
            const uint64_t prior = (seed - 1) % 2 == uint64_t(slot) ? seed : seed - 1;
            initial.ready[slot].value = prior; initial.consumed[slot].value = prior;
        }
        DeviceState producer{seed, 0}; x.epoch_base = seed;
        hip_check(hipMemcpyAsync(x.control, &initial, sizeof(initial), hipMemcpyHostToDevice, x.stream), "initialize fresh control");
        hip_check(hipMemsetAsync(x.inbox, 0xa5, 2 * kTpVramRendezvousMaxWords * 4, x.stream), "poison inbox");
        hip_check(hipMemsetAsync(x.stamp, 0, sizeof(*x.stamp), x.stream), "initialize stamp");
        hip_check(hipMemcpyAsync(x.state, &producer, sizeof(producer), hipMemcpyHostToDevice, x.stream), "initialize producer epoch");
        hip_check(hipMemsetAsync(x.witnesses, 0, kWitnessCapacity * sizeof(Witness), x.stream), "initialize witnesses");
        hip_check(hipStreamSynchronize(x.stream), "initialization drain");
    }, 60000);
}
void close_lifecycle(Workers& workers, std::array<Rank, 2>& ranks) {
    // Barrier completes BOTH drains before either allocation can be released.
    workers.run("drain-both", [&](int r) { hip_check(hipStreamSynchronize(ranks[r].stream), "drain both streams"); });
    workers.run("release-after-drain", [&](int r) {
        Rank& x = ranks[r]; x.graph_release(); x.src.release(); x.dst.release();
        hip_check(hipFree(x.control), "free control"); hip_check(hipFree(x.inbox), "free inbox");
        hip_check(hipFree(x.stamp), "free stamp"); hip_check(hipFree(x.state), "free state");
        hip_check(hipFree(x.witnesses), "free witnesses"); hip_check(hipStreamDestroy(x.stream), "destroy stream"); x = Rank{};
    });
}
uint64_t check_witnesses(Rank& x, int rank, int kind, int tokens, uint64_t start, int count) {
    require(start + uint64_t(count) <= kWitnessCapacity, "witness capacity exceeded");
    std::vector<Witness> records(count);
    hip_check(hipMemcpy(records.data(), x.witnesses + start, records.size() * sizeof(Witness), hipMemcpyDeviceToHost), "read live witnesses");
    uint64_t polls = 0;
    for (int i = 0; i < count; ++i) {
        const Witness& w = records[i]; polls += w.polls;
        require(w.epoch == x.epoch_base + start + uint64_t(i) + 1 && w.status == uint64_t(TpVramRendezvousStatus::Ready) && w.failures == 0 &&
                w.checked_words == uint64_t(result_words(kind, tokens)), "live GPU witness rejected (rank=" + std::to_string(rank) + ", epoch=" + std::to_string(start + i + 1) + ")");
    }
    DeviceState state{}; hip_check(hipMemcpy(&state, x.state, sizeof(state), hipMemcpyDeviceToHost), "read device epoch");
    require(state.epoch == x.epoch_base + start + uint64_t(count) && state.sequence == start + uint64_t(count), "producer epoch mismatch");
    x.src.verify(tokens * width(kind), [&](int i) { return pattern(kind, rank, state.epoch, i); });
    x.dst.verify(result_words(kind, tokens), [&](int i) { return expected(kind, tokens, state.epoch, i); });
    TpVramRendezvousControl control{};
    hip_check(hipMemcpy(&control, x.control, sizeof(control), hipMemcpyDeviceToHost), "read successful controls");
    require(control.epoch.value == state.epoch && control.local_failure.value == 0 && control.peer_failure.value == 0, "successful pair has sticky failure or wrong epoch");
    return polls;
}
void correct_case(Workers& workers, std::array<Rank, 2>& ranks, const Options& o, int life, int kind, int tokens, int first, bool delayed) {
    const auto start = ranks[0].calls;
    require(start == ranks[1].calls && start + o.iterations <= kWitnessCapacity, "inconsistent witness sequence");
    workers.run("capture", [&](int r) {
        auto& x = ranks[r]; x.graph_release();
        hip_check(hipStreamBeginCapture(x.stream, hipStreamCaptureModeThreadLocal), "begin capture");
        enqueue(ranks, r, kind, tokens);
        hip_check(hipStreamEndCapture(x.stream, &x.graph), "end capture");
        hip_check(hipGraphInstantiate(&x.exec, x.graph, nullptr, nullptr, 0), "instantiate graph");
    });
    std::atomic<int> turn{0};
    workers.run("graph-replay-live-witness", [&](int r) {
        auto& x = ranks[r];
        for (int i = 0; i < o.iterations; ++i) {
            const int expected_turn = 2 * i + (r == first ? 0 : 1);
            while (turn.load(std::memory_order_acquire) != expected_turn) std::this_thread::yield();
            // Native HIP graph launches are asynchronous. Publish the next turn
            // only after successful submission, establishing real issue order.
            if (delayed && r != first) std::this_thread::sleep_for(std::chrono::milliseconds(5));
            hip_check(hipGraphLaunch(x.exec, x.stream), "graph launch");
            turn.fetch_add(1, std::memory_order_release);
        }
        hip_check(hipStreamSynchronize(x.stream), "graph replay drain");
    }, std::max(o.timeout_ms, o.iterations * 20));
    std::array<uint64_t, 2> observed_polls{};
    workers.run("independent-readback", [&](int r) { observed_polls[r] = check_witnesses(ranks[r], r, kind, tokens, start, o.iterations); ranks[r].calls += o.iterations; });
    require(!delayed || observed_polls[first] > 0, "delayed peer had no observed live device wait");
    std::printf("VRAM_CASE_PASS scenario=%s life=%d kind=%d tokens=%d order=%d%d replays=%d epochs=%llu..%llu device_witness=1 guards=1 ring_slots=2\n",
                o.scenario.c_str(), life, kind, tokens, first, 1 - first, o.iterations,
                static_cast<unsigned long long>(ranks[0].epoch_base + start + 1), static_cast<unsigned long long>(ranks[0].epoch_base + start + o.iterations));
}
void alternating_shapes(Workers& workers, std::array<Rank, 2>& ranks, const Options& o, int life) {
    const uint64_t start = ranks[0].calls;
    require(start == ranks[1].calls && start + o.iterations <= kWitnessCapacity, "alternating witness capacity");
    std::array<std::array<hipGraph_t, 2>, 2> graphs{};
    std::array<std::array<hipGraphExec_t, 2>, 2> execs{};
    workers.run("alternating-shape-capture", [&](int r) {
        for (int shape = 0; shape < 2; ++shape) {
            hip_check(hipStreamBeginCapture(ranks[r].stream, hipStreamCaptureModeThreadLocal), "alternating capture begin");
            enqueue(ranks, r, 0, shape == 0 ? 8 : 1);
            hip_check(hipStreamEndCapture(ranks[r].stream, &graphs[r][shape]), "alternating capture end");
            hip_check(hipGraphInstantiate(&execs[r][shape], graphs[r][shape], nullptr, nullptr, 0), "alternating instantiate");
        }
    });
    std::atomic<int> turn{0};
    workers.run("alternating-shape-replay", [&](int r) {
        for (int i = 0; i < o.iterations; ++i) {
            const int wanted = 2 * i + (r == life ? 0 : 1);
            while (turn.load(std::memory_order_acquire) != wanted) std::this_thread::yield();
            hip_check(hipGraphLaunch(execs[r][i % 2], ranks[r].stream), "alternating graph launch");
            turn.fetch_add(1, std::memory_order_release);
        }
        hip_check(hipStreamSynchronize(ranks[r].stream), "alternating replay drain");
    });
    workers.run("alternating-shape-witness", [&](int r) {
        auto& x = ranks[r];
        std::vector<Witness> records(o.iterations);
        hip_check(hipMemcpy(records.data(), x.witnesses + start, records.size() * sizeof(Witness), hipMemcpyDeviceToHost), "alternating witness readback");
        for (int i = 0; i < o.iterations; ++i) {
            const int tokens = i % 2 == 0 ? 8 : 1; const Witness& w = records[i];
            require(w.epoch == x.epoch_base + start + uint64_t(i) + 1 && w.status == uint64_t(TpVramRendezvousStatus::Ready) &&
                    w.failures == 0 && w.checked_words == uint64_t(result_words(0, tokens)), "alternating shape live GPU witness failed");
        }
        const int last_tokens = (o.iterations - 1) % 2 == 0 ? 8 : 1;
        check_witnesses(x, r, 0, last_tokens, start + o.iterations - 1, 1);
        x.calls += o.iterations;
        for (int shape = 0; shape < 2; ++shape) {
            hip_check(hipGraphExecDestroy(execs[r][shape]), "alternating exec destroy");
            hip_check(hipGraphDestroy(graphs[r][shape]), "alternating graph destroy");
        }
    });
    std::printf("VRAM_ALTERNATING_SHAPE_PASS scenario=normal life=%d order=%d%d replays=%d tokens=8,1 fixed_stride=32768 device_witness=1\n",
                life, life, 1 - life, o.iterations);
}
TpVramRendezvousStamp read_stamp(const Rank& x) {
    TpVramRendezvousStamp stamp{}; hip_check(hipMemcpy(&stamp, x.stamp, sizeof(stamp), hipMemcpyDeviceToHost), "read failure stamp"); return stamp;
}
void failed_call(Workers& workers, std::array<Rank, 2>& ranks, int rank, TpVramRendezvousStatus status) {
    workers.run("fault-one-rank", [&](int r) {
        if (r != rank) return;
        enqueue(ranks, r, 2, 8, false);
        hip_check(hipStreamSynchronize(ranks[r].stream), "fault stream drain");
        const auto stamp = read_stamp(ranks[r]);
        require(stamp.status == uint64_t(status), "fault did not produce exact expected failure status");
        if (status == TpVramRendezvousStatus::TimedOut) require(stamp.polls > 0 && stamp.elapsed_ticks > 0, "missing peer did not enter a bounded live wait");
        ranks[r].dst.verify(0, [](int) { return 0u; });
    });
}
void missing_cases(Workers& workers, const Options& o) {
    for (int first : {0, 1}) {
        std::array<Rank, 2> ranks{}; open_lifecycle(workers, ranks, o);
        std::printf("VRAM_FAULT_ARMED scenario=missing-peer first=%d omitted=%d\n", first, 1 - first);
        failed_call(workers, ranks, first, TpVramRendezvousStatus::TimedOut);
        // The late peer uses the SAME failed allocation; neither signals nor
        // producer epochs are reset. Both ranks must stay fail-closed.
        failed_call(workers, ranks, 1 - first, TpVramRendezvousStatus::Aborted);
        failed_call(workers, ranks, first, TpVramRendezvousStatus::Aborted);
        workers.run("sticky-control-readback", [&](int r) {
            TpVramRendezvousControl c{}; hip_check(hipMemcpy(&c, ranks[r].control, sizeof(c), hipMemcpyDeviceToHost), "read sticky control");
            require(c.local_failure.value != 0 || c.peer_failure.value != 0, "sticky failure disappeared");
        });
        close_lifecycle(workers, ranks);
        std::printf("VRAM_FAULT_CASE_PASS first=%d omitted=%d late_peer_aborted=1 repeated_call_aborted=1 output_poison=1 both_drained=1\n", first, 1 - first);
    }
    // Corrupt only fresh, idle local metadata; this is explicit test injection.
    // A future publication/ack or wrapped epoch must fail before payload consume.
    for (int fault = 0; fault < 3; ++fault) {
        std::array<Rank, 2> ranks{}; open_lifecycle(workers, ranks, o);
        workers.run("negative-metadata-injection", [&](int r) {
            if (r) return;
            TpVramRendezvousControl c{};
            if (fault == 0) c.ready[0].value = 2;
            if (fault == 1) c.consumed[0].value = 2;
            if (fault == 2) c.epoch.value = std::numeric_limits<uint64_t>::max();
            hip_check(hipMemcpy(ranks[r].control, &c, sizeof(c), hipMemcpyHostToDevice), "inject idle negative metadata");
        });
        failed_call(workers, ranks, 0, TpVramRendezvousStatus::InvalidEpoch);
        close_lifecycle(workers, ranks);
        std::printf("VRAM_NEGATIVE_CASE_PASS kind=%s status=invalid-epoch output_poison=1 both_drained=1\n", fault == 0 ? "future-ready" : fault == 1 ? "future-consumed" : "epoch-wrap");
    }
    std::puts("VRAM_TIMEOUT_GATE_PASS scenario=missing-peer");
    std::puts("VRAM_STICKY_FAILURE_PASS scenario=missing-peer");
    std::puts("VRAM_DRAIN_PASS scenario=missing-peer");
}
int gpu_run(const Options& o) {
    require(tp_vram_rendezvous_supported(), "compiled HIP/gfx906/atomic ABI unsupported; no admission");
    Workers workers(o);
    std::array<std::string, 2> pci_ids;
    workers.run("hardware-admission", [&](int r) {
        int count = 0; hip_check(hipGetDeviceCount(&count), "device count"); require(count == 2, "exactly two visible GPUs required");
        hip_check(hipSetDevice(o.devices[r]), "set device"); hipDeviceProp_t prop{};
        hip_check(hipGetDeviceProperties(&prop, o.devices[r]), "properties");
        const std::string arch(prop.gcnArchName);
        require(arch == "gfx906" || arch.rfind("gfx906:", 0) == 0, "only gfx906 accepted");
        require(prop.warpSize == 64, "wave64 required");
        int peer = 0; hip_check(hipDeviceCanAccessPeer(&peer, o.devices[r], o.devices[1 - r]), "peer capability"); require(peer == 1, "bidirectional peer access required");
        char pci[32]{}; hip_check(hipDeviceGetPCIBusId(pci, sizeof(pci), o.devices[r]), "PCI identity");
        pci_ids[r] = pci;
        int runtime = 0, driver = 0; hip_check(hipRuntimeGetVersion(&runtime), "runtime version");
        hip_check(hipDriverGetVersion(&driver), "driver version");
        std::printf("VRAM_GPU rank=%d device=%d pci=%s arch=%s name=%s runtime=%d driver=%d allocation=hipDeviceMallocUncached\n", r, o.devices[r], pci, arch.c_str(), prop.name, runtime, driver);
    }, 60000);
    require(!pci_ids[0].empty() && !pci_ids[1].empty() && pci_ids[0] != pci_ids[1], "two distinct physical GPU PCI identities required");
    if (o.scenario == "missing-peer") missing_cases(workers, o);
    int cases = 0;
    for (int life = 0; life < 2; ++life) {
        const uint64_t seed = life == 1 ? 4294967294ull : 0ull;
        std::array<Rank, 2> ranks{}; open_lifecycle(workers, ranks, o, seed);
        const std::vector<int> tokens = o.scenario == "normal" ? std::vector<int>{1, 8} : std::vector<int>{8};
        for (int t : tokens) for (int first : {0, 1}) for (int kind = 0; kind < 3; ++kind) {
            correct_case(workers, ranks, o, life, kind, t, first, o.scenario == "delayed-peer"); ++cases;
        }
        if (o.scenario == "normal") alternating_shapes(workers, ranks, o, life);
        close_lifecycle(workers, ranks);
        std::printf("VRAM_FRESH_LIFECYCLE_PASS scenario=%s life=%d both_drained=1 released=1\n", o.scenario.c_str(), life);
        if (life == 1) std::printf("VRAM_EPOCH_CARRY_PASS scenario=%s life=1 seed=4294967294\n", o.scenario.c_str());
    }
    if (o.scenario == "delayed-peer") std::puts("VRAM_DELAYED_PEER_PASS scenario=delayed-peer");
    std::printf("VRAM_EXACT_GATE_PASS scenario=%s\n", o.scenario.c_str());
    std::printf("VRAM_LIFECYCLE_GATE_PASS scenario=%s fresh_lifecycles=2\n", o.scenario.c_str());
    std::printf("VRAM_PROBE_PASS scenario=%s cases=%d lifecycles=2 timing_admitted=0\n", o.scenario.c_str(), o.scenario == "missing-peer" ? 2 : cases);
    return 0;
}
#endif
} // namespace
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0); std::setvbuf(stderr, nullptr, _IONBF, 0);
    try {
        const Options o = parse(argc, argv);
        require(std::fesetround(FE_TONEAREST) == 0, "host rounding unavailable");
        if (o.selftest) return selftest();
#ifndef STRATA_VRAM_CPU_SELFTEST
        return gpu_run(o);
#else
        throw std::runtime_error("CPU-only build: use --selftest; no GPU test ran");
#endif
    } catch (const std::exception& e) {
        std::fprintf(stderr, "VRAM_PROBE_FAIL reason=%s\n", e.what()); std::fflush(nullptr);
        // No RAII GPU destruction or driver exit handlers after an uncertain error.
        std::_Exit(1);
    }
}
