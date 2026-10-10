// Standalone opt-in RCCL/HIP transport preflight. No engine/model dependency.
// Build on the stand with its EXISTING HIP compiler/RCCL; never download a runtime.
// CPU helper check: c++ -std=c++17 -DSTRATA_RCCL_CPU_SELFTEST this.cpp -o selftest
//                  ./selftest --selftest
// GPU build: hipcc -x hip -std=c++17 -ffp-contract=off this.cpp -pthread -ldl
//            <existing RCCL include/library flags> -o tp2_rccl_transport
// Use the accompanying bounded stand runner for installed-library discovery and fault cases.
// Capture API references: ROCm RCCL api-library.html (ncclSend/Recv/GroupEnd),
// HIP graph_management.html. Actual installed-library compatibility is a gate.

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
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef STRATA_RCCL_CPU_SELFTEST
#include <hip/hip_runtime.h>
#if defined(STRATA_RCCL_HEADER)
#include STRATA_RCCL_HEADER
#elif __has_include(<rccl/rccl.h>)
#include <rccl/rccl.h>
#elif __has_include(<rccl.h>)
#include <rccl.h>
#elif __has_include(<nccl.h>)
#include <nccl.h>
#else
#error "Existing RCCL development headers are required; no runtime is downloaded."
#endif
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace {
constexpr int kRanks = 2, kBanks = 2, kKinds = 3;
constexpr int kY = 3072, kOutput = 1280, kFfn = 2560, kRun = 8 * 128;
constexpr int kGuard = 16, kThreads = 256;
constexpr uint32_t kPoison = 0x7fc5a39du, kCanary = 0xc3d2e1f0u;
using Clock = std::chrono::steady_clock;

[[maybe_unused]] const char* name(int mode) {
    constexpr const char* names[] = {"y", "output", "ffn", "sequence"};
    return names[mode];
}
int width(int kind) { return kind == 0 ? kY : kind == 1 ? kOutput : kFfn; }
int full_width(int kind) { return kind == 2 ? kFfn : 2 * width(kind); }
[[maybe_unused]] bool uses(int mode, int kind) { return mode == 3 || mode == kind; }

// Independent host oracle expresses canonical destination -> packed source.
std::pair<int, int> canonical_source(int kind, int column) {
    if (kind == 0) return {(column / kRun) % 2, (column / (2 * kRun)) * kRun + column % kRun};
    return {column / kOutput, column % kOutput};
}
uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
uint32_t pattern(int kind, int rank, int bank, uint32_t epoch, int index) {
    const uint32_t h = mix(uint32_t(index) * 0x85ebca6bu ^
                           uint32_t(rank + 1) * 0xc2b2ae35u ^ uint32_t(bank + 1) * 0x27d4eb2fu ^
                           uint32_t(kind + 1) * 0x165667b1u);
    // Odd epoch step changes EVERY source word between generations. The shortest
    // payload period is 2^21, longer than the maximum legal suite's epoch count.
    const uint32_t mantissa = h + epoch * 7919u;
    // Transport-only seams include subnormals and NaN payloads. RCCL send/recv
    // must copy these bit patterns rather than perform arithmetic. NaN bit 21
    // stays set, so the receive poison (bit 21 clear) can never be a valid source.
    if (kind != 2) {
        switch (index % 17) {
            case 0: return 0x80000000u | (mantissa & 0x007fffffu);
            case 1: return 0x7fe00000u | (mantissa & 0x001fffffu);
            default: return (h & 0x80000000u) | 0x3f000000u | (mantissa & 0x007fffffu);
        }
    }
    // FFN oracle deliberately avoids NaNs/denormals and uses explicit RNE FP32.
    return (h & 0x80000000u) | (uint32_t(124 + ((h >> 24) + epoch) % 7) << 23) |
           (mantissa & 0x007fffffu);
}
uint32_t sum_bits(uint32_t a, uint32_t b) {
    float af, bf; std::memcpy(&af, &a, 4); std::memcpy(&bf, &b, 4);
    volatile float av = af, bv = bf;
    volatile float result = av + bv;
    float r = result; uint32_t bits; std::memcpy(&bits, &r, 4); return bits;
}
uint32_t expected(int kind, int bank, uint32_t epoch, int index) {
    if (kind == 2) return sum_bits(pattern(kind, 0, bank, epoch, index),
                                  pattern(kind, 1, bank, epoch, index));
    const int token = index / full_width(kind), col = index % full_width(kind);
    const auto src = canonical_source(kind, col);
    return pattern(kind, src.first, bank, epoch, token * width(kind) + src.second);
}
uint64_t hash_words(const std::vector<uint32_t>& data) {
    uint64_t h = 14695981039346656037ull;
    for (uint32_t v : data) for (int b = 0; b < 4; ++b) { h ^= (v >> (8 * b)) & 255; h *= 1099511628211ull; }
    return h;
}
struct ExactError : std::runtime_error { using std::runtime_error::runtime_error; };
void require(bool ok, const std::string& message) { if (!ok) throw ExactError(message); }

struct Options {
    std::vector<int> tokens{1, 8}, firsts{0, 1};
    std::array<int, 2> devices{0, 1};
    int iterations = 20, warmup = 3, chain = 32, lifecycles = 2, timeout_ms = 10000, init_timeout_ms = 60000, delay_ms = 200;
    std::string scenario = "normal", library;
    bool selftest = false;
};
int integer(const std::string& value) {
    size_t used = 0; int n = std::stoi(value, &used);
    if (used != value.size()) throw std::invalid_argument("invalid integer: " + value);
    return n;
}
void usage() {
    std::puts("tp2_rccl_transport [--tokens 1|8|all] [--launch-order 01|10|both]"
              " [--devices 0,1] [--iterations 20] [--warmup 3] [--chain 32]"
              " [--lifecycles 2] [--timeout-ms 10000] [--init-timeout-ms 60000] [--rccl-library ABS_REAL_PATH]"
              " [--scenario normal|delayed-rank|missing-rank] [--delay-ms 200] [--selftest]");
}
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") { o.selftest = true; continue; }
        if (a == "--help") { usage(); std::exit(0); }
        if (++i == argc) throw std::invalid_argument("missing value for " + a);
        const std::string v = argv[i];
        if (a == "--tokens") {
            if (v == "all") o.tokens = {1, 8}; else if (v == "1" || v == "8") o.tokens = {integer(v)};
            else throw std::invalid_argument("tokens must be 1, 8 or all");
        } else if (a == "--launch-order") {
            if (v == "both") o.firsts = {0, 1}; else if (v == "01") o.firsts = {0};
            else if (v == "10") o.firsts = {1}; else throw std::invalid_argument("launch-order must be 01, 10 or both");
        } else if (a == "--devices") {
            const auto comma = v.find(',');
            if (comma == std::string::npos) throw std::invalid_argument("devices must be A,B");
            o.devices = {integer(v.substr(0, comma)), integer(v.substr(comma + 1))};
        } else if (a == "--iterations") o.iterations = integer(v);
        else if (a == "--warmup") o.warmup = integer(v);
        else if (a == "--chain") o.chain = integer(v);
        else if (a == "--lifecycles") o.lifecycles = integer(v);
        else if (a == "--timeout-ms") o.timeout_ms = integer(v);
        else if (a == "--init-timeout-ms") o.init_timeout_ms = integer(v);
        else if (a == "--delay-ms") o.delay_ms = integer(v);
        else if (a == "--scenario") o.scenario = v;
        else if (a == "--rccl-library") o.library = v;
        else throw std::invalid_argument("unknown option: " + a);
    }
    if (o.iterations < 2 || o.iterations > 1000 || o.warmup < 1 || o.warmup > 100 ||
        o.chain < 2 || o.chain > 1024 || o.lifecycles < 1 || o.lifecycles > 8 ||
        (o.scenario == "normal" && o.lifecycles < 2) ||
        o.timeout_ms < 100 || o.timeout_ms > 120000 || o.init_timeout_ms < 100 || o.init_timeout_ms > 120000 || o.delay_ms < 1 ||
        (o.scenario == "delayed-rank" && o.delay_ms >= o.timeout_ms) || o.devices[0] < 0 || o.devices[1] < 0 || o.devices[0] == o.devices[1])
        throw std::invalid_argument("option out of range (iterations >=2; normal lifecycles >=2; delay < timeout)");
    if (o.scenario != "normal" && o.scenario != "delayed-rank" && o.scenario != "missing-rank")
        throw std::invalid_argument("unknown scenario");
    return o;
}
int selftest() {
    require(std::fesetround(FE_TONEAREST) == 0, "cannot set host rounding");
    size_t tested = 0;
    for (int t : {1, 8}) for (int kind = 0; kind < kKinds; ++kind) for (int bank = 0; bank < kBanks; ++bank) {
        for (uint32_t epoch : {1u, 2u, 19u, 12345u}) {
            const int n = t * full_width(kind);
            std::vector<uint32_t> actual(n, kPoison), hits(n);
            if (kind < 2) {
                // Independent source -> destination scatter, unlike oracle gather.
                for (int rank = 0; rank < 2; ++rank) for (int i = 0; i < t * width(kind); ++i) {
                    const int col = i % width(kind);
                    const int dst = (i / width(kind)) * full_width(kind) + (kind == 0 ?
                        (col / kRun) * (2 * kRun) + rank * kRun + col % kRun : rank * kOutput + col);
                    actual[dst] = pattern(kind, rank, bank, epoch, i); ++hits[dst];
                }
            } else {
                for (int i = 0; i < n; ++i) { actual[i] = sum_bits(pattern(kind, 0, bank, epoch, i),
                    pattern(kind, 1, bank, epoch, i)); hits[i] = 1; }
            }
            for (int rank = 0; rank < 2; ++rank) for (int i = 0; i < t * width(kind); ++i) {
                const auto current = pattern(kind, rank, bank, epoch, i);
                require(current != kPoison, "source pattern collides with poison");
                require(current != pattern(kind, rank, bank, epoch + 1, i), "source word did not change generation");
            }
            bool stale_detected = false;
            for (int i = 0; i < n; ++i) {
                require(hits[i] == 1 && actual[i] == expected(kind, bank, epoch, i), "oracle layout mismatch");
                require(actual[i] != kPoison, "pattern collides with poison");
                stale_detected |= actual[i] != expected(kind, bank, epoch + 1, i); ++tested;
            }
            require(stale_detected, "freshness mutation not detected");
            const auto saved = actual[n / 2]; actual[n / 2] ^= 1;
            require(actual[n / 2] != expected(kind, bank, epoch, n / 2), "single-bit mutation not detected");
            actual[n / 2] = saved;
            require(hash_words(actual) != 0, "unexpected zero fingerprint");
        }
    }
    require(sum_bits(0x3f800000u, 0x33800000u) == 0x3f800000u, "FP32 tie-even oracle mismatch");
    require(sum_bits(0x3f800001u, 0x33800000u) == 0x3f800002u, "FP32 odd tie oracle mismatch");
    std::printf("CPU_SELFTEST_PASS checked_words=%zu NOT_GPU_TEST=1 capture_tested=0 rccl_tested=0\n", tested);
    return 0;
}

#ifndef STRATA_RCCL_CPU_SELFTEST
void hip_check(hipError_t result, const char* what) {
    if (result != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(result));
}
void rccl_check(ncclResult_t result, const char* what) {
    if (result != ncclSuccess) throw std::runtime_error(std::string(what) + ": " + ncclGetErrorString(result));
}

// Initialization diagnostics stay outside captured/timed work. A BEGIN without
// its END identifies the blocked API; errors remain visible if its peer hangs.
template <typename Action>
void init_call(int life, int rank, const std::string& call, const Action& action) {
    const auto begin = Clock::now();
    std::printf("INIT_CALL_BEGIN life=%d rank=%d call=%s\n", life, rank, call.c_str());
    try { action(); }
    catch (...) {
        const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        std::fprintf(stderr, "INIT_CALL_ERROR life=%d rank=%d call=%s elapsed_ms=%.3f\n", life, rank, call.c_str(), elapsed);
        throw;
    }
    const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    std::printf("INIT_CALL_END life=%d rank=%d call=%s elapsed_ms=%.3f\n", life, rank, call.c_str(), elapsed);
}

// Each rank owns its context, communicator, stream and API calls for the entire
// process. A command timeout exits without freeing possibly live GPU buffers.
// An OUTER subprocess timeout is still mandatory (e.g. runtime/library startup).
class Workers {
public:
    explicit Workers(const Options& o) : options_(o) {
        for (int rank = 0; rank < 2; ++rank) threads_[rank] = std::thread([this, rank] { loop(rank); });
    }
    Workers(const Workers&) = delete;
    ~Workers() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        changed_.notify_all(); for (auto& t : threads_) t.join();
    }
    void run(const std::string& stage, const std::function<void(int)>& action, int timeout_ms = 0) {
        std::unique_lock<std::mutex> lock(mutex_);
        failed_.store(false); error_ = nullptr; completed_ = 0; action_ = action; stage_ = stage; ++generation_;
        const int budget_ms = timeout_ms ? timeout_ms : options_.timeout_ms;
        const auto deadline = Clock::now() + std::chrono::milliseconds(budget_ms);
        changed_.notify_all();
        if (!done_.wait_until(lock, deadline, [&] { return completed_ == 2; })) {
            std::fprintf(stderr, "WATCHDOG_TIMEOUT scenario=%s stage=%s exit=70 "
                         "teardown=process_exit_without_buffer_release budget_ms=%d completed_ranks=%d\n",
                         options_.scenario.c_str(), stage.c_str(), budget_ms, completed_);
            std::fflush(nullptr); std::_Exit(70);
        }
        if (error_) std::rethrow_exception(error_);
    }
    bool failed() const { return failed_.load(std::memory_order_acquire); }
    void check_peer() const { if (failed()) throw std::runtime_error("peer worker failed"); }
private:
    void loop(int rank) {
        unsigned seen = 0;
        for (;;) {
            std::function<void(int)> action;
            std::string stage;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                changed_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_; action = action_; stage = stage_;
            }
            try { action(rank); }
            catch (...) {
                failed_.store(true, std::memory_order_release);
                // Do not wait for a possibly blocked peer before exposing the
                // original failure. Cleanup still waits for both rank owners.
                try { throw; }
                catch (const std::exception& e) {
                    std::fprintf(stderr, "WORKER_ERROR scenario=%s stage=%s rank=%d reason=%s\n",
                                 options_.scenario.c_str(), stage.c_str(), rank, e.what());
                }
                catch (...) {
                    std::fprintf(stderr, "WORKER_ERROR scenario=%s stage=%s rank=%d reason=unknown_exception\n",
                                 options_.scenario.c_str(), stage.c_str(), rank);
                }
                std::lock_guard<std::mutex> lock(mutex_); if (!error_) error_ = std::current_exception();
            }
            { std::lock_guard<std::mutex> lock(mutex_); ++completed_; }
            done_.notify_one();
        }
    }
    const Options& options_;
    std::array<std::thread, 2> threads_;
    mutable std::mutex mutex_;
    std::condition_variable changed_, done_;
    std::function<void(int)> action_;
    std::string stage_;
    std::exception_ptr error_;
    std::atomic<bool> failed_{false};
    unsigned generation_ = 0;
    int completed_ = 0;
    bool stop_ = false;
};

struct Buffer {
    uint32_t* base = nullptr;
    int capacity = 0;
    uint32_t* data() const { return base + kGuard; }
    void allocate(int n) {
        capacity = n; hip_check(hipMalloc(reinterpret_cast<void**>(&base), (n + 2 * kGuard) * sizeof(uint32_t)), "hipMalloc");
    }
    void release() { if (base) hip_check(hipFree(base), "hipFree"); base = nullptr; }
    void fill(int active, const std::function<uint32_t(int)>& value, hipStream_t stream) const {
        std::vector<uint32_t> h(capacity + 2 * kGuard, kCanary);
        std::fill(h.begin() + kGuard, h.end() - kGuard, kPoison);
        for (int i = 0; i < active; ++i) h[kGuard + i] = value(i);
        // Host source must stay alive until upload completes; synchronous copy is
        // deliberately outside captured/timed work and the stream is idle here.
        (void)stream;
        hip_check(hipMemcpy(base, h.data(), h.size() * 4, hipMemcpyHostToDevice), "prepare upload");
    }
    std::vector<uint32_t> read() const {
        std::vector<uint32_t> h(capacity + 2 * kGuard);
        hip_check(hipMemcpy(h.data(), base, h.size() * 4, hipMemcpyDeviceToHost), "check download");
        return h;
    }
    uint64_t verify(int active, const std::function<uint32_t(int)>& value, const std::string& label) const {
        auto h = read();
        for (int i = 0; i < int(h.size()); ++i) {
            const int index = i - kGuard;
            const uint32_t want = index < 0 || index >= capacity ? kCanary : index < active ? value(index) : kPoison;
            if (h[i] != want) {
                char details[160]; std::snprintf(details, sizeof(details), " index=%d expected=%08x actual=%08x", index, want, h[i]);
                throw ExactError(label + details);
            }
        }
        return hash_words(h);
    }
};
struct Bank {
    std::array<Buffer, 3> source, result;
    Buffer attention_receive, ffn_receive;
    void allocate(int life, int rank, int bank) {
        const std::string prefix = "hipMalloc.bank" + std::to_string(bank) + ".";
        for (int k = 0; k < 3; ++k) {
            init_call(life, rank, prefix + "source." + name(k), [&] { source[k].allocate(8 * width(k)); });
            init_call(life, rank, prefix + "result." + name(k), [&] { result[k].allocate(8 * full_width(k)); });
        }
        init_call(life, rank, prefix + "attention_receive", [&] { attention_receive.allocate(8 * kY); });
        init_call(life, rank, prefix + "ffn_receive", [&] { ffn_receive.allocate(8 * kFfn); });
    }
    void release() {
        for (int k = 0; k < 3; ++k) { source[k].release(); result[k].release(); }
        attention_receive.release(); ffn_receive.release();
    }
    Buffer& receive(int kind) { return kind == 2 ? ffn_receive : attention_receive; }
};
struct Graph {
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    void instantiate() { hip_check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "hipGraphInstantiate"); }
    void release() {
        if (exec) hip_check(hipGraphExecDestroy(exec), "hipGraphExecDestroy");
        exec = nullptr;
        if (graph) hip_check(hipGraphDestroy(graph), "hipGraphDestroy");
        graph = nullptr;
    }
};
struct Rank {
    ncclComm_t comm = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, end = nullptr;
    std::array<Bank, 2> banks;
    // Stable addresses in captures: [T=1/T=8][physical bank][seam/sequence].
    Graph graphs[2][2][4], empty;
};

__global__ void unpack(const uint32_t* local, const uint32_t* peer, uint32_t* full, int tokens, int rank, int kind) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    const int w = kind == 0 ? kY : kOutput;
    if (i >= tokens * w) return;
    const int col = i % w;
    const int pair = kind == 0 ? (col / kRun) * (2 * kRun) + col % kRun : col;
    const int stride = kind == 0 ? kRun : kOutput;
    const int row = (i / w) * (2 * w);
    full[row + pair + rank * stride] = local[i];
    full[row + pair + (1 - rank) * stride] = peer[i];
}
__global__ void reduce_ordered(const float* local, const float* peer, float* result, int values, int rank) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < values) result[i] = rank == 0 ? __fadd_rn(local[i], peer[i]) : __fadd_rn(peer[i], local[i]);
}
void enqueue(Rank& r, int rank, int tokens, int bank, int mode) {
    for (int k = 0; k < 3; ++k) if (uses(mode, k)) {
        Bank& b = r.banks[bank];
        rccl_check(ncclGroupStart(), "ncclGroupStart");
        // Always close the group before reporting an enqueue error. No consumer
        // is submitted until GroupEnd succeeds and orders all grouped operations.
        const auto send = ncclSend(b.source[k].data(), size_t(tokens * width(k)), ncclFloat32, 1 - rank, r.comm, r.stream);
        const auto recv = ncclRecv(b.receive(k).data(), size_t(tokens * width(k)), ncclFloat32, 1 - rank, r.comm, r.stream);
        const auto end = ncclGroupEnd();
        rccl_check(send, "ncclSend"); rccl_check(recv, "ncclRecv"); rccl_check(end, "ncclGroupEnd");
        if (k == 2) {
            hipLaunchKernelGGL(reduce_ordered, dim3((tokens * kFfn + kThreads - 1) / kThreads), dim3(kThreads), 0, r.stream,
                reinterpret_cast<const float*>(b.source[k].data()), reinterpret_cast<const float*>(b.receive(k).data()),
                reinterpret_cast<float*>(b.result[k].data()), tokens * kFfn, rank);
        } else {
            hipLaunchKernelGGL(unpack, dim3((tokens * width(k) + kThreads - 1) / kThreads), dim3(kThreads), 0, r.stream,
                              b.source[k].data(), b.receive(k).data(), b.result[k].data(), tokens, rank, k);
        }
        hip_check(hipGetLastError(), "consumer kernel launch");
    }
}
void asynchronous_error(Rank& r) {
    ncclResult_t state = ncclSuccess;
    rccl_check(ncclCommGetAsyncError(r.comm, &state), "ncclCommGetAsyncError");
    rccl_check(state, "RCCL_ASYNC_ERROR");
}
void complete(Rank& r, Workers& workers) {
    for (;;) {
        workers.check_peer(); asynchronous_error(r);
        const auto status = hipEventQuery(r.end);
        if (status == hipSuccess) { asynchronous_error(r); return; }
        if (status != hipErrorNotReady) hip_check(status, "hipEventQuery");
        // Busy query initially; yielding keeps the two persistent workers from
        // starving one another without adding a fixed sleep to every measurement.
        std::this_thread::yield();
    }
}
void prepare(Rank& r, int rank, int tokens, int active_bank, int mode, uint32_t epoch) {
    for (int bank = 0; bank < 2; ++bank) {
        Bank& b = r.banks[bank];
        for (int k = 0; k < 3; ++k) {
            b.source[k].fill(tokens * width(k), [&](int i) { return pattern(k, rank, bank, epoch, i); }, r.stream);
            b.result[k].fill(0, [](int) { return 0u; }, r.stream);
        }
        b.attention_receive.fill(0, [](int) { return 0u; }, r.stream);
        b.ffn_receive.fill(0, [](int) { return 0u; }, r.stream);
    }
    (void)active_bank; (void)mode;
}
uint64_t verify(Rank& r, int rank, int tokens, int active_bank, int mode, uint32_t epoch, bool empty) {
    uint64_t fingerprint = 0;
    for (int bank = 0; bank < 2; ++bank) {
        Bank& b = r.banks[bank];
        const std::string label = "rank=" + std::to_string(rank) + " bank=" + std::to_string(bank) + " ";
        for (int k = 0; k < 3; ++k) {
            fingerprint ^= b.source[k].verify(tokens * width(k), [&](int i) { return pattern(k, rank, bank, epoch, i); }, label + name(k) + " source");
            const bool active = !empty && bank == active_bank && uses(mode, k);
            fingerprint ^= b.result[k].verify(active ? tokens * full_width(k) : 0,
                [&](int i) { return expected(k, bank, epoch, i); }, label + name(k) + " consumer");
        }
        const bool active = !empty && bank == active_bank;
        // Sequence intentionally reuses the same attention inbox. Its final
        // content is output in the prefix and previous Y in the remaining tail.
        const int attn_kind = mode == 0 ? 0 : 1;
        const int attn_count = active && mode != 2 ? tokens * (mode == 3 ? kY : width(attn_kind)) : 0;
        fingerprint ^= b.attention_receive.verify(attn_count, [&](int i) {
            const int kind = mode == 3 && i >= tokens * kOutput ? 0 : attn_kind;
            return pattern(kind, 1 - rank, bank, epoch, i);
        }, label + "attention receive");
        fingerprint ^= b.ffn_receive.verify(active && uses(mode, 2) ? tokens * kFfn : 0,
            [&](int i) { return pattern(2, 1 - rank, bank, epoch, i); }, label + "ffn receive");
    }
    return fingerprint;
}
struct Sample { double wall = 0; std::array<double, 2> gpu{}; };
Sample replay(Workers& workers, std::array<Rank, 2>& ranks, const Options& o,
              int tokens, int bank, int mode, int first, int count, bool empty, bool fault = false) {
    std::atomic<int> ready{0}, admission{0};
    Sample sample;
    const auto begin = Clock::now();
    workers.run(fault ? "fault-launch" : "paired-replay", [&](int rank) {
        Rank& r = ranks[rank];
        ready.fetch_add(1, std::memory_order_release);
        while (ready.load(std::memory_order_acquire) != 2) { workers.check_peer(); std::this_thread::yield(); }
        if (fault && rank == 1 - first) {
            if (o.scenario == "missing-rank") {
                std::printf("FAULT_PEER_OMITTED scenario=missing-rank rank=%d\n", rank);
                return; // Deliberate absent launch in its own subprocess.
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(o.delay_ms));
        }
        hip_check(hipEventRecord(r.start, r.stream), "start event");
        auto exec = empty ? r.empty.exec : r.graphs[tokens == 1 ? 0 : 1][bank][mode].exec;
        for (int n = 0; n < count; ++n) {
            const int turn = 2 * n + (rank == first ? 0 : 1);
            while (admission.load(std::memory_order_acquire) != turn) { workers.check_peer(); std::this_thread::yield(); }
            // Release before calling HIP: waiting for a first launch to RETURN
            // can deadlock a captured multi-GPU operation. This measures ordered
            // host admission, not a promise about driver-internal execution order.
            admission.fetch_add(1, std::memory_order_release);
            if (fault) std::printf("FAULT_GRAPH_LAUNCH_ENTER scenario=%s rank=%d\n", o.scenario.c_str(), rank);
            hip_check(hipGraphLaunch(exec, r.stream), "hipGraphLaunch");
        }
        hip_check(hipEventRecord(r.end, r.stream), "end event");
        complete(r, workers);
    });
    sample.wall = std::chrono::duration<double, std::micro>(Clock::now() - begin).count() / count;
    workers.run("collect-event-timing", [&](int rank) {
        const Rank& r = ranks[rank];
        float ms = 0; hip_check(hipEventElapsedTime(&ms, r.start, r.end), "hipEventElapsedTime");
        if (!(ms >= 0) || ms > float(o.timeout_ms)) throw std::runtime_error("invalid event timing");
        sample.gpu[rank] = double(ms) * 1000 / count;
    });
    return sample;
}
void print_stats(const std::vector<Sample>& samples, const char* control, int life, int tokens, int mode, int first, int count) {
    for (int metric = 0; metric < 3; ++metric) {
        std::vector<double> v;
        double mean = 0;
        for (const auto& s : samples) { const double x = metric == 0 ? s.wall : s.gpu[metric - 1]; v.push_back(x); mean += x; }
        mean /= v.size(); std::sort(v.begin(), v.end());
        const double median = (v[(v.size() - 1) / 2] + v[v.size() / 2]) * 0.5;
        const size_t p95 = (95 * v.size() + 99) / 100 - 1;
        std::printf("TIMING life=%d tokens=%d mode=%s order=%d%d control=%s scope=%s chain=%d metric=%s "
                    "samples=%zu mean_us=%.3f median_us=%.3f p95_us=%.3f min_us=%.3f max_us=%.3f\n",
                    life, tokens, name(mode), first, 1 - first, control, count == 1 ? "single_call" : "sustained_chain",
                    count, metric == 0 ? "host_wall" : metric == 1 ? "rank0_event" : "rank1_event",
                    v.size(), mean, median, v[p95], v.front(), v.back());
    }
}
void metadata(const Options& o, int argc, char** argv) {
    std::printf("RCCL_PREFLIGHT_BEGIN scenario=%s\nCOMMAND", o.scenario.c_str());
    for (int i = 0; i < argc; ++i) std::printf(" [%s]", argv[i]);
    std::puts("");
    int version = 0, runtime = 0, driver = 0, count = 0;
    rccl_check(ncclGetVersion(&version), "ncclGetVersion");
    if (version < 20700 || version / 10000 != NCCL_VERSION_CODE / 10000)
        throw std::runtime_error("RCCL runtime/header major ABI mismatch or runtime predates send/recv API");
    hip_check(hipRuntimeGetVersion(&runtime), "hipRuntimeGetVersion");
    hip_check(hipDriverGetVersion(&driver), "hipDriverGetVersion");
    hip_check(hipGetDeviceCount(&count), "hipGetDeviceCount");
    Dl_info info{};
    // Resolve the DSO symbol explicitly: taking the linked function's address
    // can identify a main-executable PLT trampoline on a non-PIE build.
    void* version_symbol = dlsym(RTLD_DEFAULT, "ncclGetVersion");
    if (!version_symbol || !dladdr(version_symbol, &info) || !info.dli_fname)
        throw std::runtime_error("cannot resolve loaded RCCL path");
    char* resolved = realpath(info.dli_fname, nullptr);
    if (!resolved) throw std::runtime_error("cannot canonicalize loaded RCCL path");
    const std::string path = resolved; std::free(resolved);
    if (!o.library.empty()) {
        char* requested = realpath(o.library.c_str(), nullptr);
        if (!requested) throw std::runtime_error("cannot canonicalize requested RCCL path");
        const std::string want = requested; std::free(requested);
        if (want != path) throw std::runtime_error("loaded RCCL does not match --rccl-library: " + path);
    }
    std::printf("DEPENDENCY rccl_version=%d header_version=%d hip_runtime=%d hip_driver=%d loaded_rccl=%s "
                "sha256=runner_recorded device_count=%d\n", version, NCCL_VERSION_CODE, runtime, driver, path.c_str(), count);
    for (int rank = 0; rank < 2; ++rank) {
        if (o.devices[rank] >= count) throw std::runtime_error("requested GPU unavailable");
        hipDeviceProp_t p{}; hip_check(hipGetDeviceProperties(&p, o.devices[rank]), "hipGetDeviceProperties");
        char pci[64]{}; hip_check(hipDeviceGetPCIBusId(pci, sizeof(pci), o.devices[rank]), "hipDeviceGetPCIBusId");
        int peer = 0; hip_check(hipDeviceCanAccessPeer(&peer, o.devices[rank], o.devices[1 - rank]), "hipDeviceCanAccessPeer");
        std::printf("DEVICE rank=%d logical_id=%d pci=%s name=%s arch=%s vram_bytes=%zu peer_access=%d\n",
                    rank, o.devices[rank], pci, p.name, p.gcnArchName, p.totalGlobalMem, peer);
    }
    for (const char* key : {"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE",
                          "NCCL_SOCKET_IFNAME", "NCCL_SOCKET_FAMILY", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE",
                          "NCCL_ALGO", "NCCL_PROTO", "NCCL_GRAPH_REGISTER", "NCCL_GRAPH_MIXING_SUPPORT"})
        std::printf("ENV %s=%s\n", key, std::getenv(key) ? std::getenv(key) : "<unset>");
    std::puts("CONTRACT payload_per_rank_T1_y=12288 output=5120 ffn=10240 T8_multiplier=8 bytes_type=FP32_raw "
              "consumer=canonical_unpack_or_rank0_then_rank1_fadd_rn physical_banks=2 max_inflight=1 "
              "chain_payload=constant_within_chain initialization_restore_poison_checks=outside_timing "
              "wall_scope=persistent_worker_dispatch_to_completion_ack_after_both_events event_scope=rank_local "
              "launch_order=host_admission_before_HIP_call empty_control=zero_node_graph paired_order=alternating_AB_BA subtraction=forbidden "
              "layer_speedup_claim=0 prefix_continuation_tested=0 full_layer_gate_required=1");
}
void init_rank(Rank& r, int rank, int life, const Options& o, const ncclUniqueId& id) {
    init_call(life, rank, "hipSetDevice", [&] { hip_check(hipSetDevice(o.devices[rank]), "hipSetDevice"); });
    require(std::fesetround(FE_TONEAREST) == 0, "cannot set worker rounding");
    init_call(life, rank, "hipStreamCreateWithFlags", [&] {
        hip_check(hipStreamCreateWithFlags(&r.stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
    });
    init_call(life, rank, "hipEventCreate.start", [&] { hip_check(hipEventCreate(&r.start), "hipEventCreate start"); });
    init_call(life, rank, "hipEventCreate.end", [&] { hip_check(hipEventCreate(&r.end), "hipEventCreate end"); });
    for (int bank = 0; bank < kBanks; ++bank) r.banks[bank].allocate(life, rank, bank);
    init_call(life, rank, "ncclCommInitRank", [&] {
        rccl_check(ncclCommInitRank(&r.comm, 2, id, rank), "ncclCommInitRank");
    });
}
void capture(Rank& r, int rank, const Options& o) {
    hip_check(hipGraphCreate(&r.empty.graph, 0), "hipGraphCreate empty"); r.empty.instantiate();
    for (int tokens : o.tokens) for (int bank = 0; bank < 2; ++bank) for (int mode = 0; mode < 4; ++mode) {
        Graph& g = r.graphs[tokens == 1 ? 0 : 1][bank][mode];
        hip_check(hipStreamBeginCapture(r.stream, hipStreamCaptureModeThreadLocal), "hipStreamBeginCapture");
        enqueue(r, rank, tokens, bank, mode);
        hip_check(hipStreamEndCapture(r.stream, &g.graph), "hipStreamEndCapture"); g.instantiate();
    }
}
void release_rank(Rank& r) {
    // All events were observed complete. Captured plans outlive ALL graph handles;
    // communicator and allocations are released only after exec+graph destruction.
    for (auto& by_t : r.graphs) for (auto& by_bank : by_t) for (auto& g : by_bank) g.release();
    r.empty.release();
    rccl_check(ncclCommDestroy(r.comm), "ncclCommDestroy"); r.comm = nullptr;
    for (auto& b : r.banks) b.release();
    hip_check(hipEventDestroy(r.start), "hipEventDestroy start"); r.start = nullptr;
    hip_check(hipEventDestroy(r.end), "hipEventDestroy end"); r.end = nullptr;
    hip_check(hipStreamDestroy(r.stream), "hipStreamDestroy"); r.stream = nullptr;
}
int gpu_main(const Options& o, int argc, char** argv) {
    metadata(o, argc, argv);
    Workers workers(o);
    std::array<Rank, 2> ranks;
    uint32_t epoch = 0;
    uint64_t fingerprint = 0;
    size_t checked_replays = 0;
    try {
        for (int life = 0; life < o.lifecycles; ++life) {
            std::printf("INITIALIZE_BEGIN life=%d init_budget_ms=%d command_budget_ms=%d\n", life, o.init_timeout_ms, o.timeout_ms);
            ncclUniqueId id{};
            init_call(life, -1, "ncclGetUniqueId", [&] { rccl_check(ncclGetUniqueId(&id), "ncclGetUniqueId"); });
            workers.run("initialize", [&](int rank) { init_rank(ranks[rank], rank, life, o, id); }, o.init_timeout_ms);
            std::printf("INITIALIZE_END life=%d\n", life);
            // Establish both directions and all consumers outside capture/timing.
            for (int w = 0; w < o.warmup; ++w) {
                ++epoch;
                workers.run("warmup-prepare", [&](int rank) { prepare(ranks[rank], rank, 8, 0, 3, epoch); });
                workers.run("warmup-exchange", [&](int rank) {
                    auto& r = ranks[rank]; enqueue(r, rank, 8, 0, 3);
                    hip_check(hipEventRecord(r.end, r.stream), "warmup end"); complete(r, workers);
                });
                workers.run("warmup-verify", [&](int rank) { verify(ranks[rank], rank, 8, 0, 3, epoch, false); });
            }
#ifdef STRATA_RCCL_DEBUG_WARMUP_ONLY
            // Diagnostic build only: preserve the original T8/bank-0 sequence
            // warmup, then stop before any capture, timing sweep, or lifecycle 2.
            // Process exit releases resources; do not introduce a teardown case.
            std::puts("RCCL_DEBUG_WARMUP_ONLY_COMPLETE capture_tested=0 full_suite=0");
            std::fflush(nullptr); std::_Exit(0);
#endif
            workers.run("capture-and-instantiate", [&](int rank) { capture(ranks[rank], rank, o); });
            std::printf("CAPTURE_PASS life=%d graphs_per_rank=%zu empty_graphs_per_rank=1\n", life, o.tokens.size() * 8);
            ++epoch;
            workers.run("empty-warmup-prepare", [&](int rank) { prepare(ranks[rank], rank, o.tokens.front(), 0, 3, epoch); });
            (void)replay(workers, ranks, o, o.tokens.front(), 0, 3, o.firsts.front(), 1, true);
            workers.run("empty-warmup-verify", [&](int rank) { verify(ranks[rank], rank, o.tokens.front(), 0, 3, epoch, true); });
            if (o.scenario != "normal") {
                const int t = o.tokens.front(), first = o.firsts.front(); ++epoch;
                workers.run("fault-prepare", [&](int rank) { prepare(ranks[rank], rank, t, 0, 3, epoch); });
                std::printf("FAULT_INJECTION_ARMED scenario=%s launch_rank=%d omitted_or_delayed_rank=%d\n",
                            o.scenario.c_str(), first, 1 - first);
                (void)replay(workers, ranks, o, t, 0, 3, first, 1, false, true);
                if (o.scenario == "missing-rank") throw std::runtime_error("missing peer unexpectedly completed");
                workers.run("delayed-verify", [&](int rank) { verify(ranks[rank], rank, t, 0, 3, epoch, false); });
                ++checked_replays;
                std::printf("DELAYED_RANK_PASS life=%d delay_ms=%d order=%d%d timing_admitted=0\n", life, o.delay_ms, first, 1 - first);
            } else {
                for (int t : o.tokens) for (int mode = 0; mode < 4; ++mode) for (int first : o.firsts) {
                    // First graph replay is untimed and checked (runtime may do lazy setup).
                    for (int bank = 0; bank < 2; ++bank) {
                        ++epoch; workers.run("graph-warmup-prepare", [&](int rank) { prepare(ranks[rank], rank, t, bank, mode, epoch); });
                        (void)replay(workers, ranks, o, t, bank, mode, first, 1, false);
                        workers.run("graph-warmup-verify", [&](int rank) { verify(ranks[rank], rank, t, bank, mode, epoch, false); });
                        ++checked_replays;
                    }
                    for (int count : {1, o.chain}) {
                        std::array<std::vector<Sample>, 2> samples;
                        for (int i = 0; i < o.iterations; ++i) {
                            const int bank = i % 2;
                            for (int pair = 0; pair < 2; ++pair) {
                                const bool empty = ((pair + i) % 2) == 0; ++epoch;
                                workers.run("sample-prepare", [&](int rank) { prepare(ranks[rank], rank, t, bank, mode, epoch); });
                                const auto sample = replay(workers, ranks, o, t, bank, mode, first, count, empty);
                                std::array<uint64_t, 2> hashes{};
                                workers.run("sample-exact-check", [&](int rank) { hashes[rank] = verify(ranks[rank], rank, t, bank, mode, epoch, empty); });
                                fingerprint = (fingerprint << 1) ^ hashes[0] ^ (hashes[1] + epoch);
                                samples[empty ? 1 : 0].push_back(sample);
                                if (!empty) ++checked_replays;
                            }
                        }
                        print_stats(samples[0], "transport", life, t, mode, first, count);
                        print_stats(samples[1], "empty_graph", life, t, mode, first, count);
                    }
                }
            }
            workers.run("normal-teardown", [&](int rank) { release_rank(ranks[rank]); });
            std::printf("LIFECYCLE_PASS life=%d graph_destroy_before_comm=1 comm_destroy_before_buffer_free=1\n", life);
        }
    } catch (const std::exception& e) {
        const int code = dynamic_cast<const ExactError*>(&e) ? 3 : 4;
        std::fprintf(stderr, "RCCL_PREFLIGHT_FAIL scenario=%s reason=%s exit=%d timing_admitted=0\n", o.scenario.c_str(), e.what(), code);
        // Both commands have returned before here. Each owner aborts its own
        // communicator. If abort blocks, run's watchdog terminates the subprocess.
        // Never destroy/free a possibly in-flight graph/buffer on a failed run.
        try {
            workers.run("failure-abort", [&](int rank) {
                if (ranks[rank].comm) rccl_check(ncclCommAbort(ranks[rank].comm), "ncclCommAbort");
                ranks[rank].comm = nullptr;
            });
        } catch (const std::exception& abort_error) { std::fprintf(stderr, "ABORT_FAIL reason=%s\n", abort_error.what()); }
        std::fflush(nullptr); std::_Exit(code);
    }
    std::printf("EXACT_GATE_PASS checked_replays=%zu changing_payloads=1 poison_before_replay=1 "
                "bank_isolation=1 receive_and_source_guards=1 mismatch_tolerance_bits=0 fingerprint=%016llx\n",
                checked_replays, static_cast<unsigned long long>(fingerprint));
    std::printf("RCCL_PREFLIGHT_PASS scenario=%s tokens=%zu orders=%zu lifecycles=%d "
                "full_suite=%d layer_integration_admitted=0 timing_scope=transport_only\n", o.scenario.c_str(),
                o.tokens.size(), o.firsts.size(), o.lifecycles, o.tokens.size() == 2 && o.firsts.size() == 2 && o.scenario == "normal");
    return 0;
}
#endif
} // namespace
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    Options o;
    try { o = parse(argc, argv); }
    catch (const std::exception& e) { std::fprintf(stderr, "USAGE_ERROR %s\n", e.what()); usage(); return 2; }
    try {
        if (o.selftest) return selftest();
#ifdef STRATA_RCCL_CPU_SELFTEST
        std::fprintf(stderr, "CPU_ONLY_BUILD requires --selftest; NOT_GPU_TEST=1\n"); return 2;
#else
        return gpu_main(o, argc, argv);
#endif
    } catch (const std::exception& e) {
        std::fprintf(stderr, "RCCL_PREFLIGHT_FAIL scenario=%s reason=%s exit=4 timing_admitted=0\n", o.scenario.c_str(), e.what()); return 4;
    }
}
