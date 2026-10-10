#include "tp_gdn_rccl.hpp"
#include "tp_gdn_rccl_workers.hpp"

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

#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>

namespace strata::core::tp2::detail {
namespace {

void hip_check(hipError_t result, const char* what) {
    if (result != hipSuccess)
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(result));
}
void rccl_check(ncclResult_t result, const char* what) {
    if (result != ncclSuccess)
        throw std::runtime_error(std::string(what) + ": " + ncclGetErrorString(result));
}
std::string canonical_path(const char* path, const char* what) {
    char* resolved = realpath(path, nullptr);
    if (!resolved) throw std::runtime_error(std::string("cannot canonicalize ") + what);
    const std::unique_ptr<char, decltype(&std::free)> owner(resolved, &std::free);
    return resolved;
}
void validate(const std::array<int, 2>& devices, const std::array<void*, 2>& streams,
              const std::string& library, int timeout_ms, int init_timeout_ms) {
    if (devices[0] < 0 || devices[1] < 0 || devices[0] == devices[1])
        throw std::invalid_argument("RCCL session requires two distinct device IDs");
    if (!streams[0] || !streams[1] || streams[0] == streams[1])
        throw std::invalid_argument("RCCL session requires two distinct non-default streams");
    if (library.empty() || library.front() != '/')
        throw std::invalid_argument("RCCL session requires an absolute expected library path");
    if (timeout_ms < 100 || timeout_ms > 120000 || init_timeout_ms < 100 || init_timeout_ms > 120000)
        throw std::invalid_argument("RCCL session timeouts must be between 100 and 120000 ms");
}

} // namespace

struct RcclSession::Impl {
    struct Rank {
        ncclComm_t comm = nullptr;
        hipStream_t stream = nullptr;
        hipEvent_t complete_event = nullptr;
    };
    std::array<int, 2> devices;
    std::array<Rank, 2> ranks;
    int timeout_ms, init_timeout_ms;
    bool closed = false;
    RcclWorkers workers;

    Impl(const std::array<int, 2>& device_ids, const std::array<void*, 2>& streams,
         const std::string& expected_library, int timeout, int init_timeout)
        : devices(device_ids), timeout_ms(timeout), init_timeout_ms(init_timeout) {
        for (int rank = 0; rank < 2; ++rank)
            ranks[rank].stream = reinterpret_cast<hipStream_t>(streams[rank]);
        // Even runtime identification and unique-ID creation are bounded. An
        // outer subprocess timeout is still required for loader/runtime startup.
        ncclUniqueId id{};
        run("initialize-identity", [&](int rank) {
            if (rank == 0) {
                identity(expected_library);
                rccl_check(ncclGetUniqueId(&id), "ncclGetUniqueId");
            }
        }, true);
        run("initialize-ranks", [&](int rank) {
            hip_check(hipSetDevice(devices[rank]), "hipSetDevice");
            if (std::fesetround(FE_TONEAREST) != 0)
                throw std::runtime_error("cannot set RCCL worker rounding");
            auto& r = ranks[rank];
            hip_check(hipEventCreateWithFlags(&r.complete_event, hipEventDisableTiming), "hipEventCreate completion");
            // Both permanent owners must enter communicator initialization at
            // once. The supplied compute stream remains owned by the layer.
            rccl_check(ncclCommInitRank(&r.comm, 2, id, rank), "ncclCommInitRank");
        }, true);
    }

    void identity(const std::string& expected_library) const {
        Dl_info info{};
        // Taking the linked function address can return an executable PLT stub.
        void* symbol = dlsym(RTLD_DEFAULT, "ncclGetVersion");
        if (!symbol || !dladdr(symbol, &info) || !info.dli_fname)
            throw std::runtime_error("cannot resolve loaded RCCL path");
        const auto path = canonical_path(info.dli_fname, "loaded RCCL path");
        const auto expected = canonical_path(expected_library.c_str(), "expected RCCL path");
        if (path != expected)
            throw std::runtime_error("loaded RCCL does not match requested library: " + path);
        int version = 0, runtime = 0, driver = 0, count = 0;
        rccl_check(ncclGetVersion(&version), "ncclGetVersion");
        if (version < 20700 || version / 10000 != NCCL_VERSION_CODE / 10000)
            throw std::runtime_error("RCCL runtime/header major ABI mismatch or missing send/recv API");
        hip_check(hipRuntimeGetVersion(&runtime), "hipRuntimeGetVersion");
        hip_check(hipDriverGetVersion(&driver), "hipDriverGetVersion");
        hip_check(hipGetDeviceCount(&count), "hipGetDeviceCount");
        if (devices[0] >= count || devices[1] >= count)
            throw std::runtime_error("requested RCCL GPU unavailable");
        std::printf("RCCL_LAYER_DEPENDENCY rccl_version=%d header_version=%d hip_runtime=%d "
                    "hip_driver=%d loaded_rccl=%s device_count=%d devices=%d,%d "
                    "command_budget_ms=%d init_budget_ms=%d\n",
                    version, NCCL_VERSION_CODE, runtime, driver, path.c_str(), count,
                    devices[0], devices[1], timeout_ms, init_timeout_ms);
        std::fflush(stdout);
    }

    void run(const char* stage, const std::function<void(int)>& action, bool initialization = false) {
        if (closed) throw std::logic_error("RCCL session already shut down");
        if (!stage || !*stage || !action)
            throw std::invalid_argument("invalid RCCL paired command");
        if (workers.active()) throw std::logic_error("overlapping RCCL session command");
        try { workers.run(stage, action, initialization ? init_timeout_ms : timeout_ms); }
        catch (const std::exception& e) { fail(e.what()); }
        catch (...) { fail("unknown worker exception"); }
    }

    Rank& owned_rank(int rank) {
        if (rank < 0 || rank >= 2) throw std::out_of_range("RCCL rank");
        if (!workers.owns_rank(rank))
            throw std::logic_error("RCCL rank API must run on its permanent owner");
        if (closed || !ranks[rank].comm)
            throw std::logic_error("RCCL rank communicator is not active");
        return ranks[rank];
    }

    void async_error(const Rank& rank) {
        ncclResult_t state = ncclSuccess;
        rccl_check(ncclCommGetAsyncError(rank.comm, &state), "ncclCommGetAsyncError");
        rccl_check(state, "RCCL_ASYNC_ERROR");
    }

    void complete(int rank) {
        auto& r = owned_rank(rank);
        workers.check_peer();
        hip_check(hipEventRecord(r.complete_event, r.stream), "hipEventRecord completion");
        for (;;) {
            workers.check_peer();
            async_error(r);
            const auto status = hipEventQuery(r.complete_event);
            if (status == hipSuccess) { async_error(r); return; }
            if (status != hipErrorNotReady) hip_check(status, "hipEventQuery completion");
            std::this_thread::yield();
        }
    }

    [[noreturn]] void fail(const char* reason) noexcept {
        std::fprintf(stderr, "RCCL_LAYER_FAIL reason=%s exit=4 timing_admitted=0 "
                             "teardown=process_exit_without_buffer_release\n",
                     reason ? reason : "unspecified");
        if (!workers.active()) {
            try {
                workers.run("failure-abort", [&](int rank) {
                    auto& r = ranks[rank];
                    if (r.comm) {
                        hip_check(hipSetDevice(devices[rank]), "hipSetDevice abort");
                        rccl_check(ncclCommAbort(r.comm), "ncclCommAbort");
                        r.comm = nullptr;
                    }
                }, timeout_ms);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "RCCL_LAYER_ABORT_FAIL reason=%s\n", e.what());
            } catch (...) {
                std::fprintf(stderr, "RCCL_LAYER_ABORT_FAIL reason=unknown_exception\n");
            }
        } else {
            // No cross-thread abort or unsafe recursive dispatch if a caller
            // violates fail()'s post-paired contract. Process exit is safe.
            std::fprintf(stderr, "RCCL_LAYER_ABORT_SKIPPED reason=workers_still_active\n");
        }
        std::fflush(nullptr);
        std::_Exit(4);
    }
};

RcclSession::RcclSession(const std::array<int, 2>& devices, const std::array<void*, 2>& streams,
                         const std::string& expected_library, int timeout_ms, int init_timeout_ms) {
    validate(devices, streams, expected_library, timeout_ms, init_timeout_ms);
    impl_ = std::make_unique<Impl>(devices, streams, expected_library, timeout_ms, init_timeout_ms);
}

RcclSession::~RcclSession() {
    if (impl_ && !impl_->closed)
        impl_->fail("RCCL session destroyed before shutdown destroyed external graphs");
}

void RcclSession::paired(const char* stage, const std::function<void(int)>& action, bool initialization) {
    impl_->run(stage, action, initialization);
}

void RcclSession::exchange(int rank, const float* source, float* receive, size_t words) {
    auto& r = impl_->owned_rank(rank);
    if (!source || !receive || source == receive || words == 0)
        throw std::invalid_argument("RCCL exchange requires distinct buffers and a nonempty payload");
    impl_->workers.check_peer();
    rccl_check(ncclGroupStart(), "ncclGroupStart");
    // Once a group is open, always close it before reporting any enqueue
    // status. The caller may submit its consumer only after this returns.
    const auto send = ncclSend(source, words, ncclFloat32, 1 - rank, r.comm, r.stream);
    const auto receive_status = ncclRecv(receive, words, ncclFloat32, 1 - rank, r.comm, r.stream);
    const auto end = ncclGroupEnd();
    rccl_check(send, "ncclSend");
    rccl_check(receive_status, "ncclRecv");
    rccl_check(end, "ncclGroupEnd");
}

void RcclSession::complete(int rank) { impl_->complete(rank); }
void RcclSession::check_peer() const { impl_->workers.check_peer(); }

void RcclSession::shutdown(const std::function<void(int)>& destroy_graphs,
                           const std::function<void(int)>& release_rank_resources) {
    if (impl_->closed) return;
    if (!destroy_graphs || !release_rank_resources)
        throw std::invalid_argument("RCCL shutdown requires graph and rank-resource destruction callbacks");
    // Cross-rank phase boundaries matter: no communicator can be destroyed
    // until every stream is complete and every graph handle is gone.
    impl_->run("shutdown-complete", [&](int rank) { impl_->complete(rank); });
    impl_->run("shutdown-graphs", destroy_graphs);
    impl_->run("shutdown-communicators", [&](int rank) {
        auto& r = impl_->ranks[rank];
        rccl_check(ncclCommDestroy(r.comm), "ncclCommDestroy");
        r.comm = nullptr;
    });
    impl_->run("shutdown-events", [&](int rank) {
        auto& r = impl_->ranks[rank];
        hip_check(hipEventDestroy(r.complete_event), "hipEventDestroy completion");
        r.complete_event = nullptr;
    });
    impl_->run("shutdown-rank-resources", release_rank_resources);
    impl_->closed = true;
}

[[noreturn]] void RcclSession::fail(const char* reason) noexcept { impl_->fail(reason); }

} // namespace strata::core::tp2::detail
