#pragma once

// Private transport for the explicitly enabled HIP/RCCL hybrid-layer benchmark.
// No HIP/RCCL types or dependency are exposed by the normal public layer API.
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace strata::core::tp2::detail {

class RcclSession {
public:
    RcclSession(const std::array<int, 2>& devices,
                const std::array<void*, 2>& streams,
                const std::string& expected_library,
                int timeout_ms, int init_timeout_ms);
    ~RcclSession();
    RcclSession(const RcclSession&) = delete;
    RcclSession& operator=(const RcclSession&) = delete;

    // Runs action(rank) concurrently on the two permanent rank owners. Calls
    // must not overlap. Worker/API failures abort after both callbacks return;
    // a stuck callback or GPU API exits the subprocess with watchdog status 70.
    void paired(const char* stage, const std::function<void(int)>& action,
                bool initialization = false);

    // Called only from the corresponding paired rank callback. The layer owns
    // streams, device buffers, capture/launch admission, and consumer kernels.
    void exchange(int rank, const float* source, float* receive, size_t words);
    void complete(int rank);
    // Use in host admission/barrier spins so a failed peer wakes this owner.
    void check_peer() const;

    // Mandatory before destruction. Completes both streams, destroys both
    // ranks' graphs, then destroys communicators/events. Only afterward may
    // release_rank_resources release the layer-owned buffers and streams.
    // Both callbacks run on rank owners under the command watchdog.
    void shutdown(const std::function<void(int)>& destroy_graphs,
                  const std::function<void(int)>& release_rank_resources);

    // May be called only after paired() has returned. Never frees graph/buffer
    // storage on failure: bounded owner-thread abort, then subprocess exit 4.
    [[noreturn]] void fail(const char* reason) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::core::tp2::detail
