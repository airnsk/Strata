#pragma once

// Private host owner for the isolated local-VRAM hybrid layer experiment.
// No RCCL dependency. Device handshakes never read these host worker flags.
#include <array>
#include <functional>
#include <memory>

namespace strata::core::tp2::detail {

class VramSession {
public:
    VramSession(const std::array<int,2>& devices,
                const std::array<void*,2>& compute,
                const std::array<void*,2>& copy, int watchdog_ms);
    ~VramSession();
    VramSession(const VramSession&) = delete;
    VramSession& operator=(const VramSession&) = delete;

    // Persistent owners submit concurrently, even when the first graph launch
    // blocks inside HIP. The controller watchdog covers every callback/API.
    // On a callback exception, BOTH callbacks finish before rethrow to caller.
    void paired(const char* stage, const std::function<void(int)>& action);
    void complete(int rank);
    void check_peer() const;
    // Ignores previous callback errors and drains BOTH ranks' streams before
    // returning. If drain cannot be established, exit without releasing storage.
    void drain() noexcept;
    void shutdown(const std::function<void(int)>& destroy_graphs,
                  const std::function<void(int)>& release_rank_resources) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::core::tp2::detail
