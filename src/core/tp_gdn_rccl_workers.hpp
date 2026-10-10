#pragma once

// Private, host-only scheduling component of the opt-in RCCL layer experiment.
// Derived from the persistent rank owners in tests/hip/tp2_rccl_transport.cpp.
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace strata::core::tp2::detail {

class RcclWorkers {
public:
    RcclWorkers() {
        try {
            for (int rank = 0; rank < 2; ++rank)
                threads_[rank] = std::thread([this, rank] { loop(rank); });
        } catch (...) {
            // No command has been published, so partial host construction is
            // safe to unwind without touching a device or communicator.
            stop_and_join();
            throw;
        }
    }
    RcclWorkers(const RcclWorkers&) = delete;
    RcclWorkers& operator=(const RcclWorkers&) = delete;
    ~RcclWorkers() {
        if (active() || owner_ == this) {
            std::fprintf(stderr, "RCCL_LAYER_FAIL reason=destroy_active_workers exit=4 "
                                 "teardown=process_exit_without_buffer_release\n");
            std::fflush(nullptr);
            std::_Exit(4);
        }
        stop_and_join();
    }

    // Only one controller may dispatch at a time. Nested/overlapping dispatch
    // is rejected before publishing a command, instead of deadlocking workers.
    void run(const char* stage, const std::function<void(int)>& action, int timeout_ms) {
        if (!stage || !*stage || !action || timeout_ms <= 0)
            throw std::invalid_argument("invalid RCCL worker command");
        if (owner_ == this) throw std::logic_error("nested RCCL worker dispatch");
        bool idle = false;
        if (!active_.compare_exchange_strong(idle, true, std::memory_order_acq_rel))
            throw std::logic_error("overlapping RCCL worker dispatch");
        struct Release {
            std::atomic<bool>& active;
            ~Release() { active.store(false, std::memory_order_release); }
        } release{active_};
        std::unique_lock<std::mutex> lock(mutex_);
        // Copies can throw; finish them before publishing the next generation.
        action_ = action;
        stage_ = stage;
        failed_.store(false, std::memory_order_release);
        error_ = nullptr;
        completed_ = 0;
        ++generation_;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        changed_.notify_all();
        if (!done_.wait_until(lock, deadline, [&] { return completed_ == 2; })) {
            std::fprintf(stderr, "WATCHDOG_TIMEOUT component=tp_gdn_rccl stage=%s exit=70 "
                                 "teardown=process_exit_without_buffer_release "
                                 "budget_ms=%d completed_ranks=%d\n",
                         stage_.c_str(), timeout_ms, completed_);
            std::fflush(nullptr);
            std::_Exit(70);
        }
        // Both owners are back before the caller can initiate abort or cleanup.
        if (error_) std::rethrow_exception(error_);
    }

    bool failed() const noexcept { return failed_.load(std::memory_order_acquire); }
    bool active() const noexcept { return active_.load(std::memory_order_acquire); }
    bool owns_rank(int rank) const noexcept { return owner_ == this && rank_ == rank; }
    void check_peer() const {
        if (failed()) throw std::runtime_error("RCCL peer worker failed");
    }

private:
    void stop_and_join() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        changed_.notify_all();
        for (auto& thread : threads_) if (thread.joinable()) thread.join();
    }

    void loop(int rank) {
        owner_ = this;
        rank_ = rank;
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                changed_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
            }
            // action_ and stage_ stay immutable until both completion acks.
            // Avoid a potentially throwing std::function copy in the worker.
            try { action_(rank); }
            catch (...) {
                const auto error = std::current_exception();
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (!error_) error_ = error;
                }
                // Publish the original exception before telling its peer to
                // stop, so a secondary peer error cannot mask the root cause.
                failed_.store(true, std::memory_order_release);
                try { std::rethrow_exception(error); }
                catch (const std::exception& e) {
                    std::fprintf(stderr, "WORKER_ERROR component=tp_gdn_rccl stage=%s rank=%d reason=%s\n",
                                 stage_.c_str(), rank, e.what());
                } catch (...) {
                    std::fprintf(stderr, "WORKER_ERROR component=tp_gdn_rccl stage=%s rank=%d reason=unknown_exception\n",
                                 stage_.c_str(), rank);
                }
            }
            { std::lock_guard<std::mutex> lock(mutex_); ++completed_; }
            done_.notify_one();
        }
    }

    inline static thread_local const RcclWorkers* owner_ = nullptr;
    inline static thread_local int rank_ = -1;
    std::array<std::thread, 2> threads_;
    std::atomic<bool> failed_{false}, active_{false};
    std::mutex mutex_;
    std::condition_variable changed_, done_;
    std::function<void(int)> action_;
    std::string stage_;
    std::exception_ptr error_;
    uint64_t generation_ = 0;
    int completed_ = 0;
    bool stop_ = false;
};

} // namespace strata::core::tp2::detail
