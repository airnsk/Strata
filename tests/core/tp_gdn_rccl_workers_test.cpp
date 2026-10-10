// Host-only checks; this does not establish RCCL, HIP or graph correctness.
// c++ -std=c++17 -pthread -I. tests/core/tp_gdn_rccl_workers_test.cpp -o workers_test
#include "../../src/core/tp_gdn_rccl_workers.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(__unix__) || defined(__APPLE__)
#include <cerrno>
#include <csignal>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
using strata::core::tp2::detail::RcclWorkers;
using Clock = std::chrono::steady_clock;
void require(bool ok, const char* reason) {
    if (!ok) throw std::runtime_error(reason);
}

void persistent_dispatch() {
    RcclWorkers workers;
    std::array<std::thread::id, 2> owners;
    std::array<int, 2> calls{};
    workers.run("record-owners", [&](int rank) {
        owners[rank] = std::this_thread::get_id();
        require(workers.owns_rank(rank), "worker does not own its rank");
        require(!workers.owns_rank(1 - rank), "worker owns the wrong rank");
    }, 2000);
    require(owners[0] != owners[1], "rank workers are not distinct");
    require(owners[0] != std::this_thread::get_id() && owners[1] != std::this_thread::get_id(),
            "rank work ran on the caller");
    require(!workers.owns_rank(0) && !workers.owns_rank(1), "caller owns a rank");

    for (int round = 0; round < 256; ++round) {
        std::atomic<int> entered{0};
        workers.run("repeated-paired-dispatch", [&](int rank) {
            require(std::this_thread::get_id() == owners[rank], "rank changed permanent owner");
            require(workers.active(), "running command is not marked active");
            require(calls[rank] == round, "command ran twice or skipped a generation");
            ++calls[rank];
            entered.fetch_add(1, std::memory_order_release);
            while (entered.load(std::memory_order_acquire) != 2) {
                workers.check_peer();
                std::this_thread::yield();
            }
        }, 2000);
        require(entered.load() == 2 && !workers.active(), "dispatch did not await both owners");
    }
    require(calls[0] == 256 && calls[1] == 256, "wrong final dispatch count");
}

void peer_error() {
    RcclWorkers workers;
    std::atomic<bool> peer_observed{false};
    bool original_caught = false;
    try {
        workers.run("peer-failure", [&](int rank) {
            if (rank == 0) throw std::runtime_error("original-rank-zero-error");
            while (!workers.failed()) std::this_thread::yield();
            peer_observed.store(true, std::memory_order_release);
            workers.check_peer();
        }, 2000);
    } catch (const std::runtime_error& e) {
        original_caught = std::string(e.what()) == "original-rank-zero-error";
    }
    require(original_caught, "secondary peer exception masked original failure");
    require(peer_observed.load(std::memory_order_acquire), "peer did not observe failure");
    require(!workers.active(), "failed dispatch returned before both callbacks");
    std::atomic<int> recovered{0};
    workers.run("worker-state-reset", [&](int) {
        workers.check_peer();
        recovered.fetch_add(1);
    }, 2000);
    require(recovered == 2 && !workers.failed(), "failed state leaked to next command");

    bool nonstandard_caught = false;
    try {
        workers.run("nonstandard-exception", [&](int rank) { if (rank == 1) throw 123; }, 2000);
    } catch (int value) { nonstandard_caught = value == 123; }
    require(nonstandard_caught, "nonstandard worker exception was lost");
}

void invalid_dispatch() {
    RcclWorkers workers;
    std::atomic<int> calls{0};
    const auto action = [&](int) { calls.fetch_add(1); };
    int rejected = 0;
    try { workers.run(nullptr, action, 2000); } catch (const std::invalid_argument&) { ++rejected; }
    try { workers.run("", action, 2000); } catch (const std::invalid_argument&) { ++rejected; }
    try { workers.run("no-action", {}, 2000); } catch (const std::invalid_argument&) { ++rejected; }
    try { workers.run("no-budget", action, 0); } catch (const std::invalid_argument&) { ++rejected; }
    require(rejected == 4 && calls == 0, "invalid command was published");

    std::atomic<int> nested_rejected{0};
    workers.run("nested-command", [&](int) {
        try { workers.run("nested", action, 2000); }
        catch (const std::logic_error&) { nested_rejected.fetch_add(1); }
    }, 2000);
    require(nested_rejected == 2 && calls == 0, "nested dispatch was accepted");

    // A second controller must not replace a live command or reset its state.
    std::atomic<bool> live{false}, release{false};
    std::exception_ptr controller_error;
    std::thread controller([&] {
        try {
            workers.run("live-command", [&](int) {
                live.store(true, std::memory_order_release);
                while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
            }, 2000);
        } catch (...) { controller_error = std::current_exception(); }
    });
    while (!live.load(std::memory_order_acquire)) std::this_thread::yield();
    bool overlap_rejected = false;
    try { workers.run("overlap", action, 2000); }
    catch (const std::logic_error&) { overlap_rejected = true; }
    release.store(true, std::memory_order_release);
    controller.join();
    if (controller_error) std::rethrow_exception(controller_error);
    require(overlap_rejected && calls == 0 && !workers.active(), "overlapping dispatch corrupted command");
}

void missing_worker_timeout() {
#if defined(__unix__) || defined(__APPLE__)
    // Fork only after all earlier workers have been joined. The child omits the
    // peer's protocol admission, just like the hardware missing-rank scenario.
    const auto begin = Clock::now();
    const pid_t pid = fork();
    require(pid >= 0, "cannot fork missing-worker test");
    if (pid == 0) {
        RcclWorkers workers;
        std::atomic<bool> peer_admitted{false};
        workers.run("missing-peer-admission", [&](int rank) {
            if (rank == 1) return;
            while (!peer_admitted.load(std::memory_order_acquire)) std::this_thread::yield();
        }, 100);
        std::_Exit(1);
    }
    int status = 0;
    for (;;) {
        const auto result = waitpid(pid, &status, WNOHANG);
        if (result == pid) break;
        if (result < 0 && errno != EINTR) {
            kill(pid, SIGKILL);
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
            throw std::runtime_error("waitpid failed for missing-worker child");
        }
        if (Clock::now() - begin > std::chrono::seconds(5)) {
            kill(pid, SIGKILL);
            while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
            throw std::runtime_error("worker watchdog failed to bound missing peer");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    require(WIFEXITED(status) && WEXITSTATUS(status) == 70, "missing peer did not exit with watchdog status 70");
    require(Clock::now() - begin >= std::chrono::milliseconds(90), "watchdog exited before its budget");
#else
    throw std::runtime_error("missing-worker subprocess test requires POSIX process support");
#endif
}
} // namespace

int main() {
    try {
        persistent_dispatch();
        peer_error();
        invalid_dispatch();
        missing_worker_timeout();
        std::puts("tp_gdn_rccl_workers_test: PASS persistent_owners=2 repeated_dispatches=256 "
                  "peer_failure=1 invalid_dispatch=1 missing_peer_exit=70 NOT_GPU_TEST=1");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tp_gdn_rccl_workers_test: FAIL %s\n", e.what());
        return 1;
    }
}
