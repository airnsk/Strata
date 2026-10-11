// Host-only argument/capability test. Compile tp_vram_rendezvous.cu as ordinary
// C++, WITHOUT STRATA_HIP_GFX906/__HIPCC__. This does not exercise a GPU protocol.
// Build with g++ -std=c++17 -Wall -Wextra -Werror -Iinclude -x c++, passing
// src/kernels/cuda/tp_vram_rendezvous.cu and this file as the two input sources.
#include "strata/kernels/tp_vram_rendezvous.hpp"

#include <climits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace k = strata::kernels;

struct Checks {
    int count = 0;
    void require(bool value, const char* label) {
        if (!value) throw std::runtime_error(label);
        ++count;
    }
    template<class F> void invalid(const char* label, F run) {
        try { run(); }
        catch (const std::invalid_argument&) { ++count; return; }
        throw std::runtime_error(std::string("missing invalid_argument: ") + label);
    }
    template<class F> void unsupported(const char* label, F run) {
        try { run(); }
        catch (const std::invalid_argument&) {
            throw std::runtime_error(std::string("valid descriptor rejected: ") + label);
        }
        catch (const std::runtime_error& error) {
            require(std::string(error.what()).find("requires gfx906 HIP 7.2") != std::string::npos,
                    "wrong unsupported-backend error");
            return;
        }
        throw std::runtime_error(std::string("unsupported backend did not throw: ") + label);
    }
};

int main() try {
    Checks checks;
    checks.require(!k::tp_vram_rendezvous_supported(), "test must use unsupported host-only backend");
    constexpr size_t capacity = k::kTpVramRendezvousMaxWords;
    constexpr size_t ring_words = k::kTpVramRendezvousSlots * capacity;
    std::vector<uint32_t> source(capacity), source2(capacity), output(2 * capacity),
                          output2(2 * capacity), own_inbox(ring_words), peer_inbox(ring_words);
    k::TpVramRendezvousControl own{}, peer{};
    k::TpVramRendezvousStamp stamp{};
    k::TpVramRendezvousPacket packet{};
    packet.spans[0] = {source.data(), output.data(), 1, 32, 8};
    packet.count = 1;
    packet.own_inbox = own_inbox.data();
    packet.peer_inbox = peer_inbox.data();
    packet.inbox_words = 32;
    auto call = [&](const auto& p) { k::tp_vram_rendezvous(p, &own, &peer, &stamp, 0, 100, nullptr); };
    auto bad = [&](const char* label, auto change) {
        auto p = packet;
        change(p);
        checks.invalid(label, [&] { call(p); });
    };
    checks.unsupported("valid raw run gather", [&] { call(packet); });
    bad("zero span count", [](auto& p) { p.count = 0; });
    bad("too many spans", [](auto& p) { p.count = 3; });
    bad("unknown mode", [](auto& p) { p.mode = static_cast<k::TpVramRendezvousMode>(42); });
    bad("zero rows", [](auto& p) { p.spans[0].rows = 0; });
    bad("negative rows", [](auto& p) { p.spans[0].rows = -1; });
    bad("zero half words", [](auto& p) { p.spans[0].half_words = 0; });
    bad("negative half words", [](auto& p) { p.spans[0].half_words = -1; });
    bad("zero run", [](auto& p) { p.spans[0].run_words = 0; });
    bad("negative run", [](auto& p) { p.spans[0].run_words = -1; });
    bad("nondividing run", [](auto& p) { p.spans[0].run_words = 3; });
    bad("oversized run", [](auto& p) { p.spans[0].run_words = 64; });
    bad("signed product overflow", [](auto& p) {
        p.spans[0].rows = INT_MAX; p.spans[0].half_words = INT_MAX; p.spans[0].run_words = 1;
    });
    bad("zero exact extent", [](auto& p) { p.inbox_words = 0; });
    bad("wrong exact extent", [](auto& p) { p.inbox_words = 33; });
    bad("wrapping exact extent", [](auto& p) { p.inbox_words = SIZE_MAX; });
    bad("null source", [](auto& p) { p.spans[0].src = nullptr; });
    bad("null output", [](auto& p) { p.spans[0].dst = nullptr; });
    bad("null own ring", [](auto& p) { p.own_inbox = nullptr; });
    bad("null peer ring", [](auto& p) { p.peer_inbox = nullptr; });
    bad("wrapping source address", [](auto& p) {
        p.spans[0].src = reinterpret_cast<void*>(UINTPTR_MAX - 3);
    });
    bad("misaligned source", [](auto& p) {
        p.spans[0].src = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(p.spans[0].src) + 1);
    });
    bad("input/output alias", [](auto& p) { p.spans[0].dst = const_cast<void*>(p.spans[0].src); });
    bad("partial input/output alias", [](auto& p) {
        p.spans[0].dst = static_cast<uint32_t*>(const_cast<void*>(p.spans[0].src)) + 31;
    });
    bad("ring alias", [](auto& p) { p.peer_inbox = p.own_inbox; });
    bad("overlap in fixed unused ring capacity", [](auto& p) {
        p.peer_inbox = p.own_inbox + ring_words - 1;
    });
    bad("output/control alias", [&](auto& p) { p.spans[0].dst = &own; });
    bad("output/peer control alias", [&](auto& p) { p.spans[0].dst = &peer; });
    bad("output/stamp alias", [&](auto& p) { p.spans[0].dst = &stamp; });
    bad("ring/control alias", [&](auto& p) { p.own_inbox = reinterpret_cast<uint32_t*>(&own); });
    bad("sum with mapped runs", [](auto& p) { p.mode = k::TpVramRendezvousMode::SumF32; });

    for (int rank : {-1, 2}) checks.invalid("rank", [&] {
        k::tp_vram_rendezvous(packet, &own, &peer, &stamp, rank, 100, nullptr);
    });
    for (uint64_t ticks : {uint64_t{0}, UINT64_MAX}) checks.invalid("timeout", [&] {
        k::tp_vram_rendezvous(packet, &own, &peer, &stamp, 0, ticks, nullptr);
    });
    for (auto* control : {static_cast<k::TpVramRendezvousControl*>(nullptr), &peer,
                          reinterpret_cast<k::TpVramRendezvousControl*>(reinterpret_cast<uintptr_t>(&own) + 1)})
        checks.invalid("own control null/alias/alignment", [&] {
            k::tp_vram_rendezvous(packet, control, &peer, &stamp, 0, 100, nullptr);
        });
    checks.invalid("null peer control", [&] {
        k::tp_vram_rendezvous(packet, &own, nullptr, &stamp, 0, 100, nullptr);
    });
    checks.invalid("null stamp", [&] {
        k::tp_vram_rendezvous(packet, &own, &peer, nullptr, 0, 100, nullptr);
    });
    checks.invalid("stamp/control alias", [&] {
        k::tp_vram_rendezvous(packet, &own, &peer,
            reinterpret_cast<k::TpVramRendezvousStamp*>(&own), 0, 100, nullptr);
    });
    checks.invalid("stamp alignment", [&] {
        k::tp_vram_rendezvous(packet, &own, &peer,
            reinterpret_cast<k::TpVramRendezvousStamp*>(reinterpret_cast<uintptr_t>(&stamp) + 1), 0, 100, nullptr);
    });

    for (int tokens : {1, 8}) {
        packet.spans[0] = {source.data(), output.data(), tokens, 3072, 1024};
        packet.inbox_words = size_t(tokens) * 3072;
        checks.unsupported("GDN Y", [&] { call(packet); });
        packet.spans[0] = {source.data(), output.data(), tokens, 1280, 1280};
        packet.inbox_words = size_t(tokens) * 1280;
        checks.unsupported("GDN output", [&] { call(packet); });
        packet.count = 2;
        packet.spans[0] = {source.data(), output.data(), tokens * 10, 90, 90};
        packet.spans[1] = {source2.data(), output2.data(), tokens, 90, 90};
        packet.inbox_words = size_t(tokens) * 990;
        checks.unsupported("two-span hidden packet", [&] { call(packet); });
        packet.count = 1;
        packet.mode = k::TpVramRendezvousMode::SumF32;
        packet.spans[0] = {source.data(), output.data(), tokens, 2560, 2560};
        packet.inbox_words = size_t(tokens) * 2560;
        checks.unsupported("ordered sum", [&] { call(packet); });
        bad("two-span sum", [&](auto& p) { p.count = 2; });
        packet.mode = k::TpVramRendezvousMode::GatherU32;
    }
    packet.count = 2;
    packet.spans[0] = {source.data(), output.data(), 8, 3072, 1024};
    packet.spans[1] = {source2.data(), output2.data(), 8, 1024, 1024};
    packet.inbox_words = capacity;
    checks.unsupported("exact maximum packet", [&] { call(packet); });
    bad("total exceeds bounded packet", [](auto& p) { p.spans[1].rows = 9; p.inbox_words += 1024; });
    bad("second input overlaps first output", [&](auto& p) { p.spans[1].src = output.data(); });
    checks.require(own.epoch.value == 0 && peer.epoch.value == 0 && stamp.status == 0,
                   "host-only validation must not mutate protocol state");
    std::cout << "PASS tp_vram_rendezvous host-only contract checks=" << checks.count
              << " (GPU protocol not executed)\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL tp_vram_rendezvous host-only contract: " << error.what() << '\n';
    return 1;
}
