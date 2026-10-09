// Standalone launch/barrier diagnostic. No Strata library or model is needed.
// hipcc -O3 -std=c++17 --offload-arch=gfx906 -x hip tests/hip/hc_persistent_diag.cpp -o /tmp/hc_persistent_diag
// /tmp/hc_persistent_diag [iterations=100]
// All times are microseconds per launch. clock64 deltas are raw per-CU cycles,
// NOT converted using an assumed GPU frequency. This does not benchmark HC math.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static void check(hipError_t err, const char* where) {
    if (err != hipSuccess) throw std::runtime_error(std::string(where) + ": " + hipGetErrorString(err));
}

template<int Barriers>
__global__ void __launch_bounds__(256) probe(unsigned long long* cycles, unsigned* sink) {
    extern __shared__ unsigned scratch[];
    volatile unsigned* v = scratch;
    unsigned long long begin = 0;
    if (threadIdx.x == 0) begin = clock64();
    v[threadIdx.x] = threadIdx.x + blockIdx.x;
    __syncthreads();
    if constexpr (Barriers > 0) {
        auto grid = cooperative_groups::this_grid();
#pragma unroll
        for (int i = 0; i < Barriers; ++i) grid.sync();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned value = v[255];
        const unsigned long long end = clock64();
        sink[blockIdx.x] = value;
        cycles[blockIdx.x] = end - begin;
    }
}

template<int Barriers>
static void launch(bool coop, int blocks, size_t lds, hipStream_t stream,
                   unsigned long long* cycles, unsigned* sink) {
    if (coop) {
        void* args[] = {&cycles, &sink};
        check(hipLaunchCooperativeKernel(reinterpret_cast<const void*>(probe<Barriers>),
              dim3(blocks), dim3(256), args, lds, stream), "cooperative launch");
    } else {
        hipLaunchKernelGGL((probe<Barriers>), dim3(blocks), dim3(256), lds, stream, cycles, sink);
        check(hipGetLastError(), "ordinary launch");
    }
}

// mode 0: host loop; mode 1: host loop of one-node graphs (original HC test);
// mode 2: one graph containing N serial launches, replayed 3 times. Mode 2
// tests caller replay overhead. Cooperative graph nodes may still be
// host-dispatched inside the runtime rather than replayed as GPU packets.
template<int Barriers>
static void measure(bool coop, int blocks, size_t lds, int iterations, int mode,
                    hipStream_t stream, unsigned long long* cycles, unsigned* sink) {
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    const int nodes = mode == 2 ? iterations : 1;
    const int repeats = mode == 2 ? 3 : iterations;
    if (mode) {
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), "begin capture");
        for (int i = 0; i < nodes; ++i) launch<Barriers>(coop, blocks, lds, stream, cycles, sink);
        check(hipStreamEndCapture(stream, &graph), "end capture");
        check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
    }
    const auto submit = [&] {
        if (mode) check(hipGraphLaunch(exec, stream), "graph replay");
        else launch<Barriers>(coop, blocks, lds, stream, cycles, sink);
    };
    for (int i = 0; i < (mode == 2 ? 2 : 12); ++i) submit();
    check(hipStreamSynchronize(stream), "warmup sync");
    hipEvent_t start, stop;
    check(hipEventCreate(&start), "create start");
    check(hipEventCreate(&stop), "create stop");
    const auto wall_start = std::chrono::steady_clock::now();
    check(hipEventRecord(start, stream), "record start");
    for (int i = 0; i < repeats; ++i) submit();
    check(hipEventRecord(stop, stream), "record stop");
    check(hipEventSynchronize(stop), "wait stop");
    const double wall_us = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - wall_start).count();
    float ms = 0;
    check(hipEventElapsedTime(&ms, start, stop), "event elapsed");
    std::vector<unsigned long long> raw(blocks);
    std::vector<unsigned> values(blocks);
    check(hipMemcpy(raw.data(), cycles, raw.size() * sizeof(raw[0]), hipMemcpyDeviceToHost), "read cycles");
    check(hipMemcpy(values.data(), sink, values.size() * sizeof(values[0]), hipMemcpyDeviceToHost), "read sink");
    for (int b = 0; b < blocks; ++b)
        if (values[b] != unsigned(255 + b)) throw std::runtime_error("probe output mismatch");
    std::sort(raw.begin(), raw.end());
    const double count = double(repeats) * nodes;
    std::printf("%s barriers=%d blocks=%d lds=%zu mode=%s event_us=%.3f wall_us=%.3f last_cycles_med=%llu max=%llu\n",
        coop ? "coop" : "ordinary", Barriers, blocks, lds,
        mode == 0 ? "eager" : mode == 1 ? "graph1" : "graph_batch",
        1000.0 * ms / count, wall_us / count, raw[raw.size()/2], raw.back());
    std::fflush(stdout);
    check(hipEventDestroy(start), "destroy start");
    check(hipEventDestroy(stop), "destroy stop");
    if (exec) check(hipGraphExecDestroy(exec), "destroy executable");
    if (graph) check(hipGraphDestroy(graph), "destroy graph");
}

template<int Barriers>
static void test(int cu_count, int iterations, hipStream_t stream,
                 unsigned long long* cycles, unsigned* sink) {
    hipFuncAttributes attr{};
    check(hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(probe<Barriers>)), "function attributes");
    std::printf("probe barriers=%d regs=%d static_lds=%zu local_bytes=%zu max_threads=%d\n",
                Barriers, attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes, attr.maxThreadsPerBlock);
    for (size_t lds : {size_t(5120), size_t(20480)}) {
        int active = 0;
        check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&active, probe<Barriers>, 256, lds), "occupancy");
        for (int blocks : {60, 160}) {
            if (blocks > active * cu_count) {
                std::printf("SKIP barriers=%d blocks=%d lds=%zu occupancy_bound=%d\n", Barriers, blocks, lds, active * cu_count);
                continue;
            }
            std::printf("config barriers=%d blocks=%d lds=%zu active_per_cu=%d\n", Barriers, blocks, lds, active);
            for (int mode = 0; mode < 3; ++mode) {
                if constexpr (Barriers == 0) measure<0>(false, blocks, lds, iterations, mode, stream, cycles, sink);
                measure<Barriers>(true, blocks, lds, iterations, mode, stream, cycles, sink);
            }
        }
    }
}

int main(int argc, char** argv) {
    try {
        const int iterations = argc > 1 ? std::atoi(argv[1]) : 100;
        if (argc > 2 || iterations < 10 || iterations > 10000)
            throw std::runtime_error("usage: hc_persistent_diag [iterations 10..10000]");
        int device = 0, runtime = 0, cooperative = 0;
        check(hipGetDevice(&device), "device");
        hipDeviceProp_t prop{};
        check(hipGetDeviceProperties(&prop, device), "properties");
        check(hipRuntimeGetVersion(&runtime), "runtime");
        check(hipDeviceGetAttribute(&cooperative, hipDeviceAttributeCooperativeLaunch, device), "cooperative capability");
        std::printf("device=%d name=%s arch=%s CUs=%d runtime=%d cooperative=%d iterations=%d\n",
                    device, prop.name, prop.gcnArchName, prop.multiProcessorCount, runtime, cooperative, iterations);
        if (!cooperative || runtime < 60400000) { std::puts("SKIP: cooperative graph capture requires HIP >= 6.4"); return 77; }
        std::puts("clock64 values are raw per-block cycles from LAST launch only; events/wall are batch averages.\n"
                  "Empty probes do not match the HC kernel's registers, spills, static LDS, or math.\n"
                  "Large coop0 cost with short cycles suggests scheduling; added sync cost suggests barriers.\n"
                  "graph1 vs graph_batch tests caller replay overhead; coop nodes may still be host-dispatched.\n"
                  "No speedup claim.");
        hipStream_t stream;
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create stream");
        unsigned long long* cycles = nullptr;
        unsigned* sink = nullptr;
        check(hipMalloc(&cycles, 160 * sizeof(*cycles)), "allocate cycles");
        check(hipMalloc(&sink, 160 * sizeof(*sink)), "allocate sink");
        test<0>(prop.multiProcessorCount, iterations, stream, cycles, sink);
        test<1>(prop.multiProcessorCount, iterations, stream, cycles, sink);
        test<2>(prop.multiProcessorCount, iterations, stream, cycles, sink);
        test<3>(prop.multiProcessorCount, iterations, stream, cycles, sink);
        check(hipFree(cycles), "free cycles");
        check(hipFree(sink), "free sink");
        check(hipStreamDestroy(stream), "destroy stream");
        std::puts("PASS: standalone launch/barrier diagnostic finished.");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
