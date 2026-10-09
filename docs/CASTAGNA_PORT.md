# Experimental Castagna-inspired gfx906 port

Status (2026-10-09): HIP 7.2.4 build and model-free bitwise parity **passed on two MI50 cards**.
Performance acceptance **failed**: this first implementation is approximately 4.6–9.8 times slower than
the checked production HC path in the supplied stage benchmark. Keep it disabled; do not deploy it.
Whole-model parity and performance have not been tested.
This is the first stage of a larger decode optimization effort. It does not establish a 30–50% decode gain.
The engine's existing path remains the default. The build flag also defaults to OFF so an unused experimental
kernel cannot consume VRAM under eager loading and change default expert-cache residency.

## First hardware result

Commit `634b6614`, two gfx906 MI50 32 GiB cards, HIP runtime `70253211`. Both cards passed
32 HC configurations, 32 router configurations, 24 graph replays per case, subgroup offsets,
T=5–8/Q8 fallbacks and isolated streams. This establishes only the tested model-free cases.

Representative GPU 0 graph timings (microseconds, median of five alternating pairs):

| T | Router experts | Checked baseline | Persistent |
|---|---:|---:|---:|
| 1 | none | 48.725 | 445.664 |
| 4 | 256 | 85.922 | 530.235 |
| 4 | 512 | 123.257 | 564.132 |

GPU 1 showed the same regression. Timings include the stream interval for repeated submissions;
they do not isolate kernel-body execution from cooperative-launch scheduling or host submission gaps.
The failure requires diagnosis, not another whole-model A/B run with this unchanged kernel.

## What this stage changes

Build with `-DSTRATA_HC_PERSIST_BUILD=ON`, then `STRATA_HC_PERSIST=1` selects a persistent hyper-connection (HC) read for eligible calls:

- The pending residual update, normalization, BF16 down projection, SiLU, BF16 up projection, gate and mix
  run in one cooperative GPU kernel instead of the existing three-kernel multi-token read.
- The verify loop can fold the following BF16 router projection into the same grid: **four kernel launches
  become one** for the HC-plus-router section. Top-k selection, routing semantics, expert ownership, CPU
  doorbells and the shared/routed expert kernels retain their existing order.
- When the caller asks for the mixed activation's q8_1 image, the kernel also writes it after a grid barrier.
  No global completion counters are needed in the new path.
- The original BF16 weights, FP32 activation arithmetic, accumulation order and caller-owned `xn`, `lo`,
  `rs`, injection and output buffers are retained. Bitwise parity is an acceptance requirement, not a
  result already measured on hardware.

The donor is [Castagna Veloce at 3483d462](https://github.com/benpeterson40/castagna-veloce/blob/3483d462715d61f21ed5877836be35a724446359/ggml/src/ggml-cuda/hc-persist.cu).
[Attribution and its MIT license](../third_party/castagna-veloce/README.md) are retained. Its Q4/Q5 expert
kernels do not cover Strata's IQ1/IQ3 expert layouts, so this stage keeps the existing native expert path,
including `STRATA_EXP_MODE=8` if selected. `STRATA_SH_STREAM` is not changed.

## Eligibility and safe fallback

- gfx906, `n_embd=2560`, `hc=4`, `hc_lr=320`, **1–4 tokens** per call, BF16 HC weights
- HIP runtime **6.4 or newer**, reported cooperative-launch support, and positive occupancy for the
  compiled kernel with its actual static/dynamic LDS and registers
- Optional router fusion: 256 or 512 experts, native FP32-activation/BF16 projections, batched routing;
  the verifier leaves its existing `STRATA_LFUSE` auxiliary shared-gate path alone
- The router's per-token diagnostic overrides also keep their existing projection path
- 5–8-token calls, Q8 HC overrides, unsupported hardware/runtimes and other backends keep their existing
  kernels. The router projection is skipped only after the launch reports that it actually produced logits.

The grid is bounded by the occupancy query and the work size, not a hard-coded number of CUs. All grid
barriers use HIP cooperative groups and the cooperative launch API. There are no process-global device
accumulators, peer polling loops, hand-written inter-block spin barriers or token-path scratch allocations.
Separate graphs may use separate session buffers without sharing a barrier counter. Existing buffer alias
contracts still apply: `inject_out` must not overlap `inj_prev`; concurrent sessions must own disjoint scratch.

Configure the option before graph capture. A captured graph retains its chosen kernels. The verifier warms
the capability/occupancy cache during initialization. That cache is host-thread-local: external callers
that move capture to another host thread should call `fused_gr_persistent_supported(T)` on that thread
before capture as well. Leave other HC
experiments (`STRATA_GR_V3`, `STRATA_GR_SPLIT`, `STRATA_HC_Q8`) unset for an isolated A/B test. If runtime or
occupancy requirements fail, an explicit log message reports use of the existing kernels. The acceptance
harness refuses to call that a passing persistent run. Internal norm/down/up profiling boundaries do not
exist in a fused call; use its total HC timing and the stage harness.

## Why the host's ROCm 6.3.3 is insufficient for this graph path

Basic cooperative launches exist in 6.3.3, but its direct launch function does not capture them into a graph.
Manually setting a cooperative graph attribute does not fix this: that version stores the attribute but
its graph replay does not pass a cooperative dispatch flag to the launch.

Source evidence:

- [6.3.3 direct cooperative launch](https://github.com/ROCm/clr/blob/rocm-6.3.3/hipamd/src/hip_module.cpp)
- [6.3.3 graph command creation](https://github.com/ROCm/clr/blob/rocm-6.3.3/hipamd/src/hip_graph_internal.hpp)
- [6.4 capture creates a cooperative graph node](https://github.com/ROCm/clr/blob/rocm-6.4.0/hipamd/src/hip_graph.cpp)
- [6.4 preserves the cooperative flag for replay](https://github.com/ROCm/clr/blob/rocm-6.4.0/hipamd/src/hip_graph_internal.hpp)

This experiment therefore uses an **isolated newer runtime**. It does not require replacing host ROCm,
changing the production server, or disabling graph capture to claim activation.

## Build and acceptance gates

Use a separate build directory and the same pinned llama.cpp dependency as the existing engine.
A model-free GPU harness is available even when the published checkout leaves `STRATA_BUILD_TESTS` off.
Inside an already prepared gfx906-capable ROCm 6.4+ development environment:

```sh
cmake -S . -B build-hc-persist -G Ninja \
  -DSTRATA_HIP_GFX906=ON -DSTRATA_HC_PERSIST_BUILD=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_DIR=/absolute/path/to/the/existing/pinned/llama.cpp \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build-hc-persist --target strata strata-device hc_persistent -j2
build-hc-persist/strata-device --list-devices
build-hc-persist/strata-device --selftest
# Run once per card. No model files are needed; timeout also catches a stuck grid.
HIP_VISIBLE_DEVICES=0 timeout 300 build-hc-persist/hc_persistent 100 24
HIP_VISIBLE_DEVICES=1 timeout 300 build-hc-persist/hc_persistent 100 24
```

Exit **77 means skipped**, not passed. A nonzero error or bitwise mismatch blocks model testing. The harness
checks its actual persistent submission count, graph node count, eager and repeated graph outputs, guarded
buffers, null/non-null injection, pending residual writes, optional q8 output, router logits and independent
streams. It reports alternating baseline/persistent stage timings; those are not tokens/second.

`strata-device --selftest` checks device allocation, **not hipBLAS math**. Before model inference, also verify
that the container's active hipBLAS/rocBLAS libraries and gfx906 Tensile files belong together and run a small
hipBLAS numerical check using the build's actual linked libraries. A Torch GPU test by itself does not prove
that Strata resolves the same rocBLAS library. Inspect `ldd build-hc-persist/strata`, the environment and the
runtime library search path; do not substitute an unverified Tensile directory or change the host installation.

Then run the existing backend parity suite where available (`gr_parity --selftest`, `verify_parity --selftest`
and router/expert parity), followed by fixed-model greedy output and teacher-forced logit comparisons.
Compare the same model pack, quantization, expert residency, context, prompts, MTP settings, power limits and
runtime. Use at least five interleaved A/B pairs with `STRATA_HC_PERSIST=0` and `=1`; record time to first token,
prefill, decode tokens/second, accepted drafts, total output tokens and errors. Keep any production engine
running on its existing build until the isolated trial passes. A slow or failing arm is a result to report,
not a reason to change the default.

CPU-only dispatch-policy checks can run without ROCm:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
  tests/hip/hc_persistent_policy.cpp -o /tmp/hc-persistent-policy
/tmp/hc-persistent-policy
```

## Remaining stages

1. Validate this HC/router stage on the target cards and measure its share of a whole verify window.
2. Extend persistent FFN work to Strata's actual shared and routed expert formats, retaining grouped expert
   reuse and explicit cache ownership. Gate an all-resident path separately from CPU/PCIe-miss handling.
3. If profiles justify it, add tensor-parallel weight sharding, simultaneous per-device scheduling and a
   graph-safe reduction protocol. Strata's existing layer split is not tensor parallelism; transplanting
   Castagna's all-reduce alone would not make both cards compute each layer concurrently.

Each stage needs repeatable parity/quality checks and end-to-end measurements before changing a default.

### Standalone launch/barrier diagnostic

After the two-card parity pass and performance failure, isolate launch scheduling
from math before another full-model run. This probe needs no Strata rebuild,
model files, or external downloads. In the same ROCm 7.2.4 environment:

```sh
/opt/rocm/bin/hipcc -O3 -std=c++17 --offload-arch=gfx906 -x hip \
  tests/hip/hc_persistent_diag.cpp -o /tmp/hc_persistent_diag
HIP_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=10s 120s \
  /tmp/hc_persistent_diag 100
```

For the existing read-only Docker run, keep its image, device flags and repository
mount, compile and execute inside that container, and leave the output binary in
its writable `/tmp`. One idle MI50 is enough for this diagnostic.

The same zero-barrier probe runs with ordinary and cooperative launch. Other
probes add one, two, or three grid barriers. The small matrix uses 60/160 blocks,
256 threads, and 5/20 KiB dynamic LDS; configurations beyond reported cooperative
occupancy are skipped. It prints function attributes, event and wall time per
launch, and raw `clock64` cycle deltas for each block in the last launch. Raw cycles
are not converted using an assumed frequency.

Three submission modes are compared: eager, one-node graph replay, and a graph
containing 100 serial nodes replayed three times. The last tests caller replay
overhead; cooperative graph nodes can still be host-dispatched by the runtime.
Graph construction, warmup and readback are outside the timed interval.

Interpretation:

- Slow cooperative zero-barrier launches with short in-kernel cycle counts place
  most time outside the measured kernel body. They do not alone distinguish
  queue switching, runtime dispatch and GWS initialization.
- A large cost added by one/two/three barriers implicates barrier handling.
- Fast empty probes mean the actual HC bodies and their resource usage need the
  next inspection. These probes do not reproduce HC register pressure, spills,
  static LDS or arithmetic, and do not prove a speedup.

The standalone probe compiled and ran successfully on the user's idle MI50 with
HIP 7.2.4. Batched graphs measured approximately 1.4–1.6 us for ordinary launches,
17.5–17.8 us for cooperative launches without barriers, and 18.5–19.6 us for
cooperative launches with three barriers. This rejects cooperative scheduling and
empty grid barriers alone as an explanation for the roughly 400 us HC regression.
It does not exclude slower barriers under the actual HC kernel's resource pressure.


### Actual HC body and argument-passing diagnostic

The next diagnostic runs only with the explicit test argument below. The original
persistent kernel and its dispatch remain unchanged. Incrementally rebuild the
existing experimental target in the same ROCm image, then run one idle MI50:

```sh
cmake --build build-hc-persist --target hc_persistent -j2
HIP_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=10s 300s \
  build-hc-persist/hc_persistent --diagnose 100
```

This rebuild needs a writable build directory; unlike the standalone probe,
compilation cannot use the earlier read-only repository mount. Runtime testing
can still use that read-only mount after compilation.

It compares four variants: the original production kernel, an instrumented
by-value-helper clone, an untimed const-reference-helper clone, and an instrumented
const-reference clone. The five copied helper bodies are identical except for
names and the `GrMulti` argument's passing mode. This tests whether passing the
large, dynamically indexed argument aggregate by value produces expensive private
storage or copies. Inlining may eliminate those copies; the source alone does not
establish that this is the cause.

For T=1/4 and router=0/512, the test first checks bitwise output parity with pending
writes and q8 enabled/disabled, including router selections. It reports actual
compiled registers, local bytes, static/dynamic LDS, occupancy, and selected grid
for every variant. Graph timings compare a shared grid within every variant's
capacity, plus each variant's native grid when different. Instrumentation can
change allocation and occupancy; the untimed const-reference clone is the primary
performance comparison.

Seven phase durations are reported in raw per-block `clock64` cycles: norm,
barrier 1, down, barrier 2, up, optional barrier 3, and quantization/router tail.
Only the final graph node's phase samples are retained. Active-block medians/maxima
are printed separately from idle blocks. Do not add maxima or medians from
separate CUs into a critical-path duration. Compiler-generated entry/exit work
outside the first/last clock sample is not included. Timing samples are written
only at the end of the kernel.

This actual-body diagnostic has passed source review, host syntax checks using
HIP declarations, normalized helper-body comparison, and whitespace checks. It
has not yet been compiled with HIP or run on a GPU. A lower local-memory count
plus a large untimed speedup would support the argument-copy hypothesis; otherwise
use the phase/resource evidence to select the next fix. These results are not an
end-to-end model throughput claim.
