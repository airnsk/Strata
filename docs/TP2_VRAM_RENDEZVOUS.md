# Two-MI50 VRAM rendezvous admission probe

This is an opt-in, model-free correctness experiment. It does not change the
engine, serving defaults, quantization, expert selection, or the existing TP
exchange. No performance, whole-layer speed, generation speed, or quality result
is established by this probe. There is no timing phase to admit accidentally.

## Run once on the stand

Both MI50s must already be idle. From this checkout:

```sh
bash tools/run_tp2_vram_probe_mi50.sh
```

The runner uses only the installed image
`sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca`.
It does not pull an image, install software, mount a model, use RCCL, stop a
service, change GPU clocks/power, or add a Docker security override. Missing or
ambiguous idle telemetry rejects the run. Sudo authorization happens on the
controlling terminal before the timed/captured work.

Only two HIP translation units are built with the image's compiler, gfx906,
`-fno-fast-math -ffp-contract=off`, and `BUILD_JOBS=48`. The default private
`.tp2-vram-build-cache` is incremental. Source, compiler, flags and image identify
its configuration; cached objects and the executable have verified SHA-256
records. `TP2_VRAM_CACHE=off` uses a temporary build instead. The engine and its
model/BLAS dependencies are not built by this command.

Optional flags:

- `--devices 0,1` or `--devices 1,0`: logical rank mapping
- `--iterations 128..512`: changing-input graph replays per case, default 128
- `--timeout-ms 1000..60000`: host operation watchdog allowance, default 10000;
  a device wait gets one quarter of this allowance plus an independent finite
  polling cap

Optional environment bounds, in seconds: `TP2_TIMEOUT=180` per process,
`TP2_BUILD_TIMEOUT=180`, `TP2_OUTER_TIMEOUT=900`. The outer bound covers discovery,
compilation and the three processes; separately bounded container cleanup follows.
The wrapper prints a durable `tp2-vram-probe-*.log` path and final exit status.
It attempts each scenario once. A timeout or unexpected failure is a failed run,
not a reason for an automatic retry. Inspect GPU state before trying again.

## What the gate checks

The normal matrix has two fresh allocation lifecycles, T1 and T8, both actual
host issue orders (01 and 10), and three operations:

1. Raw 3072-word-per-rank gather, interleaved in 1024-word runs
2. Raw 1280-word-per-rank gather
3. 2560-word-per-rank FP32 sum in fixed rank0-plus-rank1 order

Raw data include signed subnormals and NaN payloads. Sum data include finite
normal values, signed zeros, cancellation, and round-to-nearest-even ties. There
is no tolerance-based comparison. A host oracle checks the final payload and
canaries independently of the transport implementation.

Every graph replay first generates its inputs on the GPU from a live device
counter. Every input word changes from the preceding epoch. A consumer-side GPU
witness checks every output word, input word, inactive tail and surrounding guard,
plus the actual protocol stamp. A distinct witness record is checked for every
replay on both GPUs. Thus a static captured upload or a correct last replay cannot
hide stale intermediate output. Both failure controls are checked after both
streams drain, in addition to both completion stamps and witnesses.

The two-slot ring is reused without host signal resets. Each inbox is a separate
256 KiB uncached allocation, using a fixed 32768-word slot stride even when the
active packet changes size. Extra normal cases alternate T8/T1 graphs while work
is queued, using the same controls and inboxes. The second fresh lifecycle starts
with explicitly seeded valid prior slot history at epoch 4294967294, then crosses
the 32-bit carry boundary in live 64-bit GPU publication and polling. This is a
bounded carry test, not an exhaustive proof of arbitrarily long execution.

The delayed-peer process repeats the complete three-operation matrix at T8,
both orders and two fresh lifecycles. One rank is delayed before each graph
submission; success additionally requires observed live GPU polling by the
waiting rank.

The missing-peer process omits each rank in turn on fresh allocations. It must
observe a finite device timeout, poisoned output, an aborted late peer on the
same failed allocations, and an aborted repeat by the original rank. It never
clears failure flags. Separate fresh negative cases reject future ready tokens,
future consumed acknowledgements, and UINT64_MAX epoch wrap. Both streams must
drain before any allocation is freed. Only then do two new lifecycles repeat the
three-operation T8 correctness matrix, including the 64-bit carry test.

The processes have independent host watchdogs, and the runner wraps them with
external deadlines. A wedged driver/fabric is not recoverable merely because a
kernel has a polling limit. On a host watchdog the process exits without freeing
possibly live buffers. The wrapper removes only its uniquely named container
and verifies its absence. An outer timeout, killed child, skip, incomplete log,
duplicate/malformed gate, failed SHA check, or unverified cleanup can never be a
suite pass.

## Boundaries and validation status

The primitive uses a single CTA per rank, peer VRAM stores, explicit AMD SYSTEM
release/acquire operations, locally polled uncached flags, and consumed-slot
acknowledgements. It has a single ordered stream per rank and sticky pair failure.
Only gfx906 with the pinned HIP 7.2 compiler interface is accepted; architecture
metadata alone is not evidence that peer ordering works.

The probe exercises single-span packets only. The API's two-span gather form has
host descriptor-validation coverage, but has no GPU admission from this matrix.
The same is true of other architectures, arbitrary stream concurrency, arbitrary
packet schedules, long-duration soak, and full model execution. These cases need
separate admission. The isolated whole-layer experiment below uses only the
same one-span gather and sum operations after target-runtime admission.

Development validation in the CPU-only workspace:

```sh
c++ -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off \
  -DSTRATA_VRAM_CPU_SELFTEST tests/hip/tp2_vram_rendezvous.cpp \
  -o /tmp/vram-selftest
/tmp/vram-selftest --selftest
python3 tools/test_tp2_vram_probe_runner.py
c++ -std=c++17 -Wall -Wextra -Werror -pedantic -Iinclude -x c++ \
  src/kernels/cuda/tp_vram_rendezvous.cu \
  tests/core/tp_vram_rendezvous_contract_test.cpp -o /tmp/vram-contract
/tmp/vram-contract
```

The CPU oracle checks 811008 words, mutation detection, and FP32 tie/signed-zero
behavior. Runner tests use mocked commands and transcripts; they are not GPU or
Docker evidence. The host contract test checks invalid descriptors and the
unsupported-backend path. No real HIP compilation, MI50 execution, or throughput
measurement was available in that development workspace.

### Target-runtime protocol admission

Before using the separate whole-layer experiment, run this model-free probe on
its target GPU pair and runtime. Preserve the source and binary identities and
complete log with the experiment's private evidence. A pass admits only the
bounded protocol matrix; it contains no throughput measurements and does not
establish whole-layer correctness or model quality.

An optional repository CMake target is available with
`-DSTRATA_HIP_GFX906=ON -DSTRATA_TP2_VRAM_PROBE_BUILD=ON`, then build
`tp2_vram_rendezvous`. It links only HIP and host threads, never an engine target.
The normal repository configuration can still require its usual dependencies;
the standalone runner above avoids those requirements. GPU cases are not added
to automatic ctest because the idle and isolated-fault requirements matter.

The next separate opt-in step is the
[whole-GDN VRAM comparison](TP2_VRAM_GDN_LAYER.md). It must pass its own real-weight
prefix, continuation, state-bank and fault gates before matched end-to-end
wall timing can support a one-layer speed claim.
