# TP2 routed-expert FFN test (experimental)

This is an isolated two-GPU arithmetic and timing test, not a new Strata inference
mode. It splits native expert weights without requantization and runs the existing
expert kernels on both cards. It does **not** contain shared FFN, router, HC, attention,
KV, PLE, MTP, or production peer scheduling. No whole-decode speedup is established.

The build gate is `STRATA_TP2_BUILD=OFF` by default. With it off, production binaries
link no additional GPU kernels and allocate no TP buffers. The CPU-only
`tp2_shard_policy` target is also available with `STRATA_BUILD_TESTS=ON`.

## Hardware build and run

The reusable runner builds in a separate directory with the existing image and then
runs the CPU and GPU gates (replace the image placeholder). The updated default uses
rank-local reduction and event-joined peer writes:

```sh
bash tools/run_tp2_mi50.sh EXISTING_IMAGE --iters 2
```

For a manual build, use the existing gfx906/ROCm workflow in a separate directory, adding:

```sh
-DSTRATA_HIP_GFX906=ON -DSTRATA_TP2_BUILD=ON -DSTRATA_HC_PERSIST_BUILD=OFF
```

Build only the test targets first:

```sh
cmake --build build-tp2 --target tp2_shard_policy tp2_ffn -j 48
./build-tp2/tp2_shard_policy
HIP_VISIBLE_DEVICES=0,1 timeout 600 ./build-tp2/tp2_ffn --mode 8 --iters 2
HIP_VISIBLE_DEVICES=0,1 timeout 1200 ./build-tp2/tp2_ffn --mode 8 --iters 12
```

Do not run the GPU test beside the inference server or another GPU benchmark. Stop
the server yourself first when convenient; this test does not stop it or change its
configuration. Capture stdout/stderr and the exit code. Exit 77 means skipped because
two suitable peer-accessible GPUs are unavailable; 1 is a numerical gate failure;
2 is an argument/runtime failure. A timeout is not a pass.

CLI:

- `--mode 8` (default): explicitly select the configured gfx906 fused GU/SwiGLU/q8
  family and LDS down family. `--mode 7` is a separate diagnostic experiment.
- `--iters 12`: measured iterations per case/mode, after four diagnostic warmups.
- `--exchange reduced` (new default): local weighted reduction on both EP and TP ranks;
  direct peer input/output writes with explicit system fences and cross-device event
  dependencies. Only final completion waits on the host.
- `--exchange reduced-host`: the same local reductions, direct input push and packed
  metadata, but a host join and runtime-copy return. This is an attribution control,
  not a fully factorial decomposition of every overhead.
- `--exchange rows`: original phase-1 full per-expert return and host join, retained as
  an isolation control. It does not require the new direct-write/event probe to pass.
- Default T values: 1, 2, 4, 8; `--tokens N` restricts to one value in 1..8.
- `--experts 10` (default, supported 4..16): number of selected experts. Default top10
  is the relevant case; do not describe a different K as the production model.
- `--no-graphs`: enqueue compute kernels directly. Default captures each rank's local
  quantization/expert chain separately. There is no cross-device capture or device spin.
- Repeat `--gguf SHARD` to supply actual weight files, with `--layer N`. This replaces
  synthetic weights with the selected layer's actual tensors. Supply the shard(s)
  containing its gate, up and down; a dense/PLE replacement file need not contain them.
  The reader mmaps supplied files and copies only K distinct experts into test buffers.

Example, with placeholders replaced by actual paths:

```sh
HIP_VISIBLE_DEVICES=0,1 timeout 1200 ./build-tp2/tp2_ffn \
  --mode 8 --iters 12 --gguf /path/to/expert-shard.gguf --layer 0
```

The test does not need a pack manifest for a real-weight run: it validates GGUF role
names, types, shapes, matching expert counts and payload bounds directly. It currently
requires N=2560, F=640, GU types IQ3_XXS/IQ3_S/IQ2_S/IQ4_XS and down IQ4_NL/Q2_0.
Unsupported types are refused. Source paths are not needed in published results.

## What is checked

CPU test: all eight supported type pairs, additional dimensions, exact byte ownership
and reassembly, block alignment, invalid types/shapes/ranks/buffer lengths and overflow.
Gate/up rows split 320/320; down input columns split 320/320 on complete quant blocks.

GPU test: every supported synthetic type pair, signed nonzero random codes with finite
nonzero block scales, deterministic varied inputs and normalized router weights. Four
epochs change input values and live selected-expert pointer ordering inside captured
graphs. Weight fixtures cover ordinary normalized weights, concentrated one-hot weights,
all-zero weights and signed cancellation stress. Signed weights are a numerical test,
not a claim about real router semantics. Every replay is checked, and outputs are poisoned before each launch.
All tokens use the same selected set, with T entries/expert: arbitrary sparse per-token
routing and unused groups inside a nonempty launch are not yet covered. Empty ranks are
covered by the 10/0 and 0/10 ownership cases.

The full-width single-GPU result with the independent original ordered join remains the
arithmetic oracle. The new timed full-width local reduction must match that old join
bitwise on every epoch, checked outside timing. EP expert rows must match
it bitwise. TP per-expert partials are rejoined outside timing for numerical diagnostics.
In reduced modes, both EP and TP combine expert rows locally, then add rank0+rank1;
this changes floating-point association for EP as well as TP, so final results are
numerical, not bitwise against the full-width oracle. The rows control retains its original
per-expert join before weighting. The test combine is not a claim of matching either
production combine backend.

Stronger intermediate gate: each entry's concatenated rank0/rank1 hidden q8_1 blocks must
match the full-width hidden q8_1 blocks **byte for byte**. Mode8 fuses gate/up into those
blocks and does not materialize separate gate/up floats. This gate isolates changes
introduced after hidden activation quantization. Full-width-only V2/V2K branches and old
IQ/grouped variants are disabled explicitly within the test process, not in user config.

Predeclared provisional gross-error thresholds (not model-quality approval):

- `max_abs_error / max(max_abs_reference, 1e-12) <= 3e-3`
- `sqrt(sum(error^2) / max(sum(reference^2), 1e-24)) <= 1e-3`
- All values finite; nonzero reference expert rows required. The deliberate zero-weight
  fixture must produce exactly zero combined output; it does not waive the nonzero-row check
- Componentwise `abs_error > 1e-5 + 3e-3*abs(reference)` counts are diagnostic, to make
  cancellation visible without treating a near-zero denominator as a reliable ratio

These limits are fixed before hardware execution; do not relax them after a failure.
Full-model logit/top-k margins, greedy prompts, quality and MTP acceptance remain separate
requirements even if this gross-error gate passes. Reports include actual maximum and RMS
errors, not just PASS.

## Timing interpretation

Compare single-GPU full width, EP balanced, primary-heavy, peer-heavy, primary-only,
peer-only and TP2. Default K=10 produces EP counts 5/5, 8/2, 2/8, 10/0 and 0/10.
TP2 computes ten half-width experts on each card. Existing production EP already executes
whole experts concurrently; balanced EP is not expected to become another automatic 2x
faster merely by changing to TP.

The EP cases are an **isolated P2P ownership baseline**, not measured production
`PeerExperts` performance. Small EP fixtures retain all K full blobs on each active card
so route assignment can rotate between epochs, but compute only their assigned experts;
the printed allocated weight bytes expose that distinction. This is not a model-residency
estimate. Weight uploads, fixture construction and graph capture are outside timing.

Measured wall time starts with input already resident on GPU0 and live routing/weight
values available to the host. Reduced modes include one packed pinned H2D dynamic record
per rank (expert pointers plus router weights), direct input push, both local compute
submissions, local reductions, return exchange/dependency, primary sum and final completion.
Counts, starts and token/destination indices are uploaded once because they do not change in
this fixture. Arbitrary real routing would need any changing fields in that dynamic record.
Input preparation, diagnostic poisoning and validation readback are outside reduced timing.
Legacy rows mode retains its original pageable uploads and in-timing output poison.

Event-joined DAG: GPU0 input push (each writer system-fences remote writes) records input
ready; GPU1 waits before compute, then its weighted reduction writes directly into GPU0's
result inbox and each writer system-fences before completion; GPU1 records output ready;
GPU0 waits before adding local+peer vectors. There is no device spin or mid-protocol CPU wait.
Rank-local compute graphs are unchanged; cross-device operations/dependencies are outside
capture. Inputs, outputs and per-rank metadata have separate allocations and lifetimes.

`input_stage_us` times the input operation; `return_stage_us` in reduced modes includes
local peer reduction (and direct push for event mode). In reduced-host it excludes the
later runtime copy, separately reported as `runtime_return_copy_us`. Compute event fields
remain expert kernels only. Event intervals may include host enqueue gaps; their sum is
NOT critical-path latency. `wall_us` is the complete measured protocol.

Return payload is T*N floats in both reduced EP and reduced TP, versus K*T*N partial rows
for the original TP test. At T=8, K=10, N=2560 this is 81,920 instead of 819,200 bytes.
This optimization belongs to both EP and TP; it is not evidence that TP has beaten balanced EP.
The reduced protocol writes/returns a zero vector even for an empty peer rank, which is
included in its payload/timing. Diagnostic per-expert readbacks occur only after timing.

Both directions must report peer capability and pass runtime-copy content checks. Reduced
modes additionally verify direct writes and the cross-device event ordering with 16 distinct
monotonic patterns per direction, more than the fixture's replay period. Every iteration
checks received peer input and final values after deliberately poisoning buffers. System
fences plus event dependencies are explicit, but hardware success must still be established;
an unsupported/erroring dependency fails rather than silently falling back. No probe is
presented as a physical link-bandwidth measurement.

## First hardware result: phase 1, 2026-10-10

The original rows-return test built and passed all 32 synthetic cases on two MI50s with
mode8 and two measured iterations/case after two warmups. Every reported hidden-q8 check
was byte-exact. Worst reported normalized max error was 2.75135e-7, RMS-relative error
1.21616e-7. This establishes the raw hidden-dimension split's arithmetic foundation.

It did **not** establish a speed win. Across the eight synthetic format pairs at each T:

| T | Median TP wall us | Median balanced EP wall us | Median paired TP/EP time ratio |
| --- | ---: | ---: | ---: |
| 1 | 264.12 | 233.50 | 1.162 |
| 2 | 301.17 | 261.20 | 1.137 |
| 4 | 371.07 | 316.02 | 1.183 |
| 8 | 603.84 | 491.29 | 1.187 |

TP was slower than balanced EP in 30/32 cases and slower than single-GPU full width in
32/32. These are unweighted synthetic summaries, not a model-weighted prediction; two
iterations do not support fine ranking. TP compute itself was generally also slower than
balanced EP compute. The optimized exchange must improve the full measured protocol,
not reinterpret these negative results as a win.

The old TP return event intervals had medians 58/85/121/185 us for T=1/2/4/8, with
102/205/410/819 KB of partial rows. Zero-byte event intervals were already about 4.8 us.
The individually host-submitted operations and host join can introduce enqueue gaps;
these figures do not prove SDMA behavior or host staging, nor overturn an independent
28 GB/s PCIe benchmark. They motivate reduced payload, packed dynamic metadata and an
event-joined protocol rather than another unchanged baseline run.

## Probe initialization correction, 2026-10-10

The first reduced-protocol hardware attempt compiled and passed the CPU gate, then
stopped at the first direct peer-write/event visibility probe. No reduced FFN case or
performance result was produced.

Review found that probe initialization used default-stream H2D/memset operations without
an explicit completion dependency before another device's nonblocking producer. The
[pinned ROCm 7.2.4 `hipMemset` implementation](https://github.com/ROCm/rocm-systems/blob/rocm-7.2.4/projects/clr/hipamd/src/hip_memory.cpp#L2759-L2784)
can enqueue this device-memory operation asynchronously. This is a real ordering defect in the probe; it is not proof that it
caused the observed mismatch. The measured FFN loop already completed its poison/input
initialization explicitly and is not being retimed or changed by this correction.

Both runtime-copy and direct-write probes now initialize on named streams and explicitly
complete source upload and destination/witness poisoning before transport. Input transport
uses raw `uint4` loads/stores, and the probe uses finite FP32 word patterns. After the
cross-device event, a GPU consumer kernel copies destination data to a local witness so
host/DMA readback cannot alone certify kernel-visible data.

Failures print source/destination, sequence, first bad byte and expected/actual words.
One fully synchronized producer/consumer re-read is diagnostic only: even if that restores
the expected data, the original event-protocol failure remains a failure. There is no
automatic fallback, threshold change, or speculative event-flag change. Actual fenced
peer-write/event behavior remains a hardware gate after the initialization correction.

## Validation status of the updated reduced protocol

CPU standalone strict-warning, optimized, Release/NDEBUG and ASan/UBSan checks passed.
LeakSanitizer is unavailable under this executor's ptrace, so its check was disabled.
Host harness C++ syntax was checked with local API declarations; that is not a HIP compile.
CMake was unavailable in the authoring executor. The initial rows-return version did
compile/run on the owner's hardware as recorded above. The reduction/peer-write/event version compiled on hardware, but its initial probe failed
before arithmetic/timing. The corrected probe's GPU build/visibility and subsequent
reduced arithmetic/timing remain pending hardware validation. These test sources do not enable any production TP path.
