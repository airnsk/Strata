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
runs the CPU and GPU gates (replace the image placeholder). The default compares both input-column and output-row TP against paired single-GPU
controls, using event-joined peer writes:

```sh
bash tools/run_tp2_mi50.sh EXISTING_IMAGE --iters 2
```

For a manual build, use the existing gfx906/ROCm workflow in a separate directory, adding:

```sh
-DSTRATA_HIP_GFX906=ON -DSTRATA_TP2_BUILD=ON -DSTRATA_HC_PERSIST_BUILD=OFF
```

Build only the test targets first:

```sh
cmake --build build-tp2 --target tp2_shard_policy tp_layer_layout_test tp2_ffn -j 48
./build-tp2/tp2_shard_policy
./build-tp2/tp_layer_layout_test
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
- `--iters 12`: measured single/candidate pairs per case/mode, after four warmup pairs.
- `--tp-layout both` (default): run both `TP2-column` and `TP2-output-row`, plus EP controls.
  `column` selects the earlier hidden-dimension split; `row` selects literal output-row
  splitting. Row/both requires `--exchange reduced`; legacy exchange controls require column.
- `--order alternate` (default): alternate single/candidate order and flip each fixture’s order on successive four-epoch cycles.
  `forward` and `reverse` are attribution controls. Candidate modes still run in fixed order;
  this is not randomization of every mode or of the two ranks’ host submissions.
- `--profile-stages`: add a separate instrumented diagnostic replay, excluded from timing.
- `--exchange reduced` (default): local weighted reduction on both EP and TP ranks;
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

The separate CPU-only `tp_layer_layout_test` validates GDN/QSA head and state ownership
for the future full-layer executor; it does not execute an attention kernel. See
[FULL_LAYER_TP_IMPLEMENTATION.md](FULL_LAYER_TP_IMPLEMENTATION.md).

CPU shard test: all eight supported type pairs, additional dimensions, exact byte ownership
and reassembly, block alignment, invalid types/shapes/ranks/buffer lengths and overflow.
Gate/up rows split 320/320. Column TP splits down input columns 320/320 on complete
quant blocks; output-row TP splits down output rows 1280/1280 and preserves F=640.
Both layouts must exactly reconstruct the original bytes.

GPU test: every supported synthetic type pair, signed nonzero random codes with finite
nonzero block scales, deterministic varied inputs and normalized router weights. Four
epochs change input values and live selected-expert pointer ordering inside captured
graphs. Weight fixtures cover ordinary normalized weights, concentrated one-hot weights,
all-zero weights and signed cancellation stress. Signed weights are a numerical test,
not a claim about real router semantics. All four correctness epochs are checked and poisoned before and after the timed batch.
Performance replays are not individually downloaded or poisoned.
All tokens use the same selected set, with T entries/expert: arbitrary sparse per-token
routing and unused groups inside a nonempty launch are not yet covered. Empty ranks are
covered by the 10/0 and 0/10 ownership cases.

The full-width single-GPU result with the independent original ordered join remains the
arithmetic oracle. The new timed full-width local reduction must match that old join
bitwise on every epoch, checked outside timing. EP expert rows must match
it bitwise. Column-TP per-expert partials are rejoined outside timing for numerical diagnostics.
Output-row TP instead concatenates independent output halves, and must match the full-width
expert rows and ordered weighted combine bitwise.
In reduced EP and column-TP modes, ranks combine expert rows locally, then add rank0+rank1;
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
Column TP computes ten half-width experts on each card. Output-row TP computes ten
half-width GU projections, exchanges the q8 hidden halves, then computes ten full-width
down projections with half the output rows. Existing production EP already executes
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
Legacy rows mode retains its pageable metadata uploads and host join. Diagnostic output
poisoning is outside timing in all modes.

Event-joined DAG: GPU0 input push (each writer system-fences remote writes) records input
ready; GPU1 waits before compute, then its weighted reduction writes directly into GPU0's
result inbox and each writer system-fences before completion; GPU1 records output ready;
GPU0 waits before adding local+peer vectors. There is no device spin or mid-protocol CPU wait.
Rank-local compute graphs are unchanged; cross-device operations/dependencies are outside
capture. Inputs, outputs and per-rank metadata have separate allocations and lifetimes.

Output-row DAG: both GU projections create group-entry-ordered q8 halves. Each rank
writes its half into the appropriate per-entry positions of both full-F scratch buffers,
with a system fence and an event before the other rank consumes it. Two hidden-ready
events prevent down from reading an incomplete vector. Each down graph uses F=640 and
1280 output rows. Local ordered expert reductions produce half an output vector; the peer
half is pushed to GPU0 and concatenated after the completion event. Hidden values are not
dequantized or requantized during transport. Both gathered buffers must match the
full-width oracle byte for byte. There is no host wait between GU and down.

Performance `TIMING` lines report paired `single_wall_us`, `candidate_wall_us` and their
ratio `speedup`. Diagnostic timing events are disabled; input/output/hidden dependency
events and final completion remain included. Optional `PROFILE` lines come from a separate
replay; rank events measure full FFN for EP/column and GU only for row TP. These intervals
may include host enqueue gaps and must not be summed as critical-path latency.

At T=8, K=10, N=2560, column/EP reduced return is 81,920 bytes. Row TP returns 40,960
bytes and additionally sends 28,800 bytes of hidden q8 in each direction (360 bytes per
entry). The original full-row column protocol returned 819,200 bytes. Payload sizes are
not a throughput estimate; row TP also introduces a second graph launch per rank.

Both directions must report peer capability and pass runtime-copy content checks. Reduced
modes additionally verify direct writes and cross-device event ordering with 16 distinct
monotonic patterns per direction. Correctness passes check received peer input, poisoned
outputs, local hidden halves, gathered full hidden vectors and final values. Performance
replays run separately between those checks. An unsupported/erroring dependency fails
rather than silently falling back. No probe is a physical link-bandwidth measurement.

The new row layout and uninstrumented paired timing require fresh hardware validation.
The historical measurements below used the earlier instrumented harness; do not compare
old and new wall times as if only the tensor layout changed. No full-engine TP path has
been enabled by this test.

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

## Corrected reduced-protocol hardware result, 2026-10-10

At `1a3d24c`, the corrected probe and all 32 synthetic cases passed on the two MI50s,
mode 8, `--exchange reduced`, 12 measured iterations/case. Both directions passed 16
monotonic direct-write/event/GPU-witness sequences. All 128 reported hidden-q8 checks
were byte-exact; worst reported normalized max error was 2.61760e-7 and RMS-relative
error 1.42847e-7. The old ordered join oracle remains independent and unchanged.

The exchange correction materially reduces the microbenchmark's overhead, but TP still
loses to balanced EP in 31/32 cases. Unweighted medians across eight synthetic format
pairs (ratios are medians of paired ratios, not ratios of the displayed medians):

| T | TP wall us | Balanced EP wall us | TP/balanced EP | TP/8-primary-2-peer EP | TP/single GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 166.12 | 155.12 | 1.058 | 1.051 | 1.114 |
| 2 | 179.25 | 170.65 | 1.062 | 1.032 | 1.058 |
| 4 | 217.34 | 201.90 | 1.087 | 0.986 | 0.959 |
| 8 | 324.81 | 292.61 | 1.086 | 0.901 | 0.818 |

TP wins against the single-GPU control in 16/32 and the primary-heavy EP control in
15/32 cases. At T8 it wins all eight primary-heavy cases, about 10% by the paired median;
at T1/T2 its median is still slower. These are synthetic local-subgraph results, not a
model-weighted estimate or whole-decode result. The model's aggregate ownership hits
cannot establish its per-layer critical-path distribution.

TP input/return stage medians are now about 8.7/13.0 us at T1 and 12.0/16.3 us at T8.
The return payload is T*N floats rather than K*T*N, metadata is packed/pinned, and the
intermediate host join is removed. Those changes were tested together, so the result
cannot assign a speedup to each separately or claim physical transport bandwidth.
Balanced EP benefits from the same protocol and retains its advantage. Production TP
FFN integration is not justified by this result alone.

## Production exchange boundary and next bounded candidate

The user's `PeerExperts` decode path is different from both test modes:

- `src/core/peer_experts.cpp:227-247` uploads pinned-host input and packed metadata,
  computes peer-owned rows, then directly scatters those rows to mapped **host** output.
- `PeerExperts::finish` (`:255-267`) CPU-polls peer stream completion.
- `src/core/expert_source.cpp:3250,3362` launches/joins this work. The verifier waits
  for the host completion flag, gathers missing rows from mapped memory, adds primary
  hits, and combines routing weights (`src/core/verify.cpp:1507-1548`). With batched
  decode it already avoids copying primary-owned rows: its payload is peer_entries*N,
  not automatically K*T*N.
- A separate helper-cache optimization already reduces remote rows to T*N, then D2H
  copies and host-accumulates them (`src/core/remote_expert_opt.cu:97-119`). It attaches
  `RemoteExperts`, not `PeerExperts` (`src/program/generate.cpp:4830,4903`).

A bounded next candidate is an opt-in reduced peer return, retaining expert ownership
and the existing completion protocol: publish router weights, reduce peer-owned rows
in original token/route order, write T*N into a fixed primary-device buffer, mask these
rows out of the host gather, and add the peer sum at the existing combine boundary.
This changes FP32 association and needs the independent row/combine oracle and real
logit/greedy checks. It must support empty ownership, sparse per-token routes and replay.
It can reuse the helper optimization's mask/combine semantics rather than inventing a
second incompatible scheme. No engine changes have been made.

Retain the host completion gate initially. The verifier captures the whole layer range
for a window (`verify.cpp:1718,1982`), unlike the prototype's independently launched
per-rank compute graphs. Moving its synchronization to cross-device events requires an
explicit graph-generation/lifetime design; an event wait inserted into an already-running
graph is not automatically equivalent to the prototype. No 30–50% model gain follows
from this candidate.

## Output-row TP hardware result, 2026-10-10

At `05695a9`, the owner built the HIP harness and both CPU targets and ran 12
alternating single/candidate pairs, both layouts, reduced transport, graphs on.
All 32 synthetic cases passed. Output-row expert rows and ordered FP32 combines
were bit-exact against the original single-GPU oracle; local hidden halves and
both full-F gathered buffers were byte-exact. The direct peer/event probes passed.
This is a subgraph correctness result, not full-model quality approval.

| T | Median row/single speedup (single time / row time) | Median column/single speedup | Median EP/single speedup |
|---|---:|---:|---:|
| 1 | 0.862 | 1.033 | 1.066 |
| 2 | 0.937 | 1.120 | 1.139 |
| 4 | 1.067 | 1.190 | 1.261 |
| 8 | 1.289 | 1.260 | 1.406 |

Each entry is the median of eight ratios using that candidate's paired single
baseline. This is not a median-time ratio. Row TP beats its own single baseline
in 18/32 cases, balanced EP in 0/32, and column TP in 6/32 (all at T8).
The strongest row result is 1.342x. No case establishes 2x.

Applying actual layer-format counts to these small synthetic fixtures gives
single/row ratios 0.886, 0.962, 1.081 and 1.297 at T1/2/4/8. These are ratios
of count-weighted wall-time sums, excluding absent pair (23,42), not model
latency predictions. The common (22,20) pair still favors column at T8
(278.429 us column versus 288.693 us row). One (23,20) T8 row result is weak
at 306.425 us; no cause is established and it should not decide architecture.

The run has 12 samples/pairs but reports only means, not dispersion. EP/column
comparisons are same-run separate candidate blocks, not directly interleaved
row-versus-EP trials. The build is incremental and contains 46 compiler warnings.
No further unchanged FFN repeats are required to advance the implementation:
the next useful gate is the complete GDN-layer executor, shared FFN and
GPU-owned routing/state, as specified in `FULL_LAYER_TP_IMPLEMENTATION.md`.

Source log: `tp2-20261010T011347Z.txt`, SHA-256
`152aa7d4d2d27476bd1a968e99c418b5bb7c6123db7ff2f8924dc07284205deb`.

## Validation status

CPU strict-warning, optimized, Release/NDEBUG and ASan/UBSan checks passed.
LeakSanitizer is unavailable under this executor's ptrace, so its check was disabled.
The authoring executor has no HIP toolchain; GPU compile and runtime evidence above
comes from the owner's hardware logs for the earlier column-only revisions. The corrected
column reduced protocol passed its historical hardware gate. The output-row path and paired timing changes subsequently passed the owner’s
HIP/GPU gate at `05695a9`, documented above. The authoring environment itself still
has no HIP compiler or GPU.
Real-weight/model-quality and production integration remain untested. These test
sources do not enable any production TP path.
