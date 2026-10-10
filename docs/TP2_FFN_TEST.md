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
runs the CPU and GPU gates (replace the image placeholder):

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
- `--iters 12`: measured iterations per case/mode, after two warmups.
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
nonzero block scales, deterministic varied inputs and normalized router weights. Two
alternating epochs change input values and live selected-expert pointer ordering inside
captured graphs. Every replay is checked, and outputs are poisoned before each launch.
All tokens use the same selected set, with T entries/expert: arbitrary sparse per-token
routing and unused groups inside a nonempty launch are not yet covered. Empty ranks are
covered by the 10/0 and 0/10 ownership cases.

The full-width single-GPU result is the arithmetic oracle. EP rows and the test's ordered
FP32 weighted combine must match it bitwise. TP adds each expert's rank0/rank1 partials
before applying the same test combine. TP down-dot association changes, so TP-vs-stock
is numerical, not bitwise. The test combine is not a claim of matching either production
combine backend.

Stronger intermediate gate: each entry's concatenated rank0/rank1 hidden q8_1 blocks must
match the full-width hidden q8_1 blocks **byte for byte**. Mode8 fuses gate/up into those
blocks and does not materialize separate gate/up floats. This gate isolates changes
introduced after hidden activation quantization. Full-width-only V2/V2K branches and old
IQ/grouped variants are disabled explicitly within the test process, not in user config.

Predeclared provisional gross-error thresholds (not model-quality approval):

- `max_abs_error / max(max_abs_reference, 1e-12) <= 3e-3`
- `sqrt(sum(error^2) / max(sum(reference^2), 1e-24)) <= 1e-3`
- All values finite; nonzero reference output required
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

Measured wall time starts with input already resident on GPU0 and includes live metadata
uploads, input peer copy, both local compute submissions, host synchronization, compact
peer result copy, ordered combine on GPU0 and final completion. Input preparation and
validation readback are outside timing. Separate event times report input copy, return
copy and each rank's compute. Their sum is NOT critical-path latency. There is no claim
that two devices overlapped optimally; the explicit host join is part of this prototype.

Both directions must report peer capability, enable peer access and pass a byte-content
copy probe. Transport uses runtime peer-copy APIs, with no explicit host fallback. A
successful API probe does not independently establish the physical DMA route or reproduce
the owner's previously measured PCIe bandwidth.

## Validation status before hardware results

CPU standalone strict-warning, optimized, Release/NDEBUG and ASan/UBSan checks passed.
LeakSanitizer is unavailable under this executor's ptrace, so its check was disabled.
Host harness C++ syntax was checked with local API declarations; that is not a HIP compile.
CMake was unavailable in the initial authoring executor. GPU compilation, link, peer copy,
capture replay, arithmetic and timing all remain pending hardware validation. These test
sources do not enable any production TP path.
