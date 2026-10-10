# TP2 complete GDN-layer correctness gate

This opt-in component executes one complete non-PLE GDN layer on two gfx906
cards. It is not connected to `generate`, prefill, MTP or the server. It makes
no generation-speed or quality claim. `STRATA_TP2_GDN_BUILD=OFF` is the default.

## Implemented execution

`TpGdnWeights` loads only the selected layer: all 512 expert shards, native
quantized projection/shared rows, and canonical HC/router/scalar parameters.
Weights remain native GGUF blocks. Each rank owns its allocations; no old
configuration file or old executable is used. The canonical pack and GGUF
must correspond to the same model. Dimensions/types/spans are checked; source
identity is not inferred merely from a matching shape.

`TpGdnLayer` executes:

1. Replicated HC read, modulo-mapped QKV/gate and GDN head recurrence
2. GDN output reassembly, output-row projection, full mixer reconstruction
3. HC write/read and replicated GPU router
4. GPU resident expert grouping (all experts have local static shards)
5. Output-row shared and routed FFNs, hidden-Q8 all-gather and weighted combine
6. Full output reconstruction and final HC residual write

The hidden dimension stays full width for down. No CPU expert-pool planning
or host route selection occurs. In the baseline modes HC and router are
replicated; the optional flat-HC mode below partitions HC compute. GDN state is sharded using modulo head ownership; convolution
state uses canonical `[channel,3]` order. This component uses explicit per-call
FFN phase/mode dispatch rather than the benchmark's global phase overrides.
The legacy `native_expert_grouped` entry point keeps its existing dispatch.

A proposal leaves committed recurrence and convolution state unchanged.
`commit(keep)` handles every prefix 0..T: candidate buffers are completed on
both ranks before host pointer publication. A failed execution poisons the
component, requiring reconstruction; it cannot report a successful partial
commit. Only one window is in flight. Epochs increase between proposals.

Execution modes share the same native weights and layer contract:

- `runtime` (default): the validated schedule with individual runtime copies.
- `consolidated`: one bit-preserving peer-push kernel per exchange/rank.
- `captured`: five cached compute/push graph segments per TP rank, with four
  host-submitted inter-device event joins. One GPU uses one complete graph.
- `flat`: **one complete proposal graph per rank**, with no host intervention
  between layer phases. Dedicated uncached peer inboxes, coherent mapped
  SYSTEM acquire/release epoch signals and local unpack replace event joins.
- `flat-hc`: the same flat graph plus output-row partitioning of both large HC
  matrices. Each rank computes 160 full down rows and 1280 up channels in every
  HC stream. Full N-wide normalization and four inject dots remain replicated.
  This reduces duplicate compute and weight reads, **not immutable allocations**:
  the present loader still retains full canonical HC matrices on both ranks.
  Four added small joins make this an independently measured candidate, not an
  assumed improvement over `flat`.

The flat protocol has private monotonic nonzero generations, per-phase storage,
writer-system fences, bounded steady-clock waits and a separate finite poll cap.
Abort is sticky; failed unpack clears outputs, and a failed proposal poisons the
session before any output/commit is accepted. Downstream finite scratch work may
still drain after an abort; this is not generic device-graph cancellation.
A model-free two-GPU protocol preflight must pass before model graphs are used.
Unsupported compiler/runtime primitives fail explicitly rather than falling back.

Graph preparation is startup-only. State-sensitive graphs are keyed by their
physical state bank; commit pointer swaps cannot replay stale addresses. Profiled
and unprofiled flat graphs are distinct. New profiling stamps are absent from
unprofiled compute, and all authoritative timing uses unprofiled graphs.
The one-GPU reference aliases complete output/hidden buffers, avoiding self copies.
Both rank streams complete before the synchronous proposal/commit API returns.
These are single-layer components, not a full-model generation adapter.

## Build and run

Use the existing tested ROCm image; no downloads, host ROCm installation or
server changes are performed by the runner. Both GPUs must be idle.

```sh
bash tools/run_tp2_gdn_mi50.sh EXISTING_IMAGE /absolute/model/root \
  --pack /models/PACK_DIRECTORY \
  --gguf /models/EXPERT_SHARD \
  --gguf /models/COMPANION_SHARD \
  --layer 0 --mode 8
```

The model root is mounted read-only as `/models`. Supply all GGUF shards that
contain the selected layer's dense/shared and expert tensors. Layer 1 is PLE
and refused; layers 3,7,... are QSA and also refused. Large native projection
formats must match the existing `native_mmvq`/Verifier contract; BF16 small
HC/router/alpha/beta/scalar parameters remain canonical. An unsupported dense
format fails explicitly rather than changing activation arithmetic.

Build directory: `build-tp2-gdn`. The script logs the source commit, command
arguments, resolved image ID, build output and test exit code. Its default
compiler concurrency is 48, overridable with `BUILD_JOBS`; runtime timeout is
1200 seconds, overridable with `TP2_TIMEOUT`. A timeout/error is not a pass.
The script does not stop other services or change card power settings.

## Complete-layer timing

Add `--benchmark --profile-flat` to the command above. Benchmark mode compares
captured, flat and flat-HC paths; slow runtime/consolidated timing is not repeated
by default. `--execution all` explicitly includes every mode. Ordinary invocation
without benchmark remains runtime-only. Each selected path first passes all
44 prefix/continuation cases against the runtime arithmetic reference.

Small-window timing uses matched-input AB/BA crossover at T=1,2,4,8. Each input
is tested in both execution orders. Warmup, input upload synchronization, state
reset and snapshots are outside timers; every measured pair is parity-checked.
Snapshots between pairs create gaps, so these are not continuous-inference
measurements. An additional sustained block test runs repeated complete calls
without intermediate snapshots: an untimed shadow sequence checks every step,
while timed blocks check final outputs/state. That distinction is reported
explicitly; timed intermediate states are not individually downloaded.

Host wall includes device-input staging, launches, peer exchange, layer compute,
plan-status checking, explicit accepted-prefix commit replay and completion.
Proposal and commit times are also reported separately. Both arms use
proposal+commit even at T=1, unlike optimized autoregressive self-commit in the
production Verifier. This is not a production EP or whole-model comparison.
Execution-mode blocks remain sequential, so cross-mode differences can reflect
drift. Same-mode full/TP comparisons have matched workloads and balanced order.

`--profile-flat` runs separate diagnostic graphs with same-device phase stamps
for HC, projections, GDN, router/shared/routed FFN, push, wait and unpack. Never
subtract timestamps from different GPUs or treat instrumentation overhead as
normal inference time. The runner prints actual native HC tensor types from
GGUF headers; artifact filenames are not evidence of their quantization.

The runner records read-only ROCm telemetry outside the test, source revision
and any tracked-diff hash. Idle clock snapshots do not establish loaded clock
rates. No power or clock settings are changed. No timing is accepted if the
associated parity gate fails.

## Gates and reference independence

CPU-only tests validate both expert layouts, GDN/QSA ownership, native row
span reconstruction, invalid inputs/overflow, and explicit-phase options and
hidden-Q8 scratch offsets. Checks remain active under Release/NDEBUG.

The GPU runner loads one unsharded layer on GPU0 and rank0/rank1 shards on
GPU0/GPU1. All T=1..8 and keep=0..T give 44 proposal/commit cases, each followed
by a two-token continuation. Initial residual, recurrence and convolution are
nonzero, signed and position-varying. Intermediate outputs, exact routing,
hidden-Q8 blocks, final residual and committed state are compared. Router
margins and changing/repeated expert-ID coverage are reported.

The unsharded and TP executions share the new orchestrator, so they are not
an independent full Verifier oracle. Additional checks therefore use original
HC read/write, token-serial fused GDN, shared-expert and original full grouped
expert entry points, plus CPU HC/FFN combination formulas and actual loaded
weight-row coherence. Fixed seam tolerances are declared before hardware
execution; failures are investigated, not made green by relaxing thresholds.
Passing this gate still does not prove logits, model quality or MTP acceptance.

## Validation status

Hardware gate passed on 2026-10-10 at `67a423a89305077772589f625a2087d65f256f31`:
2 x MI50 32 GB, layer 0, mode 8, native artifact and canonical pack loaded.
The supplied stand log `tp2-gdn-20261010T014713Z.txt` reports **44 proposal/prefix
cases and 44 continuation cases, failures=0**. HIP compilation, actual weight
loading, runtime P2P exchange and GPU state/activation/routing gates therefore
passed for that revision and selected layer. This is not full-model inference,
quality approval, a generation-speed measurement, or proof for other layers.
Source log SHA-256:
`9f1210b1b4d0f3531af5b8e51ec90623e1e857a90d2daf7e9e2714dc68d8d9d6`.
It is not bundled in the repository because it includes local stand paths.
All 88 main/continuation comparisons at each TP/full numerical seam reported
zero maximum and RMS differences; router IDs, weights and hidden Q8 bytes also
matched exactly. Independent legacy/CPU checks had small nonzero differences
within their predeclared tolerances. Thus not every reference comparison was
bit-exact.

The authoring environment still has no HIP compiler/GPU. New changes after
that revision require a new hardware gate; CPU tests and declaration-only
host syntax checks cannot substitute for it.

### Captured-layer hardware result at e48e2bd

On the same two MI50s, `tp2-gdn-20261010T021055Z.txt` reports 132 prefix cases,
132 continuations and all 144 measured/36 warmup pairs passing. The SHA-256 is
`dc38727926e30381894e0498b0a187aaf965c94781ce224a55525f3306414787`.
Captured median paired full/TP speed ratios were 0.732700, 0.859802, 0.925753
and 1.083770 at T=1,2,4,8. Thus TP was slower at the first three sizes, with only
a modest T8 win; this did not meet the acceleration objective. It did not prove
host submission to be the remaining bottleneck. New flat/HC modes above are
not covered by that result and await their own hardware gate.

The remaining integration includes QSA/KV/indexer commit, PLE, full-model rank
residency, output head, prompt ingestion, MTP binding, graph capture and the
production generation loop. See `FULL_LAYER_TP_IMPLEMENTATION.md`.
