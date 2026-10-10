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
2. GDN projection with explicit output-row gathering or input-column reduction
3. HC write/read and replicated GPU router
4. GPU resident expert grouping (all experts have local static shards)
5. Shared and routed FFNs with either gathered hidden/output rows or local-input down/combine partials
6. Full output reconstruction and final HC residual write

The output-row reference uses full-width hidden input for down. The column
candidate keeps local hidden input and reduces full-width output partials. No
CPU expert-pool planning or host route selection occurs. In the baseline modes HC and router are
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
- `column`: native input-column weights, three captured compute segments per
  TP rank and two event-ordered output reductions. GDN Y and FFN hidden stay
  local. Routed down, ordered weighting, shared contribution and peer publication
  are fused. Dedicated attention/FFN peer inboxes avoid overwrite races. No
  runtime peer-copy calls or mapped-host flag polling occur between segments.
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

Add `--benchmark --profile-stages` to the command above. Benchmark mode selects
the captured output-row reference and column candidate; rejected flat/flat-HC
modes are not repeated by default. `--execution all` explicitly includes every
mode. Ordinary invocation without benchmark remains runtime-only. Each selected
path must first pass all 44 prefix/continuation cases against the original full
reference under its explicitly declared numerical contract below.

Small-window timing uses matched-input AB/BA crossover at T=1,2,4,5,8. Each input
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
The new default compares full single-GPU versus column TP, then captured-row TP
versus column TP directly, each with matched workloads and balanced order. Arm
labels identify the comparator; do not read a TP-versus-TP ratio as scaling
against one GPU. Historical mode-separated studies remain explicitly labelled.

`--profile-stages` (`--profile-flat` remains an alias) runs separate diagnostic
graphs for the selected captured modes, with same-device phase stamps for HC,
projections, GDN, router/shared/routed FFN and collective boundaries. Inter-
segment gaps include host arrival, event dependency and peer imbalance; they
are not pure transfer times. Never
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

### Column-mode numerical contract

Column reductions change floating-point association. Row-mode exact gates stay
unchanged; column mode is a distinct predeclared engineering experiment:

- Original full-reference seam, final residual, state and continuation numeric
  bounds are retained. Ordered router IDs must match exactly. Router logits and
  normalized weights use the existing HC numeric bounds when their inputs differ;
  identical inputs retain exact checks. No threshold is widened after a failure.
- The candidate's actual FFN input is also passed to an independent original
  full-weight FFN oracle. GU/hidden Q8 ownership is byte-exact against that
  same-input oracle, and fused output is compared numerically to its full result.
- Original-input hidden byte differences are explicitly diagnostic, never claimed
  as an exact pass. Local fused versus materialized down/combine checks separately
  isolate fusion from rank-sum reassociation.
- Both ranks still produce identical replicated boundaries using the same ordered
  rank0+rank1 addition. Diagnostic snapshot reconstruction happens outside timing.

Changing this contract does not establish model quality; end-to-end logits,
routing and MTP acceptance still require later full-model validation.

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

### Flat graph hardware result at 8f4ef54

`tp2-gdn-20261010T150019Z.txt` validates the actual HIP build and both flat
protocol paths on the same two MI50s. All 132 prefix cases, 132 continuations,
benchmark parity and missing/late-rank rejection gates passed. Source SHA-256:
`b4e89698ca7d50a0ae06acdd1578b2f214c981e14281dc6657edb6cc56f60999`.

The performance decision is **reject both flat candidates**. Sustained crossover
full/TP ratios at T=1,2,4,8 were:

| Mode | T1 | T2 | T4 | T8 |
|---|---:|---:|---:|---:|
| captured | 0.875 | 0.962 | 1.113 | 1.272 |
| flat | 0.807 | 0.860 | 0.954 | 1.045 |
| flat-hc | 0.674 | 0.737 | 0.824 | 0.930 |

At T8, median TP wall/call was 1.273189, 1.554211 and 1.750460 ms respectively.
These are one-layer sustained 16-call fixtures, not model tokens/s. Mode blocks
were sequential, so cross-mode comparisons are diagnostic. Same-mode full/TP
comparisons use identical input sequences in both orders.

Separate instrumented replays expose significant exchange and HC overhead.
Successful zero-retry protocol waits still execute multiple SYSTEM atomics on
mapped host control; push/unpack also repeat mapped control checks per block.
Thus reducing graph submissions introduced additional PCIe control accesses.
These reads are a source-level cost, not a causal timing decomposition of the
entire regression. HC sharding also adds four exchange boundaries; in the T8
rank0 diagnostic its approximately 54 us arithmetic saving is outweighed by
approximately 320 us of added HC exchange spans. All 48 layers' supplied HC
projection/injection tensors are actually BF16, so a Q8 HC path would require
new quantization rather than recovering an original representation.

The captured/event schedule remains the performance baseline. The next candidate
changes ownership to local-input projection/down partials and two output
reductions, without either intermediate Y or hidden all-gather. It retains
replicated HC and native weights. It must establish its own numerical and
whole-layer timing evidence; altered FP32 association is not bitwise equivalence.

The remaining integration includes QSA/KV/indexer commit, PLE, full-model rank
residency, output head, prompt ingestion, MTP binding, graph capture and the
production generation loop. See `FULL_LAYER_TP_IMPLEMENTATION.md`.
