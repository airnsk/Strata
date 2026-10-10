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
or host route selection occurs. HC and router are replicated, not accelerated
by partitioning. GDN state is sharded using modulo head ownership; convolution
state uses canonical `[channel,3]` order. This component uses explicit per-call
FFN phase/mode dispatch rather than the benchmark's global phase overrides.
The legacy `native_expert_grouped` entry point keeps its existing dispatch.

A proposal leaves committed recurrence and convolution state unchanged.
`commit(keep)` handles every prefix 0..T: candidate buffers are completed on
both ranks before host pointer publication. A failed execution poisons the
component, requiring reconstruction; it cannot report a successful partial
commit. Only one window is in flight. Epochs increase between proposals.

Three explicit execution modes retain the same arithmetic:

- `runtime` (default): the validated schedule with individual runtime copies.
- `consolidated`: one bit-preserving peer-push kernel per exchange/rank. The
  routed and shared hidden-Q8 transfers share a launch. Every writing thread
  system-fences its own writes; both consumers wait for both producer events.
- `captured`: cached rank-local compute/push graphs, with inter-device event
  dependencies outside graph capture. The one-GPU reference captures its entire
  proposal in one graph; TP uses five segments per rank. State-sensitive graphs are keyed by the
  physical state bank, so commit pointer swaps cannot replay stale addresses.

Graph preparation and first-use warmup are startup-only, before the first
proposal. There is no silent fallback if graph preparation or execution fails.
Both rank streams complete before the synchronous proposal/commit API returns.
The one-GPU reference aliases its complete output/hidden buffers instead of
copying them back to itself, in all modes. This removes avoidable baseline
transport overhead; the next gate also rechecks that reference change.
These are single-layer components, not a full-model GPU-persistent scheduler.
The new peer-push and graph modes still require their own hardware parity gate;
previous runtime-copy success does not establish their visibility/capture safety.

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

Add `--benchmark` to the command above. It first runs the full correctness suite
in runtime, consolidated and captured modes: 132 prefix cases and 132
continuations. Then it alternates one-GPU full-layer and two-GPU TP calls at
T=1,2,4,8, with three warmup pairs and twelve measured pairs per execution mode.
`--bench-warmup` and `--bench-trials` override these counts. Every pair has
matched signed input and committed initial state; parity checks run after both
arms and outside their timers. `--execution all` runs the three correctness
modes without timing; the default remains runtime-only.

Reported host wall time includes device-input staging, graph/kernel launches,
peer exchange, layer compute, plan-status checking, explicit accepted-prefix
commit replay and completion. Weight loading, graph preparation, host uploads,
state reset and diagnostic snapshots are outside the timer. Proposal and
commit times are also reported separately. Both arms use proposal+commit even
at T=1, unlike an optimized autoregressive self-commit path in the production
Verifier. This is not a production EP or whole-model generation comparison.

Full/TP pairs are alternated within each mode; execution-mode blocks are
sequential. Thus full/TP ratios within a mode are paired measurements, while
runtime-vs-captured differences remain diagnostic and can reflect drift.
The runner records read-only ROCm telemetry outside the test, the source
revision and any tracked-diff hash. It does not set clocks or power caps.
No timing result is accepted if its parity gate fails.

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

The remaining integration includes QSA/KV/indexer commit, PLE, full-model rank
residency, output head, prompt ingestion, MTP binding, graph capture and the
production generation loop. See `FULL_LAYER_TP_IMPLEMENTATION.md`.
