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

This first implementation is an **uncaptured correctness schedule**. Copy
streams use explicit producer/arrival events and runtime P2P copies, including
small strided-entry copies. It is not the final low-launch-overhead exchange
implementation. Proposals finish synchronously; this is not a full-model GPU
persistent scheduler. Do not benchmark these host-submitted copies as proof of
the eventual TP architecture's speed limit.

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

At authoring time: strict CPU Release tests, declaration-only host C++ syntax
checks and source review are available. No HIP compiler/GPU exists in the
authoring environment. HIP compilation, actual artifact loading, runtime P2P
events, memory fit and all GPU parity gates await the stand. Previous routed
FFN hardware success at `05695a9` does not validate this new whole-layer code.

The remaining integration includes QSA/KV/indexer commit, PLE, full-model rank
residency, output head, prompt ingestion, MTP binding, graph capture and the
production generation loop. See `FULL_LAYER_TP_IMPLEMENTATION.md`.
