# Full-layer TP2 implementation contract for two MI50s

Initial source review: `8cdc127`, 2026-10-10. This remains a whole-model
implementation specification, not a delivered inference mode or generation-speed
result. The opt-in harness now covers one complete non-PLE GDN layer; see
`TP2_GDN_LAYER_TEST.md` for hardware results and rejected candidates. The target here is one conversation whose layers execute on
both cards, with static shards of every expert and dense projection. It replaces
whole-expert ownership, rather than wrapping `PeerExperts` in another exchange.

The supplied artifact has N=2560, routed F=640, 48 layers, 512 experts/layer,
36 GDN layers and 12 QSA layers. Raw experts occupy 46.8384 GiB total, 23.4192
GiB/rank. Verify the manifest and every tensor's actual shape/type at startup;
these values are not an allocation budget. The target is two 32 GB, 60-CU MI50s.

Read alongside `DUAL_GPU_TP_PLAN.md`, `TP2_FFN_TEST.md`,
`MI50_SCALING_REAUDIT.md`, `AMD_HIP.md`, and `MULTI_GPU.md`. Earlier routed-only
negative measurements do not settle whole-layer performance. They also do not
justify promising a 30–50% or 2x whole-generation improvement.

## 1. Actual integration constraints

The useful mathematical boundaries already exist in
`src/core/verify.cpp::Verifier::record_window`:

1. HC read produces `mixed_` and injection coefficients.
2. GDN/QSA produces mixer output `bo_`.
3. HC write/read produces the FFN input `mixed_`.
4. Router produces ordered `ids_`, normalized `w_`.
5. Shared and routed FFNs produce `shared_` and expert `parts_`.
6. Combine produces `bo_`; the next HC write consumes it.
7. Final HC read and native head produce logits.

These are not currently rank-callable interfaces. `pre` and `post` are local
lambdas closing over verifier-owned buffers, a single table and one session.
`Verifier::init` explicitly requires `VerifyHits.d_res`, `cache_base` and a full
expert blob size. `resident_plan` expects full-expert storage. Its all-resident
fast path means primary-local residency, not collective two-card residency.
`Verifier::run` services host expert doorbells when that condition fails.

`WeightRef.native_data/native_q8_1` belong to one `NativeDense`; its scratch is
shared on one ordered session stream. `NativeDense::load` can filter layers,
not rows/columns or tensor ranks. `WeightTable` has a private map and records
canonical plane sizes and source metadata. Changing `ne0/ne1` on a copied
reference is not a loader or a valid shard descriptor.

`NativeExpertLayout` ties GU input width and down output width to `n_embd`, and
GU output width and down input width to `n_ff`. The literal-row layout below
cannot be passed as `native_expert_layout(N/2,F/2)`. The bench-only
`native_expert_set_mode(mode, phase)` is process-global dispatch state, not a
production per-rank phase API. Explicit GU and down APIs are required.

`SessionState` and `Verifier::capture_commit` derive recurrence, convolution,
KV and snapshot sizes from one `ModelGeometry`. Layer-range carve support is
not head-sharding support. Passing a half-sized geometry to an ordinary
Verifier would also change unrelated offsets and PLE/HC assumptions.

`Prefill::set_peer` and its peer streaming consume whole experts. They cannot
prefill a half-expert cache. `MtpDrafter::bind` expects a full `NativeHead` and
final `T x (HC*N)` residuals on its own device. These are mandatory integration
work, not incidental compatibility details.

## 2. Ownership and rank contracts

Use a dedicated `Tp2Session`/`Tp2Executor`, not two ordinary Verifiers. Preserve
the existing classes as the reference/fallback implementation. Both ranks own:

- Static shards for all layers and all experts, with device pointer tables
- Rank-local compute/exchange streams, events, quantization buffers and scratch
- A full replicated `R[T,HC,N]` and `mixed[T,N]` at layer boundaries initially
- Local GDN head state and local QSA KV heads
- Their own graph executables, immutable weight descriptors and error status

Replicated small work is acceptable in the first symmetric executor. It must
be described honestly: HC/router duplication is not HC/router acceleration.
Neither rank should invoke the CPU expert pool, wait for its planning doorbell,
or own a permanently serial dense/mixer/FFN stage.

All buffers carry explicit row width, row stride, capacity, device and ownership.
Each exchange has `(window_epoch, layer, phase, token_count)` identity. Expert
entries use the same `(token, top-k position, expert id)` order on both ranks.
Local pointers must never be copied as usable remote-local addresses.

### Current acceleration candidate: input-column reductions

The measured captured/event row baseline is retained. The new column-owned
GDN-layer candidate pairs local GU output rows with down input columns:

- Local GDN Y3072 feeds matching mapped columns of `ssm_out`; every rank
  produces all N2560 output partials, followed by one ordered rank0+rank1 sum.
- Local F320 hidden blocks feed full-N routed/shared down partials. Routed down,
  ordered expert weighting, shared contribution and peer publication are fused
  without materializing routed `[T,K,N]` parts in the timed path.
- Each producer publishes to a separate peer inbox for its sublayer. Captured
  compute segments use ordinary event dependencies, with no mapped-host polling.
- There are two output reductions and three captured compute segments per rank.
  HC remains replicated; the rejected HC split is not a prerequisite.

Native weight blocks remain unchanged. The partition changes floating-point
association: column-mode engineering gates use predeclared numerical bounds
against the original full reference, exact ordered router IDs, exact candidate-
same-input GU/Q8 checks and independent FFN reconstruction. Original-reference
hidden-byte differences are diagnostic, never relabelled exact passes. Original
row-mode exact gates remain intact. No quality or speed conclusion follows until
the corresponding hardware and full-model validation passes.

### Reference routed FFN: literal output-row partition

For rank r in {0,1}, source matrices are GU `[F,N]`, down `[N,F]`:

- Gate/up rows `[r*F/2,(r+1)*F/2)` retain the full N input.
- Down rows `[r*N/2,(r+1)*N/2)` retain all F inputs.
- Compute local SwiGLU `[entry,F/2]`, quantize it with the existing Q8_1
  contract, then all-gather hidden blocks into `[entry,F]` on both ranks.
- Down computes `[entry,N/2]`. Combine the K experts in the original top-k
  order, using original `w`, adding the locally owned shared output once.
- All-gather combined `[token,N/2]` into replicated `[token,N]` `bo`.

This avoids a partial-dot all-reduce and retains the original full-F down dot.
For F=640, each 320-element boundary is ten Q8_1 blocks; concatenating those
blocks is valid. Byte-for-byte hidden-Q8 parity is still a gate, not an
assumption about a new GU dispatch. No dequantize/requantize of expert weights.
The two shards together must reconstruct every original byte.

The GU/down dispatch needs independent dimensions and pitches:
`input_N`, `local_F`, `full_F`, `local_output_N`, `hidden_stride`,
`output_stride`, explicit gate/up/down pointers and entry metadata. A new
shard format must not masquerade as a full `ExpertCache` slot. Retain both row and column ownership as explicit immutable descriptors; neither
may silently reinterpret the other's dimensions or byte strides.

At T=8,K=10, each rank sends 28,800 hidden-Q8 bytes and 40,960 combined-output
bytes per FFN, excluding protocol overhead. For T=1 the totals are 3,600 and
5,120 bytes. Count latency and launches as well as bandwidth. Do not gather
K full N-vectors when each rank can first combine its own complete output rows.

### Shared expert

Read and validate `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp` separately;
do not infer their types from routed experts. Current verifier uses `g.n_ff`
for shared scratch, so enforce the supported shared shape explicitly.
Use the same row scheme with its actual F: local GU, hidden all-gather,
local down output. Replicate `ffn_gate_inp_shexp` and its scalar sigmoid or
compute it once and broadcast. Apply that scalar exactly once before addition;
it is not router-weighted or renormalized with the selected experts.

`shared_expert_multi` currently has a full-FFN contract. Factor explicit native
GU/SwiGLU/quantize and down/gate phases, preserving native BF16 gate behavior,
Q8_1 activation choice, `lfuse` semantics and existing rounding. Initially
turn off the shared side-stream fork; restore it only with separate scratch
and measured rank-local dependencies. A successful routed-only test does not
validate the shared path.

### HC and residuals

First replicate HC parameters and run existing `fused_gr_read_multi` and
`gr_write_multi` on both ranks after identical complete `bo` gathers. Preserve
pending-write ordering, injection coefficients, full N-wide RMS normalization independently in each of the HC streams,
and final output HC read. Assert replica equality at selected layer boundaries.
This uses existing validated HC code and eliminates primary-to-peer input
broadcasts from the steady layer loop, while leaving HC cost undivided.

Actual HC sharding is a separate optimization with a concrete contract:
shard residual channels within every HC stream, reduce the N-wide RMS sum separately within each HC stream,
compute low-rank down partials and reduce them, then replicate small low-rank
outputs/mixing coefficients and perform local up/inject/residual writes. This
needs rank-aware GR kernels and changes floating-point association. Do not
substitute local-half RMSNorm or halve `n_embd` in the existing fused kernel.
No persistent HC change is a prerequisite for the full-layer vertical slice.

### Router

Start by replicating BF16 router and top-k on identical `mixed`; require bitwise
identical ordered IDs/weights. This is device-owned, with no host planner.
Build the same group order on both ranks and resolve it against their respective
shard pointer tables. Reject ownership-biased `STRATA_ROUTE_RESIDENT` for TP.

To partition router compute later, shard its 512 output rows, all-gather all
512 logits, and apply the original top-k and selected-weight normalization to
that full vector. Local top-10 followed by a different tie-break/normalization
is not the original router. Existing `native_router_top10_multi` specializes
512 experts/K=10; preserve the generic path for any admitted other geometry.

### GDN: head mapping must preserve modulo groups

`verify_kernels.cu::gdn_step_norm_multi_kernel` uses
`qh = value_head % h_k`, not contiguous groups. With h_k=16,h_v=48:

- rank r owns K/Q heads `8*r+j`, j=0..7
- its local V head `v=8*b+j` maps to global `16*b+8*r+j`, b=0..2
- local h_k=8,h_v=24 preserves `v % 8 = j`

A contiguous split V=0..23/24..47 with K=0..7/8..15 is WRONG. Pack local
QKV as `[Q eight heads, K eight heads, V mapped twenty-four heads]`, for
5120 channels. Gather source weight row ranges; do not blindly bisect the
10240 concatenated QKV rows. Apply the same V mapping to `attn_gate` output
rows, `ssm_alpha/beta` rows, `ssm_dt.bias`, `ssm_a` and convolution channels.
Replicate the shared 128-element `ssm_norm` after validating its source shape.

State is addressed as `[state_row, value_head, state_col]` in the actual
kernel, with 128 x h_v x 128 floats, not a contiguous head-major slice.
Pack/unpack its head axis for checkpoint conversion. Conv history uses the
local QKV channel order and three history taps. `gdn_conv_l2_multi`,
`gdn_ab_multi`, `gdn_step_norm_multi`, `gdn_conv_commit` expose dynamic head
counts and are the first reusable kernels; test h_k=8,h_v=24 explicitly.
Their fixed state width 128 and conv width four remain requirements.

First output projection: all-gather local 3072-element GDN outputs into
original global head order, then each rank computes N/2 complete output rows
of `ssm_out` with full 6144 input. All-gather N/2 outputs for HC. This retains
full-dot arithmetic and avoids column-block boundary problems for the
noncontiguous V mapping. Quantize only after restoring original global order,
or scatter independently quantized complete Q8_1 blocks into that order.

The current GDN column candidate instead shards `ssm_out`, repacks the matching
mapped input blocks, and reduces N partial outputs. Every gathered 128-value segment
must align with that weight format's native block; 256-value formats cannot
be split/reordered on arbitrary single-head boundaries. Validate block
alignment or group heads accordingly. Measure the alternative separately.

### QSA and sparse indexer

QSA uses contiguous grouped-query mapping, unlike GDN. Rank r owns query
heads `12*r..12*r+11` and KV head r, with HD=256. Query source is interleaved
per head `[query256, gate256]`, as `record_window`'s strided copy shows; copy
whole 512-row head units, not separate matrix halves guessed from names.

Use local `QsaShapes.n_head=12,n_head_kv=1`, unchanged HD, indexer dimensions,
position and selection limits. `qsa_decode_attn.cu` accepts G=12 and dynamic
KV head count, so its shape guard admits this grouping; this is not GPU parity
proof. Each rank has local KV pages, scales/rotations and host mirrors when
that format is supported. Logical page/cell indices and committed length
must match. Q4, INT8, hybrid and elastic modes require separate coverage.

Replicate the indexer initially: all four query heads and the ONE shared
128-wide cached index key. Do not halve indexer head count with attention
head count. `qsa_select.cu` has fixed indexer shape checks. Replicate pooled
keys, tail/dead/block_pos and run identical selection, or compute it once and
broadcast selected cells; both attention shards must use the same selection.

Gather the gated attention output in original 24-head order; row-shard
`attn_output` into N/2 rows with full 6144 input, then gather `bo` for HC.
This is the QSA equivalent of the literal GDN output path. A later
column-sharded output/all-reduce is optional after numerical validation.

### Output head, PLE and MTP

Shard `output.weight` by vocabulary rows. Both ranks consume the same final
HC mixed N-vector. Greedy decoding can reduce rank-local `(max,id)` pairs with
the original tie-breaking rule. The first general path gathers full logits to
rank 0 and uses the existing sampler/logprob code: penalties, constrained
sampling and exact vocabulary probabilities must not silently change.

`NativeHead::load` currently loads the whole matrix and its `weights()` is
consumed directly by Verifier/MTP. Add a distinct `Tp2Head` row-range loader;
do not change what a `NativeHead` pointer means. Global vocabulary offsets
must be explicit in every local result and subset-head lookup.

PLE is part of the full-layer contract. Initially replicate its small
projections/history and broadcast or identically gather its host-resident
embedding inputs; apply PLE at the same layer boundary and maintain both
histories. Current NG_HC_DIM/NG_HIST and `ss.ple` paths assume full geometry.
A primary-only PLE bridge is a clearly labeled temporary baseline, not the
completed symmetric executor.

Keep MTP on rank 0 for the first runnable generation mode, with its full
residual input available there. This leaves MTP latency and extra memory
asymmetric; report both. Before freeing the original full head, give MTP
its explicitly owned draft-vocabulary rows loaded from source, or implement
a TP head adapter. `MtpDrafter::bind` cannot consume half a `NativeHead`.
The later TP drafter owns local attention/KV and row-sharded projections,
with identical draft sampling/acceptance and shared global token IDs.
Do not hold a duplicate full main head merely to conceal this dependency.

## 3. Synchronization, capture and error contract

First build a correct uncaptured two-stream layer executor, then capture its
rank-local compute segments. It must use direct device P2P transfer, not
`PeerExperts::launch`'s host activation/metadata staging and mapped-host rows.
Enable and verify access in both directions before allocation/capture.

For each exchange: enqueue producer compute, record producer-ready, have the
copy stream wait, copy into the receiving rank's stable slot, record arrival,
and make the consumer stream wait before reading. The reverse direction is
independent; do not put each rank's producer behind the other rank's consumer.
An exchange slot is reusable only after its previous consumer completes.
All-gather means disjoint destination ranges, never concurrent writes to the
same byte. Initial implementation uses one in-flight window and stable buffers.

HIP graph behavior must be tested on the actual runtime. Do not assume a CUDA
multi-device graph can simply be captured under HIP, or that event reuse in two
independently replayed graphs automatically refers to the intended epoch.
Start with explicitly orchestrated rank-local segments/events outside capture.
Then test supported cross-device event edges and graph replay before reducing
host submissions. This correctness fallback may be slow and must be timed as
such. A persistent peer signal protocol is a later implementation, requiring
system-scope release/acquire, an epoch, cancellation and bounded failure paths;
a volatile peer spin flag alone is insufficient.

Destroy graphs after draining/releasing waits, on their owning device, before
freeing weights/scratch/events. Check errors on both devices and release both
sides on cancellation. A failed rank aborts the window; do not commit a prefix
on only one rank. Use a watchdog for visibility/capture tests. No token-path
allocations, capture-time memory sizing, hot-cache mutation or adaptive swaps.

## 4. Prefill, state and commit are part of the first runnable mode

Build `Tp2Session::commit(keep)` alongside the first speculative window:
GDN replays the same accepted prefix on both local state shards and advances
local conv history; QSA restores indexer tails then appends only committed
indexer rows. Proposed KV cells may remain physically written, but committed
length/selection must exclude rejected rows identically. Disable the one-token
self-commit optimization initially so one commit protocol covers all cases.
Test keep=0..T, rollback and continuation against the reference state.

Prompt ingestion must use the TP executor too. An initially slow token-wise
prefill through the same T=1 full-layer path is a valid functional bridge; it
avoids needing an additional full-expert cache or converting half-built state.
Report prompt latency separately. Do not run existing `Prefill` against shards.
Batch/chunk TP prefill comes after the first end-to-end generation result.

Disable conversation-cache save/restore until an explicit TP state adapter
serializes GDN's mapped head axis and KV heads, replicated indexer/PLE state,
position and MTP state. Existing `conversation_snapshot.cpp`,
`conversation_state.cpp`, and size checks in `conversation_snapshot.hpp` are
integration points. Either store a tagged rank layout or reconstruct canonical
state; never load a single-device byte snapshot into a half-state arena.

## 5. Memory planning and selection/fallback

Add a header-only metadata plan before uploads, then compare requested and
actual free/device allocations on each card. Include expert shard arena and
pointer table, dense/shared/head shards, replicated HC/router/indexer/PLE,
local KV and host mirrors, GDN state, proposal/commit snapshots, exchange
buffers, graph allocations, MTP rank-0 allocation and an explicit reserve.
Log payload and allocator-rounded bytes separately. `NativeDense`'s existing
2 MiB rounding heuristic was measured on a different card; actual MI50
allocation deltas remain necessary.

Load directly into shard arenas. Do not allocate 46.84 GiB full experts plus
23.42 GiB shards/rank, retain discarded full dense matrices, or let expert
cache auto-sizing consume graph/KV reserve. First version requires all expert
shards resident and fixed context capacity. No elastic borrowing from static
TP shard arenas or prefill loans.

Keep `STRATA_TP2_BUILD=OFF` for existing tests; use an additional explicit
production mode gate when the executor exists, e.g. `--tensor-parallel 2`.
An explicit TP request must fail with a precise startup reason for unsupported
geometry/type, missing bidirectional P2P, insufficient memory or incompatible
options. Never silently reinterpret it as layer split or whole-expert peer mode.
Without the flag, execution/allocation/weights/capture remain unchanged.
A diagnostic per-op fallback may run on one rank then broadcast only when
implemented and logged; it cannot be called a fully symmetric result.

First admitted mode: one conversation, two gfx906 devices, static fully
resident native expert shards, greedy, bounded fixed KV format/context,
no pipeline windows, batch slots, adaptive caches, helper devices, residency
routing, conversation cache or coupled sampling. Add compatibility one feature
at a time, preserving the unmodified normal CLI path.

## 6. Exact integration sequence and deliverables

The next milestone is a complete layer, not another indefinite routed-FFN
benchmark series. Every stage below produces reusable production components.

1. Implemented CPU-only foundation: `include/strata/core/tp_layer_layout.hpp`
   and `tests/core/tp_layer_layout_test.cpp`: pure GDN/QSA
   global/local mapping, full source-row ownership, recurrence-state indexing,
   and output reconstruction. Reject unsupported geometry and test bijections
   independently. No runtime changes or link dependency on GPU code.
2. Proposed `tp2_weights.hpp/.cpp`: validated immutable per-rank tensor/shard
   descriptors, memory plan and direct source-to-shard loading. Reuse block
   readers and CPU shard validation; do not mutate `WeightTable` internals.
3. Explicit GU/down rank APIs in `iq_kernels.hpp/.cu` and shared phase APIs in
   `shared_expert.hpp/.cu`; reuse quantization/vecdot code. Literal-row FFN
   worker supplies the first kernel test. No process-global phase switching.
4. Proposed `tp2_exchange.hpp/.cpp` and `tp2_session.hpp/.cpp`: rank ownership,
   stable buffers, two-way event/copy DAG, local state, commit, error cleanup.
5. Proposed `tp2_executor.hpp/.cpp`: first real GDN layer vertical slice.
   Use an actual layer's HC parameters, all its dense/shared/routed weights,
   modulo-mapped GDN state, device router/groups, both FFN phases and final
   HC write. Inputs include arbitrary nonzero R/state/conv, T=1 and 2..8;
   output includes R and committed state, not just FFN floats. Compare every
   seam against the existing one-device layer/window reference. This is the
   first complete test of useful symmetric layer execution.
6. Add one actual QSA layer with local KV, replicated indexer, head mapping,
   sparse selection and commit. Cover empty and nonempty prior context,
   block/page boundaries and a rejected draft prefix.
7. Connect all 48 layers, PLE, row-sharded head and token-wise prompt ingestion.
   In `src/program/generate.cpp`, choose a dedicated TP startup/execution branch
   before ordinary `NativeDense`, session, head and ExpertCache allocations.
   Keep the ordinary `Verifier` path untouched. Deliver greedy generation and
   logits/state comparisons as the first production-capable opt-in artifact.
8. Add MTP binding/accepted-prefix protocol using rank-0-owned draft weights
   and subset head. Integrate existing generation-loop acceptance/sampling
   semantics; replace its concrete Verifier assumptions with a small explicit
   adapter only where needed. Compare acceptance and accepted tokens/window.
9. Capture the complete rank-local execution path and optimize its exchanges;
   measure whole-generation gain with the real baseline and equal residency.
   Optimize HC/router/PLE/MTP remaining serial or duplicated fractions based
   on that trace, rather than declaring the routed-only benchmark the goal.

A minimal safe full production integration is not a one-line extension of
`PeerExperts` or `Verifier::init`. Steps 1–6 can be implemented as isolated
opt-in components; step 7 is the first runnable full-model result. Keep work
moving toward that vertical slice even if a generic isolated routed kernel
loses to balanced EP, while rejecting arithmetic/state errors immediately.

## 7. Acceptance and performance evidence

CPU gates: every source head/channel is owned once, raw weight reconstruction,
checked byte arithmetic, independent expected GDN modulo mapping, recurrence
state and QSA interleaved-query reconstruction, unsupported-shape rejection.
GPU gates: actual formats, changing routes/repeated IDs, finite outputs,
predeclared errors, replica equality, hidden Q8 parity, all keep counts,
state continuation and repeated capture replay with alternating T and epochs.

Compare same-day opt-in TP and existing two-card EP with identical prompts,
context, power/clock settings, output length and expert residency. Report
prompt latency, wall time to a fixed number of accepted tokens, accepted
tokens/window, MTP time, rank compute/communication/host wait, launch count,
peak allocation and remaining free memory. Remove diagnostic synchronizations
from timing. Interleave run order. Output-head, routed-only or one-layer wins
are intermediate measurements, not a whole-generation claim.

If a run is slower, retain its verified correctness result and use the full
critical-path trace to identify the next concrete fix. Stop or narrow support
for wrong results, unbounded waits or impossible residency; do not conceal
those behind fallback or change defaults on an unmeasured hypothesis.

## Local foundation validation

The strict target-geometry CPU layout test compiles with C++20 and
`-Wall -Wextra -Werror -pedantic` and passes. AddressSanitizer/UBSan also pass
with leak detection disabled; this executor's ptrace environment prevents
LeakSanitizer from running. No GPU compiler/device or actual model payload
was used for this layout test. It establishes ownership/index contracts, not
kernel parity, allocation fit, graph safety or inference performance.
