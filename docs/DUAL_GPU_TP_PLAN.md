# Two MI50s: tensor-parallel feasibility and first-layer plan

Draft decision document, 2026-10-09. Source review: Strata `6c32b09` and Castagna Veloce
`3483d462715d61f21ed5877836be35a724446359`. No TP implementation or hardware result is
claimed here. The exact running configuration and per-layer weight metadata are still needed.

Target hardware reported by the owner: two MI50 32 GB, gfx906, 60 CUs/card, 150 W/card;
EPYC 7K62, 512 GB RAM; measured PCIe P2P 27.84 GB/s one-way, about 55 GB/s aggregate.
Those bandwidth figures do not establish the latency of a graph-captured small exchange.

## Decision before coding

Prototype a complete routed-plus-shared FFN split, then a complete GDN layer. Do not
start with an isolated output-head optimization or copy the donor's persistent FFN
kernel without adapting its quantization contract. Keep existing HC experiments unchanged.

**Existing expert parallelism is already parallel compute.** `PeerExperts::launch`
(`src/core/peer_experts.cpp`) launches `native_expert_grouped` on the peer; the primary
runs its own selected experts. With balanced ownership, each card already does roughly
half the total expert work. Changing this into two half-width shards of every expert
DOES NOT halve that balanced two-card FFN time again. Potential benefits are better
balance for a small routed set, eliminating ownership-dependent hot-card bottlenecks,
and a common layout for subsequent dense/attention TP. Communication and smaller
matrices can erase those benefits.

For expert costs c_i, a simplified EP compute bound is
`max(sum(c_i on GPU0), sum(c_i on GPU1))`; ideal TP is `sum(c_i)/2`, before its own
communication and kernel-efficiency costs. A balanced EP layer already meets that
idealized compute bound. Measure actual per-card times and ownership per layer/window.

The 30–50% whole-decode throughput goal requires a broader critical-path improvement.
Even ideal 2x acceleration of a currently serial fraction f gives only
`1 / (1 - f/2)`: 1.3x needs f >= 46.2%; 1.5x needs f >= 66.7%, before overhead.
This calculation must not count the existing parallel expert work as wholly serial.
Dense projections, GDN/QSA work, waits, routing, HC/PLE and MTP need a same-day profile.
No speed prediction follows from the donor's four-card headline numbers.

## Existing alternatives and interfaces

- `src/core/peer_experts.cpp`, `include/strata/core/peer_experts.hpp`: whole-expert tier.
  The reviewed `launch` implementation stages its input from a host pointer and can
  scatter results into portable mapped host rows. Its `open` also enables P2P, and
  `src/prefill/prefill.cpp` uses the peer's P2P capability. This is not a claim that the
  owner's entire b4 execution path, every peer transfer, or every helper path uses host
  staging. `remote_experts.cpp`/`remote_expert_opt.cu` are separate helper paths; resolve
  the actual configuration before attributing communication costs.
- `src/program/generate.cpp:3015+` and `9991+`, `include/strata/core/verify.hpp:201+`:
  `--pipeline-windows 2` already overlaps stage 0/window K+1 with stage 1/window K for
  one conversation. It guesses full acceptance and the bonus token, snapshots GDN
  state, and restores/replays the real commit when wrong. It is speculative layer
  overlap, not a split of one matrix.
- Pipeline prerequisites: `--serve`, exactly two real GPU stages, MTP and spec >= 2;
  no peer-device, helper caches or batch slots. Repetition penalties and coupled
  sampling use serial decode. Reserve includes 160 MiB/card for the second verifier
  plus two GDN snapshots on the first card. Compare equal expert residency.
- `src/core/verify.cpp:1305–1550`: useful FFN boundary is `mixed_`, routing IDs/weights,
  and selected expert pointers through shared/routed FFNs to `bo_`, before HC write.
- `include/strata/core/weights.hpp`: `WeightRef` carries
  native data/type and a shared single-stream scratch pointer. It is not already a
  two-device weight or scratch owner.

## First TP subgraph: complete FFN

Subject to confirming N=2560, F=640 and each layer's formats:

1. Each rank owns 320 complete gate rows and the same 320 up rows of EVERY expert.
2. Its down matrix contains those 320 input columns of every output row. Copy complete
   raw quant blocks into a new contiguous shard; do not dequantize/requantize weights.
3. Both ranks consume the same input and ordered selected IDs/weights. Each computes
   every selected expert's local shard, including a correspondingly sharded shared FFN.
4. Reduce partial outputs in fixed rank order before the primary's HC consumes `bo_`.
   First validate per-expert partial reduction followed by the original expert combine.
   Reducing rank-local weighted expert sums is a later fusion with an additional
   floating-point association change and its own acceptance test.

`native_expert_layout` and generic `native_expert_grouped` in
`src/kernels/cuda/iq_kernels.cu` express N=2560,F=320. This is an API/layout observation,
not proof of every optimized dispatch's correctness or speed at that width. IQ1_M is
supported by generic GU dispatch but excluded from current fused mode8 GU dispatch.

320 aligns Q2_0 down blocks (64) and IQ4_NL down blocks (32). It does not align a
256-element down type. The existing native expert contract already requires F to be
an exact multiple of its down block size, so F=640 would also reject those types
unsplit. Do not invent padding or use an asymmetric split to conceal invalid rows.
Use the actual manifest and GGUF shapes; a pack named IQ3 need not have IQ3 down weights.

The donor is not an IQ1/IQ3 full-FFN implementation ready to paste:
[hc-persist.cu:1570–1585](https://github.com/benpeterson40/castagna-veloce/blob/3483d462/ggml/src/ggml-cuda/hc-persist.cu#L1570-L1585)
accepts GU Q4_K/Q5_K/Q8_0 and down Q5_1/Q8_0. Its exchange and fusion architecture
can inform an implementation while retaining Strata's validated IQ decoding.

## Ownership, build gate and capture

Proposed files, not implemented:

- `include/strata/core/tp2_weights.hpp`, `src/core/tp2_weights.cpp`: block-aligned shard
  plans, per-rank ownership, expert pointer tables, exact device-memory budget.
- `include/strata/core/tp2_exchange.hpp`, `src/core/tp2_exchange.cpp`: input/routing
  transfer, output join, device/stream/event lifetimes.
- Later `include/strata/kernels/tp2_exchange.hpp`, `src/kernels/cuda/tp2_exchange.cu`:
  graph-safe exchange/reduction if separate-graph measurements justify it.
- `tests/hip/tp2_ffn.cpp`: model-free full-FFN parity, replay and timing gate.
- Eventual integration: `verify.hpp/cpp`, `expert_source.cpp`, `generate.cpp`; explicit
  separate shard ownership rather than overloading ordinary full ExpertCache blobs.

Proposed `STRATA_TP2_BUILD=OFF`, initially gfx906-only, with runtime opt-in. Build-off
must add no device code, allocation, weight changes or capture changes. Unsupported
formats/memory layouts must produce a clear refusal before loading, not silent fallback
that drops a shard. Start with static, fully resident expert shards, one conversation,
without simultaneous pipeline/batch/adaptive mutation. Replace full expert storage;
keeping original blobs plus shards would give misleading memory results.

First hardware harness: two rank-local graphs with explicit orchestration and measured
transfer overhead. Do not claim an uncaptured subgraph win proves captured decode wins.
Before production capture, validate cross-device dependencies, producer memory visibility,
replay epochs, buffer reuse, alternating window sizes, cleanup, cancellation and failures.
A stuck peer must not leave an unbounded wait. Do not assume CUDA capture behavior proves
HIP 7.2.4 behavior. Per-rank scratch cannot alias existing shared projection scratch.

FFN-only leaves primary KV, GDN state, PLE and MTP unchanged. Whole-layer TP is the next
substantial step, not an automatic consequence of the FFN test.

## Route to a complete layer

`ModelGeometry` in `include/strata/core/layout.hpp` currently specifies 36 GDN layers
and 12 QSA layers. Confirm against the artifact before allocating.

- GDN: split 16 K-heads into 8/rank, 48 V-heads into 24/rank. Shard Q, K and V row
  segments individually (not a blind midpoint of concatenated QKV), conv/gate parameters
  and recurrent state consistently. `ssm_out` has 3072 input columns/rank and produces
  partial 2560 outputs for reduction. Audit head-group mapping and every kernel's shape
  assumptions. Commit, rollback and snapshots must own the corresponding state shards.
- QSA: 12 query heads and one KV head/rank; shard output-projection input columns,
  maintain matching indexer selections, and adapt paged KV, streaming, checkpoints and
  restores. `verify.cpp:1075+` currently binds these to primary/session geometry.
- HC and routing can initially remain primary and distribute their outputs. Measure
  that remaining serial work. Replicated HC later requires consistent reductions and
  state; no need to port its persistent implementation as a prerequisite.
- MTP can remain primary initially. Its memory and serial latency remain in the budget.
  TP MTP is separate work involving draft weights, state, shared head ownership and
  acceptance/commit semantics. PLE host storage is not assumed to fit on either GPU.

## Illustrative expert memory, not a fit claim

Premises: exactly N=2560, F=640, 48 layers, 512 experts/layer (24,576 total), one uniform
GU/down type pair across all layers, no padding/alignment. Raw bytes/expert are
`2*F*row_bytes(GU,N) + N*row_bytes(down,F)`. GiB means 2^30 bytes.

| GU / down | Raw experts total GiB | Raw TP experts per card GiB |
| --- | ---: | ---: |
| IQ1_M / Q2_0 | 26.953125 | 13.4765625 |
| IQ1_M / IQ4_NL | 37.500000 | 18.750000 |
| IQ3_XXS / Q2_0 | 39.2578125 | 19.62890625 |
| IQ3_XXS / IQ4_NL | 49.8046875 | 24.90234375 |
| IQ3_S / Q2_0 | 42.7734375 | 21.38671875 |
| IQ3_S / IQ4_NL | 53.3203125 | 26.66015625 |

Block sizes are in `tools/gguf_reader.py`. Mixed layers require summing actual per-layer
blob sizes. This excludes allocation granules, dense/shared weights, HC/PLE, MTP, KV,
activation/scratch/graph buffers, prefill borrowing and reserve. In the final illustrative
case only about 5.34 GiB/card remains before those allocations. Having all experts across
an EP pair does not establish a full TP runtime fits. Replicating all expert weights on
both cards is not the proposed design.

## Acceptance sequence and stop conditions

1. Metadata and CPU shard reconstruction: exact raw bytes reassemble; all boundaries
   align; both device budgets fit including reserved runtime memory.
2. Model-free two-card complete FFN: IQ1_M/IQ3_XXS/IQ3_S GU with supported actual down
   types; T=1..8, repeated/scattered IDs, skewed ownership, unused/empty groups.
3. Compare full-width output against TP with finite-value, maximum absolute, relative
   (with a near-zero floor), and normalized error. Set numeric acceptance limits before
   the run using existing kernel-oracle practice; do not pick them after seeing results.
   Rank replicas should agree bitwise. Stock-vs-TP cannot be promised bitwise because
   down-dot accumulation and potentially expert combination change association.
4. Stress changing inputs/routes and captured replay; check epochs, alternating T,
   cancellation/error release and clean shutdown under a watchdog.
5. Time complete input/routing transfer, both ranks, FFNs, reduction/join and launch
   overhead against the existing two-card EP baseline, not just single-GPU full-width.
6. Actual layer weights, then full-model logits/top-k margins, multiple greedy prompts,
   quality checks, MTP acceptance and output lengths. Near ties require investigation.
7. Same-day interleaved whole-decode runs with fixed context, power, residency and reserve;
   report per-card balance and accepted tokens/window. Expand to whole-layer integration
   only if the measured critical-path saving justifies the remaining engineering cost.

Stop implementation planning at this document until the configuration, manifest and
runtime memory evidence below arrive. No GPU compiler/device was available for this review.

## Minimum metadata to unblock the decision

Send the sanitized running configuration/launch command, relevant STRATA_* environment
variables, exact model/shard and pack identity, MTP path/format, context/KV settings,
resident-KV limit, prefill size, cache limits/reserves and spec settings. Remove credentials
and unrelated private values. Also send an existing startup log and decode timing/profile
from the same run; actual per-card allocated/free memory is necessary, not file sizes alone.

Once the pack path is known, from the checkout run this read-only command (replace path):

```sh
cat /actual/pack/path/native_experts.txt
```

The manifest records type IDs and blob bytes per layer, and newer headers record expert
count. It does not explicitly record N/F, so confirm them from source GGUF tensor headers.
The following uses the repository's header-only reader, reads no tensor payload and needs
no GPU or extra Python packages. Supply every model shard exactly once, not the MTP file:

```sh
python - /actual/model/shard1.gguf /actual/model/shard2.gguf <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'tools')
from gguf_reader import GGUFFile
seen = set()
total = 0
for path in sys.argv[1:]:
    g = GGUFFile(Path(path))
    for t in g.tensors:
        if not any(s in t.name for s in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')):
            continue
        if t.name in seen:
            raise SystemExit('Duplicate expert tensor: ' + t.name)
        seen.add(t.name)
        size = t.expected_bytes()
        if size is None:
            raise SystemExit('Unknown/invalid block geometry: ' + t.name)
        print(t.name, t.type_id, t.type_name, t.shape, size)
        total += size
if not seen:
    raise SystemExit('No expert tensors found: verify the model/shard paths')
print('Expert tensor bytes supplied:', total, 'GiB:', total / 2**30)
print('Total is complete only if every model shard was supplied.')
PY
```

For GU shape `[N,F,E]` and down `[F,N,E]`, these lines establish widths, expert counts,
quant formats and raw expert footprint. The manifest maps those source tensors to the
runtime pack. Startup allocation logs and actual free memory establish dense, KV/MTP and
scratch overhead; no metadata-only command here claims to measure that runtime footprint.
