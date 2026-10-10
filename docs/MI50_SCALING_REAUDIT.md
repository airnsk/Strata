# MI50 scaling re-audit: what the tests do and do not establish

2026-10-10. Strata source examined at `a8e4b026`; Castagna Veloce at
`3483d462715d61f21ed5877836be35a724446359`. No new GPU experiment was run for
this audit. Hardware evidence is the supplied two-MI50 test and production logs.

## Corrected conclusion

The tested reuse of generic mode-8 kernels with a host-submitted exchange did not
outperform balanced EP in 31/32 cases. This does not reject optimized tensor
parallelism as an architecture.
Its successful arithmetic and peer visibility checks are reusable. Its wall time
is not a faithful reproduction of the donor's persistent full-model executor.

A whole-generation 2x gain is neither demonstrated nor ruled out. Merely dividing
the current weights between two devices does not create that executor. Fusion,
device-owned routing, balanced work and parallel dense/attention execution are
material parts of the donor design that remain unported.

## 1. The current two-card engine is not a whole-model TP executor

All 24,576 experts are collectively resident, but the verifier's zero-doorbell
test checks **local** primary residency only (`src/core/verify.cpp:639–660`).
The peer path therefore retains one host rendezvous for each of 48 layers in
the configured unsplit verify window.

GPU0 still runs HC, GDN/QSA, router, shared expert and vocabulary head. GPU1 runs
its selected routed experts. Peer ownership is deliberately biased toward the
primary's hot profile. The post-P2P log reports 350,158 primary lookups and 103,922
other/peer/PCIe lookups: 77.1:22.9 by count, **not by elapsed work**.

Under an illustrative equal-cost and perfect-overlap model, that routed-work
division gives 1/0.771 = 1.30x versus one GPU, not 2x. Balancing it to 50:50 would
improve that portion by 0.771/0.5 = 1.54x. Reuse, formats and per-window skew make
these explanatory calculations, not measured bounds.

P2P enablement did not replace the decode data path:

- `src/core/peer_experts.cpp:226–231`: pinned-host activation and metadata upload.
- `:245–252`: peer output rows scatter into mapped **host** memory.
- `:256–268`: the CPU polls completion.
- `src/core/verify.cpp:1511–1525`: primary gathers host rows and combines them.

The changed P2P startup description concerns prefill. The decode launch has no
corresponding `p2p_` branch. CPU expert arithmetic is skipped when every selected
expert is GPU-owned; adding pool workers does not remove serial host coordination.
Also, reported `CPU` time includes peer completion polling, and GPU-reach wait is
host waiting for GPU progress. Neither should be relabeled GPU idle time.

## 2. The microbenchmark introduces substantial submission cost

The reduced test passes 32 cases, including independent original-combine checks,
byte-exact hidden q8 blocks and evolving peer-write/event visibility checks.
It loses to balanced EP in 31/32 cases. That result needs narrower interpretation.

The dual-rank timed path records ten events plus two dependency waits per
iteration; the single-rank path records two events. Empty-rank event intervals
already measure about 4.8 us. The primary graph and reduction are submitted
before the peer graph. The protocol is correct, but submission is asymmetric.

Measured medians of wall minus the larger reported compute interval:

| T | Single GPU | Balanced EP | TP2 |
|---|---:|---:|---:|
| 1 | 31.25 us | 81.86 us | 84.17 us |
| 8 | 30.17 us | 86.98 us | 90.01 us |

These residuals contain communication, reductions, host submission and completion;
they are not a disjoint CPU-only measurement. Swapping EP ownership from 8/2 to
2/8 keeps the maximum compute interval similar but adds approximately 25–28 us
to wall time. Thus the current harness itself imposes an important limit.

Every iteration also drains the queues for untimed downloads, comparisons,
poisoning and input preparation. Increasing `--iters` alone does not turn that
into steady graph throughput. Conversely, enqueueing independent fixed-route
DAGs ahead of time would measure a throughput control, not autoregressive decode.

Before another performance verdict, separate correctness from performance,
remove diagnostic stage events from the latency measurement while preserving
required dependencies, randomize paired mode order, and measure actual sparse
routes and layer formats. Keep a real single-window dependency boundary.

## 3. Half-width weights do not halve this kernel's issued work

For N=2560, F=640, K=10, mode-8 GU has the same per-rank block count:

- Balanced EP: (640/32) x 5 = 100 blocks.
- TP2: (320/32) x 10 = 100 blocks.

But mode-8 down keeps a 64-output-row block and a 32-lane logical dot group:

- Balanced EP: (2560/64) x 5 = 200 blocks/rank.
- TP2: (2560/64) x 10 = 400 blocks/rank.

Q2_0 has 20 dot items per full-width row and ten per half-width row. Both fit
one loop iteration over 32 lanes. TP therefore doubles block/wave setup and
issues the same dot sequence with fewer useful lanes. IQ4_NL goes from 40 to
20 items, or two iterations to one; its dot work scales better, but setup,
barriers, reductions and per-expert row materialization still repeat.

This matches measured TP-versus-EP compute penalties: approximately 13–18% for
Q2 down versus 1–7% for IQ4 down. The source explanation is concrete, but only
separate phase measurements could assign an exact share of total latency.

A plausible F=320 specialization uses 16-lane groups and 128 output rows/block:
20 x 10 = 200 blocks/rank, with suitable width-16 reductions. Its target is
compute near balanced EP, **not another 2x over already balanced two-card EP**.
Sources: `src/kernels/cuda/iq_kernels.cu:579–590,3251–3304,3321–3402,3503–3511,3581–3585`.

The eight equally weighted synthetic format pairs also overrepresent Q2 down:
it is nine of the actual 48 layers, not half. Weighting those test times by the
actual layer-format counts gives the following calculated proxies:

| T | Single wall | EP wall | TP wall |
|---|---:|---:|---:|
| 1 | 154.10 us | 159.59 us | 164.39 us |
| 8 | 392.06 us | 296.84 us | 316.45 us |

These are not full-model timings. The fixture also gives all tokens the same ten
experts, whereas real token routes may differ. Its 15–26 MB weights do not fit
wholly in MI50's [4 MiB L2](https://rocm.docs.amd.com/en/docs-6.2.1/reference/gpu-arch-specs.html);
claiming this is simply an all-L2 benchmark is wrong.

## 4. The donor couples sharding with a different fused algorithm

The donor's persistent FFN owns output rows, traverses routed experts, accumulates
weighted partial sums in registers and writes one final T x N vector, including
the peer push. Our generic test writes K x T x N expert rows and then reduces.

The donor's fusion eligibility requires F/32 <= 16. Full F=640 fails that gate;
TP2 F=320 passes. Sharding therefore unlocks a fused implementation, rather than
only running the original kernel twice on smaller matrices.

However, its weight contract differs: HC is Q8_0; routed GU is Q4_K/Q5_K/Q8_0;
down is Q5_1/Q8_0. The current model uses IQ3_XXS/IQ3_S/IQ2_S/IQ4_XS GU and
IQ4_NL/Q2_0 down. IQ GU needs codebook handling and a separate resource budget.
Copying the donor's fixed launch geometry is not a validated adaptation.

The donor also shards shared FFN, GDN/QSA projections and recurrent/KV state.
Those are absent from the isolated Strata routed-expert test. Its T<=4 persistent
path does not cover the user's captured six- and seven-token windows unchanged.

Sources: donor `ggml/src/ggml-cuda/hc-persist.cu:54,1090–1188,1262–1274,1570–1585`
and `src/llama-model.cpp:543–627` at the pinned commit above.

An IQ-preserving next kernel experiment would keep validated GU/SwiGLU/q8, then
replace down plus rank reduction with an output-row-owned IQ4_NL/Q2_0 kernel.
First preserve per-expert dot reduction and ordered weighted sums; later
cross-expert partial fusion changes FP association and needs stronger checks.
Apply the same fusion to EP and TP. Persistent GU/down fusion and GPU routing
are later steps with explicit scratch lifetime, residency and barrier proofs.

## 5. The published numbers are not a one-to-two-card scaling test

See the donor [benchmark protocol](https://github.com/benpeterson40/castagna-veloce/blob/3483d462715d61f21ed5877836be35a724446359/docs/BENCHMARKS.md)
and [serving profiles](https://github.com/benpeterson40/castagna-veloce/blob/3483d462715d61f21ed5877836be35a724446359/docs/models/qwen3.8-flash-next.md).

The donor's four-card layer profile reports 2614.7 pp2048, 52.3 AR and 79.1 MTP
tok/s. Its different four-card TP2+2 persistent profile reports 1939.2, 79.5 and
119.0 respectively. AR uses llama-bench; MTP aggregates twenty varied, short,
greedy up-to-200-token requests. MTP acceptance is about 69–70%; the published benchmark does not report emitted
tokens/window. The user's adaptive speculative policy,
sampling, prompts and context configuration differ.

The donor Q4 model cannot simply replace this IQ model while keeping all routed
experts resident on two 32 GiB cards.
Its [audited routed-expert type mix](UNSLOTH_Q4.md) alone occupies 77,017,907,200 bytes =
71.7285 GiB, or 35.8643 GiB/rank under TP2, before dense weights, KV or MTP.
Packed block sizes are recorded in `tools/gguf_reader.py:41–43`.
The current IQ experts occupy 46.8384 GiB. File size is not used as VRAM size here.

## Next decision

Do not present the negative generic-TP test as proof that the donor architecture
cannot help. Do not promise 2x from a reduced peer return either.

The large target is a distributed, all-resident executor with device-owned routing,
fused native-IQ FFN and eventually parallel primary-only work. A corrected
microbenchmark and one donor-like fused subgraph can establish its first useful
step. The existing layer-split/pipeline mode is a separate, lower-code experiment:
local residency can remove doorbells, but plain splitting is serial and speculative
overlap depends on full-window plus bonus prediction success.

Historical single-card stage percentages are not a current two-card breakdown.
For illustration only, if unchanged serial MTP were 10% of time, exactly halving
everything else would yield 1/(0.10+0.90/2) = 1.82x. Fusion and host-work removal
change the amount of work, so this is not an impossibility proof for 2x.
