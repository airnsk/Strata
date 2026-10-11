# Targeted two-MI50 model inputs

`--model-probe` is an opt-in diagnostic for the accepted output-row attention
layout and the hybrid FFN's local hidden columns. It measures native, real-weight
layer operations at T=1 and T=8. It does not run generation, fit model quality,
change production dispatch, or select a faster default.

The earlier `--calibrate` remains a separate coarse output-row experiment. Its
routed down halves output rows while retaining K=640. This probe instead measures
the hybrid down input width K=320 against full K=640. The layouts answer different
questions; their samples must not be merged under one scaling label.

## One bounded stand run

From the existing stand checkout, using the model inputs recorded in its previous
hybrid run:

```bash
cd /home/alex/Strata-hc-persist
BUILD_JOBS=48 TP2_TIMEOUT=600 bash tools/run_tp2_model_probe_mi50.sh \
  /mnt/nvme/models \
  --pack /models/pack-iq3_s \
  --gguf /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --gguf /models/Qwen3.8-Flash-Next-GSQ-RCO-Q8_0-PLE-00002-of-00002.gguf \
  --layer 0 --mode 8 --bench-warmup 1 --bench-trials 4
```

That historical command keeps `--hc-dispatch legacy`: it does not call the
production HC startup selector. With `STRATA_HC_SPLIT` unset, an unchecked device
uses variant 1 (plain). Do not combine those HC phase measurements with production
logs that selected split or staged.

For a new dispatch-aligned baseline, add `--hc-dispatch production-check` to the
same command. This calls the same `fused_gr_check()` as `Verifier::init` on both
GPUs before layer setup, capture or timing. It logs the raw process HC/GR environment
(an unset value is not a resolved backend default),
then each device's selected variant (1=plain, 2=split, 3=staged) and
`HC_DISPATCH_INIT_PASS`. The runner requires both checked selection records and
initialization admission before phase samples, plus the existing `PROBE_GATE`.
The selector may legitimately choose different variants or fall back to plain;
the initialization marker does not claim that its optional self-test ran or that
staged was selected. The kernel's own `strata hc:` lines report those details.
Only compare HC timings when both logs establish compatible selected variants
and environment. This option changes diagnostic startup only. It neither changes
production defaults nor retroactively corrects old logs, and adds no new timing
or performance result. The probe remains T=1/8 and has no inverse-plan candidate.

The wrapper adds `--model-probe`. It refuses unrelated benchmark/execution flags
and uses only the installed image
`sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca`.
The probe requires explicit/default expert mode 8 and permits warmup 1..16 and
measured trials 2..64; other modes retain their existing CLI limits.
It verifies that exact ID, uses `sudo docker` for the stand's Snap Docker, never
pulls an image, and does not set `no-new-privileges`. Both GPUs must already be
idle. No service, driver, power, or clock setting is changed.

The existing `build-tp2-gdn` directory supplies incremental compilation. The
wrapper builds the layer test and five small host-side layout/dispatch checks.
It does not launch the legacy 44-prefix suite. Runtime has a 600-second default
timeout; compilation is outside that timeout. A timeout or skip is not a pass.

The model root is read-only at `/models`. The wrapper reuses the existing
`mi50_pipeline.model_views` validation for logical and physical root aliases,
checks explicit pack/shard paths, and preserves in-root absolute symlinks with
read-only root-view mounts. It rejects escaping aliases rather than mounting an
unrelated directory. Pack and GGUF sources must belong to the same model; matching
shapes alone do not prove that identity.

## Bounded correctness before timing

The probe loads one full layer on GPU0, one full layer on GPU1, and hybrid rank0
and rank1 weights. Actual source/shard byte coherence, including all 512 expert
shards, is checked. No extra pure-column or captured-row candidate is loaded.

The whole-layer correctness gate covers five cases:

- T1, keep=0 and keep=1
- T8, keep=0, keep=4 and keep=8
- A fresh T1 continuation after every case

The original seam tolerances, exact FFN input/router/hidden-Q8 contract, independent
legacy HC/GDN/full-FFN checks, and local fused-versus-unfused diagnostics remain in
force. Both full-device baselines and replicated hybrid boundaries are checked.
A failure prevents phase timing. This is explicitly bounded coverage, not a new
all-prefix or full-model validation claim.

Initial residual, recurrence and convolution buffers are deterministic, signed,
nonzero synthetic fixtures. They are not prompt-derived activations. Real model
weights and actual GPU router plans make the phase inputs representative of this
one-layer fixture, not evidence about production route distributions or acceptance.
FNV-1a fingerprints identify exact input buffers within the run; they are not
cryptographic model-identity proofs.

## Fine frozen phases

The coordinated replay uses full0's valid seam data as the canonical input to all
arms, copying the corresponding local slices into hybrid owners. Full0, full1,
isolated half0, isolated half1 and scheduled-concurrent halves are tested within
one mirrored crossover, with byte-identical input restored before every replay.
Inactive owners are restored first, and measured owners last, to reduce systematic
cache eviction from restoring another resident owner on the same GPU.
Each trial contains each compute arm twice. Warmup trials do not enter samples.
Preparation, restoration, and output downloads stay outside timing.
The decomposed phase outputs are checked against intact prepared full/hybrid
layer-graph outputs before replay timing. Canonical routes and hidden Q8 bytes
are exact; shared and combined FFN column reconstructions use the fixed numeric
bounds. Active replay results and inactive owners are checked, and original
storage is restored and verified exactly when the probe finishes.

The measured phases and launch-floor control are:

- `hc-attn`: full canonical attention HC read
- `hc-ffn`: pending attention write plus full canonical FFN HC read
- `router-plan-quantize`: router, resident grouping plan, and input quantization
- `shared-gu-scalar`: native shared gate/up, hidden quantization, and scalar gate
- `routed-gu`: grouped routed gate/up and hidden quantization
- `shared-down`: full K=640 or local K=320 native shared down
- `routed-down-combine`: production grouped full down plus combine against hybrid
  fused local down/combine
- `routed-down-combine-fused-control`: existing full-K fused kernel against the
  same local-K fused implementation, checked against the full production result
- `empty-graph-control`: the same five arms and event protocol without model compute

HC remains replicated: an HC `half0` or `half1` arm still does the full HC work.
GU halves own 320 hidden output rows with full N=2560 input. Hybrid down owns
K=320 input columns and produces N=2560 partial output. The production down
comparison changes both shape and kernel family; it is not pure GEMV scaling.
The same-kernel fused control helps distinguish those effects without replacing
the production full baseline.

Three separate exchange phases use frozen valid payloads, poisoned destinations,
and exact receive checks: `attn-y-exchange`, `attn-output-exchange`, and
`ffn-partial-exchange`. `exchange` and `event-join-empty` controls each occur twice
per trial in mirrored order. Exchange timing includes publication and the required
peer-event join; it is not a raw PCIe copy-bandwidth test. The FFN exchange carries
local full-width partial outputs, with reduction outside that pure exchange span.

Host wall and same-device event spans are separate observables. Even adjacent
events can include host enqueue gaps when the stream drains before the end event
is submitted. The empty-graph control exposes that measurement floor without
turning it into a universally subtractable cost. The host submits
rank graphs serially even for the scheduled-concurrent arm: actual overlap is not
verified. A small or absent concurrency penalty cannot rule out contention in a
long-running overlapped workload. Restoration changes cache residency; these are
single-replay, restore-conditioned timings, not sustained inference.

Do not add phase times into a model latency, subtract them from whole-layer time,
subtract timestamps across GPUs, or label a speedup from those operations.
Production EP inference and its baseline throughput are not measured by this TP
fixture. Production routing, QSA, PLE, full-model residency, head, MTP, and acceptance remain
outside this experiment. The data can constrain a model only under the recorded
phase/shape/kernel/input scope.

## Saved evidence and acceptance

The saved `tp2-model-probe-*.log` includes:

- `PROBE_RUN`: run ID, Git HEAD and pinned image; resolved image ID and command
- Tracked-diff hash, working-tree status, focused source hashes and built-binary hash
- Actual runtime GU/down/shared matrix types and dimensions, source HC directory
  types, selected HC variant and relevant dispatch environment
- Read-only ROCm product/utilization/clocks/power/temperature before and after the
  test when available; these snapshots do not establish loaded clock rates
- All preflight, build and test stdout and stderr
- `PROBE_CORRECTNESS_GATE PASS` only after all five cases and continuations pass
- `PROBE_FIXTURE` and `PROBE_PHASE` input fingerprints, actual group entry counts,
  paired trial/order, owner/device IDs, restoration bytes, peer bytes, wall time,
  and rank-local event time
- `PROBE_GATE PASS` only after all requested fine replays return successfully
- `PROBE_PROCESS_EXIT` for the test process and the actual outer
  `TP2_MODEL_PROBE_EXIT` footer, written into the log itself

A build or image-inspection error also gets the outer exit footer. A zero exit
alone is insufficient: require both correctness/frozen replay gates and complete
paired phase records for T1 and T8. The outer runner preserves nonzero subprocess
status instead of returning `tee`'s success. A logging failure is itself an error.

Implementation and host-only checks do not establish HIP compilation, two-GPU
correctness, or a measured performance result. This probe still needs its one
bounded run on the stand before any of its phase data can be treated as measured.
