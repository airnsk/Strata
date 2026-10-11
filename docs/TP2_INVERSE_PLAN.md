# Once-built inverse routes for captured hybrid Down

This opt-in CUDA/HIP experiment moves the grouped-route inversion out of every
output-row Down block. The GPU resident planner builds the inverse entry/blob
lookup once per proposal, then the new Down consumer reads it. The old hybrid
consumer remains available as the exact arithmetic and timing baseline.

`TpGdnLayer` takes a startup-only `inverse_down_plan` flag, default `false`.
`tp2_gdn_layer --execution hybrid-inverse` creates two separate capacity-eight
hybrid owners sharing the same immutable weights: one with the original flag,
one with the flag enabled. Both use `HybridCaptured`, the same local hidden
width 320, the same four compute segments and three event-joined GPU phase
boundaries (asynchronously enqueued by the host), the same peer transport, and
the same ordered FFN reduction. Existing runners and production
dispatch defaults are unchanged. No RCCL build, installation, or artifact is
needed. At capacity eight, inverse storage is 992 bytes per rank (80 32-bit
indices, 80 64-bit blob addresses and eight 32-bit status words). Construction
is fused into the existing resident-planner launch, so it adds no whole-layer
launch or join. These are source-level counts, not measured latency savings.

There is no hardware speedup claim yet. CPU runner mocks and compiler syntax
checks do not establish GPU correctness or speed. A result is admitted only
after all the stand gates below pass; the experiment still covers one layer,
not full-model generation, quality, or production readiness.

## One bounded stand command

Both MI50 cards must already be idle. Use the existing model pack and every
required GGUF shard from the same model root:

```bash
bash tools/run_tp2_inverse_plan_mi50.sh /absolute/model/root \
  --pack /models/PACK_DIRECTORY \
  --gguf /models/EXPERT_SHARD \
  --gguf /models/COMPANION_SHARD \
  --layer 0 --mode 8
```

The pack must be a directory and each shard a file below `/models`. In-root
logical and physical aliases are mounted read-only; escapes are rejected.
Layer 1 is PLE and layers 3, 7, 11, … are QSA; those are refused. Expert modes 7
and 8 are accepted. The runner fixes `--execution hybrid-inverse --benchmark`.
It refuses profiling, RCCL options, arbitrary execution changes and unrelated
fault injection.

## Gates before timing

1. Build `tp2_gdn_layer` and `native_down_plan_parity` in the container's private
   tmpfs, with gfx906 enabled and RCCL disabled. Run the existing six CPU
   shard/layout/weight gates plus `native_down_plan_test`.
2. Run the synthetic `native_down_plan_parity` GPU gate in both device/peer
   directions. Each process must return zero and print exactly
   `NATIVE_DOWN_PLAN_PARITY_PASS`. This checks arbitrary grouped route metadata,
   invalid/duplicate/missing destinations, planned-versus-legacy Down output,
   captured reuse and peer publication without loading model weights. A skip
   or missing marker prevents the whole-layer process from starting.
3. Run every T=1…8 on both physical state banks in the same capacity-eight
   old/new layer owners, with changing inputs. Require all 19 materialized
   snapshot fields to be byte-exact on each of the two ranks, before and after
   commits.
4. Retain the original full-reference, legacy token-serial GDN, HC and
   candidate-same-input full-FFN numerical/byte gates unchanged. For all 44
   `(T, keep)` cases, including keep=0 and keep=T, compare the new owner exactly
   with old hybrid after proposal, prefix commit, two-token continuation and
   continuation commit. The ordered route IDs, logits, weights, hidden Q8 bytes,
   residuals, recurrence/conv state and rank-local fused diagnostics all retain
   their old-hybrid bits. T below allocation capacity is exercised.
5. Only after `INVERSE_PLAN_BANK_GATE_PASS` and
   `INVERSE_PLAN_EXACT_GATE_PASS`, run both named benchmark studies. No
   instrumented graph is prepared or used for this experiment.

The original-full numerical bounds are not loosened. Diagnostic expert-part
recomputation does not replace the actual fused-output check. Any correctness
failure rejects timing.

## Matched burst and sustained pairs

The primary pair is `baseline=tp-hybrid-captured candidate=tp-hybrid-inverse`.
It holds geometry, weights, transport and arithmetic fixed. The second pair is
`baseline=single-gpu-captured candidate=tp-hybrid-inverse`, which retains the
complete original full-layer reference as context. Both run at T=1,2,4,5,8.
Use each directly paired result; comparing absolute times across the two
separate studies can include clock or thermal drift.

Burst samples use matched initialized state and uploaded inputs with alternating
ABBA/BAAB crossover order. Every measured result is downloaded and checked after
the timer. Sustained blocks use the same preuploaded sequence and evolving state;
a shadow pass checks every step, while timed blocks download no intermediate
outputs or state and check their final outputs afterwards. Each measured arm
gets an equal untimed warm block. Both accept all T tokens explicitly.

The whole-layer wall interval includes device-input copies, host launch submission and
event-ordered dependencies, resident planning **including inverse construction**, Down, peer exchange,
explicit commit replay and completion. It excludes loading, graph preparation,
initial state resets, fixture uploads and diagnostic snapshots. The plan is
rebuilt every proposal; startup capture is not an amortized plan-build claim.
Neither isolated kernel timing nor a transport measurement substitutes for
these complete-call pairs.

The runner requires all twenty `BENCH_SUMMARY`/`BLOCK_SUMMARY` combinations
(two studies, five token counts, two timing kinds), the admission markers before
any timing output, and final `BENCH_GATE PASS`. A success footer after a partial
or mislabeled result cannot pass.

## Optional frozen-phase diagnostics

A direct invocation of the same built `tp2_gdn_layer` accepts
`--model-probe-inverse --pack … --gguf …`. This separate opt-in probe applies the
same startup flag to full0, full1 and hybrid. The full owners retain their
ordinary full-layer grouped Down; their frozen phase-7 fused controls use the
same inverse consumer as hybrid. Optimized phase samples are explicitly named
`PROBE_INVERSE_PHASE`, with `PROBE_INVERSE_VARIANT` and `PROBE_INVERSE_GATE`.
The existing `--model-probe` output and default selection stay unchanged.

This bounded T1/T8 frozen probe is optional, is not run by the whole-layer
wrapper, and does not replace its exhaustive correctness gates or matched
complete-call benchmark. Do not sum phase times or use standalone prebuilt-plan
Down times as evidence for whole-layer speed.

## Bounds and isolation

Optional values and defaults:

```bash
BUILD_JOBS=48 TP2_BUILD_TIMEOUT=600 TP2_TIMEOUT=1200 TP2_OUTER_TIMEOUT=2400 \
  bash tools/run_tp2_inverse_plan_mi50.sh /absolute/model/root \
  --pack /models/PACK_DIRECTORY --gguf /models/EXPERT_SHARD \
  --gguf /models/COMPANION_SHARD \
  --bench-warmup 3 --bench-trials 12 \
  --bench-block-calls 16 --bench-block-trials 4
```

`TP2_BUILD_TIMEOUT` bounds configure/build/CPU checks together. `TP2_TIMEOUT`
bounds each synthetic GPU process and the complete real-weight process.
`TP2_OUTER_TIMEOUT` bounds the entire container, including compilation and all
processes; the smallest enclosing deadline wins. The container runs `timeout`
as its entrypoint with TERM then a 15-second kill grace, and the outer host
Docker-client guard adds 30 seconds. Image inspection, container removal and
independent absence checks also have their own bounds.

Sudo authorization happens before timed operations, outside captured log
pipelines. Prompts and input use the controlling terminal only. Cleanup may
explicitly reauthorize after a long run; there is no keepalive. Without a
terminal, only cached/passwordless sudo is accepted. Unverified cleanup fails
the suite.

The installed image is pinned to
`sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca`.
The runner uses `--pull never`, `--network none`, a read-only image/check-out/model
mounts, `--cap-drop ALL`, an executable 8 GiB `/tmp` tmpfs and explicit GPU device
nodes. It preserves host UID/GID and numeric GPU-device groups. No software is
downloaded or installed, no service is stopped, and no driver, runtime, security,
clock or power setting is changed. Telemetry is read-only.

The durable `tp2-inverse-plan-<UTC-run-id>.log` contains source hashes, build/test
output, paired samples, telemetry, process status, independently checked cleanup,
and final `TP2_INVERSE_PLAN_EXIT`. `INVERSE_PLAN_SUITE_PASS` requires successful
cleanup as well as every gate. After an unexpected watchdog or kill during GPU
work, inspect the device state before another attempt.

CPU-only runner boundary checks:

```bash
python tools/test_tp2_inverse_plan_runner.py
python tools/test_tp2_hc_dispatch.py
```

These mocks cover authorization, identity, read-only mounts, deadlines, cleanup,
argument rejection, both synthetic-device gates, mandatory complete pair labels,
profiling refusal and timing admission order. They do not execute HIP or Docker.

The runner also accepts `--hc-dispatch production-check` to run production's HC
selector on each GPU before layer setup/capture. It requires the per-device
selected variants and `HC_DISPATCH_INIT_PASS` before benchmark samples, in
addition to every existing inverse-plan gate. The default remains `legacy`,
preserving historical unchecked selection. A selector may fall back to plain;
initialization admission is not numerical or performance evidence. Do not merge
measurements from different HC policies or selected variants. No corrected GPU
measurement is supplied by this startup/provenance change. For the baseline-only
T1/8 phase probe, use the dispatch option in [the model-probe runner](TP2_MODEL_PROBE.md).

Related: [whole-GDN-layer reference gates](TP2_GDN_LAYER_TEST.md) and
[earlier frozen model probes](TP2_MODEL_PROBE.md).
