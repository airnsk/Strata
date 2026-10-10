# Captured RCCL real-weight GDN layer experiment

This opt-in diagnostic adds a captured RCCL transport to the existing hybrid
GDN-layer test. It does not integrate RCCL into generation, the server, or the
production dispatcher. `STRATA_TP2_GDN_RCCL_BUILD` defaults to `OFF`; the runner
builds a separate temporary `tp2_gdn_layer` with that option enabled.

The preceding model-free RCCL transport preflight checked **1,344 batches**.
That result admits trying the real layer, not a layer speedup. No real-weight
RCCL layer run or GPU test was performed while preparing this runner. The local
runner tests use CPU mocks. A speed claim requires the whole-layer measurements
below and all their gates to pass on the stand.

## One bounded stand command

Use the same already verified gfx906 RCCL artifact, model pack, and GGUF shards
as the preceding stand work. Both MI50 cards must already be idle. From the
stand checkout, replace the uppercase example names with those existing inputs:

```bash
TP2_RCCL_ARTIFACT=EXISTING_ARTIFACT \
  bash tools/run_tp2_rccl_gdn_mi50.sh /absolute/model/root \
  --pack /models/PACK_DIRECTORY \
  --gguf /models/EXPERT_SHARD \
  --gguf /models/COMPANION_SHARD \
  --layer 0 --mode 8
```

`TP2_RCCL_ARTIFACT` must name one direct child directory in the checkout, with
`artifact.json` and `install/`. It is mandatory. The wrapper reruns
`verify_rccl_gfx906.py`, compares the exact static manifest, then calls
`tp2_rccl_preflight.py` with that installation and the verified library SHA256.
It also checks that the discovered absolute library and header paths match the
manifest before passing `STRATA_RCCL_LIBRARY`, `STRATA_RCCL_INCLUDE_DIR`, and
`STRATA_RCCL_HEADER` to CMake. There is no system-library fallback, package
installation, RCCL rebuild, download, or runtime replacement.

The model root is mounted read-only as `/models`. The pack is a directory; every
`--gguf` is a file below that root. Repeat `--gguf` for all required shards.
Logical and physical root aliases are mounted read-only to preserve in-root
absolute model symlinks; inputs escaping the root are rejected. Layer 1 is PLE;
layers 3, 7, 11, … are QSA. Those layers are refused. Modes 7 and 8 are accepted.
The runner fixes `--execution hybrid-rccl` and always includes the benchmark.

## Gates and timing order

One suite attempt performs these steps, stopping on the first failed gate:

1. Build the isolated gfx906 target and run the existing CPU shard/layout/weight
   gates plus `tp_gdn_gather_layout_test` and `tp_gdn_rccl_workers_test`.
2. Run delayed-rank and missing-rank real-layer fault processes for launch-first
   rank 0, then rank 1. These use T=1 and delay or omit rank `1 - first`.
   Fault output cannot admit timing or normal correctness/lifecycle markers.
3. Run one normal `--benchmark` process. Before timing, it checks the full
   original layer gates and exact parity against old hybrid across T=1…8,
   every accepted prefix and continuation, both rank orders and state banks.
   It also checks two construction/capture/replay/destruction lifecycles.
4. Run the existing paired burst and sustained whole-layer benchmarks. The
   named comparisons are single-GPU captured versus RCCL hybrid, and old
   captured hybrid versus RCCL hybrid, at T=1,2,4,5,8. Paired inputs, AB/BA order,
   per-result checks, shadow block checks and final measured-block parity are
   retained. The wall interval includes complete proposal plus commit and
   worker dispatch/completion and per-call D2D input copies; graph preparation,
   weight loading, reset, fixture H2D uploads and diagnostic downloads stay
   outside it.
5. Verify the unique container is absent, then emit `RCCL_GDN_SUITE_PASS`.

Required normal-process markers are `RCCL_LAYER_LIFECYCLE_PASS life=0`,
`RCCL_LAYER_LIFECYCLE_PASS life=1`, `RCCL_LAYER_EXACT_GATE_PASS`, and final
`BENCH_GATE PASS`. Timing before the lifecycle/exactness markers is rejected.
Delayed-rank processes require their armed rank identities and
`RCCL_LAYER_FAULT_PASS scenario=delayed-rank … timing_admitted=0`.

Missing-rank is an expected failure only with exit 70, matching armed and
omitted-rank markers, `RCCL_LAYER_GRAPH_LAUNCH_ENTER rank=FIRST`, and one
`WATCHDOG_TIMEOUT` line containing `stage=layer-missing-rank`, `exit=70` and
`completed_ranks=1`. Initialization, input-upload, unrelated-stage timeouts,
outer kills and successful missing-rank exits do not pass this fault gate.
A skip, missing marker, log-write failure, nonzero normal/delayed exit or
unverified cleanup is never a suite pass.

The transport preserves mapped raw gathers for Y and attention output and the
existing rank-0-then-rank-1 FFN sum. No faster transport result alone establishes
whole-layer speed, generation throughput, model quality, or production readiness.

## Bounds, isolation and logs

Optional settings and defaults:

```bash
BUILD_JOBS=48 TP2_BUILD_TIMEOUT=600 TP2_TIMEOUT=1200 TP2_OUTER_TIMEOUT=4200 \
  TP2_RCCL_ARTIFACT=EXISTING_ARTIFACT \
  bash tools/run_tp2_rccl_gdn_mi50.sh /absolute/model/root \
  --pack /models/PACK_DIRECTORY --gguf /models/EXPERT_SHARD \
  --gguf /models/COMPANION_SHARD \
  --bench-warmup 3 --bench-trials 12 \
  --bench-block-calls 16 --bench-block-trials 4
```

`TP2_BUILD_TIMEOUT` bounds configure, compilation and CPU gates together.
`TP2_TIMEOUT` bounds the final normal layer process. Each fault process has a
105-second bound. RCCL initialization has a separate 60,000 ms watchdog; worker
commands use 10,000 ms, and delayed peer launch uses 200 ms.

`TP2_OUTER_TIMEOUT` bounds the entire container, including verification, build,
telemetry and every layer process. PID 1 is `timeout` with TERM then a 15-second
kill grace, so stopping the Docker client is not the only bound. The host guard
has another 30 seconds. The smallest enclosing bound wins. Image inspection has
a separate 20-second limit; removal and absence inspection have 15-second limits.

Sudo authorization precedes timed work. Prompts and input use the controlling
terminal directly and never enter the retained log. Long runs may require
explicit terminal reauthorization for cleanup; there is no credential keepalive.
Without a terminal, only cached/passwordless authorization is accepted, and
expired cleanup authorization produces an unverified-cleanup failure.

The pinned image is
`sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca`.
The runner uses `--pull never`, `--network none`, a read-only image/repository and
model mounts, `--cap-drop ALL`, an 8 GiB executable `/tmp` tmpfs, and explicit GPU
devices. It preserves the invoking UID/GID and numeric GPU-device groups for the
owner-only artifact manifest. Build output stays in `/tmp`; no service is
stopped and no driver, security, power or clock setting is changed.

The durable `tp2-rccl-gdn-<UTC-run-id>.log` includes source and artifact identity,
all build/test output, RCCL INFO diagnostics, read-only telemetry, process exit
codes, cleanup status and final `TP2_RCCL_GDN_EXIT`. Temporary per-process logs
are also streamed into that host log. After an unexpected watchdog or kill once
a layer process has started, inspect GPU state before another attempt.

CPU-only runner validation:

```bash
python tools/test_tp2_rccl_gdn_runner.py
c++ -std=c++17 -Iinclude tests/core/tp_gdn_gather_layout_test.cpp \
  -o /tmp/strata-tp-gdn-gather-test && /tmp/strata-tp-gdn-gather-test
c++ -std=c++17 -pthread tests/core/tp_gdn_rccl_workers_test.cpp \
  -o /tmp/strata-tp-gdn-workers-test && /tmp/strata-tp-gdn-workers-test
```

Related: [transport preflight](TP2_RCCL_PROBE.md),
[verified gfx906 artifact](TP2_RCCL_GFX906_BUILD.md), and
[existing real-weight layer test](TP2_GDN_LAYER_TEST.md).
