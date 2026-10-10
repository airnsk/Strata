# Captured two-MI50 RCCL transport preflight

This opt-in, standalone test checks whether the **already installed** RCCL in the
pinned MI50 image can capture and replay the proposed hybrid-TP exchanges. It has
no model inputs, engine link, CMake change, production dispatch change, or service.
It is the transport admission gate before a separate whole-layer integration.
Passing this test does not establish model correctness or a layer speedup.

For a separate, pinned gfx906 source build and opt-in artifact selection, see
[TP2_RCCL_GFX906_BUILD.md](TP2_RCCL_GFX906_BUILD.md). The build never starts this
probe automatically; leaving `TP2_RCCL_ARTIFACT` unset preserves this default path.

## One bounded stand run

From the existing stand checkout, with both MI50 cards already idle:

```bash
bash tools/run_tp2_rccl_probe_mi50.sh
```

This is one complete suite attempt, with initialization diagnostics enabled. It
does not retry a failure or sweep settings. The normal process runs first; either
fault subprocess runs only after the preceding process passes its gate.

Before starting any timed Docker operation, the wrapper validates its options
and checks authorization. When running as a non-root user with a controlling
terminal, it runs `sudo -v` directly on that terminal, outside the log-capture
pipeline and operation deadlines. Enter the password only at sudo's terminal
prompt. The wrapper does not read, record, or echo it. Time spent authorizing is
not part of the image-inspection, build, or probe limits.

Without a controlling terminal, only cached or passwordless authorization is
accepted through a bounded `sudo -n -v`; otherwise the wrapper stops before
Docker and asks you to rerun it from a terminal. All subsequent Docker commands,
including cleanup, use `sudo -n docker`, so expired authorization fails without
another prompt. Root runs Docker directly and does not require sudo. This also
supports the stand's Snap Docker. The wrapper verifies this installed image's
exact ID before running it:

```text
sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca
```

It uses `--pull never`, `--network none`, a read-only image and repository,
`--cap-drop ALL`, a 2 GiB executable `/tmp` tmpfs, private 512 MiB shared memory,
and explicit `/dev/kfd` and `/dev/dri` access. It deliberately does not set
`no-new-privileges`, matching the stand's known Snap/AppArmor constraint. The only
host output is a durable `tp2-rccl-probe-<UTC-run-id>.log` in the checkout. There is
no model mount. No package, compiler, library, or container image is installed or
rebuilt; only the small probe executable is compiled into temporary storage.
No existing service is stopped and no driver, clock, or power setting is changed.

Defaults and optional bounded overrides:

```bash
TP2_TIMEOUT=180 TP2_BUILD_TIMEOUT=180 TP2_OUTER_TIMEOUT=600 \
  bash tools/run_tp2_rccl_probe_mi50.sh \
  --tokens all --launch-order both --iterations 20 --warmup 3 \
  --chain 32 --lifecycles 2 --init-timeout-ms 60000 --timeout-ms 10000 --devices 0,1
```

- `TP2_TIMEOUT`: normal test-process limit in seconds
- `TP2_BUILD_TIMEOUT`: compilation limit in seconds
- `TP2_OUTER_TIMEOUT`: limit for the entire container, including discovery,
  compilation, telemetry, and all test subprocesses
- `--init-timeout-ms`: watchdog for the paired rank-initialization command only,
  including device selection, stream/event creation, allocation, and
  `ncclCommInitRank`; default 60000 ms. Each lifecycle gets this budget. The
  preceding main-thread `ncclGetUniqueId` call remains under the process bound.
- `--timeout-ms`: normal-process watchdog for other worker commands, including
  warmup, capture, replay, and teardown; default 10000 ms
- Delayed-rank and missing-rank processes use the same initialization budget but
  retain fixed 10000 ms command watchdogs. Each has an outer bound of
  `ceil(init_timeout_ms / 1000) + 45` seconds, 105 seconds by default. Increasing
  the initialization allowance does not lengthen the deliberate missing-rank
  fault-launch watchdog.
- Image inspection and container cleanup have their own short bounds; these
  commands cannot prompt for a password
- Interactive authorization is outside these bounds; noninteractive
  authorization has a separate 20-second bound

The smallest enclosing bound wins. The wrapper kills/removes only its own unique
container after failure or completion, and checks that it is absent. A cleanup
check failure is an error. A timeout, skip, missing completion marker, failed log
write, or nonzero normal/delayed test exit cannot become a successful probe just
because `tee` succeeded. An image-inspection failure is labeled
`RCCL_PROBE_SETUP_FAILURE stage=image-inspect ... container_attempted=0`; it is
not GPU execution evidence. `RCCL_PROBE_CONTAINER_LAUNCH` records the Docker run
attempt, and `RCCL_PROBE_CONTAINER_ENTER` records entry into its script. Neither
means the GPU probe started. Only `RCCL_PROBE_PROCESS_BEGIN` records a probe
launch attempt. If a timeout occurs after that marker, inspect GPU state before
another run. Earlier timeouts are reported as host or container setup failures.
If authorization expires during cleanup, absence cannot be verified; the wrapper
reports that failure instead of prompting or claiming the container stopped.

Normal-run options are restricted to tokens `1`, `8`, or `all`; launch order `01`,
`10`, or `both`; iterations 2..1000; warmup 1..100; chain 2..1024; lifecycles 2..8;
initialization and other-command timeouts 100..120000 milliseconds each; and
devices `0,1` or `1,0` within
`HIP_VISIBLE_DEVICES=0,1`. The runner selects scenarios and library paths itself.
A narrowed token/order run is explicitly `full_suite=0` and cannot satisfy the
full transport admission gate. The fixed fault subprocesses use logical devices
0,1 and order 01 independently of the normal-run order.

## Existing-dependency discovery

`tools/tp2_rccl_preflight.py` searches installed ROCm/system locations and known
Python Torch bundles, including `/opt/venv/lib/python*/site-packages/torch`.
It verifies a C API header and the communicator, group, send/receive, version,
error, destroy, and abort symbols in a loadable shared library. It excludes
Torch's C++ wrapper header. Each library candidate is checked in a fresh process,
with bounded dependency/version inspection. Missing headers, symbols, or resolved
shared-library dependencies stop the run with the actual blocker.

The log records:

- Resolved RCCL library and exact compiled header paths with SHA256 hashes
- `ncclGetVersion`, compiler version, selected loader search path, `ldd` results
  for RCCL and the executable, and the executable's SHA256
- Header/runtime versions, HIP runtime/driver, actual loaded RCCL path, device
  identities, architecture, peer capability, and relevant runtime environment
- Git HEAD, tracked-diff hash, worktree status, and focused source hashes
- Read-only product/utilization/clocks/power/temperature snapshots when
  `rocm-smi` exists; these snapshots do not establish clocks under load

The build explicitly includes the discovered absolute header and links the
resolved library. Runtime resolves `ncclGetVersion` back to its loaded library
and rejects a path mismatch. Header/runtime major compatibility is checked; a
matching version still does not establish working gfx906 capture support.
No automatic fallback download, package installation, or RCCL rebuild is allowed.

## Initialization diagnosis

The real stand run `tp2-rccl-probe-20261010T194525Z-30796.log` at commit
`9bada87` passed HIP compilation and loaded RCCL 2.27.7 on both gfx906 cards, each
reporting peer access. It then exited through the old 10-second
`stage=initialize` watchdog before capture. No transport timing or correctness
result was admitted. That stage marker alone does not identify which HIP or RCCL
call stalled. The logged missing-`iommu=pt` warning is a clue, not proof of the
cause; this diagnostic makes no kernel, IOMMU, driver, or network changes.

The wrapper now explicitly sets `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=ALL`, and
`NCCL_DEBUG_FILE=/dev/stdout`, and records those values before discovery. All
subsystems are included so early kernel setup, allocation, bootstrap, and
transport messages are not filtered out. The output shares the bounded process
log and durable host log; it is not left only in temporary container storage.
These are logging settings described in the
[RCCL 2.27.7 environment-variable reference](https://rocm.docs.amd.com/projects/rccl/en/docs-7.2.0/api-reference/env-variables.html).
No algorithm, protocol, transport, or socket-interface choice is forced. The
container still uses `--network none` and the same installed library.

Per-call initialization begin/end/error markers identify lifecycle, rank, call,
and elapsed time, including the individual buffer allocations. The unique-ID
call is marked separately on the main thread. Worker exceptions are printed
immediately, so an error on one rank is visible even if the other rank remains
blocked. After a timeout, use these markers and RCCL messages to locate the last
unfinished call; do not attribute it to graph capture without evidence that
initialization and capture were reached. The 60-second initialization budget is
a bounded diagnostic allowance, not a claim that slow setup is correct.

The same default command attempts the full normal, delayed-rank, and missing-rank
suite once. A normal initialization failure stops the suite. Inspect GPU state
and the verified container-cleanup result before deciding whether another run is
needed. No setting sweep or automatic rerun is performed.

## Exact dataflow and lifetime gates

Two persistent host workers own their GPU contexts, streams, and communicators.
Communicators are initialized concurrently and connections are warmed outside
capture. Separate graphs cover each requested token count, both physical buffer
banks, and each exchange, plus the ordered three-exchange sequence.

The payload per rank is:

| Exchange | T1 | T8 | Captured consumer |
| --- | ---: | ---: | --- |
| GDN Y | 12 KiB | 96 KiB | Reconstruct canonical modulo-head Y layout |
| Attention output rows | 5 KiB | 40 KiB | Reconstruct token-major rank0/rank1 rows |
| FFN full-width partial | 10 KiB | 80 KiB | FP32 rank0-then-rank1 addition |

Each grouped send/receive is inside its graph, with the local consumer submitted
after `ncclGroupEnd`. The two attention exchanges share their rank-local receive
storage in stream order; FFN has a separate receive buffer. There are no added
host joins between the three exchanges of the sequence graph.

Fresh deterministic bit patterns change between completed samples. Receives and
outputs are poisoned before replay; exact host oracles verify receive bits,
canonical reconstruction, ordered FFN reduction, source guards, destination
guards, and inactive-bank isolation. Attention includes unusual raw FP32 bit
patterns such as NaN payloads; FFN uses finite inputs for its numerical oracle.
Both rank admission orders, alternating banks, repeated replay, and fresh
communicator/graph lifecycles are exercised by the default normal run. Host
admission order does not prove driver-internal execution order or overlap.

Normal teardown destroys executable graphs and graphs before communicators, and
communicators before buffers. Asynchronous RCCL error checks accompany completion
polling. A worker-command watchdog exits the process without freeing potentially
live graph/buffer storage if progress stalls; an independent outer timeout also
bounds startup, compiler, driver, and teardown hangs.

After the normal run, separate processes test:

1. A rank delayed by 200 ms: the graph must finish, exact outputs must pass, and
   normal teardown must complete. These are correctness samples, not timings.
2. A deliberately omitted peer launch: the test must first log fault arming,
   entry into the surviving rank's graph launch, and omission of the other rank.
   Only `WATCHDOG_TIMEOUT scenario=missing-rank stage=fault-launch exit=70`
   accompanied by all three markers is the expected failure. A setup/capture
   timeout, different error, unexpected completion, or claimed pass fails this
   fault gate.

The missing-rank process is intentionally unsuccessful. The wrapper can admit its
bounded-failure handling only after distinguishing that fault from setup failure;
it never treats the fault process as a timing or correctness pass.

## Timing and admission

Normal runs report paired transport and empty-graph controls in alternating order,
with separate host-wall and rank-local event distributions. Each transport timing
includes its local unpack or ordered reduction. Initialization, capture,
instantiation, preparation, poisoning, and host verification are outside the timed
span. The three-exchange sequence is distinct from individual exchange timings.
INFO logging can affect measurements if messages occur inside measured intervals;
inspect the diagnostic log when deciding whether any emitted timing is usable.
The logging setting alone is not evidence that a repeat is needed.

`single_call` measures one replay. `sustained_chain` reports amortized time for a
chain on the same stream/storage with one completion boundary; payloads are fixed
inside that chain, then changed before the next sample. A chain's end check is not
proof of per-replay freshness inside the chain. Single-call changing-pattern checks
supply that separate evidence. Empty controls still pay the measured host-worker,
event, and launch protocol; they are not a universally subtractable transport cost.

Require all of the following from the **same complete default run**:

- `RCCL_PROBE_BUILD_GATE PASS`
- Normal `EXACT_GATE_PASS`, all requested lifecycle markers, and
  `RCCL_PREFLIGHT_PASS scenario=normal ... full_suite=1`
- Delayed-rank correctness, lifecycle, and final success markers
- `RCCL_PROBE_FAULT_GATE EXPECTED_FAILURE scenario=missing-rank exit=70`
- `RCCL_PROBE_SUITE_PASS normal=pass delayed_rank=pass missing_rank=expected_watchdog_failure`
- Verified container absence and the saved final `TP2_RCCL_PROBE_EXIT=0` footer

Timing lines printed before a later failure are rejected data. Never add rank-local
timestamps across devices, subtract empty controls to declare exact transfer cost,
or turn transport-only numbers into a whole-layer speedup. Rough 30/50-microsecond
exchange screens are future decision thresholds, not measurements or guarantees.

A full transport pass only permits the next opt-in layer experiment: preserve the
accepted hybrid arithmetic, integrate captured communication, then require the
existing whole-layer accepted-prefix/continuation gates and matched unprofiled
hybrid-versus-candidate wall timing. This standalone test does not load weights,
run actual recurrence, validate prefix commit semantics, or exercise generation.

## Host-only checks

```bash
bash -n tools/run_tp2_rccl_probe_mi50.sh
python3 tools/test_tp2_rccl_runner.py
c++ -std=c++17 -O2 -ffp-contract=off -DSTRATA_RCCL_CPU_SELFTEST \
  tests/hip/tp2_rccl_transport.cpp -pthread -o /tmp/tp2_rccl_cpu_selftest
/tmp/tp2_rccl_cpu_selftest --selftest
```

The runner tests use a temporary fake C shared library and mocked container/HIP
commands. They check dependency failure, exact version/symbol discovery, shell
syntax, restricted Docker arguments, process-status preservation, missing gates,
fault-stage admission, hard outer timeout, and cleanup verification. They also
exercise slow terminal-only authorization outside deadlines/log capture,
noninteractive authorization, denied/expired credentials, root operation without
sudo, and separate setup/probe timeout diagnostics. They require
a host C compiler for fake-library coverage. Additional checks cover both timeout
option ranges, initialization-budget propagation, the fixed fault-launch
watchdog, diagnostic logging flags, one-attempt failure handling, and rejection
of initialization/warmup/capture timeouts by the missing-rank fault gate. They require
no GPU. The source self-test checks only host
oracles and layout arithmetic. Neither is HIP compilation, RCCL capture testing,
GPU correctness, or performance evidence. The real stand run remains necessary.
