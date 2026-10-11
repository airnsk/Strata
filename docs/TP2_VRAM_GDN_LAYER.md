# Opt-in VRAM transport whole-GDN comparison

This isolated experiment replaces only the three hybrid TP exchanges with the
one-span VRAM rendezvous. The existing row-attention/column-FFN arithmetic,
original staged HC weights, quantization, expert selection and commit behavior
remain the comparison contract. It is default-off and has no serving adapter.

First run the [standalone protocol admission](TP2_VRAM_RENDEZVOUS.md#target-runtime-protocol-admission)
on the target pair and runtime. Its pass does not establish this integrated
layer's correctness or speed.
Hardware measurements require the complete transcript admission below; CPU
tests alone establish no hardware result.

## Run on a prepared two-GPU system

Both MI50s must already be idle. Use the existing model root and verified pack/shards:

```sh
bash tools/run_tp2_vram_gdn_mi50.sh /absolute/model-root \
  --pack /models/PACK \
  --gguf /models/WEIGHTS-00001-of-00002.gguf \
  --gguf /models/WEIGHTS-00002-of-00002.gguf
```

The runner uses only the installed pinned image and existing dependency checkout.
Build products use the private, dedicated `build-tp2-gdn-vram` incremental cache.
Image, compiler, CMake configuration and build flags key separate build directories;
unchanged objects are reused after reconfiguration. Source hashes, timestamps and
the previous built binary are checked; interrupted builds or changed bytes with
old/preserved timestamps force a clean build. The cache is locked against
concurrent runs, must be owned by the invoking user and cannot escape through a
symlink. `TP2_VRAM_GDN_CACHE=off` instead builds in private temporary storage.
The checkout and validated model-root views are read-only; only this cache is
mounted writable. It does not install or build RCCL,
run the broad phase/model probe, stop services, or change power, clocks or BIOS.
Read-only GPU inventory and idle evidence must pass before work proceeds.

`BUILD_JOBS` defaults to 48 and is bounded to 1..48. `TP2_TIMEOUT=1200` bounds
each model process; `TP2_BUILD_TIMEOUT=600` bounds compilation; and
`TP2_OUTER_TIMEOUT=2400` bounds discovery, build and all processes. Deadline
variables accept integer seconds in 1..86400. Separately bounded removal and
absence verification apply to only this run's named container. Logs are retained
as `tp2-vram-gdn-<UTC-run-id>.log`.

The only selected layer is 0, expert mode 8. The production HC selector must
choose staged variant 3 on both devices before layer preparation. A valid plain
or split fallback is not accepted for this comparison. The runner fixes warmup
crossovers at 3, measured crossovers at 4, sustained block calls at 16 and block
crossovers at 4. Profiling and alternative phase probes are not mixed into this
run. Both normal execution and fault tests use the new default-off
`STRATA_TP2_GDN_VRAM_BUILD=ON` target; RCCL remains off.

## Correctness before timing

The runner first uses separate processes for each missing/delayed-rank case in
both host issue orders. Delayed cases must observe live first-seam GPU polling
and exact continuation. Missing peers must produce a finite device protocol
failure. The poisoned owner must reject both ranks' residual/snapshot access,
positive and zero commits, checkpoint reset, another proposal, preparation and
execution-mode changes. Both GPUs must drain before destruction. Recovery means
new owners, buffers, streams, graphs and workers, with exact proposal/commit and
continuation agreement against the old hybrid. A watchdog, skip, crash, partial
log or cleanup failure never counts as a pass. The wrapper does not retry a
failed GPU run automatically.

The normal process then checks:

- Two fresh lifecycle passes, both host issue orders, changing T1/T8 shapes and
  both captured physical state banks without resetting protocol epochs
- All T1 through T8 graphs on both physical banks, both owners and both issue
  orders, with changing input bytes
- Every accepted prefix `keep=0..T` at T1 through T8, followed by a two-token
  continuation, for both old hybrid and VRAM hybrid: 88 prefix/continuation cases
- All 19 materialized fields on both owners exactly equal to the old hybrid,
  including residual, router logits/weights/ordered IDs, routed/shared hidden Q8,
  recurrence, convolution and local diagnostic partials
- The unchanged original-full numerical gates, independent serial legacy GDN
  recurrence/commit, original HC and candidate-same-input full FFN oracle

Exact old-hybrid parity is additional to the existing original-full tolerance
checks. No tolerance is widened. Both transport and owner-local diagnostics are
checked; a reconstructed expert-parts array does not replace the direct fused
FFN-output gate. All these admission gates precede the benchmark records.

## Matched whole-layer wall comparison

Two explicitly named studies each cover T1, T2, T4, T5 and T8:

1. `tp-hybrid-captured` versus `tp-hybrid-vram`: identical old hybrid compute with
   staged HC, changing only the exchange mechanism
2. `single-gpu-captured` versus `tp-hybrid-vram`: the existing full-layer reference

Each study uses alternating ABBA/BAAB crossovers. The same preuploaded residual
input bytes, recurrence and convolution checkpoint, token shape and transition
count are used in both arms and both orders. `VRAM_MATCHED_FIXTURE` records the
shared residual/sequence, recurrence and convolution FNV-1a byte identifiers for
each burst trial and sustained shape. These are identity aids, not model hashes.
Checkpoint reset is outside timing.
Each measured burst pair is preceded by matched untimed warm calls; each measured
sustained arm is preceded by an equally long untimed evolving-state warm block.

The clock covers complete proposal plus full-window commit and synchronization,
including input-device copies, CPU dispatch, worker handoff, peer exchange,
compute, accepted-prefix replay and the final per-rank completion/witness
readbacks. Sustained blocks also include host looping.
It excludes model load, graph construction, checkpoint restore and diagnostic
snapshots. No sum of separately timed kernels or phases is used as latency.

Burst snapshots validate every measured result outside the timer. Sustained
shadow blocks validate every transition before that shape's measured blocks;
measured blocks have no intermediate downloads and must match the shadow and
each other at their final state. Every sample, crossover and summary must be
present, finite and positive, correctly labeled and ordered after admission.
All results remain conditional on the final benchmark and cleanup gates.

Report the measured wall times and ratios separately for each study and T.
These are one-layer fixtures with explicit proposal/commit, even at T1. They are
not optimized autoregressive self-commit, production EP, generation tokens/s, or
full-model quality evidence. A speedup is unmeasured until the hardware run passes
all gates and its matched end-to-end wall times actually improve.

## Local validation limits

```sh
bash -n tools/run_tp2_vram_gdn_mi50.sh
python3 tools/test_tp2_vram_gdn_runner.py
```

Runner tests use CPU mocks and synthetic transcripts. Declaration-only C++
syntax checks do not compile device code or prove HIP graph behavior. The actual
HIP build, GPUs, real model, parity gates and speed measurement require a hardware
run. An offline audit reuses an existing run's evidence and does not rerun it.

## Offline audit of the compiler-comment false positive

The host validator keeps build output separate from model-process output. A
Clang warning/note can display a numbered source line containing a trailing C++
comment such as `// drop any error the attempt left behind`. The lowercase word
`error` in that comment is not a runtime diagnostic. This exception requires an
immediately preceding source-location diagnostic with the same line number, an
immediately following caret, and a source line between the build configuration
record and the first host-test pass. Quoted code is excluded. Diagnostic headers,
code before the comment, and failure/skip/watchdog tokens remain checked. All
build, process, numerical, timing and cleanup gates must still pass.

An older validator could reject an otherwise complete transcript with exactly
these final records, immediately after verified container cleanup:

```text
VRAM_GDN_ADMISSION_FAIL error output
TP2_VRAM_GDN_EXIT=4 LOG=...
```

Keep that raw log unchanged and private. Set `LOG` to its local path and
`CONTAINER` to the exact name in its verified-cleanup record, then run:

```sh
python3 tools/tp2_vram_gdn_admission.py host-revalidate "$LOG" "$CONTAINER" > revalidation.json
```

This CPU-only command reads the original bytes once. It accepts only that exact
two-record failure trailer, checks every preceding record, and requires at least
one recognized compiler-comment false positive. Earlier failures, missing gates,
nonzero model-process exits, conflicting success records, or extra trailer text
are rejected. It never rewrites the log or emits a historical suite pass.

The separate JSON result retains `original_exit=4` and the original failure,
records the raw-log and validator SHA-256 hashes, and lists the admitted source
comment line numbers. `VRAM_GDN_HOST_REVALIDATION_PASS` means the revised host
policy admitted that original transcript offline. It does not change the
original wrapper result, run a GPU workload, or establish generation throughput.
