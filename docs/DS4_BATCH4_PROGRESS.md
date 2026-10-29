# Experimental DS4 resident AR batching

Base: 6c7c63052ef6ce82e49426b7cd93da6a55e8c4c6. This branch is not a production release.

## Implemented

Four bounded request arenas share model weights, expert caches and scheduler scratch.
Request handles contain a generation. State binding waits for prior device/Engram work;
release is a completion barrier. Checkpoints, histories, source buffers, rings, images
and sampler metadata belong to the request. Images and unfinished speculative rounds
are rejected by packed AR.

decode_batch performs segmented single-request attention and a common FFN/MoE/head
pass over 1–4 independent rows. It is real shared computation, but conservative
binding barriers remain; there is no overlapping 2+2 pipeline yet.
A failure poisons participating arenas instead of silently reusing partial state.

DS4_BATCH4=1 selects the experimental all-slot AR server. MTP weights remain loaded.
DS4_BATCH_CAP defaults to 32768; each arena allocates actual bounded source capacity,
leaving 128 MiB admission margin per allocation. This is not proof that four full
262144-token contexts fit. Original base arenas and shared workspace still exist.

The ds4-v2 protocol uses BGEN2/BSTOP2 and BT2/BPP2/BDONE2/BERR2 with slot generations.
It explicitly disables legacy solo migration and speculation. Terminal completion
precedes arena release notification. Client admission is FIFO, bounded to 32 waiters;
stale/negative-slot events are discarded. A full mailbox fails closed.
Unsupported images, penalties, logprobs and control keys are rejected.

Ready decode precedes one round-robin 16-token prefill quantum per scheduler tick.
This bounds work, not milliseconds. These small feeds use the conservative prefill
path: optimized large-chunk YOCO/cone prefill is not claimed for concurrent admission.
Full logits are copied/checked for finiteness before sampling in this prototype;
that correctness check has a measurable cost. Individual wall timing is reported;
shared CPU/expert phase counters are not yet attributed per request.

## Evidence so far

- request_state_test.py: CPU ownership/lifecycle and cone boundary tests, ASan/UBSan PASS.
- resident_state_gpu.hpp: actual four-arena serial interleave at positions
  3/127/128/129, three steps and checkpoint alternate continuation: eight full
  logits/state comparisons bit-identical. Admission/reuse checks PASS.
  This is not a packed-throughput result.
- batch_protocol_test.py: strict native request parser and Python generation-aware
  mailbox, cancellation/reuse/EOF/overflow tests PASS.
- batch_scheduler_test.py: actual native scheduler with deterministic model stub,
  four streams reach common B4; independent counters, malformed admission and
  stale cancellation/reuse PASS. This does not test GPU math.
- HIP gfx906 model and separate packed GPU harness compiled/linked.
- packed_state_gpu.hpp: actual B4, three common steps with changing row order,
  mixed positions 3/127/128/129. All eight full logits/state comparisons against
  B1 (including checkpoint alternate continuation) were bit-identical. PASS.
  This is a short fixed-profile test, not all-composition/long-context validation.
- batch_client_test.py: actual StrataEngine client/pump with actual native scheduler
  and CPU model stub: four concurrent streams, per-thread statistics and
  cancellation/reuse PASS. Actual HTTP/GPU integration is described below.

## Current checkpoint and remaining release gates

Windows 7–9 below supersede the earlier failed/confounded comparisons: controlled
long packed-state parity, real four-stream HTTP, segmented RNG carry, and mixed
prefill/decode state parity passed. The adaptive-profile numerical policy still
needs an explicit B1/B1 control; fixed-profile diagnostics are not a production fix.
Actual staggered HTTP prefill latency and cancellation under that load remain to
be measured. The original DSpark service with four concurrent clients, repeat
measurements and per-request latency are still required for the final comparison.

The opt-in two-frame 2+2 implementation passed the window 11 numerical and HTTP gates below; its completion timeline remains to be measured. Independent multi-request batched DSpark is not implemented.
Do not advertise either capability or call sequential B2 rounds a pipeline.
Current 16-token admission does not retain large-chunk YOCO/cone prefill speed.

The user authorized publishing this separate branch and leaving a successful
candidate running only after the required gates. Neither action has happened.
During diagnostic windows the published YOCO10/30+cone service is restored.
The chronological entries below preserve what was known at each earlier window;
their pending-status sentences are historical, not the current checkpoint.

Local raw evidence (not committed): /var/tmp/strata-batch4-1029-evidence/.

## Actual HTTP window (one run, 2026-10-29)

All use the same synthetic chat prompt (62 tokens, cache0) and an output cap128.
These are client wall times, including prefill and server thinking/recovery.

| Mode | API-accounted completion tokens | Client wall seconds |
|---|---:|---:|
| Working single DSpark |128|5.211200698|
| Candidate single AR |128|7.846159648|
| Candidate four AR streams |512 total|25.556360745|

Four streams overlapped; common GPU batch4 steps were logged. The ratio20.034 API-accounted tokens/s is NOT a model generation throughput:
the server counts injected thinking wrap-up tokens in completion_tokens. A
clean native-token benchmark is still required. A fifth waiting request plus cancellation of one active
request completed without killing its neighbors. Default thinking budget caused
extra wrap-up/prefill and approximately7-second text-event gaps.

Real-model texts differed among same-prompt/seed requests and versus AR1; the
first difference occurred before wrap-up. Adaptive expert placement was enabled.
This remains an unresolved numerical-policy issue, not a proven quality loss
and not a passing release gate. Do not attribute it solely to HTTP recovery.

A subsequent fixed-placement diagnostic executed a B1/B1 control and128 packed
steps, but returned nonzero before HTTP service startup. Its diagnostic stdout
was discarded by the startup reader; therefore no numerical bound can be claimed
from that attempt. Reporting now uses persisted stderr for a targeted repeat.
No failing gate was bypassed. The working production server was restored.

A further protocol issue is tracked: one logical HTTP request may have multiple
native legs after thinking wrap-up. Its sampling counter must continue across
legs, rather than resetting for each BGEN. Greedy tests do not validate this.

Window4 restored production at2026-10-29T12:23:25UTC, server258571/engine258579,
health200 and smoke4. Original binary SHA256
f6db7a472bb534907a36ad68f365961b4964982790bbce5c586d23dea1b2e14b;
original configuration SHA256
6334a3bae5180a08bb3e3249175b64225fbafe0f1948d37125413657f0b27303.
Diagnostic reporting fix is compiled, but its targeted repeat has not run yet.

## 2026-10-29 12:40 UTC: explicit native RNG offset

The native BGEN2 parser now accepts a bounded counter (next sampler draw index), advertises batch_counter=1, and initializes the slot from it. This is not a context position. Counter plus max_new must not overflow uint64. CPU strict-parser and actual scheduler/stub tests passed: 8 draws equal segmented 3+5 draws with the same seed, changed prompt length and a concurrent neighbor. HTTP carry-over is still pending; this source-only change is not in the running window-5 binary. Window 5 repeats the previously instrumented long numerical gate with exact synthetic inputs saved in evidence and both native stdout/stderr retained. No production switch.

## 2026-10-29 window 5: captured long gate failure and RNG repair

Test loading began 12:31:19 UTC. Identical B1/B1 control: 128 comparisons, bit exact. Packed test: 516 comparisons, 387 different state/logit comparisons; 51 top-1 changes and 182 fixed-seed sampled changes. Worst max absolute logit error 4.77399445, RMS 0.696253064, KL 0.24320298. First: request 1, teacher-forced step 0, first different serialized window at layer 14, max 1.93373346, RMS 0.445194897, KL 0.0270519491. This is a failed numerical gate, not accepted rounding or an assertion that padding alone differs. The trace does not yet distinguish prefill-origin differences from the first packed step; the next harness explicitly compares resident prefill before packing, records the first differing window byte/row, and can retain the first state/logit pair privately. No quality or throughput readiness claim.

The candidate exited before HTTP tests. Restore began 12:41:54 UTC, server PID 261608. Readiness must be confirmed separately. Evidence: /var/tmp/strata-batch4-1029-evidence/fixed-repeat-{engine,driver,server}.log and window5-numerical-summary.json. Exact synthetic inputs are in fixed-repeat-inputs.json. No raw artifacts are committed.

HTTP stream_request now carries a logical request's seed and next draw through transport segments using the shared request cancellation context. Only delivered model tokens commit draws; inserted wrap-up tokens and discarded queued tokens do not. The native counter capability is checked; old engines fail a nonzero continuation rather than silently reset. CPU tests passed for segmentation, neighbors, inserted prompt tokens, early generator close, changed seed rejection and overflow. Existing protocol and actual-client/native-stub tests also passed. These are bookkeeping tests, not model quality tests. The GPU test binary used above predates this RNG source repair.

Window 5 restoration confirmed 2026-10-29T12:51:24.042603+00:00: original server 261608 / engine 261616, health 200, smoke answer 4, original binary/config SHA256 verified. No test server remains. The updated diagnostic main linked successfully at 12:46:46 UTC; prefill-isolation-driver.py is prepared but not started. Next action is idle/PID verification then the prefill-isolation gate. Fair prefill performance, overlapping 2+2 and multi-request DSpark remain pending behind numerical correctness.

## 2026-10-29 window 6: the first difference precedes packed decode

Loading started 12:57:16 UTC. The prefill-isolation gate failed for request 1 at position 127, before any packed decode. Exact fed tokens and model identity match (token hash a3b965831e566bf82bcbe5ff58de6457a1ad44ee91e8ce0df0b58013b1d2bd48). First changed window: layer 14, byte 33993, physical/live row 64. Layer14 compressor KV/score and source14 compressed KV/index are equal; later source20 differs. Logits: max 1.60709739, RMS 0.341750929, KL 0.025227238. Raw states/logits were saved privately as window6-first-diff-*; compact metadata is window6-summary.json. This is not padding-only, but does not yet attribute the cause to resident arenas.

Code inspection found an uncontrolled variable in the previous oracle: chunk flow_spec orders streamed experts using a learned routing profile. Disabling adaptive expert placement does not freeze that profile. A diagnostic-only DS4_BATCH_TEST_FIXED_FLOW=1 now restores the same profile before every chunked prefill and prints its hash; it requires expert_cache disabled. Existing flow initialization already resets streaming position/counters. Production is unchanged. The next harness also repeats all three long prefills in the base arena before comparing residency. HIP model compilation and diagnostic-main linking succeeded at 13:10:06 UTC. The fixed-flow driver is prepared, not yet run; window6 restoration is in progress from 13:07:16 UTC.

## 2026-10-29 window 7: controlled numerical PASS

With DS4_BATCH_TEST_FIXED_FLOW=1 and adaptive expert placement disabled, the profile hash was 88ec564c3f56222f for all chunked prefills. B1/B1 decode (128 comparisons), repeated B1 prefills of lengths 127/128/129, resident prefills and the long packed test all passed. Packed summary: 520 comparisons, zero non-bit-equal/state/top1/draw differences; max/RMS/KL all zero. This includes all row permutations repeated over 128 teacher-forced steps and restore/alternate continuation. Evidence: fixed-flow-engine.log. The earlier mismatch was not a valid isolated measure of packed decode: its learned flow profile differed before the resident prefill. No numerical tolerance was relaxed, and no production arithmetic policy was changed to obtain this control.

The following real HTTP test caught a separate integration bug: INFO parses numeric capabilities as int, while the new RNG client accepted only string "1". Native had correctly advertised batch_counter=1. The client now accepts the parsed numeric form; the actual-client/native-stub regression verifies segmented 3+5 draws and a neighbor, not only a mocked string-valued INFO. CPU tests pass. The native READY limit now advertises the allocated per-request capacity (32768 in the test), rather than the shared base arena's 262144.

Opt-in private segment diagnostics record logical request/counter boundaries and delivered model-token timestamps (no prompt/token content in this additional trace), to distinguish model output from HTTP-injected wrap-up tokens. The next prepared HTTP-only window checks a stochastic segmented request and uninterrupted AR1/AR4 with request-local reasoning_effort=none. It is not a broad benchmark or production deployment. Native HIP build/link passed 13:31:22 UTC. Window7 restoration began 13:27:52 UTC; final readiness is recorded separately.

Additional CPU scheduler regression passed: mixed prompt lengths 1/96/80/47, prefill progress deltas at most 16, ready decode emits between admission quanta and finishes before the longest prefill. This tests the actual native scheduler with a CPU model stub, not GPU latency or optimized long-prompt YOCO scheduling. Window7 restored READY 2026-10-29T13:37:07.831206Z, server 266862/engine266870, health/smoke4/hashes verified. Window8 HTTP-only loading started 13:38:04 UTC; production not deployed.

## 2026-10-29 window 8: real HTTP RNG and uninterrupted B4 PASS

The native server advertised four slots, batch_counter=1 and READY capacity 32768. The stochastic segmented request completed with 111 delivered model tokens over two segments, with contiguous logical draw indices. HTTP completion usage included 17 additional server-inserted tokens; those are excluded from the following model-token accounting.

Using the same synthetic numbered-list prompt and per-request reasoning_effort=none (no global sampling/config change), uninterrupted AR1 delivered 128 model tokens in 7.065787 s: 18.11546 model tokens/s over HTTP wall. AR4 delivered 4x128=512 model tokens in 21.692241 s: 23.60291 aggregate model tokens/s, +30.2915% versus this AR1 run. All four text outputs matched AR1. Each logical request used exactly one native segment. Text TTFT: AR1 0.5409 s, AR4 1.9343/2.0508/2.1970/2.3785 s. Delivered-model-token TPOT median: AR1 51.72 ms, AR4 155.37–155.77 ms; AR4 p95 161.46–162.03 ms and max gap 207.82 ms.

These are one-pair instrumented HTTP-wall results with adaptive expert cache disabled, AR on both sides, and moderate synthetic context. They are not production DSpark comparison, steady-state decode-only throughput, a repeatability claim, or a general task-quality evaluation. Four streams overlapped and native logs confirm shared rows=4 execution. Full numerical state/logit isolation was tested separately in window7, not inferred from matching HTTP text.

Evidence: http-rng-driver.log, http-rng-native.stdout.segments.jsonl, http-rng-*-model-tokens.json, window8-summary.json (private evidence directory). The additional segment log contains counters/timestamps, no token content. Original service restoration began 13:47:58 UTC, server269150; readiness recorded separately. Still pending: GPU mixed long-prefill/decode fairness and quality, actual overlapping 2+2 implementation/comparison, independent multi-request DSpark. No permanent switch or push.


Window 8 restoration verified at 2026-10-29 14:04:27 UTC: READY since 13:56:58, server 269150 / engine 269158, original binary and config SHA256 unchanged, health loaded. Driver smoke returned 4. Mixed-prefill isolated harness compile/link passed; GPU execution pending. It compares identical 16-token prefill partitions, not optimized full-chunk prefill. Fixed-profile parity is a diagnostic oracle, not a production adaptive-profile fix.


### Why the learned prefill profile is part of the numerical control
Static code audit (not a measured first-kernel attribution): `flow_spec` sorts likely
host experts from `prof` and assigns virtual slots in that order. `flow_layer`
forms fixed stream batches and interleaves resident sets between them; `run_set`
groups these slots under the workspace row cap, computes each set and adds its
contributions into the accumulator. Thus a changed profile can change both
accumulation partition/order and the maximum rows passed to expert dispatch.
It does not directly change the router's selected expert IDs or weights.
Downstream routing can nevertheless change after floating-point differences.
The exact first mechanism for window 6 has not been kernel-localized, and this
audit must not be presented as that measurement. The window 7 fixed-profile
oracle only establishes state isolation under the same numerical control; it
does not make adaptive production behavior tenant-invariant. A long B1/B1
comparison under different profiles remains an explicit outstanding control.


The saved-conversation admission now checks the active resident arena capacity,
not only the model-wide maximum. The shared helper passed boundary tests at
128/129, zero, negative and mismatched stored positions. This is not in the
already running window 9 binary; the next native build must include it.
Optional HTTP prefill progress timestamps use a separate diagnostic JSONL file,
so they cannot be confused with RNG draws. Actual-client CPU regression passed.
A staggered HTTP prefill/decode probe is prepared but has not run.


Window 9 GPU gate PASS at 2026-10-29 14:16:25 UTC: 480 bitwise full-logit/state comparisons; 32 partial prefill states; 96 common B4 steps. Three existing requests decode while a fourth prefills 512 tokens in 16-token quanta. Both reference and mixed execution use identical prefill partitions. Restoration began 14:16:27, server PID 272070; READY pending at this checkpoint. Evidence: /var/tmp/strata-batch4-1029-evidence/window9-summary.json and mixed-prefill-driver.log.


### 2+2 implementation boundary (code audit, not a benchmark)
The current global request_select barrier is necessary, not merely an accidental
performance cost. CPU serve_lane updates the active HostRequest's Stats while
processing its mailbox; moving HostRequest before that work retires would race.
A true two-frame implementation must first separate shared execution accounting
from request-owned state (and attribute it at completion), then use per-device
bindings, independent live packed activations and an acknowledged handoff buffer.
CPU mailbox sequences must remain globally ordered; expert-cache mutation must
wait for both frames. Device0 MTP seeding must not overwrite an unfinished second
frame's scratch. No barrier has been removed in this checkpoint. Two consecutive
decode_batch(B2) calls would still serialize and are not an implemented pipeline.


Window 9 restoration completed 2026-10-29 14:25:42 UTC: server 272070 / engine 272078, READY, health loaded, smoke 4. Binary path and original binary/config SHA256 independently verified. Candidate with resident restore bound compiled/linked at 14:17:51; SHA256 ff485c994eca01bdee3a1e356644c13883539fa22eb4abaab645dc6fdd7cdb65. No production deployment or push. Next prepared action: verify idle then run evidence/fair-prefill-driver.py (not started), which uses http_mixed_prefill_probe.py and restores production. True 2+2 and multi-request DSpark remain implementation work, not completed fallbacks.


## Window 10 and pipeline preparation
Staggered real HTTP admission passed: a 1456-token late prompt used 91 quanta
of at most 16 tokens. Each of three already decoding requests emitted 92 tokens
during its 29.5883-second prefill. Their p95 TPOT was approximately 0.333 seconds,
maximum gap 0.402 seconds. Four logical requests delivered 800 model tokens in
52.3675 seconds; this mixed workload is not comparable to the uniform AR4 rate.
Evidence: fair-prefill-fairness.json, fair-prefill-driver.log. Original service
restored with server 274717 / engine 274725 and smoke 4; hashes/health verified.

The opt-in DS4_BATCH_PIPELINE22 path is now implemented but GPU validation is
pending. It is restricted to four rows and verified 10/30 topology. Two frames
own separate retained activations, row identities, plans and taps. CPU mailbox
statistics have executor ownership, separate from movable request host state.
Device-local binding barriers retain each scratch buffer's ownership. Source
staging explicitly waits for the remote consumer before reuse; this conservative
acknowledgement has a performance cost. Expert-cache changes wait for both frames.
MTP seeding happens after a frame completes, with the other frame's activations
retained separately. Cancellation is processed only after this bounded execution
call retires; generation-tagged protocol completion still precedes reuse.

HIP model, native main and separate long-parity/ABBA harness built successfully.
Concurrent CPU accounting snapshot/completion passed ASan/UBSan. No 2+2 GPU
correctness, overlap timeline or speed claim is made yet. The prepared benchmark
uses 128 teacher-forced steps for four requests, modes B4/2+2/2+2/B4, matching saved
initial states and comparing final state/logits. It is not emitted-token throughput.


### Next speculative-state boundary (inspection only)
The existing draft path keeps sampled q distributions in shared d_specq; the
request-owned h_specq is not automatically populated for that path. Allowing a
request switch merely because host seed/counter fields are separate would be
incorrect. A resident speculative path must copy valid draft distributions into
their request-owned host storage and restore them before verification, including
the matching sampler parameters/counter. The current unfinished-round guard
remains in place; batch_spec remains zero.

Target verification needs packed causal segments, not one flattened sequence.
Two requests with up to six verify rows each fit kWin=16; four such segments do
not. Per-segment attention/plans/ring backups/partials/taps and final logits must
be kept with the corresponding request. Shared MoE/head may process the packed
rows, but rollback and seeding use only that request's accepted prefix. These
changes are not implemented by the AR two-frame patch.


## Window 11: opt-in 2+2 numerical and HTTP gate
The two-frame path passed 520 bitwise full-logit/state comparisons including
permutations and restore. ABBA, identical saved initial states and 128 steps
(512 teacher-forced rows per run): B4 19.510999811 / 19.578356554 seconds;
2+2 18.619047187 / 18.666130876 seconds. Mean compute throughput increased
4.8389%. All final comparisons were bitwise equal. These are two measurements
per mode, not a broad performance claim or emitted-token rate. GPU completion
timeline evidence remains to be recorded separately.

Four real HTTP streams each delivered 64 model tokens, total wall 11.9149 seconds,
21.4857 model tokens/s including admission; p95 TPOT 0.159–0.161 seconds.
A cancellation run completed the other three requests. The HTTP64 run cannot be
compared directly to the earlier HTTP128 run. Evidence: window11-summary.json
and pipeline22-*. Original service restoration started 15:02:38 UTC, PID 277195.

Resident sampled draft distributions now have an explicit valid-row count and
are copied from shared device workspace to request-owned h_specq before switching;
verification restores the matching q to device workspace. An unfinished target
verify still cannot switch. Draft and verify capacity use the resident bound.
Reset invalidates pending q. CPU ownership tests and HIP compilation passed;
this ownership repair has not run on GPU yet. A 32-round-comparison oracle for
four pending sampled q distributions is compiled, not executed. Its target
verification is intentionally serial as a state-isolation control. It must not
be described as completed multi-request batched DSpark.


### 2026-10-29 17:10 UTC — window 12 and segmented verify preparation

Window 12 loaded the candidate but the request-owned-q oracle stopped at the
public Model::draft AR-only guard before its resident comparisons. This is a
capability-gate failure, not a numerical failure or a DSpark PASS. The driver
started restoration at 17:02:49 UTC (server 280471); readiness remains pending
at this checkpoint.

The experimental resident draft/verify API now requires DS4_BATCH_SPEC_TEST.
The HTTP server still advertises AR only. A segmented verify API retains each
request's positions, attention, ring backups, taps and q distribution while
sharing the target MoE rows. Whole segments are partitioned at kWin=16; four
six-row segments become two groups. A separate head-output buffer avoids
overlapping scatter when segment lengths differ. The prior AR and 2+2 paths
use one row per request through the same helpers and require regression tests.

CPU: spec_segments_test.py PASS (15/16/17/24 rows, invalid lengths, overflow,
whole-segment order); batch_execution_test.py PASS; batch_rng_test.py PASS.
The initial generalized HIP model and oracle linked successfully. A rebuild
after extracting the production partition helper is in progress. GPU correctness
of the new segmented target remains NOT RUN. The prepared oracle first repeats
serial pending-q isolation, then compares segmented verify against solo outputs,
accepted counts and saved states. No throughput or quality claim follows from
these compilation and CPU results.

17:11 UTC: final isolated HIP model/native/oracle build PASS after extracting the partition helper. The binary is build-spec-batch/segmented-spec-http-test. No GPU segmented-verify result yet.
