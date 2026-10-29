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

## Remaining release gates

Actual packed logits/state and composition/permutation checks; HTTP four-stream
integration, mixed arrivals/cancellation and capacity failure; finite memory and
latency/per-request throughput measurements against B1 AR and original DSpark.
Larger/mixed ring/compressor boundaries, image rejection, independent sampler tests,
restore continuation and failure paths remain to be extended.

2+2 GPU overlap and independent multi-request DSpark are not implemented.
No new build is installed in the production launch configuration. Production
YOCO10/30+cone is restored after each exclusive test window. No push is authorized.

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
