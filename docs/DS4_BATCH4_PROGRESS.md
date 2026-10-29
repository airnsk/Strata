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
