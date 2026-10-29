# Combined MI50 YOCO10/30 + shrinking tail
Base c4a85a825854912ebbc1a5b18c5f7d1eb012ae6f. Integrates the dedicated10/30 topology guard from dfc29cfc and cone ranges/views from35c5b6e4. No balance/replay/phase instrumentation is added to model code.
Enable DS4_SPLIT=10,30, DS4_YOCO_PIPELINE10=1, DS4_YOCO=1, DS4_YOCO_CONE=1; omit PIPELINE15 and KEEP override. Preserve original precision, HIPK, DSpark, selected experts and KV policy. Exact F16 cache remains0.
Cone is admitted only after yoco_layer validates the complete dedicated topology and physical source/workspace allocations, and only for a supported last-chunk tail. Both pipeline flags at once are rejected. Unsupported lengths fall back to rectangular YOCO.
Source8 CKV crosses GPU0->GPU1; no IK8 on GPU1. Source14/20 and Engram14 are localGPU1. All chunks execute L10..19, consume handoff/source8 and acknowledge slots before late YOCO skip. L20 source history remains complete. Cone modifies only retained late row views; topk/cand values remain absolute compressed-key IDs.
For20160 tokens: three6720 chunks, finalcut4052, late rows2668,2541,...,255. Shared helper governs execution and streaming row counts.
CPU tests: pipeline15, pipeline10, cone dependency oracle and combined topology/cut/ack tests PASS.
Numerical limitation: reduced row groups and changed expert residency can change backend dispatch and floating-point trajectories. This combination is not proven bit-exact or quality-equivalent. User explicitly requested deployment with that limitation. No intentional new quantization or routing change is introduced.
Validation/deployment evidence: /var/tmp/strata-yoco10-cone-1029-evidence. Retain published15/25 binary/config/profile for rollback. No push.
