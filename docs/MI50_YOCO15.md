# MI50 YOCO 15/25 prototype

This opt-in prototype was tested on DeepSeek-V4.1-Flash-1017 with two 32 GiB
MI50 cards on Linux/HIP gfx906. Engine implementation: badbd3c4504abe3a17eabbae6b249ab87fc7cb87.
It supports the guarded 40-layer topology only: GPU 0 owns trunk layers 0-14,
GPU 1 owns 15-39 and the head; MTP layers remain on GPU 0. Source 14 crosses
this boundary; source 20 is local to GPU 1.

## Existing server configuration

Start from the existing validated Strata server JSON for the same pack. Keep
its model arguments, tokenizer, context, sampling, expert profile, DSpark,
HIPK and KV policy. Set `exe` to the production `strata-dsv41` built from this
branch, not a diagnostic harness. Merge these entries into its `env` object:

```json
{
  "DS4_SPLIT": "15,25",
  "DS4_YOCO_PIPELINE15": "1",
  "DS4_YOCO": "1",
  "DS4_EXACT_F16_CACHE": "0"
}
```

Remove `DS4_YOCO_KEEP`; do not use approximate mode 2. Remove test-only
`CODEX_AB_CONTROL`, `DS4_PIPELINE_TRACE` and `DS4_YOCO_PIPELINE_DIAG` entries.
Keep other existing environment settings. The tested deployment retained
`STRATA_DS4_HIPK=1`, `STRATA_DS4_SLOT_GB=4`, `DS4_PROF=1` and `DS4_FLOW_STATS=1`.
The timed comparison used DS4_PROF=0 for both controls and YOCO.

Using the existing Python environment, working directory and endpoint:

```sh
python3 -m serve.server --engine strata --config /path/to/server.json \
  --tokenizer /path/to/tokenizer --port 8000 --host 127.0.0.1
```

Paths are placeholders. Preserve the existing deployment's host/port and
access controls when replacing it. This command does not install autostart.
Before replacement, retain the prior JSON, binary and expert profile for
rollback. Wait for READY, verify the engine executable and effective env,
and send a small request. A short request checks serving, not YOCO skipping.

## Validation and limits (28 October 2026)

The CPU topology/cut/two-slot mock test passed:
`python3 tests/dsv41/yoco_pipeline15_test.py`. HIP gfx906 build passed.
CUDA and SYCL builds were not run. No default path is enabled by this opt-in.

For one 20160-token input with cold prefix cache and three 6720-row chunks,
the same split 15/25 took 114.5601 s and 113.8569 s with YOCO disabled,
and 74.4561 s with YOCO enabled. Historical split 21/19 took 107.6017 s.
These are single-run measurements, not a stable speed guarantee. The exact
keep rule retained 2668 rows (cut chunk 2, row 4052). Requests completed
without deadlock or OOM, including source-14 handoff on all three chunks.

All eight emitted token IDs matched. Full 129280-element final-prompt logits
were finite but not bit-identical. Control/control max absolute difference
was 1.98449 (RMS 0.45549); control1/YOCO was 1.68606 (RMS 0.39628).
This does not prove general quality preservation or byte-identical state.
Only short decode continuation was tested; no steady decode gain is claimed.
The mode is called exact because of its dependency retention rule, not a
claim that measured logits were bit-identical. HIPK arithmetic was unchanged.

The production binary used in that deployment had SHA256
`de859f9609b306c6ec181a381d11071141b4df450e61ff62d7533a6f831fa887`.
A rebuild may have a different hash. Raw prompts, weights, logits, logs,
credentials and machine-specific launch files are intentionally not included.
