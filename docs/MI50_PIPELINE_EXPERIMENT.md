# Two MI50s: isolated layer-pipeline configuration

This helper prepares an experiment, not a performance claim. It uses the existing
`--pipeline-windows 2` implementation, with layers 0–24 on GPU0 and 25–47 on GPU1.
The supplied mixed IQ3_S artifact has 23.0835 / 23.7549 GiB of routed experts at that
split. Both cards need additional space for their weights, state, graphs and buffers.
Full-model fit and decode speed remain hardware checks.

## Build first, without loading the GPUs

From the intended current checkout, using the already-installed gfx906-capable
development image (replace `EXISTING_IMAGE`):

```sh
bash tools/run_mi50_pipeline.sh build EXISTING_IMAGE
```

This checks the image's Python server imports, builds the current `strata` target in
`build-mi50-pipeline/engine`, and verifies the needed CLI flags in that same image.
It records `build.log`, `engine-help.txt`, `linked-libraries.txt`, the image ID, a
source fingerprint and the binary hash. It neither loads a model nor passes GPU
device nodes into the build container. No packages/images/dependencies are downloaded.
It requires a clean Git checkout of `deps/llama.cpp` at the repository pin
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, and rechecks that before serving.
Source/dependencies are read-only in the build container; only the separate build
directory is writable.
`BUILD_JOBS=48` is the default; reduce it if needed.

Review the actual `ldd` hipBLAS/rocBLAS paths and Tensile inventory before proceeding.
The inventory prepends `/opt/rocm/lib` exactly as the experimental server does for
its engine process; the build does not inventory a different Torch-first search path.
A successful Torch GEMM through Torch's bundled rocBLAS does not validate Strata's
linked `/opt/rocm` libraries. The helper does not substitute Torch libraries into
`LD_LIBRARY_PATH`, change the driver, or claim that its CPU-only check validates a
gfx906 GEMM. If the required gfx906 BLAS library/kernel support is missing, stop here.
The runtime uses precisely the image ID selected during the build; it never launches
the HIP 7.x binary against the host's ROCm 6.x installation.

## Prepare a standalone config

```sh
bash tools/run_mi50_pipeline.sh prepare /absolute/model/directory
```

The defaults reproduce the supplied b4 model filenames and settings, replacing peer
mode with `--layer-split 25 --split-device 1 --trim-stage-weights --pipeline-windows 2`
and adding `--adapt-every 0`. The directory must contain `pack-iq3_s`, its tokenizer,
the IQ3_S native shard, the Q8_0-PLE dense shard, `ple-fp8.gguf`, `expert-profile.bin`
and `mtp/rt`. Paths are checked before a config is written.
The current engine also requires **every same-stem sibling** of the `--native` shard.
In particular, an `IQ3_S-00001-of-00002.gguf` input requires the matching
`IQ3_S-00002-of-00002.gguf` filename to exist; the differently named Q8_0-PLE dense
input is not a substitute. Preparation refuses a missing sibling and never creates
aliases or changes any model file. Symlinks within the model directory keep their
original split basenames. The same read-only model directory is also visible at its
original/physical root path inside the container, so ordinary absolute in-root shard
aliases work. Aliases escaping those views and mount destinations obscuring runtime
directories are refused. Container-side model and tokenizer paths are checked again
before engine construction.

An optional third argument supplies an existing JSON config solely as a parameter
source. It is read once, never modified or mounted during serving. Its old executable,
working directory, logs, API keys, hooks and server options are not copied. Model
paths must exist under the supplied model directory; other engine configurations
are refused for separate review. The resulting config is fully standalone.

The experiment keeps MTP/spec 5, `spec-min-p 0.5`, 131072 context, int8 KV with
32768 resident cells, the original native-dense and PLE inputs, and the source's
sampling values. It explicitly enables the default all-resident graph selection,
`STRATA_IQ_MT_MIN=1` for the documented row-consistent comparison, and decode timing.
The latter guard can change kernel dispatch relative to b4, so compare identical
settings when attributing a pipeline gain.

The b4 environment names `STRATA_PREFILL_GEMM_F16`, `STRATA_FDOT2_SW`, and
`STRATA_FDOT2_SW_VERBOSE` are retained as provenance but are **inert in this checkout**.
The current opt-in prompt FP16 setting is `STRATA_HIP_PROMPT_F16`; this helper does not
silently substitute it or claim numerical/kernel equivalence to the old patched b4.

Generated config and logs live only under `build-mi50-pipeline/run` (ignored by Git).
The original expert profile and all model files remain read-only. No profile-save
option is admitted. The engine works in this new run directory, not production's.

## Explicitly start the experiment

Only after the BLAS/runtime review approves loading, and after you have stopped
conflicting GPU work yourself and checked both MI50s are idle:

```sh
MI50_BLAS_REVIEWED=1 bash tools/run_mi50_pipeline.sh serve EXISTING_IMAGE
```

The helper asks you to confirm that the cards are idle, then asks for an API key
locally with input hidden (or uses `STRATA_API_KEY` already in your environment).
Do not paste the key into a chat or command line. It is not written into the config
or logs. Docker receives it in the ephemeral container environment, which is visible
to local Docker administrators while the container exists. Noninteractive callers
also need `MI50_GPUS_IDLE_CONFIRMED=1`.

The API uses model name `qwen3.8-flash-next-mi50-pipeline-k25`, binds on 0.0.0.0:8001,
and requires the key on `/v1/*`. This is HTTP, not TLS; use a trusted network or an
already-configured secure transport. It does not reconfigure networking/firewalls.
Port 8000 and the production service/configuration are untouched. Sharing the cards
with a running production engine would invalidate memory/timing results.

The foreground container uses the caller's UID/GID and device groups, read-only
root filesystem, all capabilities dropped, unlimited memlock, a 2 GiB temporary
filesystem and private 1 GiB shared memory. It mounts the checkout and the model
directory's limited views read-only, and only its own run directory writable. It uses no broad host-root mount,
host networking, privileged mode, new image pull or package installation. The known
Snap Docker workflow does not use `no-new-privileges` here. Ctrl+C stops the container.

Before the API starts serving, the wrapper requires:

- Both trimmed layer ranges and exactly 12800 / 11776 filled expert slots.
- All four verifiers reporting `100% VRAM resident: zero-doorbell graph`.
- `decode too; the GDN snapshots beside the window` in the pipeline activation line.
- No reported cache shrink or pipeline fallback.

Failure closes the newly started engine rather than accepting CPU fallback. Automatic
engine restart is disabled. A later `check` command rechecks the log without loading a
model or sending a request. There is no stock engine flag that otherwise guarantees
complete residency; `--no-pool` is a diagnostic, not a safe substitute for this gate.

Startup does **not** prove graph replay, output parity, rollback correctness or speed:
pipeline graph capture is lazy until a request. The helper sends no inference request.
When an explicit smoke request is authorized, require its completed response and a
`strata pipeline: N windows` timing line with N > 0, with no serial fallback/error.
Repetition penalties and coupled draft sampling can force serial decode.

For a later controlled serial-versus-pipeline comparison, the prepared
`pipeline-switch.txt` selects `pw=2`; changing that file to `pw=0` selects serial decode
on the same allocation at the next request. Do not change it during an active request.
Forced rollback checks require separate, explicit test settings. Equal residency,
same sampling/seed/prompts, completed captures and repeated interleaved runs are needed
before attributing any throughput gain. Mean tokens per window alone is insufficient.

An explicit new `serve` invocation archives only this experiment's old engine/server
logs with a timestamp, along with its config/switch/build identity, before starting
fresh logs. It does not require a new checkout or rebuild when the checked inputs are
unchanged. A nonblocking `flock` prevents simultaneous build/serve helpers for this
experiment. Preparation never overwrites an existing config. Previous results are
retained, and production's files are never archived or changed.

CPU-only tests: `python3 -m unittest tools.test_mi50_pipeline`.
