# Isolated RCCL 2.27.7 build for MI50

The inspected ROCm 7.2.4 Ubuntu 24.04 RCCL library has SHA256
`53149bc0a64faa580876b0f6b745ebfc02c4371be4e2a24daf90c23782ed96d3`.
Its embedded bundle contains 12 GPU targets and no gfx906 code. The debugger
captured HIP falling back from the missing static function lookup and interpreting
RCCL's host `ncclDevKernel_Generic_4` symbol as a device-function object. This
explains the observed host fault, but rebuilding has not yet established working
MI50 collectives, capture/replay correctness, or performance.

This opt-in workflow builds an isolated replacement artifact. It does not change
the engine, host libraries, installed image, drivers, GPU state, or default probe.
It never runs a GPU test automatically.

## Build on the stand

From the existing Strata checkout:

```bash
bash tools/build_rccl_gfx906_mi50.sh
```

The command creates a fresh `build-rccl-gfx906-<UTC timestamp>-<pid>` directory in
the checkout. Existing directories are never reused or removed. The final line
names the directory and exit status; retain its `build.log` on failure.

Defaults are 24 make jobs, a 128 GiB container memory limit with no additional
swap allowance, and a two-hour build bound. A separate timeout inside each
container bounds its processes even if the host Docker client disconnects or
cleanup is waiting for terminal authorization. The host bound is 30 seconds
longer; cleanup has separate short bounds. The known stand has 48 CPU cores and
512 GiB RAM. To use all 48 cores while keeping the same memory limit:

```bash
RCCL_BUILD_JOBS=48 bash tools/build_rccl_gfx906_mi50.sh
```

48 jobs can use more peak memory. A memory kill is a failed build, not an artifact
pass. Use a new output directory and lower parallelism after inspecting a failure.
Do not change GPU settings or retry a GPU test to diagnose a CPU build failure.
`RCCL_BUILD_MEMORY_GIB` accepts 16..256; `RCCL_BUILD_JOBS` accepts 1..48.
`RCCL_BUILD_TIMEOUT` accepts 1..86400 seconds; `RCCL_FETCH_TIMEOUT` accepts
1..3600 seconds (default 600). `--output NEW_DIRECTORY` selects another new
user-owned directory whose parent already exists. For the later probe integration,
keep it as a direct child of this checkout with a simple name.

Sudo authorization is requested on the controlling terminal before timed/logged
Docker operations. Root uses Docker directly; without a terminal only existing
noninteractive sudo authorization is accepted. This matches the probe's existing
stand flow. A long build may request sudo authorization again directly on the
terminal before cleanup verification. No credential keepalive runs. Without a
terminal, expired authorization makes cleanup unverified and fails the command.
Containers run with the invoking UID/GID, leaving user-owned artifacts.

## Exact build inputs and isolation

- Existing Docker image ID:
  `sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca`.
  The wrapper checks the local image ID and uses `--pull never`.
- RCCL source:
  [`96a25b5fd6f73fba58c7d83eb57cf19a50230aa4`](https://github.com/ROCm/rccl/commit/96a25b5fd6f73fba58c7d83eb57cf19a50230aa4),
  the official ROCm 7.2.4 revision, declaring RCCL 2.27.7.
- fmt source:
  [`e69e5f977d458f2650bb346dadf2ad30c5320281`](https://github.com/fmtlib/fmt/commit/e69e5f977d458f2650bb346dadf2ad30c5320281),
  exactly the dependency revision in RCCL's `cmake/Dependencies.cmake`.
- Compiler/runtime inputs remain the pinned image's ROCm 7.2.4 / HIP 7.2.53211.
  Version checks stop a mismatch. The existing rocm-cmake config must be present;
  its path/hash and tool versions are recorded.

The first container has network access only for the two explicit public Git
fetches, verifies detached full commit IDs, and runs `git fsck`. It does not fetch
submodules, install dependencies, run RCCL code, or access GPU devices. MSCCL++ and
unit tests are disabled, so their optional source dependencies are not needed.

The second container has `--network none`, no GPU devices, a read-only image,
read-only checkout, dropped capabilities, and only the new artifact directory
writable. It uses CMake directly, never `install.sh --dependencies`, apt, pip,
ldconfig, or a host/system install. The missing-dependency path stops with its
actual error instead of upgrading anything. All compiler libraries come from the
pinned image; fmt is forced to its pinned local source. Unanticipated downloads
cannot succeed in this container.

The unmodified upstream CMake logic generates its NetIB source file in the fetched
source directory using `patch` and `sed`; this is why that source copy is writable.
The workflow records the resulting source diff. No RCCL algorithm patch is applied.
It keeps the normal full set of collective functions and MSCCL kernel build option.

`GPU_TARGETS=gfx906` is explicit, local-GPU autodetection is off, and the actual
compile commands must contain exactly `--offload-arch=gfx906` before compilation.
`HAVE_PARALLEL_JOBS=OFF` disables upstream's additional compiler/linker parallelism;
Docker also enforces the requested CPU and memory bounds.

## Static artifact gate

Installation is only into `<output>/install`. The verifier reads this installed
library, never the system library or a host kernel stub. It records in
`artifact.json`:

- Resolved library/header relative paths and SHA256 hashes
- Header version 2.27.7
- Complete embedded bundle target inventory, with gfx906 required and other GPU
  architectures rejected
- Extracted gfx906 AMDGPU ELF code-object hashes
- Defined `ncclDevKernel_Generic_4` device function and its kernel descriptor

Missing tools, unreadable code objects, missing targets/symbols, or ambiguous
artifacts fail closed. A host `ncclDevKernel_Generic_4` object is insufficient.
The manifest's source/image fields are workflow provenance; binary inspection
alone cannot prove which source produced a library. Keep `sources.lock`, source
checkouts, `workflow.sha256`, compiler/version records, CMake cache and compile
commands with the manifest.

Only `RCCL_BUILD_STATIC_PASS gpu_tests=not_run ...`, verified container cleanup,
and `RCCL_BUILD_EXIT=0` establish build/static-gate completion. Neither marker
establishes that the library can load, initialize two ranks, or run GPU code.

## A later, separate transport run

Stop after the build and inspect its static manifest and log first. When ready to
make a separate GPU attempt with both cards idle, use the existing bounded probe
with this explicit artifact selection (replace the example directory name):

```bash
TP2_RCCL_ARTIFACT=build-rccl-gfx906-YYYYMMDDTHHMMSSZ-PID \
  bash tools/run_tp2_rccl_probe_mi50.sh
```

The override accepts only a direct, non-symlink child directory of the checkout.
Run the wrapper as the same user who built the artifact, without an outer `sudo`;
the wrapper obtains Docker authorization itself. Only this artifact path runs the
container as the invoking UID:GID, preserving access to its owner-only manifest.
It carries numeric host supplementary groups and the groups of the attached GPU
devices, so `video`/`render` names need not match the image. Files stay read-only,
capabilities stay dropped, and no artifact permissions or host groups are changed.
It reruns static inspection and matches the exact manifest before library loading
or probe compilation, then discovers the C API only under this artifact's install
root and checks the recorded library hash. There is no fallback to the image's
RCCL. The probe's existing exact loaded-path check, watchdogs, correctness gates,
full-suite requirement, and container cleanup remain in force. With the override
unset, the previous installed-library path is unchanged. The debug runner is
unchanged and does not select this artifact.

If a new build passes static inspection but fails loading, warmup, capture, replay,
or teardown, save the full probe log and stop. Do not interpret the absence of the
original host crash as transport admission. Follow [TP2_RCCL_PROBE.md](TP2_RCCL_PROBE.md)
for the full pass criteria before model integration.

## Checks available without ROCm or Docker

```bash
bash -n tools/build_rccl_gfx906_mi50.sh
bash -n tools/run_tp2_rccl_probe_mi50.sh
python3 tools/test_build_rccl_gfx906.py
python3 tools/test_verify_rccl_gfx906.py
python3 tools/test_tp2_rccl_runner.py
python3 tools/test_tp2_rccl_debug_runner.py
```

These are shell/parser/mocked-container/static-parser tests. Actual RCCL/HIP
compilation and the real LLVM verifier on the new artifact must run on the stand.
GPU behavior and performance remain untested by this build workflow.
