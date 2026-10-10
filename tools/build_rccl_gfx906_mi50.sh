#!/usr/bin/env bash
# Opt-in source build only. No GPU devices, package installs, or runtime tests.
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: bash tools/build_rccl_gfx906_mi50.sh [--output NEW_DIRECTORY]
Build RCCL 2.27.7 for gfx906 in the already-installed pinned MI50 Docker image.
Fetches exactly pinned RCCL and fmt sources; compilation has no network or GPUs.
Writes a new user-owned artifact directory. Never replaces installed libraries.
Defaults: RCCL_BUILD_JOBS=24 RCCL_BUILD_MEMORY_GIB=128 RCCL_BUILD_TIMEOUT=7200
         RCCL_FETCH_TIMEOUT=600 (seconds). Jobs: 1..48; memory: 16..256 GiB.
No GPU probe is run, even after the static artifact gate passes.
EOF
}
if [[ ${1:-} == --help && $# == 1 ]]; then usage; exit 0; fi
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
IMAGE=sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca
RCCL_COMMIT=96a25b5fd6f73fba58c7d83eb57cf19a50230aa4
FMT_COMMIT=e69e5f977d458f2650bb346dadf2ad30c5320281
OUT="$ROOT/build-rccl-gfx906-$RUN_ID"
JOBS=${RCCL_BUILD_JOBS:-24}
MEMORY=${RCCL_BUILD_MEMORY_GIB:-128}
BUILD_TIMEOUT=${RCCL_BUILD_TIMEOUT:-7200}
FETCH_TIMEOUT=${RCCL_FETCH_TIMEOUT:-600}
DOCKER=(docker)
ACTIVE=""
if (( $# )); then
  [[ $# == 2 && $1 == --output ]] || { usage >&2; exit 2; }
  OUT=$2
fi
for item in "$JOBS:48" "$MEMORY:256" "$BUILD_TIMEOUT:86400" "$FETCH_TIMEOUT:3600"; do
  value=${item%:*}; upper=${item#*:}
  [[ $value =~ ^[1-9][0-9]{0,4}$ ]] && (( value <= upper )) || {
    echo 'RCCL_BUILD_BLOCKED invalid jobs, memory, or timeout setting' >&2; exit 2;
  }
done
(( MEMORY >= 16 )) || { echo 'RCCL_BUILD_BLOCKED memory must be at least 16 GiB' >&2; exit 2; }
for tool in docker timeout realpath git sha256sum tee id; do
  command -v "$tool" >/dev/null || { echo "RCCL_BUILD_BLOCKED missing host tool=$tool" >&2; exit 2; }
done
OUT=$(realpath -m -- "$OUT")
[[ $OUT != *','* && $OUT != *$'\n'* && $ROOT != *','* && ! -e $OUT && ! -L $OUT ]] || {
  echo 'RCCL_BUILD_BLOCKED output must be a new path without commas or newlines' >&2; exit 2;
}
[[ -d $(dirname -- "$OUT") ]] || { echo 'RCCL_BUILD_BLOCKED output parent must already exist' >&2; exit 2; }
[[ -f $ROOT/tools/verify_rccl_gfx906.py ]] || { echo 'RCCL_BUILD_BLOCKED missing static verifier' >&2; exit 2; }
# Keep sudo prompts outside all logs and timed operations, as in the probe runner.
authorize_docker() {
  command -v sudo >/dev/null || { echo 'RCCL_BUILD_BLOCKED missing sudo' >&2; exit 2; }
  if { exec {auth_tty}<>/dev/tty; } 2>/dev/null; then
    rc=0; sudo -v <&"$auth_tty" >&"$auth_tty" 2>&1 || rc=$?
    exec {auth_tty}>&-
    (( rc == 0 )) || return "$rc"
  else
    timeout --signal=TERM --kill-after=3s 20s sudo -n -v || {
      echo 'Run from a terminal to authorize Docker. No password is read by this script.' >&2; return 2;
    }
  fi
}
if (( $(id -u) != 0 )); then
  DOCKER=(sudo -n docker)
  authorize_docker
fi
mkdir -- "$OUT"
LOG="$OUT/build.log"
cleanup() {
  local status=$? check_rc=0 remaining
  trap - EXIT
  set +e
  if [[ -n $ACTIVE ]]; then
    # A source build can outlast sudo's credential lifetime. Reauthorization is
    # explicit on the controlling terminal, never a keepalive or logged prompt.
    if (( $(id -u) != 0 )); then
      timeout --signal=TERM --kill-after=3s 20s sudo -n -v >/dev/null 2>&1 || authorize_docker
    fi
    timeout --signal=TERM --kill-after=3s 15s "${DOCKER[@]}" rm -f "$ACTIVE" >/dev/null 2>&1
    remaining=$(timeout --signal=TERM --kill-after=3s 15s "${DOCKER[@]}" ps -a --filter "name=^/$ACTIVE$" --format '{{.Names}}') || check_rc=$?
    if (( check_rc != 0 )) || [[ -n $remaining ]]; then
      echo "RCCL_BUILD_CLEANUP_UNVERIFIED name=$ACTIVE"
      (( status != 0 )) || status=125
    else
      echo "RCCL_BUILD_CLEANUP verified_absent=$ACTIVE"
    fi
  fi
  exit "$status"
}
main() {
  trap cleanup EXIT
  local actual
  actual=$(timeout --signal=TERM --kill-after=3s 20s "${DOCKER[@]}" image inspect --format '{{.Id}}' "$IMAGE")
  [[ $actual == "$IMAGE" ]] || { echo 'RCCL_BUILD_BLOCKED pinned image mismatch'; return 2; }
  printf 'RCCL_BUILD_PLAN image=%s rccl=%s fmt=%s arch=gfx906 jobs=%s memory_gib=%s gpu_access=0\n' "$IMAGE" "$RCCL_COMMIT" "$FMT_COMMIT" "$JOBS" "$MEMORY"
  git -C "$ROOT" rev-parse HEAD > "$OUT/strata-head.txt"
  git -C "$ROOT" diff --binary HEAD -- > "$OUT/strata-worktree.diff"
  sha256sum "$ROOT/tools/build_rccl_gfx906_mi50.sh" "$ROOT/tools/verify_rccl_gfx906.py" > "$OUT/workflow.sha256"
  local common=(--rm --pull never --read-only --cap-drop ALL --user "$(id -u):$(id -g)"
    --tmpfs /tmp:rw,exec,nosuid,size=2g --shm-size=128m -e HOME=/tmp
    -e GIT_CONFIG_NOSYSTEM=1 -e GIT_CONFIG_GLOBAL=/dev/null -e GIT_TERMINAL_PROMPT=0
    --mount "type=bind,src=$ROOT,dst=/work,readonly"
    --mount "type=bind,src=$OUT,dst=/out" --workdir /out --entrypoint /usr/bin/timeout)
  ACTIVE="strata-rccl-fetch-$RUN_ID"
  timeout --signal=TERM --kill-after=15s "$((FETCH_TIMEOUT + 30))s" "${DOCKER[@]}" run --name "$ACTIVE" \
    "${common[@]}" --network bridge --cpus 2 --memory 2g --memory-swap 2g -i "$IMAGE" \
    --signal=TERM --kill-after=15s "${FETCH_TIMEOUT}s" /bin/bash -s -- "$RCCL_COMMIT" "$FMT_COMMIT" <<'FETCH'
set -euo pipefail
command -v git >/dev/null || { echo 'RCCL_BUILD_BLOCKED missing image tool=git'; exit 2; }
mkdir /out/sources
fetch() {
  local name=$1 url=$2 commit=$3
  git -c core.hooksPath=/dev/null init -q "/out/sources/$name"
  git -C "/out/sources/$name" remote add origin "$url"
  git -C "/out/sources/$name" -c credential.helper= -c protocol.file.allow=never fetch --depth 1 origin "$commit"
  git -C "/out/sources/$name" -c core.hooksPath=/dev/null checkout --detach "$commit"
  [[ $(git -C "/out/sources/$name" rev-parse HEAD) == "$commit" ]]
  git -C "/out/sources/$name" fsck --no-reflogs
  printf '%s %s %s\n' "$name" "$commit" "$url" >> /out/sources.lock
}
fetch rccl https://github.com/ROCm/rccl.git "$1"
fetch fmt https://github.com/fmtlib/fmt.git "$2"
echo 'RCCL_SOURCE_FETCH_PASS pinned_repositories=2 submodules=disabled'
FETCH
  # Verify daemon-side completion before moving to the network-free container.
  local remaining
  remaining=$(timeout --signal=TERM --kill-after=3s 15s "${DOCKER[@]}" ps -a --filter "name=^/$ACTIVE$" --format '{{.Names}}')
  [[ -z $remaining ]] || { echo 'RCCL_BUILD_BLOCKED fetch container still present'; return 125; }
  ACTIVE="strata-rccl-build-$RUN_ID"
  timeout --signal=TERM --kill-after=15s "$((BUILD_TIMEOUT + 30))s" "${DOCKER[@]}" run --name "$ACTIVE" \
    "${common[@]}" --network none --cpus "$JOBS" --memory "${MEMORY}g" --memory-swap "${MEMORY}g" -i "$IMAGE" \
    --signal=TERM --kill-after=15s "${BUILD_TIMEOUT}s" /bin/bash -s -- "$RCCL_COMMIT" "$FMT_COMMIT" "$JOBS" <<'BUILD'
set -euo pipefail
export PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/opt/rocm/llvm/bin:/opt/venv/bin:$PATH
export ROCM_PATH=/opt/rocm CXX=/opt/rocm/bin/hipcc
for tool in git cmake make python3 hipcc hipconfig hipify-perl patch sed llvm-objcopy clang-offload-bundler llvm-readelf; do
  command -v "$tool" >/dev/null || { echo "RCCL_BUILD_BLOCKED missing image tool=$tool; nothing installed"; exit 2; }
done
[[ ! -e /dev/kfd && ! -e /dev/dri ]] || { echo 'RCCL_BUILD_BLOCKED unexpected GPU device'; exit 2; }
[[ $(git -C /out/sources/rccl rev-parse HEAD) == "$1" && $(git -C /out/sources/fmt rev-parse HEAD) == "$2" ]]
for source in rccl fmt; do
  [[ -z $(git -C "/out/sources/$source" status --porcelain --untracked-files=all) ]] || { echo "RCCL_BUILD_BLOCKED dirty source=$source"; exit 2; }
done
hipcc --version | tee /out/hipcc-version.txt
hipconfig -v | tee /out/hipconfig-version.txt
cat /opt/rocm/.info/version | tee /out/rocm-version.txt
grep -Eq '^HIP version: 7\.2\.53211([-.]|$)' /out/hipcc-version.txt || { echo 'RCCL_BUILD_BLOCKED HIP version mismatch'; exit 2; }
grep -Eq '^7\.2\.4([-.]|$)' /out/rocm-version.txt || { echo 'RCCL_BUILD_BLOCKED ROCm version mismatch'; exit 2; }
# No mutable rocm-cmake fallback or dependency installation is permitted.
ROCM_CONFIG=$(find /opt/rocm/share /opt/rocm/lib -name ROCMConfig.cmake -print -quit 2>/dev/null)
[[ -n $ROCM_CONFIG ]] || { echo 'RCCL_BUILD_BLOCKED installed rocm-cmake config missing'; exit 2; }
printf 'RCCL_BUILD_ROCM_CONFIG %s\n' "$ROCM_CONFIG"
sha256sum "$ROCM_CONFIG" /opt/rocm/bin/hipcc "$(command -v clang-offload-bundler)" > /out/toolchain.sha256
cmake --version > /out/cmake-version.txt
# HAVE_PARALLEL_JOBS=OFF prevents upstream's extra 12 compiler jobs per make job.
# fmt uses its pinned fetched source even if an unrelated system copy exists.
cmake -S /out/sources/rccl -B /out/build -G 'Unix Makefiles' \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/out/install \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -DROCM_PATH=/opt/rocm \
  -DROCM_DIR="$(dirname "$ROCM_CONFIG")" -DGPU_TARGETS=gfx906 \
  -DBUILD_LOCAL_GPU_TARGET_ONLY=OFF -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF \
  -DINSTALL_DEPENDENCIES=OFF -DBUILD_BFD=OFF -DENABLE_MSCCLPP=OFF \
  -DENABLE_MSCCL_KERNEL=ON -DHAVE_PARALLEL_JOBS=OFF \
  -DCMAKE_DISABLE_FIND_PACKAGE_fmt=ON -DFETCHCONTENT_SOURCE_DIR_FMT=/out/sources/fmt \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DONLY_FUNCS= -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  2>&1 | tee /out/configure.log
# CMake can silently drop or broaden unsupported targets. Check generated commands.
python3 - <<'PY'
import json, pathlib, re
commands = json.loads(pathlib.Path('/out/build/compile_commands.json').read_text())
targets = set()
for entry in commands:
    command = entry.get('command', ' '.join(entry.get('arguments', [])))
    targets.update(re.findall(r'--offload-arch[= ]([^\s"\']+)', command))
if targets != {'gfx906'}:
    raise SystemExit('RCCL_BUILD_BLOCKED unexpected generated target set: ' + repr(targets))
print('RCCL_BUILD_TARGET_GATE gfx906_only=1')
PY
cmake --build /out/build --parallel "$3" 2>&1 | tee /out/compile.log
cmake --install /out/build 2>&1 | tee /out/install.log
# No binary is executed by this gate; inspect only the installed ELF/code object.
python3 -B /work/tools/verify_rccl_gfx906.py --prefix /out/install --output /out/artifact.json
cp /out/build/CMakeCache.txt /out/CMakeCache.txt
cp /out/build/compile_commands.json /out/compile_commands.json
git -C /out/sources/rccl diff --binary HEAD -- > /out/rccl-generated.diff
printf 'RCCL_BUILD_STATIC_PASS gpu_tests=not_run artifact=/out/artifact.json\n'
BUILD
}
set +e
(set -e; main) 2>&1 | tee "$LOG"
status=("${PIPESTATUS[@]}")
rc=${status[0]}
if (( rc == 0 && status[1] != 0 )); then rc=${status[1]}; fi
if (( rc == 0 )) && ! grep -Fxq 'RCCL_BUILD_STATIC_PASS gpu_tests=not_run artifact=/out/artifact.json' "$LOG"; then rc=4; fi
printf 'RCCL_BUILD_EXIT=%s OUTPUT=%s GPU_TESTS=not_run\n' "$rc" "$OUT" | tee -a "$LOG"
footer=("${PIPESTATUS[@]}")
if (( rc == 0 && footer[1] != 0 )); then rc=${footer[1]}; fi
exit "$rc"
