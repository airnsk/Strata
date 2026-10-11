#!/usr/bin/env bash
# Opt-in whole-layer hybrid VRAM comparison. No production dispatch change.
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: bash tools/run_tp2_vram_gdn_mi50.sh MODEL_ROOT \
  --pack /models/PACK --gguf /models/SHARD [--gguf /models/SHARD ...]
Reuses existing model roots/aliases and pinned dependencies without downloads.
Only pack/GGUF inputs are configurable. Fixed layer=0, mode=8, production-check,
staged HC variant=3 on both cards, execution=hybrid-vram, warmup=3, trials=4,
block-calls=16, block-trials=4. Both GPUs must already be idle.
Four fresh bounded delayed/missing-rank processes precede one normal process.
Exact old-hybrid, lifecycle, both-bank and unchanged full-reference gates precede
complete burst/sustained paired whole-layer timings at T=1,2,4,5,8 against
old hybrid and single-GPU captured baselines. This is not generation throughput.
Uses a dedicated incremental build-tp2-gdn-vram cache (private direct child),
existing deps, and the installed pinned MI50 image. No RCCL build, broad
model-probe, profile, service stops,
packages, runtime replacement, network, BIOS, clock or power changes.
TP2_VRAM_GDN_CACHE=off uses a private tmpfs build instead; another direct child
name selects a separate cache. Configurations are keyed by image/compiler/flags.
Optional BUILD_JOBS=48 (1..48), TP2_TIMEOUT=1200, TP2_BUILD_TIMEOUT=600,
TP2_OUTER_TIMEOUT=2400 (integer seconds, 1..86400). Outer covers discovery,
build and every subprocess; bounded cleanup follows. Missing inventory or idle
evidence fails closed. Sudo authorization precedes timed/captured operations.
EOF
}
if [[ ${1:-} == --help && $# == 1 ]]; then usage; exit 0; fi
umask 077
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
LOG="$ROOT/tp2-vram-gdn-$RUN_ID.log"
IMAGE=sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca
CONTAINER_NAME="strata-tp2-vram-gdn-$RUN_ID"
CONTAINER_STARTED=0
DOCKER=(docker)
authorize_docker() {
  if (( $(id -u) == 0 )); then
    return 0
  fi
  DOCKER=(sudo -n docker)
  local auth_tty auth_rc=0
  # Open the controlling terminal directly, even if the caller pipes stdout or
  # stdin. Never put a password prompt/input in the timed or captured pipeline.
  if { exec {auth_tty}<>/dev/tty; } 2>/dev/null; then
    printf 'Checking sudo authorization outside timed operations.\n' >&"$auth_tty"
    sudo -v <&"$auth_tty" >&"$auth_tty" 2>&1 || auth_rc=$?
    exec {auth_tty}>&-
  else
    # No terminal: accept cached/NOPASSWD authorization only, with a bound on PAM.
    timeout --signal=TERM --kill-after=3s 20s sudo -n -v || auth_rc=$?
    if (( auth_rc != 0 )); then
      echo 'Sudo authorization unavailable without a terminal. Run this wrapper from a terminal to authorize Docker.' >&2
    fi
  fi
  return "$auth_rc"
}
cleanup() {
  local rc=$?
  trap - EXIT
  set +e
  if (( CONTAINER_STARTED )); then
    # Long runs can outlast cached sudo. No keepalive: explicit reauthorization
    # uses only the controlling terminal, never the captured output pipeline.
    if (( $(id -u) != 0 )); then
      timeout --signal=TERM --kill-after=3s 20s sudo -n -v >/dev/null 2>&1 || authorize_docker
    fi
    # Killing a docker client does not prove its daemon-side container stopped.
    # Remove only this unique run's container; bound even a stuck daemon request.
    timeout --signal=TERM --kill-after=3s 15s "${DOCKER[@]}" rm -f "$CONTAINER_NAME" >/dev/null 2>&1
    local cleanup_rc=$?
    # --rm normally removed it already. Verify absence independently.
    local inspect
    inspect=$(timeout --signal=TERM --kill-after=3s 15s "${DOCKER[@]}" ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' 2>&1)
    local inspect_rc=$?
    if (( inspect_rc != 0 )) || [[ -n "$inspect" ]]; then
      printf 'VRAM_GDN_CONTAINER_CLEANUP_UNVERIFIED name=%s remove_exit=%s inspect_exit=%s\n' "$CONTAINER_NAME" "$cleanup_rc" "$inspect_rc" >&2
      (( rc != 0 )) || rc=125
    else
      printf 'VRAM_GDN_CONTAINER_CLEANUP verified_absent=%s\n' "$CONTAINER_NAME"
    fi
  fi
  exit "$rc"
}
model_mount_views() {
  python3 -B - "$ROOT" "$@" <<'MODEL_VIEWS'
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "tools"))
from mi50_pipeline import model_views
root = Path(sys.argv[2])
physical = root.resolve(strict=True)
views = [Path(p) for p in model_views(root)]
for flag, value in zip(sys.argv[3::2], sys.argv[4::2]):
    if flag not in {"--pack", "--gguf"}:
        continue
    path = root / Path(value).relative_to("/models")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(physical):
        raise SystemExit("Model input escapes MODEL_ROOT: " + value)
    # Absolute alias targets must stay inside the same roots we mount.
    current = Path(os.path.abspath(root))
    for component in Path(value).relative_to("/models").parts:
        current /= component
        if current.is_symlink() and os.path.isabs(os.readlink(current)):
            target = Path(os.path.abspath(os.readlink(current)))
            if not any(target == view or view in target.parents for view in views):
                raise SystemExit("Absolute model alias escapes supported root views: " + value)
print(physical)
for view in views:
    print(view)
MODEL_VIEWS
}
validate_host() {
  local deadline=${TP2_TIMEOUT:-1200} build_deadline=${TP2_BUILD_TIMEOUT:-600} outer_deadline=${TP2_OUTER_TIMEOUT:-2400}
  local value flag
  for value in "$deadline" "$build_deadline" "$outer_deadline"; do
    [[ "$value" =~ ^[1-9][0-9]{0,4}$ ]] && (( value <= 86400 )) || {
      echo 'Timeout variables must be integer seconds in 1..86400.' >&2; return 2;
    }
  done
  value=${BUILD_JOBS:-48}
  [[ "$value" =~ ^[1-9][0-9]{0,2}$ ]] && (( value <= 48 )) || {
    echo 'BUILD_JOBS must be an integer in 1..48.' >&2; return 2;
  }
  value=${TP2_VRAM_GDN_CACHE:-build-tp2-gdn-vram}
  [[ $value =~ ^[a-zA-Z0-9_.-]+$ && $value != . && $value != .. ]] || {
    echo 'TP2_VRAM_GDN_CACHE must be off or one direct child directory name.' >&2; return 2;
  }
  (( $# >= 5 )) || { usage; return 2; }
  local original_args=("$@")
  local model_root=$1; shift
  [[ "$ROOT" != *,* && "$ROOT" != *$'\n'* && "$model_root" == /* && -d "$model_root" && "$model_root" != / &&
     "$model_root" != *,* && "$model_root" != *$'\n'* && "$model_root" != *$'\r'* ]] || {
    echo 'MODEL_ROOT must be an existing absolute directory without commas/newlines, other than /.' >&2; return 2;
  }
  model_root=$(cd -- "$model_root" && pwd -P)
  [[ "$model_root" != / && "$model_root" != *,* && "$model_root" != *$'\n'* ]] || {
    echo 'Resolved MODEL_ROOT is not a safe model mount.' >&2; return 2;
  }
  (( $# % 2 == 0 )) || { usage; return 2; }
  local -A seen=()
  local ggufs=0
  while (( $# )); do
    flag=$1; value=$2; shift 2
    [[ "$flag" == --* ]] || { echo "Unsupported option: $flag" >&2; return 2; }
    [[ "$flag" == --gguf || -z ${seen[$flag]:-} ]] || { echo "Repeated option: $flag" >&2; return 2; }
    seen[$flag]=1
    case "$flag" in
      --pack|--gguf)
        [[ "$value" == /models/* && "$value" != *$'\n'* && "$value" != *$'\r'* &&
           "/${value#/models/}/" != */../* && "/${value#/models/}/" != */./* ]] || {
          echo "Model input must be an absolute path below /models without dot components: $flag" >&2; return 2;
        }
        if [[ "$flag" == --pack ]]; then
          [[ -d "$model_root/${value#/models/}" ]] || { echo "Missing model pack directory: $value" >&2; return 2; }
        else
          [[ -f "$model_root/${value#/models/}" ]] || { echo "Missing GGUF shard: $value" >&2; return 2; }
        fi
        if [[ "$flag" == --gguf ]]; then ggufs=$((ggufs + 1)); fi
        ;;
      *) echo "Unsupported option: $flag" >&2; return 2 ;;
    esac
  done
  [[ ${seen[--pack]:-} == 1 && $ggufs -gt 0 ]] || { echo '--pack and at least one --gguf are required.' >&2; return 2; }
  for value in timeout git sha256sum tee id stat docker python3; do
    command -v "$value" >/dev/null || { echo "Missing host command: $value" >&2; return 2; }
  done
  if (( $(id -u) != 0 )); then
    command -v sudo >/dev/null || { echo 'Missing host command: sudo' >&2; return 2; }
  fi
  for value in deps/llama.cpp/ggml/CMakeLists.txt tools/tp2_vram_gdn_admission.py; do
    [[ -f "$ROOT/$value" ]] || { echo "Missing source or pinned dependency: $value" >&2; return 2; }
  done
  model_mount_views "${original_args[@]}" >/dev/null || return 2
  timeout --signal=TERM --kill-after=3s 30s python3 -B "$ROOT/tools/tp2_vram_gdn_admission.py" manifest "$ROOT" >/dev/null || return 2

}
# Bash dynamic scope supplies main's start/outer budget to discovery operations.
bounded() {
  local limit=$1; shift
  local remaining=$((outer - (SECONDS - start)))
  (( remaining > 0 )) || { echo 'VRAM_GDN_SETUP_FAILURE discovery exhausted outer deadline' >&2; return 124; }
  (( remaining >= limit )) || limit=$remaining
  timeout --signal=TERM --kill-after=3s "${limit}s" "$@"
}

main() {
  local start=$SECONDS deadline=${TP2_TIMEOUT:-1200} build_deadline=${TP2_BUILD_TIMEOUT:-600} outer=${TP2_OUTER_TIMEOUT:-2400}
  local model_root view views_text source_sha image_id uid gid host_groups device_groups group node remaining cache
  views_text=$(bounded 20 bash -c "$(declare -f model_mount_views); ROOT=\$1; shift; model_mount_views \"\$@\"" views "$ROOT" "$@")
  local views=() mounts=()
  mapfile -t views <<< "$views_text"
  model_root=${views[0]}; shift
  mounts=(--mount "type=bind,src=$model_root,dst=/models,readonly")
  for view in "${views[@]:1}"; do
    [[ $view == /models ]] || mounts+=(--mount "type=bind,src=$model_root,dst=$view,readonly")
  done
  printf 'VRAM_GDN_RUN id=%s image=%s log=%q\n' "$RUN_ID" "$IMAGE" "$LOG"
  printf 'VRAM_GDN_HEAD '; bounded 20 git -C "$ROOT" rev-parse HEAD
  printf 'VRAM_GDN_ARGUMENTS'; printf ' %q' "$@"; printf '\n'
  printf 'VRAM_GDN_BOUNDS runtime_s=%s build_s=%s outer_s=%s jobs=%s\n' "$deadline" "$build_deadline" "$outer" "${BUILD_JOBS:-48}"
  printf 'VRAM_GDN_TRACKED_DIFF_SHA256 '
  bounded 20 git -C "$ROOT" diff --binary HEAD -- | sha256sum
  source_sha=$(bounded 30 python3 -B "$ROOT/tools/tp2_vram_gdn_admission.py" manifest "$ROOT" --verbose)
  printf 'VRAM_GDN_HOST_SOURCE_SHA256 %s\n' "$source_sha"
  image_id=$(bounded 20 "${DOCKER[@]}" image inspect --format '{{.Id}}' "$IMAGE")
  [[ $image_id == "$IMAGE" ]] || { echo 'VRAM_GDN_SETUP_FAILURE stage=image-verification' >&2; return 2; }
  printf 'VRAM_GDN_RESOLVED_IMAGE %s\n' "$image_id"
  uid=$(id -u); gid=$(id -g); host_groups=$(id -G)
  [[ $uid =~ ^[0-9]+$ && $gid =~ ^[0-9]+$ && -n $host_groups ]] || return 2
  local nodes=(/dev/kfd /dev/dri) identity=(--user "$uid:$gid")
  for node in /dev/dri/card[0-9]* /dev/dri/renderD[0-9]*; do [[ ! -e $node ]] || nodes+=("$node"); done
  device_groups=$(bounded 10 stat -Lc '%g' -- "${nodes[@]}") || {
    echo 'VRAM_GDN_SETUP_FAILURE stage=device-identity' >&2; return 2;
  }
  [[ -n $device_groups ]] || return 2
  local -A groups=(["$gid"]=1)
  for group in $host_groups $device_groups; do
    [[ $group =~ ^[0-9]+$ ]] || { echo 'VRAM_GDN_SETUP_FAILURE stage=device-identity' >&2; return 2; }
    if [[ -z ${groups[$group]:-} ]]; then identity+=(--group-add "$group"); groups[$group]=1; fi
  done
  cache=${TP2_VRAM_GDN_CACHE:-build-tp2-gdn-vram}
  if [[ $cache != off ]]; then
    bounded 10 python3 -B "$ROOT/tools/tp2_vram_gdn_admission.py" cache "$ROOT/$cache"
    mounts+=(--mount "type=bind,src=$ROOT/$cache,dst=/build-cache")
  fi
  printf 'VRAM_GDN_CACHE directory=%s\n' "$cache"
  remaining=$((outer - (SECONDS - start)))
  (( remaining > 0 )) || { echo 'VRAM_GDN_SETUP_FAILURE discovery exhausted outer deadline' >&2; return 124; }
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  CONTAINER_STARTED=1
  printf 'VRAM_GDN_CONTAINER_LAUNCH name=%s\n' "$CONTAINER_NAME"
  timeout --signal=TERM --kill-after=15s "$((remaining + 30))s" "${DOCKER[@]}" run \
    --name "$CONTAINER_NAME" --rm --pull never --network none --read-only --cap-drop ALL \
    --tmpfs /tmp:rw,exec,nosuid,size=8g --shm-size=512m -e HOME=/tmp \
    --device=/dev/kfd --device=/dev/dri -e HIP_VISIBLE_DEVICES=0,1 \
    -e STRATA_HC_PERSIST=0 -e STRATA_GR_V3=0 -e STRATA_GR_SPLIT=0 \
    --mount "type=bind,src=$ROOT,dst=/work,readonly" "${identity[@]}" "${mounts[@]}" \
    --workdir /work --entrypoint /usr/bin/timeout -i "$IMAGE" \
    --signal=TERM --kill-after=15s "${remaining}s" /bin/bash -s -- \
    "$deadline" "$build_deadline" "${BUILD_JOBS:-48}" "$source_sha" "$cache" "$IMAGE" "$@" <<'VRAM_GDN_CONTAINER'
set -euo pipefail
umask 077
export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH
runtime_deadline=$1; build_deadline=$2; jobs=$3; expected_sources=$4; cache_mode=$5; image=$6; shift 6
model_args=("$@")
admission=/work/tools/tp2_vram_gdn_admission.py
for tool in python3 cmake ninja hipcc timeout sha256sum tee rocm-smi ldd flock; do
  command -v "$tool" >/dev/null || { echo "VRAM_GDN_PREFLIGHT_BLOCKED missing installed tool: $tool" >&2; exit 2; }
done
echo 'VRAM_GDN_CONTAINER_ENTER'
actual_sources=$(python3 -B "$admission" manifest /work)
[[ $actual_sources == "$expected_sources" ]] || { echo 'VRAM_GDN_SOURCE_SHA_MISMATCH' >&2; exit 4; }
printf 'VRAM_GDN_CONTAINER_SOURCE_SHA256 %s\n' "$actual_sources"
# No inherited image override may silently alter this fixed HC/kernel study.
unset STRATA_HC_SPLIT STRATA_GR_FAST STRATA_GR_DOWN_MAX4 STRATA_NO_MULTI_GR
unset STRATA_TSUM STRATA_SM70_TABLE STRATA_HC_Q8 STRATA_QFUSE
unset HIPCC_COMPILE_FLAGS_APPEND HIPCC_LINK_FLAGS_APPEND HIPCC_COMPILE_FLAGS_PREPEND HIPCC_LINK_FLAGS_PREPEND
export STRATA_HC_PERSIST=0 STRATA_GR_V3=0 STRATA_GR_SPLIT=0 HIP_PLATFORM=amd
idle_gate() {
  local stage=$1
  timeout --signal=TERM --kill-after=3s 10s rocm-smi --showuse --json > /tmp/vram-gdn-idle.json
  cat /tmp/vram-gdn-idle.json
  timeout --signal=TERM --kill-after=3s 10s python3 -B "$admission" idle \
    /tmp/vram-gdn-idle.json /sys/class/kfd/kfd/proc
  printf 'VRAM_GDN_IDLE_PASS stage=%s cards=2 utilization=0 kfd_processes=0\n' "$stage"
}
# Read-only telemetry failures are blockers; never stop a service or change clocks.
idle_gate before-build
timeout --signal=TERM --kill-after=3s 10s rocm-smi --showproductname --showbus --showuse --showclocks --showpower --showtemp
timeout --signal=TERM --kill-after=3s 20s hipcc --version > /tmp/vram-gdn-compiler.txt
cat /tmp/vram-gdn-compiler.txt
cache=/tmp/build-tp2-vram-gdn
[[ $cache_mode == off ]] || cache=/build-cache
python3 -B "$admission" cache "$cache"
exec {cache_lock}>>"$cache/.lock"
flock -n "$cache_lock" || { echo 'VRAM_GDN_CACHE_BUSY' >&2; exit 3; }
# Only a private configuration-keyed directory is writable. Reconfigure before
# each incremental build; unrelated production and RCCL objects are never read.
cat > /tmp/vram-gdn-build.sh <<'VRAM_GDN_BUILD'
set -euo pipefail
build=$2
cmake -S /work -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_HIP_GFX906=ON \
  -DSTRATA_HC_PERSIST_BUILD=OFF -DSTRATA_TP2_BUILD=OFF -DSTRATA_TP2_GDN_BUILD=ON \
  -DSTRATA_TP2_GDN_VRAM_BUILD=ON -DSTRATA_TP2_GDN_RCCL_BUILD=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 -DSTRATA_GGML_DIR=/work/deps/llama.cpp \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
host_gates=(tp2_shard_policy tp_layer_layout_test native_expert_call_policy
  tp_gdn_weights_test tp_hc_layout_test tp_gdn_gather_layout_test)
if [[ $3 == clean ]]; then cmake --build "$build" --target clean; fi
cmake --build "$build" --target "${host_gates[@]}" tp2_gdn_layer --parallel "$1"
for gate in "${host_gates[@]}"; do
  "$build/$gate"
  printf "VRAM_GDN_HOST_GATE_PASS target=%s\n" "$gate"
done
VRAM_GDN_BUILD
build=$(python3 -B "$admission" cache-config "$cache" "$image" /tmp/vram-gdn-compiler.txt /tmp/vram-gdn-build.sh /work/CMakeLists.txt)
rebuild=$(python3 -B "$admission" cache-prepare "$build" /work "$expected_sources")
printf 'VRAM_GDN_BUILD_CONFIG directory=%s jobs=%s mode=%s\n' "$build" "$jobs" "$rebuild"
timeout --signal=TERM --kill-after=10s "${build_deadline}s" bash /tmp/vram-gdn-build.sh "$jobs" "$build" "$rebuild"
python3 -B "$admission" cache-commit "$build" /work "$expected_sources"
binary=$build/tp2_gdn_layer
binary_sha=$(sha256sum "$binary"); binary_sha=${binary_sha%% *}
printf 'VRAM_GDN_BINARY_SHA256 %s\n' "$binary_sha"
timeout --signal=TERM --kill-after=3s 10s ldd "$binary" | tee /tmp/vram-gdn-dependencies.log
if grep -Eq 'not found|lib(rccl|nccl)' /tmp/vram-gdn-dependencies.log; then
  echo 'VRAM_GDN_PREFLIGHT_BLOCKED unresolved or forbidden RCCL dependency' >&2; exit 2
fi
echo 'VRAM_GDN_BUILD_GATE PASS'
fixed_args=(--execution hybrid-vram --layer 0 --mode 8 --hc-dispatch production-check
  --vram-timeout-us 100000 --vram-watchdog-ms 10000 --vram-delay-ms 20)
run_case() {
  local scenario=$1 first=$2 label=$3 log="/tmp/vram-gdn-$3.log"
  local args=("${model_args[@]}" "${fixed_args[@]}" --vram-scenario "$scenario" --vram-launch-first "$first")
  if [[ $scenario == normal ]]; then
    args+=(--benchmark --bench-warmup 3 --bench-trials 4 --bench-block-calls 16 --bench-block-trials 4)
  fi
  idle_gate "$label"
  [[ $(python3 -B "$admission" manifest /work) == "$expected_sources" ]] || { echo 'VRAM_GDN_SOURCE_SHA_MISMATCH' >&2; return 4; }
  printf '%s  %s\n' "$binary_sha" "$binary" | sha256sum -c -
  printf 'VRAM_GDN_PROCESS_BEGIN scenario=%s bound_s=%s\n' "$label" "$runtime_deadline"
  set +e
  timeout --signal=TERM --kill-after=10s "${runtime_deadline}s" "$binary" "${args[@]}" 2>&1 | tee "$log"
  status=("${PIPESTATUS[@]}")
  set -e
  printf 'VRAM_GDN_PROCESS_EXIT scenario=%s code=%s\n' "$label" "${status[0]}"
  (( status[1] == 0 )) || { echo 'VRAM_GDN_LOG_FAILURE' >&2; return "${status[1]}"; }
  (( status[0] == 0 )) || return "${status[0]}"
  if [[ $scenario == normal ]]; then python3 -B "$admission" normal "$log"
  else python3 -B "$admission" fault "$log" "$scenario" "$first"; fi
}
# Fault cases get fresh processes and must pass before any timing is attempted.
for scenario in delayed-rank missing-rank; do
  for first in 0 1; do run_case "$scenario" "$first" "$scenario-$first"; done
done
run_case normal 0 normal
[[ $(python3 -B "$admission" manifest /work) == "$expected_sources" ]] || { echo 'VRAM_GDN_SOURCE_SHA_MISMATCH' >&2; exit 4; }
echo 'VRAM_GDN_CONTAINER_COMPLETE fault_processes=4 normal_processes=1'
VRAM_GDN_CONTAINER
}
# Authorization is outside captured output and operation deadlines.
set +e
(set -e; validate_host "$@") 2>&1 | tee "$LOG"
STATUS=("${PIPESTATUS[@]}"); RC=${STATUS[0]}
if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
if (( RC == 0 )); then
  authorize_docker; RC=$?
  printf 'VRAM_GDN_AUTH_EXIT code=%s\n' "$RC" | tee -a "$LOG"
  STATUS=("${PIPESTATUS[@]}")
  if (( RC != 0 )); then
    printf 'VRAM_GDN_SETUP_FAILURE stage=authorization code=%s container_attempted=0\n' "$RC" | tee -a "$LOG"
  elif (( STATUS[1] != 0 )); then RC=${STATUS[1]}
  else
    (set -e; main "$@") 2>&1 | tee -a "$LOG"
    STATUS=("${PIPESTATUS[@]}"); RC=${STATUS[0]}
    if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
  fi
fi
if (( RC == 0 )); then
  # Re-read the durable full transcript after verified daemon-side cleanup.
  python3 -B "$ROOT/tools/tp2_vram_gdn_admission.py" host "$LOG" "$CONTAINER_NAME" 2>&1 | tee -a "$LOG"
  STATUS=("${PIPESTATUS[@]}"); RC=${STATUS[0]}
  if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
fi
if (( RC == 77 )); then echo 'SKIPPED is not a pass.' | tee -a "$LOG"; fi
if (( RC == 70 || RC == 124 || RC == 137 || RC == 143 )); then
  echo 'Unexpected watchdog/timeout/kill: inspect GPU state and verified container cleanup before another run.' | tee -a "$LOG"
fi
if (( RC == 0 )); then
  echo 'VRAM_GDN_SUITE_PASS staged=3,3 fault_processes=4 normal_processes=1 studies=2 shapes=1,2,4,5,8 scope=one-layer-wall-time generation-throughput=unmeasured' | tee -a "$LOG"
  STATUS=("${PIPESTATUS[@]}"); if (( STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
fi
printf 'TP2_VRAM_GDN_EXIT=%s LOG=%s\n' "$RC" "$LOG" | tee -a "$LOG"
STATUS=("${PIPESTATUS[@]}")
if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
exit "$RC"
