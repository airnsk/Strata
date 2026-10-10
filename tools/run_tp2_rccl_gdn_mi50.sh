#!/usr/bin/env bash
# Opt-in real-weight RCCL GDN-layer gates and whole-layer paired benchmark.
# Requires an already verified isolated artifact. Never installs or falls back.
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: TP2_RCCL_ARTIFACT=ARTIFACT bash tools/run_tp2_rccl_gdn_mi50.sh MODEL_ROOT \
  --pack /models/PACK --gguf /models/SHARD [--gguf /models/SHARD ...] [option value ...]
Only the installed pinned MI50 image and this exact verified RCCL artifact are used.
MODEL_ROOT is an existing absolute directory, mounted read-only. Both GPUs must be idle.
Required: TP2_RCCL_ARTIFACT is one direct child directory of this checkout.
Options: --layer 0..47 (non-PLE GDN only) --mode 7|8
  --bench-warmup 1..100 --bench-trials 2..1000
  --bench-block-calls 2..64 --bench-block-trials 2..20
Defaults: layer=0, mode=8, warmup=3, trials=12, block-calls=16, block-trials=4.
Runs both launch orders of delayed/missing-rank real-layer faults, normal full
correctness and two lifecycle checks, then whole-layer paired timing in that process.
No timings are accepted from fault processes. A skip or missing gate is a failure.
Optional env: BUILD_JOBS=48 TP2_TIMEOUT=1200 TP2_BUILD_TIMEOUT=600 TP2_OUTER_TIMEOUT=4200.
Deadlines are seconds; outer bounds discovery, build and every layer subprocess.
Sudo authorization precedes timed operations and is never captured in logs.
Cleanup may explicitly reauthorize on the controlling terminal after a long run.
No downloads, packages, runtime replacement, service stops, or clock/power changes.
EOF
}
if [[ ${1:-} == --help && $# == 1 ]]; then usage; exit 0; fi
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG="$ROOT/tp2-rccl-gdn-$RUN_ID.log"
IMAGE=sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca
CONTAINER_NAME="strata-tp2-rccl-gdn-$RUN_ID"
CONTAINER_STARTED=0
DOCKER=(docker)
SOURCE_FILES=(CMakeLists.txt tests/hip/tp2_gdn_layer.cpp src/core/tp_gdn_layer.cpp
  include/strata/core/tp_gdn_layer.hpp src/core/tp_gdn_rccl.cpp src/core/tp_gdn_rccl.hpp
  src/core/tp_gdn_rccl_workers.hpp src/kernels/cuda/tp_gdn_exchange.cu
  include/strata/kernels/tp_gdn_exchange.hpp include/strata/kernels/tp_gdn_gather_layout.hpp
  tests/core/tp_gdn_gather_layout_test.cpp tests/core/tp_gdn_rccl_workers_test.cpp
  tools/tp2_rccl_preflight.py tools/verify_rccl_gfx906.py tools/mi50_pipeline.py
  tools/run_tp2_rccl_gdn_mi50.sh)
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
      printf 'RCCL_CONTAINER_CLEANUP_UNVERIFIED name=%s remove_exit=%s inspect_exit=%s\n' "$CONTAINER_NAME" "$cleanup_rc" "$inspect_rc" >&2
      (( rc != 0 )) || rc=125
    else
      printf 'RCCL_CONTAINER_CLEANUP verified_absent=%s\n' "$CONTAINER_NAME"
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
  local deadline=${TP2_TIMEOUT:-1200} build_deadline=${TP2_BUILD_TIMEOUT:-600} outer_deadline=${TP2_OUTER_TIMEOUT:-4200}
  local value flag
  for value in "$deadline" "$build_deadline" "$outer_deadline"; do
    [[ "$value" =~ ^[1-9][0-9]{0,4}$ ]] && (( value <= 86400 )) || {
      echo 'Timeout variables must be integer seconds in 1..86400.' >&2; return 2;
    }
  done
  value=${BUILD_JOBS:-48}
  [[ "$value" =~ ^[1-9][0-9]{0,2}$ ]] && (( value <= 256 )) || {
    echo 'BUILD_JOBS must be an integer in 1..256.' >&2; return 2;
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
      --mode) [[ "$value" == 7 || "$value" == 8 ]] || { echo 'Mode must be 7 or 8.' >&2; return 2; } ;;
      --layer|--bench-warmup|--bench-trials|--bench-block-calls|--bench-block-trials)
        [[ "$value" =~ ^(0|[1-9][0-9]{0,3})$ ]] || { echo "Invalid integer: $flag=$value" >&2; return 2; }
        local lower=1 upper=1
        case "$flag" in
          --layer) lower=0; upper=47 ;; --bench-warmup) upper=100 ;;
          --bench-trials) lower=2; upper=1000 ;; --bench-block-calls) lower=2; upper=64 ;;
          --bench-block-trials) lower=2; upper=20 ;;
        esac
        (( value >= lower && value <= upper )) || { echo "Out of range: $flag=$value" >&2; return 2; }
        if [[ "$flag" == --layer ]] && (( value == 1 || value % 4 == 3 )); then
          echo 'Only non-PLE GDN layers are supported.' >&2; return 2
        fi
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
  for value in deps/llama.cpp/ggml/CMakeLists.txt "${SOURCE_FILES[@]}"; do
    [[ -f "$ROOT/$value" ]] || { echo "Missing source or pinned dependency: $value" >&2; return 2; }
  done
  model_mount_views "${original_args[@]}" >/dev/null || return 2
  [[ ${TP2_RCCL_ARTIFACT:-} =~ ^[a-zA-Z0-9_.-]+$ && $TP2_RCCL_ARTIFACT != . && $TP2_RCCL_ARTIFACT != .. ]] || {
    echo 'TP2_RCCL_ARTIFACT is required and must name one direct child directory in this checkout.' >&2; return 2;
  }
  [[ -d "$ROOT/$TP2_RCCL_ARTIFACT" && ! -L "$ROOT/$TP2_RCCL_ARTIFACT" &&
     -d "$ROOT/$TP2_RCCL_ARTIFACT/install" && ! -L "$ROOT/$TP2_RCCL_ARTIFACT/install" &&
     -f "$ROOT/$TP2_RCCL_ARTIFACT/artifact.json" && ! -L "$ROOT/$TP2_RCCL_ARTIFACT/artifact.json" ]] || {
    echo 'Missing isolated RCCL installation/manifest, or an unexpected artifact symlink.' >&2; return 2;
  }
}
main() {
  local deadline=${TP2_TIMEOUT:-1200} build_deadline=${TP2_BUILD_TIMEOUT:-600} outer_deadline=${TP2_OUTER_TIMEOUT:-4200}
  local model_root view views_text
  views_text=$(model_mount_views "$@")
  local views=() model_mount=()
  mapfile -t views <<< "$views_text"
  model_root=${views[0]}; shift
  local args=("$@")
  model_mount=(--mount "type=bind,src=$model_root,dst=/models,readonly")
  for view in "${views[@]:1}"; do
    [[ "$view" == /models ]] || model_mount+=(--mount "type=bind,src=$model_root,dst=$view,readonly")
  done
  printf 'RCCL_GDN_RUN id=%s head=%s image=%s\n' "$RUN_ID" "$(git -C "$ROOT" rev-parse HEAD)" "$IMAGE"
  printf 'RCCL_GDN_STAND repo=%q model-root=%q log=%q\n' "$ROOT" "$model_root" "$LOG"
  printf 'RCCL_GDN_ARGUMENTS'; printf ' %q' "${args[@]}"; printf '\n'
  printf 'RCCL_GDN_BOUNDS runtime_s=%s build_s=%s outer_s=%s host_guard_s=%s\n' "$deadline" "$build_deadline" "$outer_deadline" "$((outer_deadline + 30))"
  echo 'Both MI50 cards must be idle. No service is stopped and no power/clock setting is changed.'
  printf 'RCCL_GDN_TRACKED_DIFF_SHA256 '; git -C "$ROOT" diff --binary HEAD -- | sha256sum
  git -C "$ROOT" status --short
  local source_file
  for source_file in "${SOURCE_FILES[@]}"; do sha256sum "$ROOT/$source_file"; done
  local image_id image_rc=0
  echo 'RCCL_GDN_IMAGE_INSPECT_BEGIN bound_s=20'
  image_id=$(timeout --signal=TERM --kill-after=3s 20s "${DOCKER[@]}" image inspect --format '{{.Id}}' "$IMAGE") || image_rc=$?
  printf 'RCCL_GDN_IMAGE_INSPECT_EXIT code=%s\n' "$image_rc"
  if (( image_rc != 0 )); then
    printf 'RCCL_GDN_SETUP_FAILURE stage=image-inspect code=%s container_attempted=0\n' "$image_rc" >&2
    return "$image_rc"
  fi
  printf 'RCCL_GDN_RESOLVED_IMAGE %s\n' "$image_id"
  [[ "$image_id" == "$IMAGE" ]] || {
    echo 'RCCL_GDN_SETUP_FAILURE stage=image-verification code=2 container_attempted=0: installed image ID differs from the pinned stand image.' >&2
    return 2
  }
  local artifact_args=()
  # The builder writes an owner-only manifest. Root with all capabilities
  # dropped cannot bypass its mode bits; use the same invoking user instead.
  local uid gid host_groups device_groups group node
  uid=$(id -u); gid=$(id -g); host_groups=$(id -G)
  [[ $uid =~ ^[0-9]+$ && $gid =~ ^[0-9]+$ && -n $host_groups ]] || {
    echo 'RCCL_GDN_SETUP_FAILURE stage=artifact-identity container_attempted=0: invalid host identity' >&2; return 2;
  }
  local device_nodes=(/dev/kfd /dev/dri)
  for node in /dev/dri/card[0-9]* /dev/dri/renderD[0-9]*; do
    [[ ! -e $node ]] || device_nodes+=("$node")
  done
  device_groups=$(stat -Lc '%g' -- "${device_nodes[@]}") || {
    echo 'RCCL_GDN_SETUP_FAILURE stage=artifact-identity container_attempted=0: cannot read GPU device groups' >&2; return 2;
  }
  [[ -n $device_groups ]] || {
    echo 'RCCL_GDN_SETUP_FAILURE stage=artifact-identity container_attempted=0: missing GPU device groups' >&2; return 2;
  }
  artifact_args=(--user "$uid:$gid" -e "TP2_RCCL_ARTIFACT=/work/$TP2_RCCL_ARTIFACT")
  local -A seen_groups=(["$gid"]=1)
  local added_groups=()
  # Numeric IDs preserve host group permissions without assuming image names.
  for group in $host_groups $device_groups; do
    [[ $group =~ ^[0-9]+$ ]] || {
      echo 'RCCL_GDN_SETUP_FAILURE stage=artifact-identity container_attempted=0: invalid group ID' >&2; return 2;
    }
    if [[ -z ${seen_groups[$group]:-} ]]; then
      artifact_args+=(--group-add "$group"); added_groups+=("$group"); seen_groups[$group]=1
    fi
  done
  printf 'RCCL_GDN_ARTIFACT_IDENTITY uid=%s gid=%s supplementary_groups=%s\n' "$uid" "$gid" "${added_groups[*]}"
  trap cleanup EXIT
  CONTAINER_STARTED=1
  printf 'RCCL_GDN_CONTAINER_LAUNCH name=%s\n' "$CONTAINER_NAME"
  timeout --signal=TERM --kill-after=15s "$((outer_deadline + 30))s" "${DOCKER[@]}" run \
    --name "$CONTAINER_NAME" --rm --pull never --network none --read-only --cap-drop ALL \
    --tmpfs /tmp:rw,exec,nosuid,size=8g --shm-size=512m -e HOME=/tmp \
    --device=/dev/kfd --device=/dev/dri -e HIP_VISIBLE_DEVICES=0,1 \
    -e STRATA_HC_PERSIST=0 -e STRATA_GR_V3=0 -e STRATA_GR_SPLIT=0 \
    -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=ALL -e NCCL_DEBUG_FILE=/dev/stdout \
    --mount "type=bind,src=$ROOT,dst=/work,readonly" \
    "${artifact_args[@]}" "${model_mount[@]}" --workdir /work --entrypoint /usr/bin/timeout -i "$IMAGE" \
    --signal=TERM --kill-after=15s "${outer_deadline}s" /bin/bash -s -- "$deadline" "$build_deadline" "${BUILD_JOBS:-48}" "${args[@]}" <<'RCCL_CONTAINER'
set -euo pipefail
echo 'RCCL_GDN_CONTAINER_ENTER'
export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH
runtime_deadline=$1; build_deadline=$2; jobs=$3; shift 3
model_args=(); bench_args=()
while (( $# )); do
  case "$1" in
    --bench-*) bench_args+=("$1" "$2") ;;
    *) model_args+=("$1" "$2") ;;
  esac
  shift 2
done
for key in NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE; do
  printf 'RCCL_GDN_DIAGNOSTIC_ENV %s=%s\n' "$key" "${!key:-<unset>}"
done
for tool in python3 cmake ninja hipcc timeout ldd sha256sum awk; do
  command -v "$tool" >/dev/null || { echo "RCCL_GDN_PREFLIGHT_BLOCKED missing installed tool: $tool" >&2; exit 2; }
done
# Mandatory exact artifact; neither installed RCCL discovery nor a rebuild is a fallback.
[[ -n ${TP2_RCCL_ARTIFACT:-} ]] || { echo 'RCCL_GDN_PREFLIGHT_BLOCKED no isolated artifact' >&2; exit 2; }
python3 -B /work/tools/verify_rccl_gfx906.py --prefix "$TP2_RCCL_ARTIFACT/install" \
  --check-manifest "$TP2_RCCL_ARTIFACT/artifact.json" --output /tmp/rccl-artifact-reverified.json
artifact_hash=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["library"]["sha256"])' \
  /tmp/rccl-artifact-reverified.json)
python3 -B /work/tools/tp2_rccl_preflight.py --root "$TP2_RCCL_ARTIFACT/install" \
  --expected-sha256 "$artifact_hash" --output /tmp/rccl-env
source /tmp/rccl-env
# Exact discovered paths must remain the same verified library AND header.
python3 - "$TP2_RCCL_ARTIFACT/install" /tmp/rccl-artifact-reverified.json \
  "$RCCL_LIBRARY" "$RCCL_INCLUDE_DIR" "$RCCL_HEADER" <<'IDENTITY'
import json, pathlib, sys
root, manifest, library, include, header = sys.argv[1:]
root = pathlib.Path(root).resolve(strict=True)
m = json.loads(pathlib.Path(manifest).read_text())
for key, selected in (("library", library), ("header", header)):
    actual = pathlib.Path(selected)
    expected = (root / m[key]["path"]).resolve(strict=True)
    if not actual.is_absolute() or actual.resolve(strict=True) != expected or not expected.is_relative_to(root):
        raise SystemExit("RCCL_GDN_PREFLIGHT_BLOCKED discovered " + key + " identity differs")
if pathlib.Path(include) != pathlib.Path(header).parent:
    raise SystemExit("RCCL_GDN_PREFLIGHT_BLOCKED unexpected header directory")
print("RCCL_GDN_ARTIFACT_IDENTITY_PASS")
IDENTITY
printf 'RCCL_GDN_ISOLATED_ARTIFACT path=%s sha256=%s\n' "$TP2_RCCL_ARTIFACT" "$artifact_hash"
printf 'RCCL_GDN_LOADER_PATH %s\n' "$LD_LIBRARY_PATH"
timeout --signal=TERM --kill-after=5s 20s hipcc --version
# Build and all generated sources stay in the private container tmpfs.
timeout --signal=TERM --kill-after=10s "${build_deadline}s" bash -c '
set -euo pipefail
cmake -S /work -B /tmp/build-tp2-rccl-gdn -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_HIP_GFX906=ON \
  -DSTRATA_HC_PERSIST_BUILD=OFF -DSTRATA_TP2_BUILD=OFF -DSTRATA_TP2_GDN_BUILD=ON \
  -DSTRATA_TP2_GDN_RCCL_BUILD=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 -DSTRATA_GGML_DIR=/work/deps/llama.cpp \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DSTRATA_RCCL_LIBRARY="$RCCL_LIBRARY" -DSTRATA_RCCL_INCLUDE_DIR="$RCCL_INCLUDE_DIR" \
  -DSTRATA_RCCL_HEADER="$RCCL_HEADER"
host_gates=(tp2_shard_policy tp_layer_layout_test native_expert_call_policy tp_gdn_weights_test
  tp_hc_layout_test tp_gdn_gather_layout_test tp_gdn_rccl_workers_test)
cmake --build /tmp/build-tp2-rccl-gdn --target "${host_gates[@]}" tp2_gdn_layer --parallel "$1"
for gate in "${host_gates[@]}"; do
  "/tmp/build-tp2-rccl-gdn/$gate"
  printf "RCCL_GDN_HOST_GATE_PASS target=%s\n" "$gate"
done
' build "$jobs"
binary=/tmp/build-tp2-rccl-gdn/tp2_gdn_layer
sha256sum "$binary"
ldd "$binary" | tee /tmp/binary-dependencies.log
if grep -q 'not found' /tmp/binary-dependencies.log; then
  echo 'RCCL_GDN_PREFLIGHT_BLOCKED executable has unresolved dependencies' >&2; exit 2
fi
echo 'RCCL_GDN_BUILD_GATE PASS'
telemetry() {
  date -u +RCCL_GDN_GPU_TELEMETRY_%Y-%m-%dT%H:%M:%SZ
  if command -v rocm-smi >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=3s 10s rocm-smi --showproductname --showuse --showclocks --showpower --showtemp || true
  else
    echo 'rocm-smi unavailable; utilization/clocks/power/temperature unavailable'
  fi
}
run_case() {
  local phase=$1 scenario=$2 first=$3 bound=$4; shift 4
  local log="/tmp/${phase}-${first}.log" rc=0 pipe_status life peer=$((1 - first)) watchdog
  printf 'RCCL_GDN_PROCESS_BEGIN phase=%s scenario=%s launch_first=%s bound_s=%s\n' "$phase" "$scenario" "$first" "$bound"
  set +e
  timeout --signal=TERM --kill-after=10s "${bound}s" "$binary" \
    "${model_args[@]}" --execution hybrid-rccl --rccl-library "$RCCL_LIBRARY" \
    --rccl-timeout-ms 10000 --rccl-init-timeout-ms 60000 \
    --rccl-scenario "$scenario" --rccl-launch-first "$first" --rccl-delay-ms 200 "$@" 2>&1 | tee "$log"
  pipe_status=("${PIPESTATUS[@]}"); rc=${pipe_status[0]}
  set -e
  printf 'RCCL_GDN_PROCESS_EXIT phase=%s scenario=%s launch_first=%s code=%s\n' "$phase" "$scenario" "$first" "$rc"
  (( pipe_status[1] == 0 )) || { echo 'RCCL_GDN_LOG_FAILURE' >&2; return "${pipe_status[1]}"; }
  if [[ "$scenario" != normal ]]; then
    if grep -Eq '^(BENCH_|BLOCK_|RCCL_LAYER_EXACT_GATE_PASS|RCCL_LAYER_LIFECYCLE_PASS)' "$log"; then
      echo 'RCCL_GDN_FAULT_GATE FAIL unexpected normal/timing gate in fault process' >&2; return 4
    fi
    if [[ "$scenario" == missing-rank ]]; then
      watchdog=$(grep '^WATCHDOG_TIMEOUT ' "$log" | grep -E '(^| )stage=layer-missing-rank( |$)' |
        grep -E '(^| )exit=70( |$)' | grep -E '(^| )completed_ranks=1( |$)' || true)
      if (( rc != 70 )) || [[ -z "$watchdog" ]] ||
          ! grep -Fxq "RCCL_LAYER_FAULT_ARMED scenario=missing-rank launch_rank=$first omitted_or_delayed_rank=$peer" "$log" ||
          ! grep -Fxq "RCCL_LAYER_PEER_OMITTED rank=$peer" "$log" ||
          ! grep -Fxq "RCCL_LAYER_GRAPH_LAUNCH_ENTER rank=$first" "$log" ||
          grep -q '^RCCL_LAYER_FAULT_PASS' "$log"; then
        echo 'RCCL_GDN_FAULT_GATE FAIL missing-rank did not reach the expected real-layer watchdog' >&2
        (( rc != 0 )) && return "$rc"
        return 4
      fi
      printf 'RCCL_GDN_FAULT_GATE EXPECTED_FAILURE scenario=missing-rank launch_first=%s exit=70\n' "$first"
    else
      (( rc == 0 )) || return "$rc"
      grep -Eq "^RCCL_LAYER_FAULT_PASS scenario=delayed-rank first=$first .*timing_admitted=0([[:space:]]|$)" "$log" &&
        grep -Fxq "RCCL_LAYER_FAULT_ARMED scenario=delayed-rank launch_rank=$first omitted_or_delayed_rank=$peer" "$log" || {
        echo 'RCCL_GDN_FAULT_GATE FAIL absent delayed-rank pass' >&2; return 4;
      }
    fi
  else
    (( rc == 0 )) || return "$rc"
    for life in 0 1; do
      grep -Eq "^RCCL_LAYER_LIFECYCLE_PASS life=$life([[:space:]]|$)" "$log" || {
        echo "RCCL_GDN_ADMISSION_FAIL absent lifecycle $life" >&2; return 4;
      }
    done
    grep -Eq '^RCCL_LAYER_EXACT_GATE_PASS([[:space:]]|$)' "$log" || {
      echo 'RCCL_GDN_ADMISSION_FAIL absent full exactness gate' >&2; return 4;
    }
    if [[ "$phase" == benchmark ]]; then
      grep -Eq '^BENCH_GATE PASS([[:space:]]|$)' "$log" || {
        echo 'RCCL_GDN_ADMISSION_FAIL absent whole-layer benchmark gate' >&2; return 4;
      }
      # Markers emitted after timing cannot retroactively admit earlier samples.
      awk '
        /^RCCL_LAYER_LIFECYCLE_PASS life=0([[:space:]]|$)/ { life0=1 }
        /^RCCL_LAYER_LIFECYCLE_PASS life=1([[:space:]]|$)/ { life1=1 }
        /^RCCL_LAYER_EXACT_GATE_PASS([[:space:]]|$)/ { exact=1 }
        /^(BENCH_|BLOCK_)/ { if (!life0 || !life1 || !exact) bad=1 }
        END { exit bad }
      ' "$log" || { echo 'RCCL_GDN_ADMISSION_FAIL timing precedes admission gates' >&2; return 4; }
    elif grep -q '^BENCH_' "$log"; then
      echo 'RCCL_GDN_ADMISSION_FAIL unexpected timing in correctness process' >&2; return 4
    fi
  fi
}
telemetry
# Fresh processes keep fault captures and expected aborts out of normal timings.
# 60 s initialization + 10 s execution watchdog + model/capture/cleanup allowance.
for first in 0 1; do
  run_case delayed-rank delayed-rank "$first" 105
  run_case missing-rank missing-rank "$first" 105
done
# Normal checks all prefixes, both rank orders and two lifecycles before timing.
run_case benchmark normal 0 "$runtime_deadline" --benchmark "${bench_args[@]}"
telemetry
echo 'RCCL_GDN_LAYER_GATES_PASS'
RCCL_CONTAINER
}
# Keep failures from the real runner and from tee. The final outer status is
# written into the durable host log, including preflight/build/timeout failures.
set +e
(set -e; validate_host "$@") 2>&1 | tee "$LOG"
STATUS=("${PIPESTATUS[@]}")
RC=${STATUS[0]}
if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
if (( RC == 0 )); then
  # This call is deliberately outside every pipeline and operation deadline.
  # Only its status is logged; sudo reads/writes the controlling terminal itself.
  authorize_docker
  RC=$?
  printf 'RCCL_GDN_AUTH_EXIT code=%s\n' "$RC" | tee -a "$LOG"
  STATUS=("${PIPESTATUS[@]}")
  if (( RC != 0 )); then
    printf 'RCCL_GDN_SETUP_FAILURE stage=authorization code=%s container_attempted=0\n' "$RC" | tee -a "$LOG"
  elif (( STATUS[1] != 0 )); then
    RC=${STATUS[1]}
  else
    (set -e; main "$@") 2>&1 | tee -a "$LOG"
    STATUS=("${PIPESTATUS[@]}")
    RC=${STATUS[0]}
    if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
  fi
fi
if (( RC == 0 )) && ! grep -Fxq 'RCCL_GDN_LAYER_GATES_PASS' "$LOG"; then
  echo 'RCCL_GDN_ADMISSION_FAIL missing suite completion marker' | tee -a "$LOG"
  RC=4
fi
if (( RC == 77 )); then echo 'SKIPPED is not a pass.' | tee -a "$LOG"; fi
if (( RC == 70 || RC == 124 || RC == 137 )); then
  if grep -q '^RCCL_GDN_PROCESS_BEGIN ' "$LOG"; then
    echo 'Unexpected watchdog/timeout/kill after real-layer launch: inspect GPU state before another run.' | tee -a "$LOG"
  elif grep -q '^RCCL_GDN_CONTAINER_LAUNCH ' "$LOG"; then
    echo 'Container startup/setup timeout or kill before observed real-layer launch: inspect the setup log and container cleanup status.' | tee -a "$LOG"
  else
    echo 'Host setup timeout or kill before container launch. No real-layer process ran.' | tee -a "$LOG"
  fi
fi
if (( RC == 0 )); then
  echo 'RCCL_GDN_SUITE_PASS' | tee -a "$LOG"
  STATUS=("${PIPESTATUS[@]}")
  if (( STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
fi
printf 'TP2_RCCL_GDN_EXIT=%s LOG=%s\n' "$RC" "$LOG" | tee -a "$LOG"
FOOTER_STATUS=("${PIPESTATUS[@]}")
if (( RC == 0 && FOOTER_STATUS[1] != 0 )); then RC=${FOOTER_STATUS[1]}; fi
exit "$RC"
