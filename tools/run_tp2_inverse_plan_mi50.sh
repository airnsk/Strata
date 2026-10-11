#!/usr/bin/env bash
# Opt-in same-geometry old-hybrid versus once-built inverse Down route plan.
# No RCCL, transport replacement, installation, or production dispatch change.
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: bash tools/run_tp2_inverse_plan_mi50.sh MODEL_ROOT \
  --pack /models/PACK --gguf /models/SHARD [--gguf /models/SHARD ...] [option value ...]
Only the installed pinned MI50 image is used. No RCCL artifact is needed.
MODEL_ROOT is an existing absolute directory, mounted read-only. Both GPUs must be idle.
Options: --layer 0..47 (non-PLE GDN only) --mode 7|8
  --hc-dispatch legacy|production-check (default legacy)
  --bench-warmup 1..100 --bench-trials 2..1000
  --bench-block-calls 2..64 --bench-block-trials 2..20
Defaults: layer=0, mode=8, warmup=3, trials=12, block-calls=16, block-trials=4.
Checks exact old-hybrid parity on both owners for T1..8, all prefixes and
continuations, both banks; unchanged full-reference oracles precede timing.
Runs unprofiled burst and sustained whole-layer pairs at T1/2/4/5/8, including
inverse plan generation in the timed proposal. A skip or missing gate fails.
production-check uses the production HC selector on each GPU before captures;
it can select plain/split/staged. Legacy preserves historical unchecked dispatch.
Optional env: BUILD_JOBS=48 TP2_TIMEOUT=1200 TP2_BUILD_TIMEOUT=600 TP2_OUTER_TIMEOUT=2400.
Deadlines are seconds; outer bounds discovery, build and all subprocesses.
Sudo authorization precedes timed operations and is never captured in logs.
Cleanup may explicitly reauthorize on the controlling terminal after a long run.
No downloads, packages, runtime replacement, service stops, or clock/power changes.
EOF
}
if [[ ${1:-} == --help && $# == 1 ]]; then usage; exit 0; fi
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG="$ROOT/tp2-inverse-plan-$RUN_ID.log"
IMAGE=sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca
CONTAINER_NAME="strata-tp2-inverse-plan-$RUN_ID"
CONTAINER_STARTED=0
DOCKER=(docker)
SOURCE_FILES=(CMakeLists.txt tests/hip/tp2_gdn_layer.cpp src/core/tp_gdn_layer.cpp
  src/kernels/cuda/fused_gr.cu include/strata/kernels/fused_gr.hpp
  include/strata/core/tp_gdn_layer.hpp src/kernels/cuda/iq_kernels.cu
  src/kernels/cuda/verify_kernels.cu src/kernels/cuda/native_down_plan.cuh
  include/strata/kernels/iq_kernels.hpp
  include/strata/kernels/native_down_plan.hpp include/strata/kernels/verify_kernels.hpp
  tests/hip/native_down_plan_parity.cpp tests/core/native_down_plan_test.cpp
  src/kernels/cuda/tp_gdn_exchange.cu include/strata/kernels/tp_gdn_exchange.hpp
  include/strata/kernels/tp_gdn_gather_layout.hpp tests/core/tp_gdn_gather_layout_test.cpp
  tools/mi50_pipeline.py tools/run_tp2_inverse_plan_mi50.sh)
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
      printf 'INVERSE_CONTAINER_CLEANUP_UNVERIFIED name=%s remove_exit=%s inspect_exit=%s\n' "$CONTAINER_NAME" "$cleanup_rc" "$inspect_rc" >&2
      (( rc != 0 )) || rc=125
    else
      printf 'INVERSE_CONTAINER_CLEANUP verified_absent=%s\n' "$CONTAINER_NAME"
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
      --hc-dispatch) [[ "$value" == legacy || "$value" == production-check ]] || {
        echo 'HC dispatch must be legacy or production-check.' >&2; return 2;
      } ;;
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

}
main() {
  local deadline=${TP2_TIMEOUT:-1200} build_deadline=${TP2_BUILD_TIMEOUT:-600} outer_deadline=${TP2_OUTER_TIMEOUT:-2400}
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
  printf 'INVERSE_PLAN_RUN id=%s head=%s image=%s\n' "$RUN_ID" "$(git -C "$ROOT" rev-parse HEAD)" "$IMAGE"
  printf 'INVERSE_PLAN_STAND repo=%q model-root=%q log=%q\n' "$ROOT" "$model_root" "$LOG"
  printf 'INVERSE_PLAN_ARGUMENTS'; printf ' %q' "${args[@]}"; printf '\n'
  printf 'INVERSE_PLAN_BOUNDS runtime_s=%s build_s=%s outer_s=%s host_guard_s=%s\n' "$deadline" "$build_deadline" "$outer_deadline" "$((outer_deadline + 30))"
  echo 'Both MI50 cards must be idle. No service is stopped and no power/clock setting is changed.'
  printf 'INVERSE_PLAN_TRACKED_DIFF_SHA256 '; git -C "$ROOT" diff --binary HEAD -- | sha256sum
  git -C "$ROOT" status --short
  local source_file
  for source_file in "${SOURCE_FILES[@]}"; do sha256sum "$ROOT/$source_file"; done
  local image_id image_rc=0
  echo 'INVERSE_PLAN_IMAGE_INSPECT_BEGIN bound_s=20'
  image_id=$(timeout --signal=TERM --kill-after=3s 20s "${DOCKER[@]}" image inspect --format '{{.Id}}' "$IMAGE") || image_rc=$?
  printf 'INVERSE_PLAN_IMAGE_INSPECT_EXIT code=%s\n' "$image_rc"
  if (( image_rc != 0 )); then
    printf 'INVERSE_PLAN_SETUP_FAILURE stage=image-inspect code=%s container_attempted=0\n' "$image_rc" >&2
    return "$image_rc"
  fi
  printf 'INVERSE_PLAN_RESOLVED_IMAGE %s\n' "$image_id"
  [[ "$image_id" == "$IMAGE" ]] || {
    echo 'INVERSE_PLAN_SETUP_FAILURE stage=image-verification code=2 container_attempted=0: installed image ID differs from the pinned stand image.' >&2
    return 2
  }
  local identity_args=()
  # Preserve ordinary host and GPU-node group access with all capabilities dropped.
  local uid gid host_groups device_groups group node
  uid=$(id -u); gid=$(id -g); host_groups=$(id -G)
  [[ $uid =~ ^[0-9]+$ && $gid =~ ^[0-9]+$ && -n $host_groups ]] || {
    echo 'INVERSE_PLAN_SETUP_FAILURE stage=device-identity container_attempted=0: invalid host identity' >&2; return 2;
  }
  local device_nodes=(/dev/kfd /dev/dri)
  for node in /dev/dri/card[0-9]* /dev/dri/renderD[0-9]*; do
    [[ ! -e $node ]] || device_nodes+=("$node")
  done
  device_groups=$(stat -Lc '%g' -- "${device_nodes[@]}") || {
    echo 'INVERSE_PLAN_SETUP_FAILURE stage=device-identity container_attempted=0: cannot read GPU device groups' >&2; return 2;
  }
  [[ -n $device_groups ]] || {
    echo 'INVERSE_PLAN_SETUP_FAILURE stage=device-identity container_attempted=0: missing GPU device groups' >&2; return 2;
  }
  identity_args=(--user "$uid:$gid")
  local -A seen_groups=(["$gid"]=1)
  local added_groups=()
  # Numeric IDs preserve host group permissions without assuming image names.
  for group in $host_groups $device_groups; do
    [[ $group =~ ^[0-9]+$ ]] || {
      echo 'INVERSE_PLAN_SETUP_FAILURE stage=device-identity container_attempted=0: invalid group ID' >&2; return 2;
    }
    if [[ -z ${seen_groups[$group]:-} ]]; then
      identity_args+=(--group-add "$group"); added_groups+=("$group"); seen_groups[$group]=1
    fi
  done
  printf 'INVERSE_PLAN_DEVICE_IDENTITY uid=%s gid=%s supplementary_groups=%s\n' "$uid" "$gid" "${added_groups[*]}"
  trap cleanup EXIT
  CONTAINER_STARTED=1
  printf 'INVERSE_PLAN_CONTAINER_LAUNCH name=%s\n' "$CONTAINER_NAME"
  timeout --signal=TERM --kill-after=15s "$((outer_deadline + 30))s" "${DOCKER[@]}" run \
    --name "$CONTAINER_NAME" --rm --pull never --network none --read-only --cap-drop ALL \
    --tmpfs /tmp:rw,exec,nosuid,size=8g --shm-size=512m -e HOME=/tmp \
    --device=/dev/kfd --device=/dev/dri -e HIP_VISIBLE_DEVICES=0,1 \
    -e STRATA_HC_PERSIST=0 -e STRATA_GR_V3=0 -e STRATA_GR_SPLIT=0 \
    --mount "type=bind,src=$ROOT,dst=/work,readonly" \
    "${identity_args[@]}" "${model_mount[@]}" --workdir /work --entrypoint /usr/bin/timeout -i "$IMAGE" \
    --signal=TERM --kill-after=15s "${outer_deadline}s" /bin/bash -s -- "$deadline" "$build_deadline" "${BUILD_JOBS:-48}" "${args[@]}" <<'INVERSE_CONTAINER'
set -euo pipefail
echo 'INVERSE_PLAN_CONTAINER_ENTER'
export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH
runtime_deadline=$1; build_deadline=$2; jobs=$3; shift 3
model_args=(); bench_args=(); hc_dispatch=legacy
while (( $# )); do
  case "$1" in
    --bench-*) bench_args+=("$1" "$2") ;;
    --hc-dispatch) hc_dispatch=$2; model_args+=("$1" "$2") ;;
    *) model_args+=("$1" "$2") ;;
  esac
  shift 2
done
for tool in python3 cmake ninja hipcc timeout ldd sha256sum awk; do
  command -v "$tool" >/dev/null || { echo "INVERSE_PLAN_PREFLIGHT_BLOCKED missing installed tool: $tool" >&2; exit 2; }
done
timeout --signal=TERM --kill-after=5s 20s hipcc --version
# Build and all generated sources stay in the private container tmpfs.
timeout --signal=TERM --kill-after=10s "${build_deadline}s" bash -c '
set -euo pipefail
cmake -S /work -B /tmp/build-tp2-inverse-plan -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_HIP_GFX906=ON \
  -DSTRATA_HC_PERSIST_BUILD=OFF -DSTRATA_TP2_BUILD=OFF -DSTRATA_TP2_GDN_BUILD=ON \
  -DSTRATA_TP2_GDN_RCCL_BUILD=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx906 -DSTRATA_GGML_DIR=/work/deps/llama.cpp \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
host_gates=(tp2_shard_policy tp_layer_layout_test native_expert_call_policy tp_gdn_weights_test
  tp_hc_layout_test tp_gdn_gather_layout_test native_down_plan_test)
cmake --build /tmp/build-tp2-inverse-plan --target "${host_gates[@]}" native_down_plan_parity tp2_gdn_layer --parallel "$1"
for gate in "${host_gates[@]}"; do
  "/tmp/build-tp2-inverse-plan/$gate"
  printf "INVERSE_PLAN_HOST_GATE_PASS target=%s\n" "$gate"
done
' build "$jobs"
binary=/tmp/build-tp2-inverse-plan/tp2_gdn_layer
sha256sum "$binary"
ldd "$binary" | tee /tmp/binary-dependencies.log
if grep -q 'not found' /tmp/binary-dependencies.log; then
  echo 'INVERSE_PLAN_PREFLIGHT_BLOCKED executable has unresolved dependencies' >&2; exit 2
fi
echo 'INVERSE_PLAN_BUILD_GATE PASS'
telemetry() {
  date -u +INVERSE_PLAN_GPU_TELEMETRY_%Y-%m-%dT%H:%M:%SZ
  if command -v rocm-smi >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=3s 10s rocm-smi --showproductname --showuse --showclocks --showpower --showtemp || true
  else
    echo 'rocm-smi unavailable; utilization/clocks/power/temperature unavailable'
  fi
}
run_case() {
  local log=/tmp/inverse-plan-benchmark.log rc=0 pipe_status
  printf 'INVERSE_PLAN_PROCESS_BEGIN phase=benchmark bound_s=%s\n' "$runtime_deadline"
  set +e
  timeout --signal=TERM --kill-after=10s "${runtime_deadline}s" "$binary" \
    "${model_args[@]}" --execution hybrid-inverse --benchmark "${bench_args[@]}" 2>&1 | tee "$log"
  pipe_status=("${PIPESTATUS[@]}"); rc=${pipe_status[0]}
  set -e
  printf 'INVERSE_PLAN_PROCESS_EXIT phase=benchmark code=%s\n' "$rc"
  (( pipe_status[1] == 0 )) || { echo 'INVERSE_PLAN_LOG_FAILURE' >&2; return "${pipe_status[1]}"; }
  (( rc == 0 )) || return "$rc"
  if [[ "$hc_dispatch" == production-check ]]; then
    # Selection provenance is additional to every existing numerical gate.
    # A late, duplicate or malformed marker cannot admit earlier samples.
    awk '
      /^HC_DISPATCH_SELECTED/ {
        if ($0 !~ /^HC_DISPATCH_SELECTED policy=production-check device=[01] hc-variant=[123] hc-variant-name=(plain|split|staged) check-invoked=1$/) bad=1
        if (($4=="hc-variant=1" && $5!="hc-variant-name=plain") ||
            ($4=="hc-variant=2" && $5!="hc-variant-name=split") ||
            ($4=="hc-variant=3" && $5!="hc-variant-name=staged")) bad=1
        if (++selected[$3]!=1 || initialized) bad=1
      }
      /^HC_DISPATCH_INIT_PASS/ {
        if ($0!="HC_DISPATCH_INIT_PASS policy=production-check devices=2 before-layer-setup=1" ||
            selected["device=0"]!=1 || selected["device=1"]!=1 || initialized++) bad=1
      }
      /^(BENCH_|BLOCK_)/ { if (!initialized) bad=1 }
      END { exit (bad || initialized!=1 || selected["device=0"]!=1 || selected["device=1"]!=1) }
    ' "$log" || { echo 'INVERSE_PLAN_ADMISSION_FAIL HC production selection missing, malformed or late' >&2; return 4; }
  fi
  for marker in INVERSE_PLAN_BANK_GATE_PASS INVERSE_PLAN_EXACT_GATE_PASS; do
    grep -Eq "^$marker([[:space:]]|$)" "$log" || {
      echo "INVERSE_PLAN_ADMISSION_FAIL absent $marker" >&2; return 4;
    }
  done
  grep -Eq '^BENCH_GATE PASS([[:space:]]|$)' "$log" || {
    echo 'INVERSE_PLAN_ADMISSION_FAIL absent whole-layer benchmark gate' >&2; return 4;
  }
  # A late marker cannot retroactively admit earlier samples. No instrumented
  # graph is prepared or run for this candidate, even outside timing.
  awk '
    /^INVERSE_PLAN_BANK_GATE_PASS([[:space:]]|$)/ { banks=1 }
    /^INVERSE_PLAN_EXACT_GATE_PASS([[:space:]]|$)/ { exact=1 }
    /^(BENCH_|BLOCK_)/ { if (!banks || !exact) bad=1 }
    /^(CAPTURE_PROFILE|PROFILE_DIAGNOSTIC|PROFILE_SCOPE)/ { bad=1 }
    END { exit bad }
  ' "$log" || { echo 'INVERSE_PLAN_ADMISSION_FAIL timing precedes admission or instrumentation found' >&2; return 4; }
  # Require both complete paired studies and every sustained/burst shape, rather
  # than accepting a success footer after a partial or mislabeled run.
  for baseline in tp-hybrid-captured single-gpu-captured; do
    for tokens in 1 2 4 5 8; do
      for kind in BENCH_SUMMARY BLOCK_SUMMARY; do
        grep -Eq "^$kind baseline=$baseline candidate=tp-hybrid-inverse mode=hybrid T=$tokens([[:space:]]|$)" "$log" || {
          echo "INVERSE_PLAN_ADMISSION_FAIL missing $kind baseline=$baseline T=$tokens" >&2; return 4;
        }
      done
    done
  done
}
telemetry
for device in 0 1; do
  peer=$((1 - device)); kernel_log="/tmp/inverse-plan-kernel-$device.log"
  printf 'INVERSE_PLAN_KERNEL_BEGIN device=%s peer=%s bound_s=%s\n' "$device" "$peer" "$runtime_deadline"
  set +e
  timeout --signal=TERM --kill-after=10s "${runtime_deadline}s" \
    /tmp/build-tp2-inverse-plan/native_down_plan_parity --device "$device" --peer-device "$peer" 2>&1 | tee "$kernel_log"
  kernel_status=("${PIPESTATUS[@]}")
  set -e
  printf 'INVERSE_PLAN_KERNEL_EXIT device=%s code=%s\n' "$device" "${kernel_status[0]}"
  (( kernel_status[1] == 0 )) || exit "${kernel_status[1]}"
  (( kernel_status[0] == 0 )) || exit "${kernel_status[0]}"
  grep -Fxq 'NATIVE_DOWN_PLAN_PARITY_PASS' "$kernel_log" || {
    echo 'INVERSE_PLAN_ADMISSION_FAIL missing synthetic GPU route validation pass' >&2; exit 4;
  }
done
run_case
telemetry
echo 'INVERSE_PLAN_LAYER_GATES_PASS'
INVERSE_CONTAINER
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
  printf 'INVERSE_PLAN_AUTH_EXIT code=%s\n' "$RC" | tee -a "$LOG"
  STATUS=("${PIPESTATUS[@]}")
  if (( RC != 0 )); then
    printf 'INVERSE_PLAN_SETUP_FAILURE stage=authorization code=%s container_attempted=0\n' "$RC" | tee -a "$LOG"
  elif (( STATUS[1] != 0 )); then
    RC=${STATUS[1]}
  else
    (set -e; main "$@") 2>&1 | tee -a "$LOG"
    STATUS=("${PIPESTATUS[@]}")
    RC=${STATUS[0]}
    if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
  fi
fi
if (( RC == 0 )) && ! grep -Fxq 'INVERSE_PLAN_LAYER_GATES_PASS' "$LOG"; then
  echo 'INVERSE_PLAN_ADMISSION_FAIL missing suite completion marker' | tee -a "$LOG"
  RC=4
fi
if (( RC == 77 )); then echo 'SKIPPED is not a pass.' | tee -a "$LOG"; fi
if (( RC == 70 || RC == 124 || RC == 137 )); then
  if grep -q '^INVERSE_PLAN_PROCESS_BEGIN ' "$LOG"; then
    echo 'Unexpected watchdog/timeout/kill after real-layer launch: inspect GPU state before another run.' | tee -a "$LOG"
  elif grep -q '^INVERSE_PLAN_CONTAINER_LAUNCH ' "$LOG"; then
    echo 'Container startup/setup timeout or kill before observed real-layer launch: inspect the setup log and container cleanup status.' | tee -a "$LOG"
  else
    echo 'Host setup timeout or kill before container launch. No real-layer process ran.' | tee -a "$LOG"
  fi
fi
if (( RC == 0 )); then
  echo 'INVERSE_PLAN_SUITE_PASS' | tee -a "$LOG"
  STATUS=("${PIPESTATUS[@]}")
  if (( STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
fi
printf 'TP2_INVERSE_PLAN_EXIT=%s LOG=%s\n' "$RC" "$LOG" | tee -a "$LOG"
FOOTER_STATUS=("${PIPESTATUS[@]}")
if (( RC == 0 && FOOTER_STATUS[1] != 0 )); then RC=${FOOTER_STATUS[1]}; fi
exit "$RC"
