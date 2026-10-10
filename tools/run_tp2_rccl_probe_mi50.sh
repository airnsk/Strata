#!/usr/bin/env bash
# Standalone, model-free RCCL captured transport preflight. No production changes.
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: bash tools/run_tp2_rccl_probe_mi50.sh [option value ...]
Only the installed pinned MI50 image is used. Both GPUs must already be idle.
Normal run options (all optional):
  --tokens 1|8|all       --launch-order 01|10|both
  --iterations 2..1000  --warmup 1..100  --chain 2..1024
  --lifecycles 2..8     --timeout-ms 100..120000  --devices 0,1|1,0
Defaults: all, both, 20, 3, 32, 2, 10000, 0,1.
Also runs separate delayed-rank and missing-rank fault checks.
Optional env: TP2_TIMEOUT=180 TP2_BUILD_TIMEOUT=180 TP2_OUTER_TIMEOUT=600 (seconds).
The outer bound includes discovery, compilation and all subprocesses.
No downloads, model mounts, service stops, or clock/power changes.
EOF
}
if [[ ${1:-} == --help && $# == 1 ]]; then usage; exit 0; fi
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG="$ROOT/tp2-rccl-probe-$RUN_ID.log"
IMAGE=sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca
CONTAINER_NAME="strata-tp2-rccl-$RUN_ID"
CONTAINER_STARTED=0
cleanup() {
  local rc=$?
  trap - EXIT
  set +e
  if (( CONTAINER_STARTED )); then
    # Killing a docker client does not prove its daemon-side container stopped.
    # Remove only this unique run's container; bound even a stuck daemon request.
    timeout --signal=TERM --kill-after=3s 15s sudo docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1
    local cleanup_rc=$?
    # --rm normally removed it already. Verify absence independently.
    local inspect
    inspect=$(timeout --signal=TERM --kill-after=3s 15s sudo docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' 2>&1)
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
main() {
  local deadline=${TP2_TIMEOUT:-180} build_deadline=${TP2_BUILD_TIMEOUT:-180} outer_deadline=${TP2_OUTER_TIMEOUT:-600}
  local value flag
  for value in "$deadline" "$build_deadline" "$outer_deadline"; do
    [[ "$value" =~ ^[1-9][0-9]{0,4}$ ]] && (( value <= 86400 )) || {
      echo 'Timeout variables must be integer seconds in 1..86400.' >&2; return 2;
    }
  done
  (( $# % 2 == 0 )) || { usage; return 2; }
  local -A seen=()
  local args=("$@")
  while (( $# )); do
    flag=$1; value=$2; shift 2
    [[ -z ${seen[$flag]:-} ]] || { echo "Repeated option: $flag" >&2; return 2; }
    seen[$flag]=1
    case "$flag:$value" in
      --tokens:1|--tokens:8|--tokens:all|--launch-order:01|--launch-order:10|--launch-order:both|--devices:0,1|--devices:1,0) ;;
      --iterations:*|--warmup:*|--chain:*|--lifecycles:*|--timeout-ms:*)
        [[ "$value" =~ ^[1-9][0-9]{0,5}$ ]] || { echo "Invalid positive integer: $flag=$value" >&2; return 2; }
        local lower=1 upper=1
        case "$flag" in
          --iterations) lower=2; upper=1000 ;; --warmup) upper=100 ;;
          --chain) lower=2; upper=1024 ;; --lifecycles) lower=2; upper=8 ;; --timeout-ms) lower=100; upper=120000 ;;
        esac
        (( value >= lower && value <= upper )) || { echo "Out of range: $flag=$value" >&2; return 2; }
        ;;
      *) echo "Unsupported option/value: $flag=$value" >&2; return 2 ;;
    esac
  done
  for value in timeout sudo git sha256sum tee; do
    command -v "$value" >/dev/null || { echo "Missing host command: $value" >&2; return 2; }
  done
  [[ -f "$ROOT/tests/hip/tp2_rccl_transport.cpp" && -f "$ROOT/tools/tp2_rccl_preflight.py" ]] || {
    echo 'Missing standalone RCCL probe source or dependency-discovery helper.' >&2; return 2;
  }
  printf 'RCCL_PROBE_RUN id=%s head=%s image=%s\n' "$RUN_ID" "$(git -C "$ROOT" rev-parse HEAD)" "$IMAGE"
  printf 'RCCL_PROBE_STAND repo=%q log=%q\n' "$ROOT" "$LOG"
  printf 'RCCL_PROBE_ARGUMENTS'; printf ' %q' "${args[@]}"; printf '\n'
  printf 'RCCL_PROBE_BOUNDS runtime_s=%s build_s=%s outer_s=%s\n' "$deadline" "$build_deadline" "$outer_deadline"
  echo 'Both MI50 cards must be idle. No service is stopped and no power/clock setting is changed.'
  printf 'RCCL_PROBE_TRACKED_DIFF_SHA256 '; git -C "$ROOT" diff --binary HEAD -- | sha256sum
  git -C "$ROOT" status --short
  sha256sum "$ROOT/tests/hip/tp2_rccl_transport.cpp" "$ROOT/tools/tp2_rccl_preflight.py" "$ROOT/tools/run_tp2_rccl_probe_mi50.sh"
  local image_id
  image_id=$(timeout --signal=TERM --kill-after=3s 20s sudo docker image inspect --format '{{.Id}}' "$IMAGE")
  printf 'RCCL_PROBE_RESOLVED_IMAGE %s\n' "$image_id"
  [[ "$image_id" == "$IMAGE" ]] || { echo 'Installed image ID differs from the pinned stand image.' >&2; return 2; }
  trap cleanup EXIT
  CONTAINER_STARTED=1
  timeout --signal=TERM --kill-after=15s "${outer_deadline}s" sudo docker run \
    --name "$CONTAINER_NAME" --rm --pull never --network none --read-only --cap-drop ALL \
    --tmpfs /tmp:rw,exec,nosuid,size=2g --shm-size=512m -e HOME=/tmp \
    --device=/dev/kfd --device=/dev/dri -e HIP_VISIBLE_DEVICES=0,1 -e NCCL_DEBUG=WARN \
    --mount "type=bind,src=$ROOT,dst=/work,readonly" \
    --workdir /work --entrypoint /bin/bash -i "$IMAGE" -s -- "$deadline" "$build_deadline" "${args[@]}" <<'RCCL_CONTAINER'
set -euo pipefail
export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH
runtime_deadline=$1; build_deadline=$2; shift 2
for tool in python3 hipcc timeout ldd sha256sum; do
  command -v "$tool" >/dev/null || { echo "RCCL_PREFLIGHT_BLOCKED missing installed tool: $tool" >&2; exit 2; }
done
# The repo and image remain read-only; all build products disappear with /tmp.
python3 -B /work/tools/tp2_rccl_preflight.py --output /tmp/rccl-env
source /tmp/rccl-env
printf 'RCCL_PREFLIGHT_HIPCC path=%s\n' "$(command -v hipcc)"
timeout --signal=TERM --kill-after=5s 20s hipcc --version
printf 'RCCL_PREFLIGHT_LOADER_PATH %s\n' "$LD_LIBRARY_PATH"
# nccl.h can be a compatibility spelling in a Torch bundle. The source supports
# both spellings; no headers or shared libraries are installed or rebuilt.
timeout --signal=TERM --kill-after=5s "${build_deadline}s" hipcc -x hip -std=c++17 -O2 -ffp-contract=off \
  --offload-arch=gfx906 -I "$RCCL_INCLUDE_DIR" -DSTRATA_RCCL_HEADER=\""$RCCL_HEADER"\" /work/tests/hip/tp2_rccl_transport.cpp \
  -pthread -ldl -x none "$RCCL_LIBRARY" -Wl,-rpath,"$(dirname "$RCCL_LIBRARY")" \
  -o /tmp/tp2_rccl_transport
sha256sum /tmp/tp2_rccl_transport
ldd /tmp/tp2_rccl_transport | tee /tmp/binary-dependencies.log
if grep -q 'not found' /tmp/binary-dependencies.log; then
  echo 'RCCL_PREFLIGHT_BLOCKED executable has unresolved dependencies' >&2; exit 2
fi
echo 'RCCL_PROBE_BUILD_GATE PASS'
telemetry() {
  date -u +RCCL_GPU_TELEMETRY_%Y-%m-%dT%H:%M:%SZ
  if command -v rocm-smi >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=3s 10s rocm-smi --showproductname --showuse --showclocks --showpower --showtemp || true
  else
    echo 'rocm-smi unavailable; utilization/clocks/power/temperature unavailable'
  fi
}
run_case() {
  local scenario=$1 bound=$2 expected=$3; shift 3
  local rc=0 pipe_status full_suite=0 option value
  if [[ "$scenario" == normal ]]; then
    full_suite=1
    local scan=("$@") index
    for (( index=0; index<${#scan[@]}; index+=2 )); do
      option=${scan[index]}; value=${scan[index+1]}
      if [[ "$option" == --tokens && "$value" != all ]] || [[ "$option" == --launch-order && "$value" != both ]]; then
        full_suite=0
      fi
    done
  fi
  printf 'RCCL_PROBE_PROCESS_BEGIN scenario=%s bound_s=%s\n' "$scenario" "$bound"
  set +e
  timeout --signal=TERM --kill-after=5s "${bound}s" /tmp/tp2_rccl_transport \
    --rccl-library "$RCCL_LIBRARY" --scenario "$scenario" "$@" 2>&1 | tee "/tmp/$scenario.log"
  pipe_status=("${PIPESTATUS[@]}")
  rc=${pipe_status[0]}
  set -e
  printf 'RCCL_PROBE_PROCESS_EXIT scenario=%s code=%s\n' "$scenario" "$rc"
  (( pipe_status[1] == 0 )) || { echo 'RCCL_PROBE_LOG_FAILURE' >&2; return "${pipe_status[1]}"; }
  if [[ "$scenario" == missing-rank ]]; then
    if (( rc != expected )) || ! grep -Eq '^WATCHDOG_TIMEOUT scenario=missing-rank stage=fault-launch exit=70([[:space:]]|$)' "/tmp/$scenario.log" \
        || ! grep -Fxq 'FAULT_INJECTION_ARMED scenario=missing-rank launch_rank=0 omitted_or_delayed_rank=1' "/tmp/$scenario.log" \
        || ! grep -Fxq 'FAULT_GRAPH_LAUNCH_ENTER scenario=missing-rank rank=0' "/tmp/$scenario.log" \
        || ! grep -Fxq 'FAULT_PEER_OMITTED scenario=missing-rank rank=1' "/tmp/$scenario.log" \
        || grep -q '^RCCL_PREFLIGHT_PASS' "/tmp/$scenario.log"; then
      echo 'RCCL_PROBE_FAULT_GATE FAIL: absent rank did not fail through the expected watchdog' >&2
      (( rc != 0 )) && return "$rc"
      return 4
    fi
    echo 'RCCL_PROBE_FAULT_GATE EXPECTED_FAILURE scenario=missing-rank exit=70'
  else
    (( rc == 0 )) || return "$rc"
    grep -q '^EXACT_GATE_PASS ' "/tmp/$scenario.log" && grep -q '^LIFECYCLE_PASS ' "/tmp/$scenario.log" || {
      echo "RCCL_PROBE_ADMISSION_FAIL missing correctness/lifecycle gate for $scenario" >&2; return 4;
    }
    if [[ "$scenario" == delayed-rank ]] && ! grep -q '^DELAYED_RANK_PASS .*timing_admitted=0' "/tmp/$scenario.log"; then
      echo 'RCCL_PROBE_ADMISSION_FAIL missing delayed-rank check' >&2; return 4
    fi
    grep -Eq "^RCCL_PREFLIGHT_PASS scenario=$scenario .* full_suite=$full_suite([[:space:]]|$)" "/tmp/$scenario.log" || {
      echo "RCCL_PROBE_ADMISSION_FAIL missing success marker for $scenario" >&2; return 4;
    }
  fi
}
telemetry
normal_rc=0
run_case normal "$runtime_deadline" 0 "$@" || normal_rc=$?
telemetry
(( normal_rc == 0 )) || exit "$normal_rc"
# Fault checks are independent processes. Their output cannot admit normal timings.
run_case delayed-rank 45 0 --tokens 1 --launch-order 01 --iterations 2 --warmup 1 --chain 2 \
  --lifecycles 1 --timeout-ms 10000 --delay-ms 200 --devices 0,1
run_case missing-rank 45 70 --tokens 1 --launch-order 01 --iterations 2 --warmup 1 --chain 2 \
  --lifecycles 1 --timeout-ms 10000 --delay-ms 200 --devices 0,1
telemetry
echo 'RCCL_PROBE_SUITE_PASS normal=pass delayed_rank=pass missing_rank=expected_watchdog_failure'
RCCL_CONTAINER
}
# Keep failures from the real runner and from tee. The final outer status is
# written into the durable host log, including preflight/build/timeout failures.
set +e
(set -e; main "$@") 2>&1 | tee "$LOG"
STATUS=("${PIPESTATUS[@]}")
RC=${STATUS[0]}
if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
if (( RC == 0 )) && ! grep -Fxq 'RCCL_PROBE_SUITE_PASS normal=pass delayed_rank=pass missing_rank=expected_watchdog_failure' "$LOG"; then
  echo 'RCCL_PROBE_ADMISSION_FAIL missing suite completion marker' | tee -a "$LOG"
  RC=4
fi
if (( RC == 77 )); then echo 'SKIPPED is not a pass.' | tee -a "$LOG"; fi
if (( RC == 70 || RC == 124 || RC == 137 )); then echo 'Unexpected watchdog/timeout/kill: inspect GPU state before another run.' | tee -a "$LOG"; fi
printf 'TP2_RCCL_PROBE_EXIT=%s LOG=%s\n' "$RC" "$LOG" | tee -a "$LOG"
FOOTER_STATUS=("${PIPESTATUS[@]}")
if (( RC == 0 && FOOTER_STATUS[1] != 0 )); then RC=${FOOTER_STATUS[1]}; fi
exit "$RC"
