#!/usr/bin/env bash
# One bounded diagnostic attempt. Requires both stand GPUs to be idle.
set -euo pipefail
if [[ ${1:-} == --help && $# == 1 ]]; then
  cat <<'EOF'
Usage: bash tools/run_tp2_rccl_debug_mi50.sh
Run one original T8/bank-0 sequence warmup under an installed rocgdb/gdb.
On the first SIGSEGV, print registers, stack words, mappings, and backtraces.
Otherwise exit before graph capture or timing. This is not a full probe pass.
Existing pinned image only; no downloads, models, service stops, or clock changes.
Container only: drop all capabilities, add SYS_PTRACE, keep default seccomp.
Both GPUs must be idle. Sudo authorization precedes logging and timeouts.
The container has a 180-second outer deadline plus bounded cleanup.
EOF
  exit 0
fi
(( $# == 0 )) || { echo 'Only --help is supported.' >&2; exit 2; }
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
IMAGE=sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
NAME="strata-rccl-debug-$RUN_ID"
LOG="$ROOT/tp2-rccl-debug-$RUN_ID.log"
PRIV=(); started=0
for tool in docker timeout tee git sha256sum grep; do
  command -v "$tool" >/dev/null || { echo "Missing host command: $tool" >&2; exit 2; }
done
if (( $(id -u) != 0 )); then
  command -v sudo >/dev/null || { echo 'Missing host command: sudo' >&2; exit 2; }
  PRIV=(sudo -n)
  if { exec {tty_fd}<>/dev/tty; } 2>/dev/null; then
    sudo -v <&"$tty_fd" >&"$tty_fd" 2>&1
    exec {tty_fd}>&-
  else
    timeout --signal=TERM --kill-after=3s 20s sudo -n -v || {
      echo 'Authorize sudo in a terminal first.' >&2; exit 2;
    }
  fi
fi
bounded() { timeout --signal=TERM --kill-after=3s 20s "$@"; }
cleanup() {
  local rc=$? names
  trap - EXIT
  if (( started )); then
    bounded "${PRIV[@]}" docker rm -f "$NAME" >/dev/null 2>&1 || true
    if names=$(bounded "${PRIV[@]}" docker ps -a --filter "name=^/${NAME}$" --format '{{.Names}}') && [[ -z $names ]]; then
      echo 'RCCL_DEBUG_CLEANUP verified_absent=1'
    else
      echo "RCCL_DEBUG_CLEANUP unverified=$NAME" >&2
      (( rc != 0 )) || rc=125
    fi
  fi
  exit "$rc"
}
main() {
  local resolved
  printf 'RCCL_DEBUG_RUN id=%s head=%s image=%s log=%s\n' "$RUN_ID" "$(git -C "$ROOT" rev-parse HEAD)" "$IMAGE" "$LOG"
  echo 'RCCL_DEBUG_SCOPE one_T8_bank0_sequence_warmup outer_s=180 cap_add=SYS_PTRACE default_seccomp=1'
  sha256sum "$ROOT/tests/hip/tp2_rccl_transport.cpp" "$ROOT/tools/tp2_rccl_preflight.py" "$ROOT/tools/run_tp2_rccl_debug_mi50.sh"
  resolved=$(bounded "${PRIV[@]}" docker image inspect --format '{{.Id}}' "$IMAGE")
  [[ $resolved == "$IMAGE" ]] || { echo 'Pinned image mismatch.' >&2; return 2; }
  trap cleanup EXIT
  started=1
  timeout --signal=TERM --kill-after=5s 180s "${PRIV[@]}" docker run \
    --name "$NAME" --rm --pull never --network none --read-only \
    --cap-drop ALL --cap-add SYS_PTRACE --ulimit core=0:0 \
    --tmpfs /tmp:rw,exec,nosuid,size=2g --shm-size=512m -e HOME=/tmp \
    --device=/dev/kfd --device=/dev/dri -e HIP_VISIBLE_DEVICES=0,1 \
    -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=ALL -e NCCL_DEBUG_FILE=/dev/stdout \
    --mount "type=bind,src=$ROOT,dst=/work,readonly" --workdir /work \
    --entrypoint /bin/bash -i "$IMAGE" -s <<'CONTAINER'
set -euo pipefail
export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH
for tool in python3 hipcc timeout sha256sum tee; do
  command -v "$tool" >/dev/null || { echo "RCCL_DEBUG_BLOCKED missing=$tool" >&2; exit 2; }
done
DEBUGGER=$(command -v rocgdb || command -v gdb || true)
[[ -n $DEBUGGER ]] || { echo 'RCCL_DEBUG_BLOCKED no installed rocgdb/gdb; nothing installed.' >&2; exit 2; }
printf 'RCCL_DEBUG_DEBUGGER %s\n' "$DEBUGGER"
timeout --signal=TERM --kill-after=3s 10s "$DEBUGGER" --version
# CPU-only ptrace/command support gate, before building or launching the probe.
# If the existing container security policy blocks this, stop; do not relax it.
timeout --signal=TERM --kill-after=3s 15s "$DEBUGGER" -nx -nh -batch \
  -iex 'set auto-load off' -iex 'set debuginfod enabled off' \
  -ex 'set disable-randomization off' -ex 'set non-stop off' \
  -ex 'python print("RCCL_DEBUG_PYTHON_READY")' -ex starti -ex continue --args /bin/true
python3 -B /work/tools/tp2_rccl_preflight.py --output /tmp/rccl-env
source /tmp/rccl-env
# Host-only debug information avoids changing HIP device debug code generation.
# Retain -O2, payload arithmetic, original initialization, and original warmup.
timeout --signal=TERM --kill-after=5s 60s hipcc -x hip -std=c++17 -O2 -Xarch_host -g -ffp-contract=off \
  --offload-arch=gfx906 -DSTRATA_RCCL_DEBUG_WARMUP_ONLY -I "$RCCL_INCLUDE_DIR" \
  -DSTRATA_RCCL_HEADER=\""$RCCL_HEADER"\" /work/tests/hip/tp2_rccl_transport.cpp \
  -pthread -ldl -x none "$RCCL_LIBRARY" -Wl,-rpath,"$(dirname "$RCCL_LIBRARY")" -o /tmp/tp2_rccl_debug
sha256sum /tmp/tp2_rccl_debug
HIP_LIBRARY=$(readlink -f /opt/rocm-7.2.4/lib/libamdhip64.so.7)
printf 'RCCL_DEBUG_HIP_LIBRARY %s\n' "$HIP_LIBRARY"
sha256sum "$HIP_LIBRARY"
if command -v readelf >/dev/null; then
  readelf -n "$HIP_LIBRARY" | awk '/Build ID:/ {print "RCCL_DEBUG_HIP_" $0}'
fi
cat > /tmp/capture.gdb <<'GDB'
set pagination off
set confirm off
set disable-randomization off
set non-stop off
set print thread-events off
set print elements 16
set print frame-arguments scalars
handle SIGSEGV stop print nopass
catch signal SIGSEGV
commands
silent
echo RCCL_DEBUG_SIGSEGV_BEGIN\n
python
import gdb
failed_commands = []
gdb.write('RCCL_DEBUG_FAULT_THREAD ' + str(gdb.selected_thread().ptid) + '\n')
for command in ('info registers rip rdi rsp rbp rax', 'x/9bx $rip', 'x/8gx $rsp',
                'bt 24', 'info proc mappings', 'info sharedlibrary', 'thread apply all -c bt 24'):
    gdb.write('RCCL_DEBUG_COMMAND ' + command + '\n')
    try:
        gdb.execute(command)
    except Exception as error:
        failed_commands.append(command)
        gdb.write('RCCL_DEBUG_CAPTURE_ERROR ' + str(error) + '\n')
gdb.execute('set $capture_incomplete = ' + str(int(bool(failed_commands))))
end
echo RCCL_DEBUG_SIGSEGV_END\n
if $capture_incomplete
  echo RCCL_DEBUG_CAPTURE_PARTIAL\n
  quit 87
end
quit 86
end
run
echo RCCL_DEBUG_RUN_RETURNED_WITHOUT_SIGSEGV\n
quit 0
GDB
echo 'RCCL_DEBUG_PROBE_BEGIN original_warmup=T8_bank0_sequence_once first_seam=y'
set +e
timeout --signal=TERM --kill-after=5s 120s "$DEBUGGER" -nx -nh -batch \
  -iex 'set auto-load off' -iex 'set debuginfod enabled off' -x /tmp/capture.gdb \
  --args /tmp/tp2_rccl_debug --rccl-library "$RCCL_LIBRARY" --scenario normal \
  --tokens 8 --launch-order 01 --iterations 2 --warmup 1 --chain 2 --lifecycles 2 \
  --init-timeout-ms 60000 --timeout-ms 10000 --devices 0,1 </dev/null 2>&1 | tee /tmp/debug.log
status=("${PIPESTATUS[@]}")
set -e
printf 'RCCL_DEBUG_DEBUGGER_EXIT code=%s\n' "${status[0]}"
(( status[1] == 0 )) || exit "${status[1]}"
if (( status[0] == 86 )) && grep -Fxq 'RCCL_DEBUG_SIGSEGV_END' /tmp/debug.log; then
  echo 'RCCL_DEBUG_RESULT first_sigsegv_captured=1 probe_pass=0 rerun=not_performed'
elif (( status[0] == 87 )) && grep -Fxq 'RCCL_DEBUG_CAPTURE_PARTIAL' /tmp/debug.log; then
  echo 'RCCL_DEBUG_RESULT first_sigsegv_captured=1 capture_partial=1 probe_pass=0 rerun=not_performed'
  exit 87
elif (( status[0] == 0 )) && grep -Fxq 'RCCL_DEBUG_WARMUP_ONLY_COMPLETE capture_tested=0 full_suite=0' /tmp/debug.log; then
  echo 'RCCL_DEBUG_RESULT first_sigsegv_captured=0 warmup_finished=1 probe_pass=0 rerun=not_performed'
else
  echo 'RCCL_DEBUG_RESULT incomplete=1 probe_pass=0 rerun=not_performed'
  (( status[0] != 0 )) && exit "${status[0]}"
  exit 4
fi
CONTAINER
}
# Authentication above is deliberately outside this captured/timed pipeline.
set +e
(set -e; main) 2>&1 | tee "$LOG"
status=("${PIPESTATUS[@]}")
rc=${status[0]}
(( rc != 0 || status[1] == 0 )) || rc=${status[1]}
if (( rc == 0 )) && { [[ $(grep -c '^RCCL_DEBUG_RESULT ' "$LOG") != 1 ]] ||
    ! grep -Eq '^RCCL_DEBUG_RESULT (first_sigsegv_captured=1|first_sigsegv_captured=0 warmup_finished=1) probe_pass=0 rerun=not_performed$' "$LOG"; }; then
  echo 'RCCL_DEBUG_ADMISSION_FAIL missing or ambiguous completion result' | tee -a "$LOG"
  rc=4
fi
printf 'RCCL_DEBUG_EXIT=%s\n' "$rc" | tee -a "$LOG"
footer=("${PIPESTATUS[@]}")
if (( rc == 0 )); then
  (( footer[0] == 0 )) || rc=${footer[0]}
  (( footer[1] == 0 )) || rc=${footer[1]}
fi
exit "$rc"
