#!/usr/bin/env bash
# Opt-in, model-free two-MI50 VRAM protocol admission. Never a performance run.
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: bash tools/run_tp2_vram_probe_mi50.sh [option value ...]
  --iterations 128..512     (default 128)
  --timeout-ms 1000..60000  (default 10000, host-operation watchdog)
  --devices 0,1|1,0         (default 0,1)
Always runs the complete normal, delayed-peer and missing-peer matrix in three
fresh processes. Missing-peer passes only after device timeout, sticky failure,
drain, exact recovery and fresh lifecycle validation. A watchdog is a failure.
Uses only the installed pinned image, two standalone HIP sources and -j48.
Device wait is finite and shorter than the host-operation watchdog.
No engine/model/RCCL builds, downloads, timings, service stops or clock changes.
Both GPUs must already be idle; missing/ambiguous idle evidence fails closed.
Default incremental cache: .tp2-vram-build-cache (private, inside this checkout).
Optional TP2_VRAM_CACHE=off or another direct child directory name.
Optional BUILD_JOBS=48 TP2_TIMEOUT=180 TP2_BUILD_TIMEOUT=180 TP2_OUTER_TIMEOUT=900.
All deadlines are seconds. Outer covers discovery/build/processes; bounded
cleanup follows. Sudo authorization uses the controlling terminal before logs
or operation deadlines; cleanup can explicitly reauthorize there if needed.
EOF
}
if [[ ${1:-} == --help && $# == 1 ]]; then usage; exit 0; fi
umask 077
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUN_ID="$(date -u +%Y%m%dT%H%M%S)-$$-$RANDOM"
LOG="$ROOT/tp2-vram-probe-$RUN_ID.log"
IMAGE=sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca
CONTAINER_NAME="strata-tp2-vram-$RUN_ID"
CONTAINER_STARTED=0
DOCKER=(docker)
# BEGIN_SHARED_FUNCTIONS
source_manifest() {
  python3 -B - "$1" "${2:-quiet}" <<'SOURCE_MANIFEST'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
paths = [root / p for p in ('tests/hip/tp2_vram_rendezvous.cpp',
         'src/kernels/cuda/tp_vram_rendezvous.cu', 'tools/run_tp2_vram_probe_mi50.sh')]
headers = root / 'include'
if not headers.is_dir() or headers.is_symlink():
    raise SystemExit('VRAM_SOURCE_FAILURE missing or symlinked include tree')
paths += sorted(headers.rglob('*'))
manifest = hashlib.sha256()
for path in paths:
    if path.is_symlink():
        raise SystemExit('VRAM_SOURCE_FAILURE symlinked source: ' + str(path))
    if path.is_dir():
        continue
    relative = path.relative_to(root).as_posix()
    if '\n' in relative or '\r' in relative:
        raise SystemExit('VRAM_SOURCE_FAILURE invalid source name')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.update((digest + '  ' + relative + '\n').encode())
    if sys.argv[2] == 'log':
        print('VRAM_SOURCE_FILE path=' + relative + ' sha256=' + digest, file=sys.stderr)
print(manifest.hexdigest())
SOURCE_MANIFEST
}
validate_gates() {
  python3 -B - "$@" <<'VALIDATE_GATES'
import pathlib, re, sys
mode, path, iterations, devices = sys.argv[1:5]
iterations = int(iterations)
devices = [int(x) for x in devices.split(',')]
raw = pathlib.Path(path).read_bytes()
if not raw.endswith(b'\n') or b'\r' in raw:
    raise SystemExit('VRAM_RUNNER_ADMISSION_FAIL partial or invalid line ending')
lines = raw.decode().splitlines()
def fail(reason):
    raise ValueError(reason)
def forbidden(line):
    token = line.split(' ', 1)[0]
    if re.search(r'(?:^|[_\s])(?:PERF|BENCH|TIMING|SKIP|SKIPPED|WATCHDOG)(?:\b|_)', line):
        fail('forbidden performance/skip/watchdog output')
    if re.search(r'(?:^|_)(?:FAIL|FAILURE|SKIPPED)(?:\b|_)', token) and line != 'VRAM_STICKY_FAILURE_PASS scenario=missing-peer':
        fail('failure output')
def child(scenario, segment):
    expected = []
    if scenario == 'missing-peer':
        for first in (0, 1):
            expected += [f'VRAM_FAULT_ARMED scenario=missing-peer first={first} omitted={1-first}',
                         f'VRAM_FAULT_CASE_PASS first={first} omitted={1-first} late_peer_aborted=1 repeated_call_aborted=1 output_poison=1 both_drained=1']
        expected += ['VRAM_NEGATIVE_CASE_PASS kind=' + kind + ' status=invalid-epoch output_poison=1 both_drained=1'
                     for kind in ('future-ready', 'future-consumed', 'epoch-wrap')]
        expected += [name + ' scenario=missing-peer' for name in
                     ('VRAM_TIMEOUT_GATE_PASS', 'VRAM_STICKY_FAILURE_PASS', 'VRAM_DRAIN_PASS')]
    for life in (0, 1):
        epoch = 0 if life == 0 else 4294967294
        for tokens in ((1, 8) if scenario == 'normal' else (8,)):
            for first in (0, 1):
                for kind in (0, 1, 2):
                    expected.append(f'VRAM_CASE_PASS scenario={scenario} life={life} kind={kind} tokens={tokens} order={first}{1-first} replays={iterations} epochs={epoch+1}..{epoch+iterations} device_witness=1 guards=1 ring_slots=2')
                    epoch += iterations
        if scenario == 'normal':
            expected.append(f'VRAM_ALTERNATING_SHAPE_PASS scenario=normal life={life} order={life}{1-life} replays={iterations} tokens=8,1 fixed_stride=32768 device_witness=1')
        expected.append(f'VRAM_FRESH_LIFECYCLE_PASS scenario={scenario} life={life} both_drained=1 released=1')
        if life == 1:
            expected.append(f'VRAM_EPOCH_CARRY_PASS scenario={scenario} life=1 seed=4294967294')
    if scenario == 'delayed-peer':
        expected.append('VRAM_DELAYED_PEER_PASS scenario=delayed-peer')
    expected += ['VRAM_EXACT_GATE_PASS scenario=' + scenario,
                 'VRAM_LIFECYCLE_GATE_PASS scenario=' + scenario + ' fresh_lifecycles=2',
                 'VRAM_PROBE_PASS scenario=' + scenario + ' cases=' +
                 {'normal': '24', 'delayed-peer': '12', 'missing-peer': '2'}[scenario] +
                 ' lifecycles=2 timing_admitted=0']
    actual, gpu = [], {}
    pci = set()
    for line in segment:
        forbidden(line)
        token = line.split(' ', 1)[0]
        if token.startswith('VRAM_GPU'):
            m = re.fullmatch(r'VRAM_GPU rank=([01]) device=([01]) pci=([0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]) arch=gfx906(?::[^ ]+)? name=.+ runtime=[1-9][0-9]* driver=[1-9][0-9]* allocation=hipDeviceMallocUncached', line)
            if not m or actual:
                fail('missing/malformed/late GPU admission')
            rank, device = int(m[1]), int(m[2])
            if rank in gpu or devices[rank] != device or m[3].lower() in pci:
                fail('duplicate/wrong GPU identity')
            gpu[rank] = device; pci.add(m[3].lower())
        elif token.startswith('VRAM_') and (token.endswith('_PASS') or token.startswith('VRAM_FAULT_ARMED')):
            if set(gpu) != {0, 1}:
                fail('matrix precedes GPU admission')
            actual.append(line)
    if actual != expected or not segment or segment[-1] != expected[-1]:
        fail('partial/duplicate/malformed/out-of-order matrix or completion: ' + scenario)
try:
    if mode != 'host':
        child(mode, lines)
    else:
        def one(prefix):
            found = [line for line in lines if line.startswith(prefix)]
            if len(found) != 1:
                fail('missing/duplicate ' + prefix)
            return found[0]
        host_sha = one('VRAM_HOST_SOURCE_SHA256 ').split()[-1]
        if not re.fullmatch('[0-9a-f]{64}', host_sha) or one('VRAM_CONTAINER_SOURCE_SHA256 ') != 'VRAM_CONTAINER_SOURCE_SHA256 ' + host_sha:
            fail('source SHA mismatch')
        if not re.fullmatch(r'VRAM_BINARY_SHA256 [0-9a-f]{64}', one('VRAM_BINARY_SHA256 ')):
            fail('missing binary SHA')
        if one('VRAM_RUNNER_BUILD_GATE') != 'VRAM_RUNNER_BUILD_GATE PASS':
            fail('invalid build gate')
        if one('VRAM_RUNNER_CONTAINER_COMPLETE') != 'VRAM_RUNNER_CONTAINER_COMPLETE normal=pass delayed_peer=pass missing_peer=finite_timeout_pass':
            fail('invalid completion gate')
        if one('VRAM_CONTAINER_CLEANUP') != 'VRAM_CONTAINER_CLEANUP verified_absent=' + sys.argv[5]:
            fail('cleanup unverified')
        previous = lines.index('VRAM_RUNNER_BUILD_GATE PASS')
        before = one('VRAM_RUNNER_IDLE_PASS stage=before-build ')
        if before != 'VRAM_RUNNER_IDLE_PASS stage=before-build cards=2 utilization=0 kfd_processes=0' or lines.index(before) >= previous:
            fail('missing/late build idle check')
        included = set()
        for scenario in ('normal', 'delayed-peer', 'missing-peer'):
            begin = one('VRAM_RUNNER_PROCESS_BEGIN scenario=' + scenario + ' ')
            end = one('VRAM_RUNNER_PROCESS_EXIT scenario=' + scenario + ' ')
            if not re.fullmatch(r'VRAM_RUNNER_PROCESS_BEGIN scenario=' + scenario + r' bound_s=[1-9][0-9]*', begin):
                fail('invalid process begin')
            if end != 'VRAM_RUNNER_PROCESS_EXIT scenario=' + scenario + ' code=0':
                fail('nonzero process exit')
            idle = one('VRAM_RUNNER_IDLE_PASS stage=' + scenario + ' ')
            if idle != 'VRAM_RUNNER_IDLE_PASS stage=' + scenario + ' cards=2 utilization=0 kfd_processes=0':
                fail('invalid idle check')
            first, last = lines.index(begin), lines.index(end)
            if not previous < lines.index(idle) < first < last:
                fail('wrong process or idle order')
            child(scenario, lines[first+1:last])
            included.update(range(first+1, last))
            previous = last
        for i, line in enumerate(lines):
            forbidden(line)
            token = line.split(' ', 1)[0]
            if i not in included and token.startswith('VRAM_') and not token.startswith(('VRAM_RUNNER_', 'VRAM_CONTAINER_')) and (token.endswith('_PASS') or token.startswith(('VRAM_FAULT_ARMED', 'VRAM_GPU'))):
                fail('probe output outside process boundary')
        if not previous < lines.index(one('VRAM_RUNNER_CONTAINER_COMPLETE')) < lines.index(one('VRAM_CONTAINER_CLEANUP')):
            fail('completion/cleanup out of order')
except (ValueError, IndexError, KeyError) as error:
    print('VRAM_RUNNER_ADMISSION_FAIL ' + str(error), file=sys.stderr)
    raise SystemExit(4)
VALIDATE_GATES
}
# END_SHARED_FUNCTIONS
validate_host() {
  local value flag lower upper
  for value in "${TP2_TIMEOUT:-180}" "${TP2_BUILD_TIMEOUT:-180}" "${TP2_OUTER_TIMEOUT:-900}"; do
    [[ $value =~ ^[1-9][0-9]{0,4}$ ]] && (( value <= 86400 )) || {
      echo 'Timeout variables must be integer seconds in 1..86400.' >&2; return 2;
    }
  done
  value=${BUILD_JOBS:-48}
  [[ $value =~ ^[1-9][0-9]{0,2}$ ]] && (( value <= 256 )) || { echo 'BUILD_JOBS must be in 1..256.' >&2; return 2; }
  [[ $ROOT != *,* && $ROOT != *$'\n'* && $ROOT != *$'\r'* ]] || { echo 'Unsafe checkout mount path.' >&2; return 2; }
  value=${TP2_VRAM_CACHE:-.tp2-vram-build-cache}
  [[ $value =~ ^[a-zA-Z0-9_.-]+$ && $value != . && $value != .. ]] || {
    echo 'TP2_VRAM_CACHE must be off or one direct child directory name.' >&2; return 2;
  }
  (( $# % 2 == 0 )) || { usage; return 2; }
  local -A seen=()
  while (( $# )); do
    flag=$1; value=$2; shift 2
    [[ -z ${seen[$flag]:-} ]] || { echo "Repeated option: $flag" >&2; return 2; }
    seen[$flag]=1
    case "$flag" in
      --devices) [[ $value == 0,1 || $value == 1,0 ]] || { echo 'Devices must be 0,1 or 1,0.' >&2; return 2; } ;;
      --iterations|--timeout-ms)
        [[ $value =~ ^[1-9][0-9]{0,4}$ ]] || { echo "Invalid integer: $flag=$value" >&2; return 2; }
        lower=128; upper=512
        if [[ $flag == --timeout-ms ]]; then lower=1000; upper=60000; fi
        (( value >= lower && value <= upper )) || { echo "Out of range: $flag=$value" >&2; return 2; } ;;
      *) echo "Unsupported option: $flag" >&2; return 2 ;;
    esac
  done
  for value in timeout docker python3 git sha256sum tee id stat; do
    command -v "$value" >/dev/null || { echo "Missing host command: $value" >&2; return 2; }
  done
  if (( $(id -u) != 0 )); then command -v sudo >/dev/null || return 2; fi
  for value in tests/hip/tp2_vram_rendezvous.cpp src/kernels/cuda/tp_vram_rendezvous.cu include/strata/kernels/tp_vram_rendezvous.hpp; do
    [[ -f $ROOT/$value && ! -L $ROOT/$value ]] || { echo "Missing/symlinked standalone source: $value" >&2; return 2; }
  done
}
authorize_docker() {
  if (( $(id -u) == 0 )); then return 0; fi
  DOCKER=(sudo -n docker)
  local auth_tty auth_rc=0
  if { exec {auth_tty}<>/dev/tty; } 2>/dev/null; then
    printf 'Checking sudo authorization outside timed operations.\n' >&"$auth_tty"
    sudo -v <&"$auth_tty" >&"$auth_tty" 2>&1 || auth_rc=$?
    exec {auth_tty}>&-
  else
    timeout --signal=TERM --kill-after=3s 20s sudo -n -v || auth_rc=$?
    if (( auth_rc != 0 )); then
      echo 'Sudo authorization unavailable without a terminal. Run from a terminal to authorize Docker.' >&2
    fi
  fi
  return "$auth_rc"
}
cleanup() {
  local rc=$? inspect inspect_rc cleanup_rc
  trap - EXIT
  set +e
  if (( CONTAINER_STARTED )); then
    if (( $(id -u) != 0 )); then
      timeout --signal=TERM --kill-after=3s 20s sudo -n -v >/dev/null 2>&1 || authorize_docker
    fi
    timeout --signal=TERM --kill-after=3s 15s "${DOCKER[@]}" rm -f "$CONTAINER_NAME" >/dev/null 2>&1
    cleanup_rc=$?
    inspect=$(timeout --signal=TERM --kill-after=3s 15s "${DOCKER[@]}" ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' 2>&1)
    inspect_rc=$?
    if (( inspect_rc != 0 )) || [[ -n $inspect ]]; then
      printf 'VRAM_CONTAINER_CLEANUP_UNVERIFIED name=%s remove_exit=%s inspect_exit=%s\n' "$CONTAINER_NAME" "$cleanup_rc" "$inspect_rc" >&2
      (( rc != 0 )) || rc=125
    else
      printf 'VRAM_CONTAINER_CLEANUP verified_absent=%s\n' "$CONTAINER_NAME"
    fi
  fi
  exit "$rc"
}
main() {
  local start=$SECONDS deadline=${TP2_TIMEOUT:-180} build_deadline=${TP2_BUILD_TIMEOUT:-180} outer=${TP2_OUTER_TIMEOUT:-900}
  local source_sha image_id uid gid host_groups device_groups group node cache remaining
  printf 'VRAM_RUNNER_RUN id=%s image=%s log=%q\n' "$RUN_ID" "$IMAGE" "$LOG"
  printf 'VRAM_RUNNER_HEAD '; timeout 20s git -C "$ROOT" rev-parse HEAD
  printf 'VRAM_RUNNER_ARGUMENTS'; printf ' %q' "$@"; printf '\n'
  printf 'VRAM_RUNNER_BOUNDS runtime_s=%s build_s=%s outer_s=%s jobs=%s\n' "$deadline" "$build_deadline" "$outer" "${BUILD_JOBS:-48}"
  source_sha=$(timeout --signal=TERM --kill-after=3s 20s bash -c "$(declare -f source_manifest); source_manifest \"\$1\" log" manifest "$ROOT")
  printf 'VRAM_HOST_SOURCE_SHA256 %s\n' "$source_sha"
  image_id=$(timeout --signal=TERM --kill-after=3s 20s "${DOCKER[@]}" image inspect --format '{{.Id}}' "$IMAGE")
  [[ $image_id == "$IMAGE" ]] || { echo 'VRAM_RUNNER_SETUP_FAILURE stage=image-verification' >&2; return 2; }
  printf 'VRAM_RUNNER_RESOLVED_IMAGE %s\n' "$image_id"
  uid=$(id -u); gid=$(id -g); host_groups=$(id -G)
  [[ $uid =~ ^[0-9]+$ && $gid =~ ^[0-9]+$ && -n $host_groups ]] || return 2
  local nodes=(/dev/kfd /dev/dri) identity=(--user "$uid:$gid") mounts=()
  for node in /dev/dri/card[0-9]* /dev/dri/renderD[0-9]*; do [[ ! -e $node ]] || nodes+=("$node"); done
  device_groups=$(timeout 10s stat -Lc '%g' -- "${nodes[@]}") || { echo 'VRAM_RUNNER_SETUP_FAILURE stage=device-identity' >&2; return 2; }
  [[ -n $device_groups ]] || return 2
  local -A groups=(["$gid"]=1)
  for group in $host_groups $device_groups; do
    [[ $group =~ ^[0-9]+$ ]] || { echo 'VRAM_RUNNER_SETUP_FAILURE stage=device-identity' >&2; return 2; }
    if [[ -z ${groups[$group]:-} ]]; then identity+=(--group-add "$group"); groups[$group]=1; fi
  done
  cache=${TP2_VRAM_CACHE:-.tp2-vram-build-cache}
  if [[ $cache != off ]]; then
    timeout 10s python3 -B - "$ROOT/$cache" <<'CACHE_DIRECTORY'
import os, pathlib, stat, sys
path = pathlib.Path(sys.argv[1])
if not path.exists() and not path.is_symlink():
    path.mkdir(mode=0o700)
s = path.lstat()
if not stat.S_ISDIR(s.st_mode) or s.st_uid != os.getuid() or stat.S_IMODE(s.st_mode) != 0o700:
    raise SystemExit('VRAM_RUNNER_SETUP_FAILURE cache must be owned private directory mode 0700')
CACHE_DIRECTORY
    mounts=(--mount "type=bind,src=$ROOT/$cache,dst=/cache")
  fi
  printf 'VRAM_RUNNER_CACHE directory=%s\n' "$cache"
  remaining=$((outer - (SECONDS - start)))
  (( remaining > 0 )) || { echo 'VRAM_RUNNER_SETUP_FAILURE discovery exhausted outer deadline' >&2; return 124; }
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  CONTAINER_STARTED=1
  printf 'VRAM_RUNNER_CONTAINER_LAUNCH name=%s\n' "$CONTAINER_NAME"
  {
    declare -f source_manifest validate_gates
    cat <<'VRAM_CONTAINER'
set -euo pipefail
umask 077
export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH
runtime_deadline=$1; build_deadline=$2; jobs=$3; expected_sources=$4; image=$5; cache_mode=$6; shift 6
iterations=128; command_timeout=10000; devices=0,1
while (( $# )); do
  case "$1" in --iterations) iterations=$2 ;; --timeout-ms) command_timeout=$2 ;; --devices) devices=$2 ;; esac
  shift 2
done
normal_args=(--iterations "$iterations" --timeout-ms "$command_timeout" --devices "$devices")
for tool in python3 hipcc make timeout sha256sum tee flock rocm-smi ldd; do
  command -v "$tool" >/dev/null || { echo "VRAM_RUNNER_BLOCKED missing installed tool: $tool" >&2; exit 2; }
done
printf 'VRAM_RUNNER_CONTAINER_ENTER\n'
actual_sources=$(source_manifest /work)
[[ $actual_sources == "$expected_sources" ]] || { echo 'VRAM_RUNNER_SOURCE_SHA_MISMATCH' >&2; exit 4; }
printf 'VRAM_CONTAINER_SOURCE_SHA256 %s\n' "$actual_sources"
idle_gate() {
  local stage=$1
  timeout --signal=TERM --kill-after=3s 10s rocm-smi --showuse --json > /tmp/vram-idle.json
  cat /tmp/vram-idle.json
  timeout --signal=TERM --kill-after=3s 10s python3 -B - "$stage" <<'IDLE_GATE'
import json, os, pathlib, re, sys
def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate telemetry key: ' + key)
        result[key] = value
    return result
try:
    root = pathlib.Path('/sys/class/kfd/kfd/proc')
    if not root.is_dir() or not os.access(root, os.R_OK | os.X_OK):
        raise ValueError('KFD process visibility unavailable')
    entries = list(root.iterdir())
    if entries:
        raise ValueError('KFD processes present: ' + ','.join(p.name for p in entries))
    data = json.loads(pathlib.Path('/tmp/vram-idle.json').read_text(), object_pairs_hook=unique_object)
    cards = {key: value for key, value in data.items() if key.startswith('card')}
    # ROCm SMI uses DRM card numbers, which are not HIP logical ordinals.
    if len(cards) != 2 or any(not re.fullmatch(r'card(?:0|[1-9][0-9]*)', name) for name in cards):
        raise ValueError('expected exactly two distinct numeric DRM GPU records')
    for name, card in cards.items():
        value = card.get('GPU use (%)')
        if isinstance(value, bool) or str(value).strip() not in {'0', '0.0'}:
            raise ValueError(name + ' nonzero or unknown utilization: ' + str(value))
except (OSError, ValueError, TypeError, AttributeError) as error:
    print('VRAM_RUNNER_IDLE_FAIL ' + str(error), file=sys.stderr)
    raise SystemExit(3)
print('VRAM_RUNNER_IDLE_PASS stage=' + sys.argv[1] + ' cards=2 utilization=0 kfd_processes=0')
IDLE_GATE
}
idle_gate before-build
cache=/tmp/tp2-vram-build-cache
[[ $cache_mode == off ]] || cache=/cache
mkdir -p "$cache"
# Lock is nonblocking: concurrent probes never queue or share writable objects.
python3 -B - "$cache" <<'CACHE_LOCK'
import os, pathlib, stat, sys
root = pathlib.Path(sys.argv[1])
if root.is_symlink() or root.stat().st_uid != os.getuid() or stat.S_IMODE(root.stat().st_mode) != 0o700:
    raise SystemExit('VRAM_RUNNER_CACHE_FAILURE unsafe cache directory')
lock = root / '.lock'
if lock.is_symlink() or (lock.exists() and (not lock.is_file() or lock.stat().st_uid != os.getuid() or lock.stat().st_nlink != 1)):
    raise SystemExit('VRAM_RUNNER_CACHE_FAILURE unsafe lock file')
CACHE_LOCK
exec {cache_lock}>>"$cache/.lock"
flock -n "$cache_lock" || { echo 'VRAM_RUNNER_CACHE_BUSY' >&2; exit 3; }
timeout --signal=TERM --kill-after=3s 20s hipcc --version > /tmp/vram-compiler.txt
cat /tmp/vram-compiler.txt
cat > /tmp/vram-build.py <<'BUILD_SCRIPT'
import hashlib, json, os, pathlib, stat, subprocess, sys
root, cache, image, jobs = pathlib.Path('/work'), pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
flags = '-x hip -std=c++17 -O2 -DSTRATA_HIP_GFX906=1 -fno-fast-math -ffp-contract=off --offload-arch=gfx906 -I/work/include -I/work -pthread'
makefile = '''.DELETE_ON_ERROR:
.PHONY: all
all: tp2_vram_rendezvous
probe.o: /work/tests/hip/tp2_vram_rendezvous.cpp
\thipcc FLAGS -MMD -MP -MF probe.d -c $< -o $@
protocol.o: /work/src/kernels/cuda/tp_vram_rendezvous.cu
\thipcc FLAGS -MMD -MP -MF protocol.d -c $< -o $@
tp2_vram_rendezvous: probe.o protocol.o
\thipcc --offload-arch=gfx906 -pthread $^ -o $@
-include probe.d protocol.d
'''.replace('FLAGS', flags)
sha = lambda data: hashlib.sha256(data).hexdigest()
config = sha((image + '\n' + makefile).encode() + pathlib.Path('/tmp/vram-compiler.txt').read_bytes()
             + (root / 'tools/run_tp2_vram_probe_mi50.sh').read_bytes())
build = cache / config
build.mkdir(exist_ok=True, mode=0o700)
if build.is_symlink() or build.stat().st_uid != os.getuid() or stat.S_IMODE(build.stat().st_mode) != 0o700:
    raise SystemExit('VRAM_RUNNER_CACHE_FAILURE unsafe configuration directory')
allowed = {'Makefile', 'probe.o', 'protocol.o', 'probe.d', 'protocol.d', 'tp2_vram_rendezvous', 'manifest.json'}
for path in build.iterdir():
    if path.name not in allowed or path.is_symlink() or not path.is_file():
        raise SystemExit('VRAM_RUNNER_CACHE_FAILURE unexpected cache entry')
manifest_path = build / 'manifest.json'
old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
artifact_names = {'probe.o', 'protocol.o', 'probe.d', 'protocol.d', 'tp2_vram_rendezvous'}
if old and (set(old) != {'sources', 'artifacts'} or set(old['artifacts']) != artifact_names
            or set(old['sources']) != {'probe.o', 'protocol.o'}):
    raise SystemExit('VRAM_RUNNER_CACHE_FAILURE malformed manifest')
if not old:
    # An interrupted first build is not trusted input to make.
    for name in artifact_names:
        (build / name).unlink(missing_ok=True)
for name, digest in old.get('artifacts', {}).items():
    path = build / name
    if name not in allowed or not path.is_file() or sha(path.read_bytes()) != digest:
        raise SystemExit('VRAM_RUNNER_BINARY_SHA_MISMATCH cached artifact=' + name)
# Remove the old commit record before mutating outputs. An interrupted build
# is discarded safely on the next explicitly requested invocation.
manifest_path.unlink(missing_ok=True)
headers = hashlib.sha256()
for path in sorted((root / 'include').rglob('*')):
    if path.is_file():
        headers.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes())
fingerprints = {}
for name, source in (('probe.o', 'tests/hip/tp2_vram_rendezvous.cpp'),
                     ('protocol.o', 'src/kernels/cuda/tp_vram_rendezvous.cu')):
    fingerprints[name] = sha(config.encode() + headers.digest() + (root / source).read_bytes())
    if fingerprints[name] != old.get('sources', {}).get(name):
        (build / name).unlink(missing_ok=True)
        (build / name.replace('.o', '.d')).unlink(missing_ok=True)
# Depfiles are re-generated artifacts; verified above before make reads them.
(build / 'Makefile').write_text(makefile)
print('VRAM_RUNNER_BUILD_CONFIG sha256=' + config + ' jobs=' + jobs, flush=True)
subprocess.run(['make', '-C', str(build), '--no-print-directory', '-j', jobs, 'all'], check=True)
artifacts = {}
for name in ('probe.o', 'protocol.o', 'probe.d', 'protocol.d', 'tp2_vram_rendezvous'):
    artifacts[name] = sha((build / name).read_bytes())
manifest_path.write_text(json.dumps({'sources': fingerprints, 'artifacts': artifacts}, sort_keys=True) + '\n')
pathlib.Path('/tmp/vram-binary-path').write_text(str(build / 'tp2_vram_rendezvous'))
pathlib.Path('/tmp/vram-binary-sha').write_text(artifacts['tp2_vram_rendezvous'])
BUILD_SCRIPT
# The only build is two translation units and their link. All flags are explicit.
unset HIPCC_COMPILE_FLAGS_APPEND HIPCC_LINK_FLAGS_APPEND HIPCC_COMPILE_FLAGS_PREPEND HIPCC_LINK_FLAGS_PREPEND
export HIP_PLATFORM=amd
timeout --signal=TERM --kill-after=10s "${build_deadline}s" python3 -B /tmp/vram-build.py "$cache" "$image" "$jobs"
binary=$(cat /tmp/vram-binary-path); binary_sha=$(cat /tmp/vram-binary-sha)
printf 'VRAM_BINARY_SHA256 %s\n' "$binary_sha"
printf '%s  %s\n' "$binary_sha" "$binary" | sha256sum -c -
timeout --signal=TERM --kill-after=3s 10s ldd "$binary" | tee /tmp/vram-dependencies.log
if grep -q 'not found' /tmp/vram-dependencies.log; then echo 'VRAM_RUNNER_BLOCKED unresolved binary dependency' >&2; exit 2; fi
echo 'VRAM_RUNNER_BUILD_GATE PASS'
for scenario in normal delayed-peer missing-peer; do
  idle_gate "$scenario"
  [[ $(source_manifest /work) == "$expected_sources" ]] || { echo 'VRAM_RUNNER_SOURCE_SHA_MISMATCH' >&2; exit 4; }
  printf '%s  %s\n' "$binary_sha" "$binary" | sha256sum -c -
  printf 'VRAM_RUNNER_PROCESS_BEGIN scenario=%s bound_s=%s\n' "$scenario" "$runtime_deadline"
  set +e
  timeout --signal=TERM --kill-after=5s "${runtime_deadline}s" "$binary" --scenario "$scenario" "${normal_args[@]}" 2>&1 | tee "/tmp/vram-$scenario.log"
  status=("${PIPESTATUS[@]}")
  set -e
  printf 'VRAM_RUNNER_PROCESS_EXIT scenario=%s code=%s\n' "$scenario" "${status[0]}"
  (( status[1] == 0 )) || { echo 'VRAM_RUNNER_LOG_FAILURE' >&2; exit "${status[1]}"; }
  (( status[0] == 0 )) || exit "${status[0]}"
  validate_gates "$scenario" "/tmp/vram-$scenario.log" "$iterations" "$devices"
done
[[ $(source_manifest /work) == "$expected_sources" ]] || { echo 'VRAM_RUNNER_SOURCE_SHA_MISMATCH' >&2; exit 4; }
echo 'VRAM_RUNNER_CONTAINER_COMPLETE normal=pass delayed_peer=pass missing_peer=finite_timeout_pass'
VRAM_CONTAINER
  } | timeout --signal=TERM --kill-after=15s "$((remaining + 30))s" "${DOCKER[@]}" run \
    --name "$CONTAINER_NAME" --rm --pull never --network none --read-only --cap-drop ALL \
    --tmpfs /tmp:rw,exec,nosuid,size=2g --shm-size=512m -e HOME=/tmp \
    --device=/dev/kfd --device=/dev/dri -e HIP_VISIBLE_DEVICES=0,1 \
    --mount "type=bind,src=$ROOT,dst=/work,readonly" "${identity[@]}" "${mounts[@]}" \
    --workdir /work --entrypoint /usr/bin/timeout -i "$IMAGE" \
    --signal=TERM --kill-after=15s "${remaining}s" /bin/bash -s -- \
    "$deadline" "$build_deadline" "${BUILD_JOBS:-48}" "$source_sha" "$IMAGE" "$cache" "$@"
}
# Authorization itself is never part of the captured/timed pipeline.
set +e
(set -e; validate_host "$@") 2>&1 | tee "$LOG"
STATUS=("${PIPESTATUS[@]}"); RC=${STATUS[0]}
if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
if (( RC == 0 )); then
  authorize_docker; RC=$?
  printf 'VRAM_RUNNER_AUTH_EXIT code=%s\n' "$RC" | tee -a "$LOG"
  STATUS=("${PIPESTATUS[@]}")
  if (( RC != 0 )); then
    printf 'VRAM_RUNNER_SETUP_FAILURE stage=authorization code=%s container_attempted=0\n' "$RC" | tee -a "$LOG"
  elif (( STATUS[1] != 0 )); then RC=${STATUS[1]}
  else
    (set -e; main "$@") 2>&1 | tee -a "$LOG"
    STATUS=("${PIPESTATUS[@]}"); RC=${STATUS[0]}
    if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
  fi
fi
if (( RC == 0 )); then
  # Revalidate the durable host transcript, not merely a claimed success footer.
  iterations=128; devices=0,1
  args=("$@")
  for ((index=0; index<${#args[@]}; index+=2)); do
    case "${args[index]}" in --iterations) iterations=${args[index+1]} ;; --devices) devices=${args[index+1]} ;; esac
  done
  validate_gates host "$LOG" "$iterations" "$devices" "$CONTAINER_NAME" 2>&1 | tee -a "$LOG"
  STATUS=("${PIPESTATUS[@]}"); RC=${STATUS[0]}
  if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
fi
if (( RC == 77 )); then echo 'SKIPPED is not a pass.' | tee -a "$LOG"; fi
if (( RC == 70 || RC == 124 || RC == 137 || RC == 143 )); then
  echo 'Unexpected watchdog/timeout/kill: inspect GPU state and verified container cleanup before another run.' | tee -a "$LOG"
fi
if (( RC == 0 )); then
  echo 'VRAM_RUNNER_SUITE_PASS normal=pass delayed_peer=pass missing_peer=finite_timeout_pass timing_admitted=0' | tee -a "$LOG"
  STATUS=("${PIPESTATUS[@]}"); if (( STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
fi
printf 'TP2_VRAM_PROBE_EXIT=%s LOG=%s\n' "$RC" "$LOG" | tee -a "$LOG"
STATUS=("${PIPESTATUS[@]}")
if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
exit "$RC"
