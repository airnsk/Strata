#!/usr/bin/env bash
# Read-only inspection of the Oct 10 RCCL host segfault; never launches GPU work.
set -euo pipefail
if [[ ${1:-} == --help && $# == 1 ]]; then
  cat <<'EOF'
Usage: bash tools/inspect_rccl_hip_crash.sh
Inspect the already-installed pinned MI50 image and host /var/crash filenames.
No downloads, GPU devices, probe rerun, core contents, or system-setting changes.
Sudo authorization occurs before timed operations; all later sudo is noninteractive.
Output goes to stdout. Save it with shell redirection or tee if desired.
EOF
  exit 0
fi
(( $# == 0 )) || { echo 'Only --help is supported.' >&2; exit 2; }
IMAGE=sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca
NAME="strata-hip-inspect-$(date -u +%Y%m%dT%H%M%SZ)-$$"
PRIV=(); started=0
for tool in docker timeout find date; do
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
      echo 'HIP_INSPECT_CLEANUP verified_absent=1'
    else
      echo "HIP_INSPECT_CLEANUP unverified=$NAME" >&2
      (( rc != 0 )) || rc=125
    fi
  fi
  exit "$rc"
}
printf 'HIP_INSPECT_HOST '; date --iso-8601=seconds
printf 'HIP_INSPECT_CORE_PATTERN '; cat /proc/sys/kernel/core_pattern
# Metadata only, not Apport report contents. The user's journal query used UTC;
# do not interpret this window in the stand's local timezone.
echo 'HIP_INSPECT_APPORT_FILES utc_window=2026-10-10T20:10:00Z..20:25:00Z (maximum 20 filenames)'
if [[ -d /var/crash ]]; then
  bounded "${PRIV[@]}" find /var/crash -maxdepth 1 -type f \
    -newermt '2026-10-10 20:10:00 UTC' ! -newermt '2026-10-10 20:25:00 UTC' \
    -printf '%f\tbytes=%s\tmtime=%TY-%Tm-%TdT%TH:%TM:%TS%Tz\n' |
    awk 'NR<=20 {print} END {printf "HIP_INSPECT_APPORT_FILES matched=%d\n", NR}'
else
  echo 'HIP_INSPECT_APPORT_FILES directory_absent=1'
fi
resolved=$(bounded "${PRIV[@]}" docker image inspect --format '{{.Id}}' "$IMAGE")
[[ $resolved == "$IMAGE" ]] || { echo 'Installed image does not match pinned ID.' >&2; exit 2; }
printf 'HIP_INSPECT_IMAGE %s\n' "$resolved"
trap cleanup EXIT
started=1
# Intentionally no no-new-privileges option: the stand's Snap Docker rejects it.
# No bind mounts, /dev/kfd, /dev/dri, network, installation, or HIP runtime loading.
timeout --signal=TERM --kill-after=5s 120s "${PRIV[@]}" docker run \
  --name "$NAME" --rm --pull never --network none --read-only --cap-drop ALL \
  --user "$(id -u):$(id -g)" -e HOME=/tmp --entrypoint /bin/bash -i "$IMAGE" \
  -c 'export PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:/opt/venv/bin:$PATH; exec python3 -B -u -' <<'PY'
import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess

IP, VMA, VMA_SIZE = 0x701be2ec5634, 0x701be2ab0000, 0x489000
EXPECTED = bytes.fromhex('48 8b 47 20 48 83 c0 38 c3')
# The existing probe's ldd output identifies this installation. Do not substitute
# another library or infer that the kernel's basename uniquely identifies it.
lib = Path('/opt/rocm-7.2.4/lib/libamdhip64.so.7').resolve(strict=True)
if lib.name != 'libamdhip64.so.7.2.70204':
    raise SystemExit('Unexpected HIP library basename: ' + str(lib))
print('HIP_INSPECT_LIBRARY path=' + str(lib))
with lib.open('rb') as f:
    digest = hashlib.sha256()
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        digest.update(chunk)
print('HIP_INSPECT_SHA256 ' + digest.hexdigest())

def tool(*names):
    return next((p for name in names if (p := shutil.which(name))), None)

def run(exe, *args):
    if not exe:
        print('HIP_INSPECT_TOOL unavailable args=' + repr(args))
        return ''
    print('HIP_INSPECT_TOOL ' + repr([exe, *map(str, args)]))
    r = subprocess.run([exe, *map(str, args)], text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=25, check=False)
    if r.stderr:
        print(r.stderr.rstrip())
    if r.returncode:
        print('HIP_INSPECT_TOOL exit=' + str(r.returncode))
    return r.stdout

readelf = tool('llvm-readelf', 'readelf')
objdump = tool('llvm-objdump', 'objdump')
addr2line = tool('llvm-addr2line', 'addr2line')
nm = tool('llvm-nm', 'nm')
notes = run(readelf, '-n', lib)
build_ids = [line.strip()[:160] for line in notes.splitlines() if 'Build ID:' in line]
print('HIP_INSPECT_BUILD_ID ' + ('; '.join(build_ids[:8]) or 'unavailable'))
headers = run(readelf, '-lW', lib).splitlines()
print('\n'.join(headers[:128]))
if len(headers) > 128:
    print('HIP_INSPECT_PROGRAM_HEADERS truncated_after_lines=128')
# Read ELF fields directly so formatting differences between GNU/LLVM readelf
# cannot silently corrupt the load-bias calculation. No library is dlopened.
with lib.open('rb') as f:
    header = f.read(64)
    if header[:6] != b'\x7fELF\x02\x01':
        raise SystemExit('Expected a little-endian ELF64 library')
    h = struct.unpack('<16sHHIQQQIHHHHHH', header)
    if h[2] != 62 or h[1] != 3:
        raise SystemExit('Expected an x86-64 ET_DYN library')
    page = os.sysconf('SC_PAGE_SIZE')
    delta = IP - VMA
    print(f'HIP_INSPECT_CRASH ip={IP:#x} vma={VMA:#x} vma_size={VMA_SIZE:#x} delta={delta:#x} page={page:#x}')
    candidates = []
    for i in range(h[10]):
        f.seek(h[5] + i * h[9])
        kind, flags, offset, vaddr, _, filesz, memsz, align = struct.unpack('<IIQQQQQQ', f.read(56))
        if kind != 1 or not flags & 1:  # PT_LOAD and PF_X.
            continue
        vpage = vaddr // page * page
        bias = VMA - vpage
        address = IP - bias
        span = ((vaddr + memsz + page - 1) // page * page) - vpage
        if not vaddr <= address <= vaddr + filesz - len(EXPECTED):
            continue
        file_offset = offset + address - vaddr
        f.seek(file_offset)
        actual = f.read(len(EXPECTED))
        byte_match = actual == EXPECTED
        matches = byte_match and span == VMA_SIZE
        print(f'HIP_INSPECT_CANDIDATE phdr={i} p_offset={offset:#x} p_vaddr={vaddr:#x} '
              f'p_filesz={filesz:#x} p_memsz={memsz:#x} p_align={align:#x} '
              f'load_bias={bias:#x} elf_ip={address:#x} file_offset={file_offset:#x} '
              f'mapped_span={span:#x} span_match={int(span == VMA_SIZE)} '
              f'bytes={actual.hex(" ")} bytes_match={int(byte_match)} mapping_match={int(matches)}')
        candidates.append((address, matches))
if not candidates:
    raise SystemExit('No executable PT_LOAD candidate contains the fault-site delta')
for address, matches in candidates:
    if not matches:
        print(f'HIP_INSPECT_REJECT elf_ip={address:#x}: fault-site bytes or executable VMA span differ')
        continue
    print(f'HIP_INSPECT_SYMBOLIZE elf_ip={address:#x} fault_bytes_match={int(matches)}')
    print(run(addr2line, '-f', '-C', '-i', '-e', lib, hex(address)))
    print(run(objdump, '-d', '-C', '--start-address=' + hex(address),
              '--stop-address=' + hex(address + 96), lib))
    for extra in ([], ['-D']):
        symbols = run(nm, *extra, '-n', '-C', '--defined-only', lib).splitlines()
        parsed = []
        for line in symbols:
            fields = line.split(None, 2)
            try:
                value = int(fields[0], 16)
            except (ValueError, IndexError):
                continue
            if len(fields) == 3 and fields[1] in ('t', 'T', 'w', 'W'):
                parsed.append((value, line))
        before = [x for x in parsed if x[0] <= address][-3:]
        after = [x for x in parsed if x[0] > address][:3]
        print('HIP_INSPECT_NEARBY_SYMBOLS dynamic=' + str(bool(extra)))
        for _, line in before + after:
            print(line)
print('HIP_INSPECT_MATCHING_CANDIDATES ' + str(sum(matches for _, matches in candidates)))
print('HIP_INSPECT_LIMIT: VMA/segment candidates and bytes do not establish the runtime caller.')
print('A saved core/registers/backtrace and matching mapped-object identity are needed for the caller stack.')
if not any(matches for _, matches in candidates):
    raise SystemExit('No candidate matches both the reported fault-site bytes and executable VMA span; do not trust a symbol name alone')
PY
