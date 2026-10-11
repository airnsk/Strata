#!/usr/bin/env python3
"""CPU boundary tests only: no Docker daemon, HIP compiler, GPU or model needed."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import pty
import shlex
import shutil
import subprocess
import sys
import tempfile
import termios
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'tools/run_tp2_vram_probe_mi50.sh'
IMAGE = 'sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca'
SUITE = 'VRAM_RUNNER_SUITE_PASS'
SOURCE = RUNNER.read_text()
SHARED = SOURCE.split('# BEGIN_SHARED_FUNCTIONS\n', 1)[1].split('# END_SHARED_FUNCTIONS', 1)[0]
INNER = SOURCE.split("<<'VRAM_CONTAINER'\n", 1)[1].split('\nVRAM_CONTAINER\n', 1)[0]


def executable(path, contents):
    path.write_text(contents)
    path.chmod(0o755)


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def transcript(scenario, iterations=128, devices=(0, 1)):
    lines = [f'VRAM_GPU rank={r} device={devices[r]} pci=0000:{r+3:02x}:00.0 arch=gfx906 name=CPU fixture runtime=70200 driver=70200 allocation=hipDeviceMallocUncached'
             for r in (0, 1)]
    if scenario == 'missing-peer':
        for first in (0, 1):
            lines += [f'VRAM_FAULT_ARMED scenario=missing-peer first={first} omitted={1-first}',
                      f'VRAM_FAULT_CASE_PASS first={first} omitted={1-first} late_peer_aborted=1 repeated_call_aborted=1 output_poison=1 both_drained=1']
        lines += ['VRAM_NEGATIVE_CASE_PASS kind=' + kind + ' status=invalid-epoch output_poison=1 both_drained=1'
                  for kind in ('future-ready', 'future-consumed', 'epoch-wrap')]
        lines += [name + ' scenario=missing-peer' for name in
                  ('VRAM_TIMEOUT_GATE_PASS', 'VRAM_STICKY_FAILURE_PASS', 'VRAM_DRAIN_PASS')]
    for life in (0, 1):
        epoch = 0 if life == 0 else 4294967294
        for tokens in ((1, 8) if scenario == 'normal' else (8,)):
            for first in (0, 1):
                for kind in (0, 1, 2):
                    lines.append(f'VRAM_CASE_PASS scenario={scenario} life={life} kind={kind} tokens={tokens} order={first}{1-first} replays={iterations} epochs={epoch+1}..{epoch+iterations} device_witness=1 guards=1 ring_slots=2')
                    epoch += iterations
        if scenario == 'normal':
            lines.append(f'VRAM_ALTERNATING_SHAPE_PASS scenario=normal life={life} order={life}{1-life} replays={iterations} tokens=8,1 fixed_stride=32768 device_witness=1')
        lines.append(f'VRAM_FRESH_LIFECYCLE_PASS scenario={scenario} life={life} both_drained=1 released=1')
        if life == 1:
            lines.append(f'VRAM_EPOCH_CARRY_PASS scenario={scenario} life=1 seed=4294967294')
    if scenario == 'delayed-peer':
        lines.append('VRAM_DELAYED_PEER_PASS scenario=delayed-peer')
    lines += ['VRAM_EXACT_GATE_PASS scenario=' + scenario,
              'VRAM_LIFECYCLE_GATE_PASS scenario=' + scenario + ' fresh_lifecycles=2',
              f"VRAM_PROBE_PASS scenario={scenario} cases={dict(normal=24, **{'delayed-peer': 12, 'missing-peer': 2})[scenario]} lifecycles=2 timing_admitted=0"]
    return '\n'.join(lines) + '\n'


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / 'bin'; self.bin.mkdir()
        self.repo = self.root / 'repo'; self.repo.mkdir()
        for name in ('tools/run_tp2_vram_probe_mi50.sh', 'tests/hip/tp2_vram_rendezvous.cpp',
                     'src/kernels/cuda/tp_vram_rendezvous.cu', 'include/strata/kernels/tp_vram_rendezvous.hpp'):
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(SOURCE if name.endswith('.sh') else '// CPU fixture\n')
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=CPU Test', '-c',
                        'user.email=cpu@example.invalid', 'commit', '-qm', 'fixture', '--allow-empty'], check=True)
        self.calls, self.sudo_calls = self.root / 'docker.jsonl', self.root / 'sudo.jsonl'
        self.timeout_calls, self.compile_calls = self.root / 'timeouts.log', self.root / 'hipcc.jsonl'
        self.fixtures = self.root / 'fixtures'; self.fixtures.mkdir()
        self.write_fixtures()
        self.env = dict(os.environ, PATH=str(self.bin) + ':' + os.environ['PATH'],
                        MOCK_CALLS=str(self.calls), MOCK_SUDO_CALLS=str(self.sudo_calls),
                        MOCK_TIMEOUT_CALLS=str(self.timeout_calls), MOCK_FIXTURES=str(self.fixtures),
                        MOCK_COMPILE_CALLS=str(self.compile_calls))
        for key in ('TP2_TIMEOUT', 'TP2_BUILD_TIMEOUT', 'TP2_OUTER_TIMEOUT', 'TP2_VRAM_CACHE', 'BUILD_JOBS'):
            self.env.pop(key, None)
        executable(self.bin / 'id', '''#!/bin/bash
case "$1" in -u) echo "${MOCK_UID:-1000}" ;; -g) echo 1000 ;; -G) echo "${MOCK_GROUPS:-1000 27 44}" ;; *) exit 2 ;; esac
''')
        executable(self.bin / 'stat', '''#!/bin/bash
[[ ${MOCK_STAT_EXIT:-0} == 0 ]] || exit "$MOCK_STAT_EXIT"
printf '%s\n' "${MOCK_DEVICE_GROUPS-109 44 0}"
''')
        executable(self.bin / 'sudo', f'''#!{sys.executable}
import json, os, pathlib, sys, time
args = sys.argv[1:]
state = pathlib.Path(os.environ['MOCK_SUDO_CALLS'] + '.refreshed')
calls = pathlib.Path(os.environ['MOCK_CALLS'])
started = calls.exists() and any(json.loads(s)[0] == 'run' for s in calls.read_text().splitlines())
expired = os.environ.get('MOCK_EXPIRE_AUTH') and started and not state.exists()
with open(os.environ['MOCK_SUDO_CALLS'], 'a') as f: f.write(json.dumps(args) + '\\n')
if args == ['-v']:
    if os.environ.get('MOCK_TIMED_OPERATION') or not all(os.isatty(fd) for fd in (0,1,2)): sys.exit(93)
    print('AUTH_TERMINAL_ONLY', flush=True)
    if expired: state.touch()
    time.sleep(float(os.environ.get('MOCK_AUTH_SLEEP', '0')))
    sys.exit(int(os.environ.get('MOCK_AUTH_EXIT', '0')))
if args == ['-n', '-v']: sys.exit(1 if expired else int(os.environ.get('MOCK_AUTH_EXIT', '0')))
if args[:2] != ['-n', 'docker']: sys.exit(95)
if expired: sys.exit(1)
os.execvp(args[1], args[1:])
''')
        self.real_timeout = shutil.which('timeout')
        executable(self.bin / 'timeout', f'''#!/bin/bash
export MOCK_TIMED_OPERATION=1
printf '%s\n' "$*" >> "$MOCK_TIMEOUT_CALLS"
if [[ " $* " == *" docker run "* && -n ${{MOCK_SLEEP:-}} ]]; then exec {shlex.quote(self.real_timeout)} 1s "${{@:4}}"; fi
if [[ " $* " == *" docker image inspect "* && -n ${{MOCK_IMAGE_TIMEOUT:-}} ]]; then exit "$MOCK_IMAGE_TIMEOUT"; fi
exec {shlex.quote(self.real_timeout)} "$@"
''')
        executable(self.bin / 'docker', f'''#!{sys.executable}
import json, os, pathlib, sys, time
args = sys.argv[1:]
with open(os.environ['MOCK_CALLS'], 'a') as f: f.write(json.dumps(args) + '\\n')
if args[:2] == ['image', 'inspect']:
    print(os.environ.get('MOCK_IMAGE', {IMAGE!r}))
elif args[0] == 'run':
    pathlib.Path(os.environ['MOCK_CALLS'] + '.container').write_text(sys.stdin.read())
    start = args.index('-s') + 2
    source_sha = args[start+3]
    print('CPU_DOCKER_STDERR_DIAGNOSTIC', file=sys.stderr, flush=True)
    print('VRAM_RUNNER_CONTAINER_ENTER', flush=True)
    print('VRAM_CONTAINER_SOURCE_SHA256 ' + ('f'*64 if os.environ.get('MOCK_SOURCE_MISMATCH') else source_sha))
    print('VRAM_RUNNER_IDLE_PASS stage=before-build cards=2 utilization=0 kfd_processes=0')
    print('VRAM_BINARY_SHA256 ' + 'a'*64)
    print('VRAM_RUNNER_BUILD_GATE PASS')
    if os.environ.get('MOCK_SLEEP'): time.sleep(5)
    for scenario in ('normal', 'delayed-peer', 'missing-peer'):
        if os.environ.get('MOCK_OMIT_PROCESS') == scenario: continue
        print('VRAM_RUNNER_IDLE_PASS stage=' + scenario + ' cards=2 utilization=0 kfd_processes=0')
        print('VRAM_RUNNER_PROCESS_BEGIN scenario=' + scenario + ' bound_s=180')
        text = (pathlib.Path(os.environ['MOCK_FIXTURES']) / scenario).read_text()
        print(text, end='')
        print('VRAM_RUNNER_PROCESS_EXIT scenario=' + scenario + ' code=' + os.environ.get('MOCK_CHILD_EXIT', '0'))
    if os.environ.get('MOCK_COMPLETE', '1') == '1':
        print('VRAM_RUNNER_CONTAINER_COMPLETE normal=pass delayed_peer=pass missing_peer=finite_timeout_pass')
    sys.exit(int(os.environ.get('MOCK_RUN_EXIT', '0')))
elif args[0] == 'rm': sys.exit(int(os.environ.get('MOCK_REMOVE_EXIT', '0')))
elif args[0] == 'ps':
    if os.environ.get('MOCK_CLEANUP_FAIL'): print('still-running')
    sys.exit(int(os.environ.get('MOCK_INSPECT_EXIT', '0')))
''')

    def write_fixtures(self, iterations=128, devices=(0, 1)):
        for scenario in ('normal', 'delayed-peer', 'missing-peer'):
            (self.fixtures / scenario).write_text(transcript(scenario, iterations, devices))

    def host(self, args=(), interactive=False, **overrides):
        for log in self.repo.glob('tp2-vram-probe-*.log'): log.unlink()
        for path in (self.calls, self.sudo_calls): path.unlink(missing_ok=True)
        command = ['bash', str(self.repo / 'tools' / RUNNER.name), *args]
        kwargs = dict(env=dict(self.env, **overrides), text=True, capture_output=True, timeout=20)
        self.auth_output = ''
        if interactive:
            master, slave = pty.openpty()
            def session():
                os.setsid(); fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
            try:
                result = subprocess.run(command, stdin=slave, preexec_fn=session, **kwargs)
                os.set_blocking(master, False)
                self.auth_output = os.read(master, 65536).decode()
            finally:
                os.close(slave); os.close(master)
        else:
            result = subprocess.run(command, start_new_session=True, **kwargs)
        logs = list(self.repo.glob('tp2-vram-probe-*.log'))
        self.assertEqual(len(logs), 1)
        saved = logs[0].read_text()
        self.assertIn(f'TP2_VRAM_PROBE_EXIT={result.returncode} ', saved)
        return result, saved

    def gate(self, scenario='normal', text=None, iterations=128, devices='0,1'):
        log = self.root / 'gate.log'; log.write_text(transcript(scenario) if text is None else text)
        return subprocess.run(['bash', '-c', SHARED + '\nvalidate_gates "$@"', 'gate', scenario,
                               str(log), str(iterations), devices], text=True, capture_output=True)

    def test_syntax_and_usage(self):
        subprocess.run(['bash', '-n', str(RUNNER)], check=True)
        result = subprocess.run(['bash', str(RUNNER), '--help'], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('--iterations 128..512', result.stdout)

    def test_host_pinned_restricted_container_and_verified_cleanup(self):
        result, saved = self.host()
        self.assertEqual(result.returncode, 0, saved + result.stderr)
        self.assertIn(SUITE, saved)
        self.assertIn('CPU_DOCKER_STDERR_DIAGNOSTIC', saved)
        calls = records(self.calls); run = next(c for c in calls if c[0] == 'run')
        self.assertEqual(sum(c[0] == 'run' for c in calls), 1)
        for value in ('--rm', '--read-only', '--device=/dev/kfd', '--device=/dev/dri'):
            self.assertIn(value, run)
        for flag, value in (('--pull', 'never'), ('--network', 'none'), ('--cap-drop', 'ALL'),
                            ('--entrypoint', '/usr/bin/timeout')):
            self.assertEqual(run[run.index(flag)+1], value)
        for unsafe in ('--privileged', '--cap-add', '--ipc', '--security-opt', '--pid'):
            self.assertNotIn(unsafe, run)
        self.assertIn(IMAGE, run)
        self.assertIn(f'type=bind,src={self.repo},dst=/work,readonly', run)
        self.assertIn(f'type=bind,src={self.repo}/.tp2-vram-build-cache,dst=/cache', run)
        self.assertEqual((self.repo / '.tp2-vram-build-cache').stat().st_mode & 0o777, 0o700)
        self.assertTrue(all(not any(word in a.lower() for word in ('model', 'rccl', 'nccl')) for a in run))
        self.assertIn('verified_absent=', saved)
        subprocess.run(['bash', '-n', str(self.calls) + '.container'], check=True)
        for call in calls:
            if call[0] == 'rm': self.assertEqual(call[2], run[run.index('--name')+1])

    def test_host_all_explicit_options_and_ephemeral_cache(self):
        self.write_fixtures(256, (1, 0))
        result, saved = self.host(['--iterations', '256', '--timeout-ms', '12000', '--devices', '1,0'], TP2_VRAM_CACHE='off')
        self.assertEqual(result.returncode, 0, saved + result.stderr)
        run = next(c for c in records(self.calls) if c[0] == 'run')
        self.assertFalse(any('dst=/cache' in a for a in run))
        self.assertIn('jobs=48', saved)

    def test_host_validation_precedes_authorization(self):
        for args, env in ((['--scenario', 'normal'], {}), (['--iterations', '127'], {}),
                          (['--iterations', '513'], {}), (['--iterations', '0128'], {}),
                          (['--timeout-ms', '999'], {}), (['--devices', '0,0'], {}),
                          (['--devices', '0,1', '--devices', '0,1'], {}), (['--devices'], {}),
                          ([], {'BUILD_JOBS': '257'}), ([], {'TP2_TIMEOUT': '0'}),
                          ([], {'TP2_OUTER_TIMEOUT': '01'}), ([], {'TP2_BUILD_TIMEOUT': '86401'}),
                          ([], {'TP2_VRAM_CACHE': '../outside'})):
            with self.subTest(args=args, env=env):
                result, saved = self.host(args, **env)
                self.assertEqual(result.returncode, 2, saved)
                self.assertFalse(self.calls.exists()); self.assertFalse(self.sudo_calls.exists())

    def test_auth_prompt_is_terminal_only_and_outside_deadline(self):
        result, saved = self.host(interactive=True, MOCK_AUTH_SLEEP='1.1', TP2_OUTER_TIMEOUT='5')
        self.assertEqual(result.returncode, 0, saved + result.stderr)
        self.assertIn('AUTH_TERMINAL_ONLY', self.auth_output)
        self.assertNotIn('AUTH_TERMINAL_ONLY', saved + result.stdout + result.stderr)
        self.assertEqual(records(self.sudo_calls)[0], ['-v'])

    def test_no_tty_auth_is_noninteractive(self):
        result, saved = self.host()
        self.assertEqual(result.returncode, 0, saved)
        self.assertTrue(all(call[0] == '-n' for call in records(self.sudo_calls)))

    def test_auth_failure_and_root_path(self):
        result, saved = self.host(MOCK_AUTH_EXIT='1')
        self.assertEqual(result.returncode, 1, saved)
        self.assertFalse(self.calls.exists())
        result, saved = self.host(MOCK_UID='0')
        self.assertEqual(result.returncode, 0, saved)
        self.assertFalse(self.sudo_calls.exists())

    def test_cleanup_expired_auth_reauthorizes_only_on_tty(self):
        result, saved = self.host(interactive=True, MOCK_EXPIRE_AUTH='1')
        self.assertEqual(result.returncode, 0, saved)
        self.assertEqual(self.auth_output.count('AUTH_TERMINAL_ONLY'), 2)
        self.assertNotIn('AUTH_TERMINAL_ONLY', saved)

    def test_cleanup_failure_is_not_success(self):
        for env in ({'MOCK_CLEANUP_FAIL': '1'}, {'MOCK_INSPECT_EXIT': '1'}, {'MOCK_EXPIRE_AUTH': '1'}):
            with self.subTest(env=env):
                Path(str(self.sudo_calls) + '.refreshed').unlink(missing_ok=True)
                result, saved = self.host(**env)
                self.assertEqual(result.returncode, 125, saved)
                self.assertNotIn(SUITE, saved)
                self.assertIn('CLEANUP_UNVERIFIED', saved)

    def test_rm_failure_is_safe_if_independent_absence_verified(self):
        result, saved = self.host(MOCK_REMOVE_EXIT='1')
        self.assertEqual(result.returncode, 0, saved)

    def test_image_mismatch_timeout_and_gpu_identity_fail_before_run(self):
        for env, code in (({'MOCK_IMAGE': 'other'}, 2), ({'MOCK_IMAGE_TIMEOUT': '124'}, 124),
                          ({'MOCK_STAT_EXIT': '1'}, 2), ({'MOCK_DEVICE_GROUPS': 'unknown'}, 2)):
            with self.subTest(env=env):
                result, saved = self.host(**env)
                self.assertEqual(result.returncode, code, saved)
                self.assertFalse(any(c[0] == 'run' for c in records(self.calls)))

    def test_host_completion_source_mismatch_and_nonzero_exit(self):
        for env, code in (({'MOCK_COMPLETE': '0'}, 4), ({'MOCK_SOURCE_MISMATCH': '1'}, 4),
                          ({'MOCK_RUN_EXIT': '42'}, 42), ({'MOCK_RUN_EXIT': '77'}, 77),
                          ({'MOCK_OMIT_PROCESS': 'missing-peer'}, 4), ({'MOCK_CHILD_EXIT': '70'}, 4)):
            with self.subTest(env=env):
                result, saved = self.host(**env)
                self.assertEqual(result.returncode, code, saved)
                self.assertNotIn(SUITE, saved)

    def test_outer_timeout_removes_unique_container(self):
        result, saved = self.host(MOCK_SLEEP='1')
        self.assertEqual(result.returncode, 124, saved)
        self.assertIn('verified_absent=', saved); self.assertNotIn(SUITE, saved)

    def test_host_cache_symlink_or_open_permissions_rejected(self):
        cache = self.repo / '.tp2-vram-build-cache'
        cache.symlink_to(self.root, target_is_directory=True)
        result, saved = self.host()
        self.assertNotEqual(result.returncode, 0, saved)
        cache.unlink(); cache.mkdir(mode=0o755)
        result, saved = self.host()
        self.assertNotEqual(result.returncode, 0, saved)
        self.assertFalse(any(c[0] == 'run' for c in records(self.calls)))

    def test_complete_ordered_child_matrices(self):
        for scenario in ('normal', 'delayed-peer', 'missing-peer'):
            result = self.gate(scenario)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_summary_only_malformed_or_partial_matrix_cannot_pass(self):
        for scenario in ('normal', 'delayed-peer', 'missing-peer'):
            original = transcript(scenario)
            lines = original.splitlines()
            for prefix in ('VRAM_GPU', 'VRAM_CASE_PASS', 'VRAM_FRESH_LIFECYCLE_PASS', 'VRAM_PROBE_PASS',
                           'VRAM_NEGATIVE_CASE_PASS', 'VRAM_FAULT_CASE_PASS', 'VRAM_TIMEOUT_GATE_PASS',
                           'VRAM_STICKY_FAILURE_PASS', 'VRAM_DRAIN_PASS', 'VRAM_DELAYED_PEER_PASS',
                           'VRAM_ALTERNATING_SHAPE_PASS', 'VRAM_EPOCH_CARRY_PASS'):
                matching = [i for i, line in enumerate(lines) if line.startswith(prefix)]
                if not matching: continue
                for mutation in ('drop', 'duplicate', 'malformed'):
                    with self.subTest(scenario=scenario, prefix=prefix, mutation=mutation):
                        copy = list(lines); index = matching[0]
                        if mutation == 'drop': copy.pop(index)
                        elif mutation == 'duplicate': copy.insert(index, copy[index])
                        else: copy[index] += ' junk=1'
                        result = self.gate(scenario, '\n'.join(copy) + '\n')
                        self.assertNotEqual(result.returncode, 0, result.stderr)
            self.assertNotEqual(self.gate(scenario, '\n'.join(lines[-3:]) + '\n').returncode, 0)
            self.assertNotEqual(self.gate(scenario, original.rstrip()).returncode, 0)

    def test_wrong_devices_epoch_replays_order_and_hidden_watchdog_rejected(self):
        original = transcript('normal')
        mutations = [original.replace('device=1', 'device=0'), original.replace('03:00.0', '04:00.0'),
                     original.replace('epochs=1..128', 'epochs=2..129'), original.replace('replays=128', 'replays=8'),
                     original.replace('order=01', 'order=10'), original.replace('gfx906', 'gfx90a'),
                     original.replace('guards=1', 'guards=0'), original + 'late output\n']
        for marker in ('VRAM_WATCHDOG_TIMEOUT scenario=normal stage=hang', 'VRAM_PERF sample=1',
                       'BENCH_SUMMARY sample=1', 'SKIPPED no device', 'VRAM_PROBE_FAIL ignored',
                       'noise WATCHDOG_TIMEOUT hidden=1', 'VRAM_CASE_PASS nonsense'):
            mutations.append(marker + '\n' + original)
        for text in mutations:
            with self.subTest(text=text[:100]): self.assertNotEqual(self.gate(text=text).returncode, 0)

    def test_host_revalidates_partial_success_transcript(self):
        fixture = self.fixtures / 'normal'
        fixture.write_text('\n'.join(transcript('normal').splitlines()[-3:]) + '\n')
        result, saved = self.host()
        self.assertEqual(result.returncode, 4, saved)
        self.assertIn('VRAM_RUNNER_ADMISSION_FAIL', saved)
        self.assertNotIn(SUITE, saved)

    def inner(self, args=(), **overrides):
        tmp = self.root / 'inner-tmp'; tmp.mkdir(exist_ok=True, mode=0o700)
        self.cache = self.root / 'inner-cache'; self.cache.mkdir(exist_ok=True, mode=0o700)
        self.kfd = self.root / 'kfd-proc'; self.kfd.mkdir(exist_ok=True)
        binary = self.root / 'mock-probe'
        executable(binary, f'''#!{sys.executable}
import os, pathlib, sys
args = sys.argv[1:]; scenario = args[args.index('--scenario')+1]
with open(os.environ['MOCK_COMPILE_CALLS'], 'a') as f: f.write(__import__('json').dumps(['probe', *args]) + '\\n')
print((pathlib.Path(os.environ['MOCK_FIXTURES']) / scenario).read_text(), end='')
sys.exit(int(os.environ.get('MOCK_' + scenario.replace('-', '_').upper() + '_EXIT', '0')))
''')
        executable(self.bin / 'hipcc', f'''#!{sys.executable}
import json, os, pathlib, shutil, sys, time
args = sys.argv[1:]
with open(os.environ['MOCK_COMPILE_CALLS'], 'a') as f: f.write(json.dumps(['hipcc', *args]) + '\\n')
if args == ['--version']:
    print(os.environ.get('MOCK_COMPILER_VERSION', 'CPU HIP fixture'))
    sys.exit(0)
if os.environ.get('MOCK_BUILD_SLEEP'): time.sleep(5)
if os.environ.get('MOCK_COMPILE_EXIT'): sys.exit(int(os.environ['MOCK_COMPILE_EXIT']))
out = pathlib.Path(args[args.index('-o')+1])
if '-c' in args:
    out.write_text('CPU object ' + str(out))
    dep = pathlib.Path(args[args.index('-MF')+1])
    dep.write_text(str(out) + ': ' + args[args.index('-c')+1] + '\\n')
else:
    shutil.copyfile({str(binary)!r}, out); out.chmod(0o755)
''')
        executable(self.bin / 'rocm-smi', f'''#!{sys.executable}
import json, os, sys
sys.exit(int(os.environ['MOCK_SMI_EXIT'])) if os.environ.get('MOCK_SMI_EXIT') else None
print(os.environ.get('MOCK_SMI', json.dumps({{'card1': {{'GPU use (%)': '0'}}, 'card2': {{'GPU use (%)': '0'}}}})))
''')
        executable(self.bin / 'ldd', '#!/bin/bash\necho "${MOCK_LDD-fixture: all dependencies resolved}"\n')
        real_tee = shutil.which('tee')
        executable(self.bin / 'tee', f'''#!/bin/bash
{shlex.quote(real_tee)} "$@"
if [[ "$1" == */vram-normal.log && ${{MOCK_TEE_FAIL:-0}} == 1 ]]; then exit 71; fi
exit 0
''')
        script = SHARED + '\n' + INNER
        script = script.replace('export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH', ': # Keep CPU mocks first')
        script = script.replace('/tmp', str(tmp)).replace('/work', str(self.repo)).replace('/cache', str(self.cache))
        script = script.replace('/sys/class/kfd/kfd/proc', str(self.kfd))
        inner_file = self.root / 'inner.sh'; inner_file.write_text(script)
        source_sha = subprocess.run(['bash', '-c', SHARED + '\nsource_manifest "$1"', 'manifest', str(self.repo)],
                                    text=True, capture_output=True, check=True).stdout.strip()
        if overrides.pop('WRONG_SOURCE_SHA', ''): source_sha = '0'*64
        if overrides.pop('ACTIVE_KFD', ''): (self.kfd / '1234').mkdir(exist_ok=True)
        return subprocess.run(['bash', str(inner_file), '5', overrides.pop('BUILD_DEADLINE', '5'), '48',
                               source_sha, IMAGE, 'persistent', *args], env=dict(self.env, **overrides),
                              text=True, capture_output=True, timeout=20)

    def test_inner_two_source_build_full_matrix_and_incremental_reuse(self):
        result = self.inner()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = records(self.compile_calls)
        compiler = [c for c in calls if c[0] == 'hipcc' and '--version' not in c]
        self.assertEqual(len(compiler), 3)
        for call in compiler[:2]:
            for flag in ('-DSTRATA_HIP_GFX906=1', '-fno-fast-math', '-ffp-contract=off', '--offload-arch=gfx906', '-pthread'):
                self.assertIn(flag, call)
        self.assertEqual(len([c for c in calls if c[0] == 'probe']), 3)
        self.assertIn('jobs=48', result.stdout)
        self.compile_calls.unlink()
        result = self.inner()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len([c for c in records(self.compile_calls) if c[0] == 'hipcc' and '--version' not in c]), 0)
        source = self.repo / 'tests/hip/tp2_vram_rendezvous.cpp'; source.write_text('// changed content\n')
        self.compile_calls.unlink()
        result = self.inner()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        compiled = [c for c in records(self.compile_calls) if c[0] == 'hipcc' and '-c' in c]
        self.assertEqual(len(compiled), 1)
        self.assertTrue(compiled[0][compiled[0].index('-c')+1].endswith('tp2_vram_rendezvous.cpp'))

    def test_inner_source_mismatch_and_build_failure_never_execute(self):
        for env in ({'WRONG_SOURCE_SHA': '1'}, {'MOCK_COMPILE_EXIT': '34'}, {'MOCK_LDD': 'libhip.so => not found'}):
            with self.subTest(env=env):
                self.compile_calls.unlink(missing_ok=True)
                result = self.inner(**env)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse(any(c[0] == 'probe' for c in records(self.compile_calls)))

    def test_inner_cached_binary_and_depfile_sha_tampering_fails(self):
        result = self.inner(); self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        binary = next(self.cache.glob('*/tp2_vram_rendezvous'))
        binary.write_text(binary.read_text() + '\n# modified\n')
        self.compile_calls.unlink()
        result = self.inner()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('BINARY_SHA_MISMATCH', result.stderr)
        self.assertFalse(any(c[0] == 'probe' for c in records(self.compile_calls)))

    def test_inner_build_timeout_is_failure_without_probe(self):
        result = self.inner(MOCK_BUILD_SLEEP='1', BUILD_DEADLINE='1')
        self.assertEqual(result.returncode, 124, result.stdout + result.stderr)
        self.assertFalse(any(c[0] == 'probe' for c in records(self.compile_calls)))

    def test_inner_idle_check_fails_closed(self):
        for env in ({'MOCK_SMI': 'not-json'}, {'MOCK_SMI': '{"card0":{"GPU use (%)":"0"}}'},
                    {'MOCK_SMI': '{"card0":{"GPU use (%)":"1"},"card1":{"GPU use (%)":"0"}}'},
                    {'MOCK_SMI_EXIT': '1'}, {'ACTIVE_KFD': '1'}):
            with self.subTest(env=env):
                self.compile_calls.unlink(missing_ok=True)
                result = self.inner(**env)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(c[0] == 'probe' for c in records(self.compile_calls)))

    def test_cpu_selftest_real_harness_without_hip(self):
        compiler = shutil.which('g++') or shutil.which('c++')
        if not compiler:
            self.skipTest('No installed host C++ compiler; no installation attempted')
        output = self.root / 'cpu-selftest'
        build = subprocess.run([compiler, '-std=c++17', '-O2', '-fno-fast-math', '-ffp-contract=off',
                                '-DSTRATA_VRAM_CPU_SELFTEST', '-I', str(ROOT / 'include'),
                                str(ROOT / 'tests/hip/tp2_vram_rendezvous.cpp'), '-o', str(output)],
                               text=True, capture_output=True, timeout=60)
        self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
        result = subprocess.run([str(output), '--selftest'], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('VRAM_CPU_SELFTEST_PASS', result.stdout)
        self.assertIn('NOT_GPU_TEST=1', result.stdout)

    def test_inner_header_and_compiler_config_changes_invalidate_cache(self):
        result = self.inner(); self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        header = self.repo / 'include/strata/kernels/tp_vram_rendezvous.hpp'
        header.write_text('// changed header content\n')
        self.compile_calls.unlink()
        result = self.inner(); self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len([c for c in records(self.compile_calls) if c[0] == 'hipcc' and '-c' in c]), 2)
        self.compile_calls.unlink()
        result = self.inner(MOCK_COMPILER_VERSION='changed compiler identity')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len([c for c in records(self.compile_calls) if c[0] == 'hipcc' and '-c' in c]), 2)

    def test_inner_cache_depfile_manifest_and_lock_hardening(self):
        result = self.inner(); self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        depfile = next(self.cache.glob('*/probe.d'))
        original = depfile.read_text(); depfile.write_text('$(shell touch /tmp/never-execute)\n')
        result = self.inner()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('BINARY_SHA_MISMATCH', result.stderr)
        depfile.write_text(original)
        manifest = next(self.cache.glob('*/manifest.json'))
        data = json.loads(manifest.read_text()); data['artifacts'].pop('probe.d')
        manifest.write_text(json.dumps(data))
        result = self.inner(); self.assertNotEqual(result.returncode, 0)
        self.assertIn('malformed manifest', result.stderr)
        manifest.unlink()
        # No manifest: discard every incomplete artifact before invoking make.
        depfile.write_text('$(error unverified depfile was consumed)\n')
        result = self.inner(); self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lock = self.cache / '.lock'; lock.unlink(); lock.symlink_to(self.root / 'outside-lock')
        result = self.inner(); self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'outside-lock').exists())

    def test_inner_idle_drm_numbers_and_duplicate_records(self):
        # The real stand uses DRM card1/card2, while HIP logical ordinals are0/1.
        valid = {'card1': {'GPU use (%)': '0'}, 'card2': {'GPU use (%)': 0}}
        result = self.inner(MOCK_SMI=json.dumps(valid))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        invalid = [dict(valid, card3={'GPU use (%)': '0'}),
                   {'card1': valid['card1'], 'cardx': valid['card2']},
                   {'card1': valid['card1'], 'card01': valid['card2']},
                   {'card1': valid['card1'], 'card2': {'GPU use (%)': True}},
                   {'card1': valid['card1'], 'card2': {'GPU use (%)': 'unknown'}}]
        for data in invalid:
            with self.subTest(data=data):
                self.compile_calls.unlink(missing_ok=True)
                result = self.inner(MOCK_SMI=json.dumps(data))
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(c[0] == 'probe' for c in records(self.compile_calls)))
        duplicate = json.dumps(valid)[:-1] + ', "card2":' + json.dumps(valid['card2']) + '}'
        result = self.inner(MOCK_SMI=duplicate)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('duplicate telemetry key', result.stderr)

    def test_inner_watchdog_skip_log_failure_never_pass(self):
        for env, code in (({'MOCK_NORMAL_EXIT': '70'}, 70), ({'MOCK_NORMAL_EXIT': '77'}, 77),
                          ({'MOCK_MISSING_PEER_EXIT': '124'}, 124), ({'MOCK_TEE_FAIL': '1'}, 71)):
            with self.subTest(env=env):
                result = self.inner(**env)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                self.assertNotIn('VRAM_RUNNER_CONTAINER_COMPLETE', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
