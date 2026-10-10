#!/usr/bin/env python3
"""CPU-only boundary tests for the opt-in isolated RCCL source build."""
import json
import fcntl
import pty
import termios
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'tools/build_rccl_gfx906_mi50.sh'
IMAGE = 'sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca'
PASS = 'RCCL_BUILD_STATIC_PASS gpu_tests=not_run artifact=/out/artifact.json'


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        (self.repo / 'tools').mkdir(parents=True)
        shutil.copy2(SCRIPT, self.repo / 'tools' / SCRIPT.name)
        (self.repo / 'tools/verify_rccl_gfx906.py').write_text('# fixture\n')
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=CPU Test', '-c',
                        'user.email=cpu@example.invalid', 'commit', '-qm', 'fixture', '--allow-empty'], check=True)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.calls = self.root / 'calls.jsonl'
        self.env = dict(os.environ, PATH=str(self.bin) + ':' + os.environ['PATH'], CALLS=str(self.calls))
        self.write_tool('id', '#!/bin/sh\necho ${MOCK_UID:-0}\n')
        self.env['EXPIRY_FILE'] = str(self.root / 'expired')
        self.write_tool('sudo', '''#!/usr/bin/env python3
import os, sys
from pathlib import Path
args = sys.argv[1:]
expired = Path(os.environ['EXPIRY_FILE'])
if args == ['-v']:
 assert all(os.isatty(fd) for fd in (0, 1, 2))
 print('MOCK_TERMINAL_AUTH', flush=True)
 expired.unlink(missing_ok=True)
 sys.exit(0)
if args == ['-n', '-v']: sys.exit(1 if expired.exists() else 0)
if args[:2] != ['-n', 'docker'] or expired.exists(): sys.exit(1)
os.execvp('docker', args[1:])
''')
        self.write_tool('docker', f'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['CALLS'], 'a') as f: f.write(json.dumps(args) + '\\n')
if args[:2] == ['image', 'inspect']:
 print(os.environ.get('MOCK_IMAGE', {IMAGE!r}))
elif args[0] == 'run':
 body = sys.stdin.read()
 stage = 'fetch' if '--network' in args and args[args.index('--network')+1] == 'bridge' else 'build'
 Path(os.environ['CALLS'] + '.' + stage).write_text(body)
 if stage == 'build' and os.environ.get('MOCK_EXPIRE'): Path(os.environ['EXPIRY_FILE']).touch()
 if stage == 'build' and os.environ.get('MOCK_MARKER', '1') == '1': print({PASS!r})
 sys.exit(int(os.environ.get('MOCK_' + stage.upper() + '_EXIT', '0')))
elif args[0] == 'ps' and os.environ.get('MOCK_CLEANUP_FAIL'): print('still-present')
''')

    def write_tool(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(0o755)

    def run_build(self, args=(), interactive=False, **env):
        command = ['bash', str(self.repo / 'tools' / SCRIPT.name), *args]
        kwargs = dict(env=dict(self.env, **env), capture_output=True, text=True, timeout=10)
        if not interactive:
            return subprocess.run(command, start_new_session=True, **kwargs)
        master, slave = pty.openpty()
        def session():
            os.setsid()
            fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
        try:
            result = subprocess.run(command, stdin=slave, preexec_fn=session, **kwargs)
            os.set_blocking(master, False)
            self.auth_output = os.read(master, 65536).decode()
            return result
        finally:
            os.close(slave)
            os.close(master)

    def recorded(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def test_isolated_two_stage_contract(self):
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        runs = [a for a in self.recorded() if a[0] == 'run']
        self.assertEqual(len(runs), 2)
        for run, network in zip(runs, ('bridge', 'none')):
            self.assertEqual(run[run.index('--network') + 1], network)
            self.assertEqual(run[run.index('--pull') + 1], 'never')
            self.assertIn('--read-only', run)
            self.assertIn('--cap-drop', run)
            self.assertIn('--user', run)
            self.assertEqual(run[run.index('--entrypoint') + 1], '/usr/bin/timeout')
            self.assertIn('/bin/bash', run)
            self.assertIn('--kill-after=15s', run)
            self.assertFalse(any(a.startswith('--device') or a == '--privileged' for a in run))
            self.assertIn(IMAGE, run)
        self.assertEqual(runs[1][runs[1].index('--cpus') + 1], '24')
        self.assertEqual(runs[1][runs[1].index('--memory') + 1], '128g')
        self.assertEqual(runs[1][runs[1].index('--memory-swap') + 1], '128g')
        for stage in ('fetch', 'build'):
            subprocess.run(['bash', '-n', str(self.calls) + '.' + stage], check=True)
        body = Path(str(self.calls) + '.build').read_text()
        for flag in ('-DGPU_TARGETS=gfx906', '-DHAVE_PARALLEL_JOBS=OFF', '-DBUILD_TESTS=OFF',
                     '-DINSTALL_DEPENDENCIES=OFF', '-DFETCHCONTENT_FULLY_DISCONNECTED=ON',
                     '-DCMAKE_INSTALL_PREFIX=/out/install'):
            self.assertIn(flag, body)
        self.assertLess(body.index('RCCL_BUILD_TARGET_GATE'), body.index('cmake --build'))
        self.assertLess(body.index('verify_rccl_gfx906.py'), body.index('RCCL_BUILD_STATIC_PASS'))
        self.assertIn('RCCL_BUILD_CLEANUP verified_absent=', result.stdout)

    def test_long_build_expiry_reauthorizes_only_on_terminal(self):
        result = self.run_build(interactive=True, MOCK_UID='1000', MOCK_EXPIRE='1')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.auth_output.count('MOCK_TERMINAL_AUTH'), 2)
        self.assertNotIn('MOCK_TERMINAL_AUTH', result.stdout + result.stderr)
        self.assertIn('RCCL_BUILD_CLEANUP verified_absent=', result.stdout)

    def test_long_build_expiry_without_terminal_fails_cleanup(self):
        result = self.run_build(MOCK_UID='1000', MOCK_EXPIRE='1')
        self.assertEqual(result.returncode, 125, result.stdout + result.stderr)
        self.assertIn('RCCL_BUILD_CLEANUP_UNVERIFIED', result.stdout)

    def test_48_jobs_allowed(self):
        result = self.run_build(RCCL_BUILD_JOBS='48')
        self.assertEqual(result.returncode, 0, result.stdout)
        run = [a for a in self.recorded() if a[0] == 'run'][-1]
        self.assertEqual(run[run.index('--cpus') + 1], '48')

    def test_invalid_bounds_before_docker(self):
        for key, value in [('RCCL_BUILD_JOBS', '49'), ('RCCL_BUILD_JOBS', '024'),
                           ('RCCL_BUILD_MEMORY_GIB', '15'), ('RCCL_BUILD_MEMORY_GIB', '257'),
                           ('RCCL_BUILD_TIMEOUT', '0'), ('RCCL_FETCH_TIMEOUT', '3601')]:
            with self.subTest(key=key, value=value):
                self.assertEqual(self.run_build(**{key: value}).returncode, 2)
                self.assertFalse(self.calls.exists())

    def test_existing_output_is_never_reused(self):
        result = self.run_build(['--output', str(self.repo)])
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.calls.exists())

    def test_image_mismatch_does_not_fetch(self):
        result = self.run_build(MOCK_IMAGE='wrong')
        self.assertEqual(result.returncode, 2)
        self.assertFalse(any(a[0] == 'run' for a in self.recorded()))

    def test_fetch_failure_stops_before_build(self):
        result = self.run_build(MOCK_FETCH_EXIT='41')
        self.assertEqual(result.returncode, 41)
        self.assertEqual(sum(a[0] == 'run' for a in self.recorded()), 1)
        self.assertIn('RCCL_BUILD_CLEANUP verified_absent=', result.stdout)

    def test_build_failure_preserved(self):
        self.assertEqual(self.run_build(MOCK_BUILD_EXIT='29').returncode, 29)

    def test_missing_gate_fails_even_on_zero_exit(self):
        self.assertEqual(self.run_build(MOCK_MARKER='0').returncode, 4)

    def test_unverified_cleanup_fails(self):
        self.assertEqual(self.run_build(MOCK_CLEANUP_FAIL='1').returncode, 125)

    def test_no_automatic_probe_or_package_install(self):
        source = SCRIPT.read_text()
        self.assertNotIn('--device=', source)
        self.assertNotIn('apt-get', source)
        self.assertNotIn('tp2_rccl_transport', source)
        self.assertNotIn('run_tp2_rccl_probe', source)
        self.assertIn('96a25b5fd6f73fba58c7d83eb57cf19a50230aa4', source)
        self.assertIn('e69e5f977d458f2650bb346dadf2ad30c5320281', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
