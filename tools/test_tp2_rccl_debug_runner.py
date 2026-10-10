#!/usr/bin/env python3
"""CPU-only debug-runner admission/cleanup tests. Never invokes Docker or HIP."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'tools/run_tp2_rccl_debug_mi50.sh'
IMAGE = 'sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca'
WARMUP = 'RCCL_DEBUG_WARMUP_ONLY_COMPLETE capture_tested=0 full_suite=0'


def executable(path, text):
    path.write_text(text)
    path.chmod(0o755)


class DebugRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.env = dict(os.environ, PATH=str(self.bin) + ':' + os.environ['PATH'], MOCK_ROOT=str(self.root))
        executable(self.bin / 'id', '#!/bin/bash\necho "${MOCK_UID:-0}"\n')
        executable(self.bin / 'sudo', '''#!/bin/bash
printf '%q ' "$@" >> "$MOCK_ROOT/sudo.calls"; echo >> "$MOCK_ROOT/sudo.calls"
if [[ "$*" == '-n -v' ]]; then exit "${MOCK_AUTH_EXIT:-0}"; fi
[[ "$1" == -n ]] || exit 93
shift; exec "$@"
''')

    def host(self, **env):
        repo = self.root / 'repo'
        (repo / 'tools').mkdir(parents=True)
        (repo / 'tests/hip').mkdir(parents=True)
        shutil.copy2(RUNNER, repo / 'tools' / RUNNER.name)
        (repo / 'tests/hip/tp2_rccl_transport.cpp').write_text('fixture')
        (repo / 'tools/tp2_rccl_preflight.py').write_text('fixture')
        executable(self.bin / 'git', '#!/bin/bash\necho fixture-head\n')
        executable(self.bin / 'docker', f'''#!/bin/bash
printf '%q ' "$@" >> "$MOCK_ROOT/docker.calls"; echo >> "$MOCK_ROOT/docker.calls"
case "$1" in
 image) echo "${{MOCK_IMAGE:-{IMAGE}}}";;
 run) cat > "$MOCK_ROOT/container.sh"; echo fixture
      [[ ${{MOCK_RESULT:-1}} == 1 ]] && echo 'RCCL_DEBUG_RESULT first_sigsegv_captured=1 probe_pass=0 rerun=not_performed'
      exit "${{MOCK_EXIT:-0}}";;
 ps) [[ ${{MOCK_CLEANUP_FAIL:-0}} == 1 ]] && echo still-running; exit 0;;
 rm) exit 0;;
esac
''')
        result = subprocess.run(['bash', str(repo / 'tools' / RUNNER.name)],
                                env=dict(self.env, **env), capture_output=True, text=True,
                                timeout=10, start_new_session=True)
        calls = self.root / 'docker.calls'
        return result, [shlex.split(line) for line in calls.read_text().splitlines()] if calls.exists() else []

    def inner(self, **env):
        tmp = self.root / 'tmp'
        tmp.mkdir()
        source = RUNNER.read_text().split("<<'CONTAINER'\n", 1)[1].split('\nCONTAINER\n', 1)[0]
        source = source.replace('/tmp', str(tmp)).replace('export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH', ':')
        (self.root / 'inner.sh').write_text(source)
        executable(self.bin / 'python3', f'''#!/bin/bash
printf 'RCCL_LIBRARY=%q\\nRCCL_INCLUDE_DIR=%q\\nRCCL_HEADER=rccl.h\\n' {str(tmp / 'librccl.so')!r} {str(tmp)!r} > {str(tmp / 'rccl-env')!r}
''')
        (tmp / 'librccl.so').write_text('fixture')
        (tmp / 'libamdhip64.so.7').write_text('fixture')
        executable(self.bin / 'hipcc', f'''#!/bin/bash
printf '%q ' "$@" > "$MOCK_ROOT/compiler.args"
echo fixture > {str(tmp / 'tp2_rccl_debug')!r}
''')
        executable(self.bin / 'readlink', '#!/bin/bash\necho "$MOCK_ROOT/tmp/libamdhip64.so.7"\n')
        executable(self.bin / 'readelf', '#!/bin/bash\necho " Build ID: fixture"\n')
        executable(self.bin / 'rocgdb', f'''#!/bin/bash
[[ "$1" == --version ]] && {{ echo fixture-gdb; exit 0; }}
if [[ " $* " != *' -x '* ]]; then exit "${{MOCK_PTRACE_EXIT:-0}}"; fi
printf '%q ' "$@" > "$MOCK_ROOT/debugger.args"
if read -r input; then echo 'ERROR: debugger inherited script stdin'; exit 94; fi
case "${{MOCK_MODE:-capture}}" in
 capture) echo RCCL_DEBUG_SIGSEGV_BEGIN; echo RCCL_DEBUG_SIGSEGV_END; exit 86;;
 partial) echo RCCL_DEBUG_SIGSEGV_BEGIN; echo RCCL_DEBUG_SIGSEGV_END; echo RCCL_DEBUG_CAPTURE_PARTIAL; exit 87;;
 no-crash) echo {WARMUP!r}; exit 0;;
 no-marker) exit 0;;
 failed) exit 124;;
esac
''')
        return subprocess.run(['bash', str(self.root / 'inner.sh')], env=dict(self.env, **env),
                              capture_output=True, text=True, timeout=10, input='must not reach debugger\n')

    def test_host_container_scope(self):
        result, calls = self.host()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        launch = next(c for c in calls if c[0] == 'run')
        self.assertEqual(sum(c[0] == 'run' for c in calls), 1)
        self.assertIn('--cap-drop', launch)
        self.assertEqual(launch[launch.index('--cap-add') + 1], 'SYS_PTRACE')
        self.assertEqual(launch[launch.index('--network') + 1], 'none')
        self.assertEqual(launch[launch.index('--pull') + 1], 'never')
        for denied in ('--privileged', '--security-opt', '--pid', '--ipc'):
            self.assertNotIn(denied, launch)
        self.assertIn('verified_absent=1', result.stdout)
        self.assertEqual([c[0] for c in calls], ['image', 'run', 'rm', 'ps'])
        subprocess.run(['bash', '-n', str(self.root / 'container.sh')], check=True)

    def test_host_zero_without_result_is_failure(self):
        result, _ = self.host(MOCK_RESULT='0')
        self.assertEqual(result.returncode, 4)
        self.assertIn('RCCL_DEBUG_ADMISSION_FAIL', result.stdout)

    def test_footer_write_failure_is_not_success(self):
        real_tee = shutil.which('tee')
        executable(self.bin / 'tee', f'#!/bin/bash\n{real_tee!r} \"$@\"\n[[ \"$1\" == -a ]] && exit 71\nexit 0\n')
        result, _ = self.host()
        self.assertEqual(result.returncode, 71)

    def test_cleanup_failure_is_not_success(self):
        result, _ = self.host(MOCK_CLEANUP_FAIL='1')
        self.assertEqual(result.returncode, 125)

    def test_timeout_preserved_and_cleanup_attempted(self):
        result, calls = self.host(MOCK_EXIT='124')
        self.assertEqual(result.returncode, 124)
        self.assertEqual([c[0] for c in calls][-2:], ['rm', 'ps'])

    def test_denied_authorization_never_launches(self):
        result, calls = self.host(MOCK_UID='1000', MOCK_AUTH_EXIT='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [])

    def test_no_terminal_uses_noninteractive_sudo(self):
        result, _ = self.host(MOCK_UID='1000')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [shlex.split(line) for line in (self.root / 'sudo.calls').read_text().splitlines()]
        self.assertTrue(all(c[0] == '-n' for c in calls))

    def test_capture_first_sigsegv_and_original_warmup(self):
        result = self.inner()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('first_sigsegv_captured=1', result.stdout)
        args = shlex.split((self.root / 'debugger.args').read_text())
        for flag, value in (('--tokens', '8'), ('--warmup', '1'), ('--scenario', 'normal')):
            self.assertEqual(args[args.index(flag) + 1], value)
        compile_args = shlex.split((self.root / 'compiler.args').read_text())
        self.assertIn('-DSTRATA_RCCL_DEBUG_WARMUP_ONLY', compile_args)
        self.assertEqual(compile_args[compile_args.index('-Xarch_host') + 1], '-g')
        gdb = (self.root / 'tmp/capture.gdb').read_text()
        for text in ('set non-stop off', 'catch signal SIGSEGV', 'x/9bx $rip', 'x/8gx $rsp',
                     'info proc mappings', 'thread apply all -c bt 24', 'quit 86'):
            self.assertIn(text, gdb)
        self.assertNotIn('continue', gdb)

    def test_partial_capture_is_marked_and_nonzero(self):
        result = self.inner(MOCK_MODE='partial')
        self.assertEqual(result.returncode, 87)
        self.assertIn('capture_partial=1', result.stdout)

    def test_completed_warmup_is_not_probe_pass(self):
        result = self.inner(MOCK_MODE='no-crash')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('warmup_finished=1 probe_pass=0', result.stdout)

    def test_missing_completion_marker_is_failure(self):
        self.assertEqual(self.inner(MOCK_MODE='no-marker').returncode, 4)

    def test_debugger_failure_preserved(self):
        self.assertEqual(self.inner(MOCK_MODE='failed').returncode, 124)

    def test_ptrace_failure_prevents_gpu_build_and_launch(self):
        self.assertEqual(self.inner(MOCK_PTRACE_EXIT='1').returncode, 1)
        self.assertFalse((self.root / 'compiler.args').exists())
        self.assertFalse((self.root / 'debugger.args').exists())

    def test_capture_python_marks_errors_and_keeps_collecting(self):
        block = RUNNER.read_text().split("<<'GDB'\n", 1)[1].split('\nGDB\n', 1)[0]
        python = block.split('\npython\n', 1)[1].split('\nend\n', 1)[0]
        for fail in (False, True):
            calls, output = [], []
            def execute(command):
                calls.append(command)
                if fail and command == 'info registers rip rdi rsp rbp rax':
                    raise RuntimeError('fixture register read failure')
            gdb = SimpleNamespace(write=output.append, execute=execute,
                                  selected_thread=lambda: SimpleNamespace(ptid=(123, 456, 0)))
            with mock.patch.dict('sys.modules', gdb=gdb):
                exec(compile(python, 'capture.gdb', 'exec'), {})
            self.assertIn('thread apply all -c bt 24', calls)
            self.assertEqual(calls[-1], 'set $capture_incomplete = ' + str(int(fail)))
            self.assertEqual(any('CAPTURE_ERROR' in line for line in output), fail)

    def test_source_exit_precedes_capture_and_preserves_sequence(self):
        source = (ROOT / 'tests/hip/tp2_rccl_transport.cpp').read_text()
        self.assertIn('enqueue(r, rank, 8, 0, 3);', source)
        guard = source.split('#ifdef STRATA_RCCL_DEBUG_WARMUP_ONLY\n', 1)[1].split('#endif', 1)[0]
        self.assertIn('std::_Exit(0)', guard)
        self.assertLess(source.index('#ifdef STRATA_RCCL_DEBUG_WARMUP_ONLY'),
                        source.index('workers.run("capture-and-instantiate"'))
        subprocess.run(['bash', '-n', str(RUNNER)], check=True)


if __name__ == '__main__':
    unittest.main()
