#!/usr/bin/env python3
"""CPU-only tests for installed-dependency discovery and RCCL runner boundaries."""
from __future__ import annotations
import contextlib
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import pty
import shlex
import shutil
import subprocess
import tempfile
import termios
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_tp2_rccl_probe_mi50.sh"
SPEC = importlib.util.spec_from_file_location("rccl_preflight", ROOT / "tools/tp2_rccl_preflight.py")
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)
IMAGE = "sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca"
SUITE = "RCCL_PROBE_SUITE_PASS normal=pass delayed_rank=pass missing_rank=expected_watchdog_failure"


def executable(path: Path, contents: str):
    path.write_text(contents)
    path.chmod(0o755)


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "torch"
        (self.root / "include/rccl").mkdir(parents=True)
        (self.root / "lib").mkdir()
        self.header = self.root / "include/rccl/rccl.h"
        self.header.write_text("int ncclGetVersion(int*); int ncclSend(); int ncclRecv();\n")

    def fake_library(self, symbols=None, version=22703, status=0, noisy=False, extra_c=""):
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("A host C compiler is needed to build the CPU-only fake shared library")
        symbols = PREFLIGHT.REQUIRED_SYMBOLS if symbols is None else symbols
        code = "#include <stdio.h>\n"
        code += "\n".join(f"int {name}(void) {{ return 0; }}" for name in symbols if name != "ncclGetVersion")
        code += f"\nint ncclGetVersion(int *v) {{ *v = {version}; return {status}; }}\n"
        if noisy:
            code += r'''
__attribute__((constructor)) static void diagnostics_before(void) {
    fputs("native stdout before version: rocprofiler registration 22707", stdout);
    fflush(stdout);
    fputs("native stderr before version 22707", stderr);
    fflush(stderr);
    fputs("native stdout buffered until exit 22707\n", stdout);
}
__attribute__((destructor)) static void diagnostics_after(void) {
    fputs("native stdout after version 22707\n", stdout);
    fputs("native stderr after version 22707\n", stderr);
}
'''
        code += extra_c
        source = self.root / "stub.c"
        source.write_text(code)
        library = self.root / "lib/librccl.so.1"
        subprocess.run([compiler, "-shared", "-fPIC", str(source), "-o", str(library)], check=True, capture_output=True)
        (self.root / "lib/librccl.so").symlink_to(library.name)
        return library

    def discover(self):
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            result = PREFLIGHT.discover([self.root])
        self.stderr = err.getvalue()
        return result, out.getvalue()

    def test_existing_torch_style_library_and_header(self):
        library = self.fake_library()
        result, output = self.discover()
        self.assertEqual(result["RCCL_LIBRARY"], str(library.resolve()))
        self.assertEqual(result["RCCL_INCLUDE_DIR"], str(self.header.parent))
        self.assertEqual(result["RCCL_RUNTIME_VERSION"], 22703)
        self.assertIn("sha256=" + PREFLIGHT.sha256(library), output)
        self.assertIn("RCCL_PREFLIGHT_DEPENDENCIES_BEGIN", output)
        self.assertEqual(len(PREFLIGHT.unique_paths([library, library.parent / "librccl.so"])), 1)

    def test_missing_header_is_clear(self):
        self.header.unlink()
        with self.assertRaisesRegex(RuntimeError, "No installed RCCL C API header"):
            self.discover()

    def test_wrapper_header_is_not_c_api(self):
        self.header.write_text("namespace torch { void send(); }\n")
        with self.assertRaisesRegex(RuntimeError, "No installed RCCL C API header"):
            self.discover()

    def test_missing_library_is_clear(self):
        with self.assertRaisesRegex(RuntimeError, "No installed RCCL shared library"):
            self.discover()

    def test_missing_required_symbol_rejected(self):
        self.fake_library([name for name in PREFLIGHT.REQUIRED_SYMBOLS if name != "ncclSend"])
        with self.assertRaisesRegex(RuntimeError, "No loadable installed RCCL C API library"):
            self.discover()

    def test_bad_version_rejected(self):
        self.fake_library(version=0)
        with self.assertRaisesRegex(RuntimeError, "No loadable installed RCCL C API library"):
            self.discover()

    def test_native_diagnostics_before_and_after_version_are_retained(self):
        self.fake_library(version=22707, noisy=True)
        with mock.patch.dict(os.environ, NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="ALL", NCCL_DEBUG_FILE="/dev/stdout"):
            result, output = self.discover()
        self.assertEqual(result["RCCL_RUNTIME_VERSION"], 22707)
        marker = PREFLIGHT.CHECK_VERSION_MARKER + "=22707"
        self.assertLess(output.index("native stdout before version"), output.index(marker))
        self.assertLess(output.index(marker), output.index("native stdout buffered until exit"))
        self.assertLess(output.index(marker), output.index("native stdout after version"))
        for diagnostic in ("native stderr before version", "native stderr after version"):
            self.assertIn(diagnostic, self.stderr)

    def test_nonzero_symbol_status_rejects_valid_looking_native_record(self):
        self.fake_library(version=22707, status=3, noisy=True, extra_c=f'''
__attribute__((constructor)) static void misleading_record(void) {{
    puts("\\n{PREFLIGHT.CHECK_VERSION_MARKER}=22707");
    fflush(stdout);
}}
''')
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaisesRegex(RuntimeError, "ncclGetVersion status=3 version=22707"):
                PREFLIGHT.discover([self.root])
        self.assertIn(PREFLIGHT.CHECK_VERSION_MARKER + "=22707", out.getvalue())
        self.assertIn("native stdout before version", out.getvalue())
        self.assertIn("native stdout after version", out.getvalue())
        self.assertIn("native stderr before version", err.getvalue())
        self.assertIn("native stderr after version", err.getvalue())
        self.assertNotIn("RCCL_PREFLIGHT_VERSION value=", out.getvalue())

    def test_version_record_requires_one_strict_positive_integer(self):
        marker = PREFLIGHT.CHECK_VERSION_MARKER
        valid = marker + "=22707"
        rejected = (
            "", "22707\n", "rocprofiler registration 22707\n22707\n",
            valid + "\n" + valid, valid + valid,
            valid + "\n" + marker + "=bad", marker, marker + "=",
            marker + "=0", marker + "=-1", marker + "=+22707",
            marker + "=022707", marker + "=22707.0", marker + "=22707 trailing",
            "prefix " + valid, valid + " ", marker + "= 22707",
        )
        for output in rejected:
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    PREFLIGHT.parse_checked_version(output)
        self.assertEqual(PREFLIGHT.parse_checked_version("22703\n" + valid + "\n99999\n"), 22707)

    def test_stderr_version_record_cannot_replace_missing_stdout_record(self):
        self.fake_library()
        real_run = subprocess.run
        def run(command, **kwargs):
            if "--check-library" in command:
                return subprocess.CompletedProcess(command, 0, "22707\n",
                                                   PREFLIGHT.CHECK_VERSION_MARKER + "=22707\n")
            return real_run(command, **kwargs)
        with mock.patch.object(PREFLIGHT.subprocess, "run", side_effect=run):
            with self.assertRaisesRegex(RuntimeError, "Expected exactly one RCCL check version record"):
                self.discover()

    def test_helper_timeout_retains_both_output_streams(self):
        self.fake_library()
        real_run = subprocess.run
        def run(command, **kwargs):
            if "--check-library" in command:
                raise subprocess.TimeoutExpired(command, 20, b"native partial stdout", b"native partial stderr")
            return real_run(command, **kwargs)
        with mock.patch.object(PREFLIGHT.subprocess, "run", side_effect=run), \
                contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaisesRegex(RuntimeError, "No loadable installed RCCL C API library"):
                PREFLIGHT.discover([self.root])
        self.assertIn("native partial stdout", out.getvalue())
        self.assertIn("native partial stderr", err.getvalue())


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.env = dict(os.environ, PATH=str(self.bin) + ":" + os.environ["PATH"])
        self.repo = self.root / "repo"
        (self.repo / "tools").mkdir(parents=True)
        (self.repo / "tests/hip").mkdir(parents=True)
        shutil.copy2(RUNNER, self.repo / "tools" / RUNNER.name)
        (self.repo / "tests/hip/tp2_rccl_transport.cpp").write_text("// CPU test fixture only\n")
        (self.repo / "tools/tp2_rccl_preflight.py").write_text("# CPU test fixture only\n")
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=CPU Test", "-c", "user.email=cpu@example.invalid", "commit", "-qm", "fixture", "--allow-empty"], check=True)
        self.calls = self.root / "docker-calls.jsonl"
        self.env["MOCK_CALLS"] = str(self.calls)
        self.sudo_calls = self.root / "sudo-calls.jsonl"
        self.env["MOCK_SUDO_CALLS"] = str(self.sudo_calls)
        executable(self.bin / "id", '#!/bin/bash\necho "${MOCK_UID:-1000}"\n')
        executable(self.bin / "sudo", '''#!/usr/bin/env python3
import json, os, sys, time
args = sys.argv[1:]
with open(os.environ["MOCK_SUDO_CALLS"], "a") as f: f.write(json.dumps(args) + "\\n")
if args == ["-v"]:
 if os.environ.get("MOCK_TIMED_OPERATION"): sys.exit(93)
 if not all(os.isatty(fd) for fd in (0, 1, 2)): sys.exit(94)
 print("MOCK_AUTH_TERMINAL_ONLY", flush=True)
 time.sleep(float(os.environ.get("MOCK_AUTH_SLEEP", "0")))
 sys.exit(int(os.environ.get("MOCK_AUTH_EXIT", "0")))
if args == ["-n", "-v"]:
 sys.exit(int(os.environ.get("MOCK_AUTH_EXIT", "0")))
if args[:2] != ["-n", "docker"]: sys.exit(95)
if args[2] in os.environ.get("MOCK_EXPIRE_ON", "").split(","): sys.exit(1)
os.execvp(args[1], args[1:])
''')
        real_timeout = shutil.which("timeout")
        executable(self.bin / "timeout", f'''#!/bin/bash
export MOCK_TIMED_OPERATION=1
if [[ " $* " == *" docker image inspect "* && -n ${{MOCK_IMAGE_TIMEOUT:-}} ]]; then
  exit "$MOCK_IMAGE_TIMEOUT"
fi
exec {real_timeout!r} "$@"
''')
        executable(self.bin / "docker", f'''#!/usr/bin/env python3
import json, os, sys, time
with open(os.environ["MOCK_CALLS"], "a") as f: f.write(json.dumps(sys.argv[1:]) + "\\n")
args = sys.argv[1:]
if args[:2] == ["image", "inspect"]:
 print(os.environ.get("MOCK_IMAGE", {IMAGE!r}))
elif args[0] == "run":
 script = sys.stdin.read()
 open(os.environ["MOCK_CALLS"] + ".container", "w").write(script)
 print("RCCL_PROBE_CONTAINER_ENTER", flush=True)
 if os.environ.get("MOCK_PROCESS_BEGIN"): print("RCCL_PROBE_PROCESS_BEGIN scenario=normal bound_s=1", flush=True)
 if os.environ.get("MOCK_SLEEP"): time.sleep(5)
 if os.environ.get("MOCK_SUITE", "1") == "1": print({SUITE!r})
 sys.exit(int(os.environ.get("MOCK_RUN_EXIT", "0")))
elif args[0] == "ps" and os.environ.get("MOCK_CLEANUP_FAIL"):
 print("still-running")
''')

    def host(self, args=(), interactive=False, **env):
        command = ["bash", str(self.repo / "tools" / RUNNER.name), *args]
        kwargs = dict(env=dict(self.env, **env), text=True, capture_output=True, timeout=12)
        self.auth_output = ""
        if interactive:
            master, slave = pty.openpty()
            def terminal_session():
                os.setsid()
                fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
            try:
                process = subprocess.run(command, stdin=slave, preexec_fn=terminal_session, **kwargs)
                os.set_blocking(master, False)
                self.auth_output = os.read(master, 65536).decode()
            finally:
                os.close(slave)
                os.close(master)
        else:
            # Deterministically exercise the no-terminal path even from a shell.
            process = subprocess.run(command, start_new_session=True, **kwargs)
        logs = list(self.repo.glob("tp2-rccl-probe-*.log"))
        self.assertEqual(len(logs), 1)
        saved = logs[0].read_text()
        self.assertIn(f"TP2_RCCL_PROBE_EXIT={process.returncode} ", saved)
        return process, saved

    def test_host_success_and_restricted_flags(self):
        process, saved = self.host(["--tokens", "8", "--devices", "1,0"])
        self.assertEqual(process.returncode, 0, process.stderr + saved)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        run = next(call for call in calls if call[0] == "run")
        for option in ("--rm", "--read-only", "--cap-drop", "--network", "--tmpfs", "--device=/dev/kfd", "--device=/dev/dri"):
            self.assertIn(option, run)
        self.assertEqual(run[run.index("--network") + 1], "none")
        self.assertEqual(run[run.index("--pull") + 1], "never")
        self.assertIn(IMAGE, run)
        self.assertNotIn("no-new-privileges", " ".join(run))
        self.assertNotIn("/models", " ".join(run))
        self.assertEqual([run[i + 1] for i, value in enumerate(run) if value == "-e"], [
            "HOME=/tmp", "HIP_VISIBLE_DEVICES=0,1", "NCCL_DEBUG=INFO",
            "NCCL_DEBUG_SUBSYS=ALL", "NCCL_DEBUG_FILE=/dev/stdout",
        ])
        for forbidden in ("--privileged", "--ipc", "--security-opt"):
            self.assertNotIn(forbidden, run)
        self.assertEqual(sum(call[0] == "run" for call in calls), 1)
        subprocess.run(["bash", "-n", str(self.calls) + ".container"], check=True)

    def test_slow_interactive_auth_precedes_deadlines_and_log_capture(self):
        started = time.monotonic()
        process, saved = self.host(interactive=True, TP2_OUTER_TIMEOUT="1", MOCK_AUTH_SLEEP="1.2")
        self.assertGreaterEqual(time.monotonic() - started, 1.2)
        self.assertEqual(process.returncode, 0, process.stderr + saved)
        self.assertIn("MOCK_AUTH_TERMINAL_ONLY", self.auth_output)
        self.assertNotIn("MOCK_AUTH_TERMINAL_ONLY", saved + process.stdout + process.stderr)
        calls = [json.loads(line) for line in self.sudo_calls.read_text().splitlines()]
        self.assertEqual(calls[0], ["-v"])
        self.assertTrue(all(call[:2] == ["-n", "docker"] for call in calls[1:]), calls)
        self.assertEqual([call[2] for call in calls[1:]], ["image", "run", "rm", "ps"])

    def test_no_terminal_uses_only_noninteractive_authorization(self):
        process, saved = self.host()
        self.assertEqual(process.returncode, 0, process.stderr + saved)
        calls = [json.loads(line) for line in self.sudo_calls.read_text().splitlines()]
        self.assertEqual(calls[0], ["-n", "-v"])
        self.assertTrue(all(call[0] == "-n" for call in calls), calls)

    def test_no_terminal_and_no_authorization_stops_before_docker(self):
        process, saved = self.host(MOCK_AUTH_EXIT="1")
        self.assertEqual(process.returncode, 1)
        self.assertIn("stage=authorization code=1 container_attempted=0", saved)
        self.assertIn("Run this wrapper from a terminal", process.stderr)
        self.assertFalse(self.calls.exists())

    def test_interactive_authorization_failure_stops_before_docker(self):
        process, saved = self.host(interactive=True, MOCK_AUTH_EXIT="1")
        self.assertEqual(process.returncode, 1)
        self.assertIn("stage=authorization", saved)
        self.assertNotIn("MOCK_AUTH_TERMINAL_ONLY", saved)
        self.assertFalse(self.calls.exists())

    def test_root_does_not_invoke_sudo(self):
        (self.bin / "sudo").unlink()
        process, saved = self.host(MOCK_UID="0")
        self.assertEqual(process.returncode, 0, process.stderr + saved)
        self.assertFalse(self.sudo_calls.exists())

    def test_expired_authorization_never_reprompts_for_docker_or_cleanup(self):
        process, saved = self.host(MOCK_EXPIRE_ON="run,rm,ps")
        self.assertEqual(process.returncode, 1)
        self.assertIn("RCCL_CONTAINER_CLEANUP_UNVERIFIED", saved)
        calls = [json.loads(line) for line in self.sudo_calls.read_text().splitlines()]
        self.assertEqual([call[2] for call in calls[1:]], ["image", "run", "rm", "ps"])
        self.assertTrue(all(call[0] == "-n" for call in calls), calls)

    def test_image_inspection_timeout_and_kill_are_not_gpu_failures(self):
        for code in ("124", "137"):
            with self.subTest(code=code):
                # host() requires one saved log per invocation.
                for log in self.repo.glob("tp2-rccl-probe-*.log"):
                    log.unlink()
                process, saved = self.host(MOCK_IMAGE_TIMEOUT=code)
                self.assertEqual(process.returncode, int(code))
                self.assertIn(f"stage=image-inspect code={code} container_attempted=0", saved)
                self.assertIn("Host setup timeout or kill before container launch", saved)
                self.assertNotIn("inspect GPU state", saved)
                self.assertNotIn("RCCL_PROBE_CONTAINER_LAUNCH", saved)
                self.assertFalse(self.calls.exists())

    def test_host_nonzero_propagates(self):
        process, _ = self.host(MOCK_RUN_EXIT="29")
        self.assertEqual(process.returncode, 29)

    def test_host_zero_without_suite_fails(self):
        process, _ = self.host(MOCK_SUITE="0")
        self.assertEqual(process.returncode, 4)

    def test_host_outer_timeout(self):
        process, saved = self.host(TP2_OUTER_TIMEOUT="1", MOCK_SLEEP="1")
        self.assertEqual(process.returncode, 124)
        self.assertIn("verified_absent=", saved)
        self.assertIn("Container startup/setup timeout", saved)
        self.assertNotIn("inspect GPU state", saved)

    def test_host_probe_timeout_warns_about_gpu_state(self):
        process, saved = self.host(TP2_OUTER_TIMEOUT="1", MOCK_SLEEP="1", MOCK_PROCESS_BEGIN="1")
        self.assertEqual(process.returncode, 124)
        self.assertIn("after probe launch: inspect GPU state", saved)

    def test_host_cleanup_must_be_verified(self):
        process, _ = self.host(MOCK_CLEANUP_FAIL="1")
        self.assertEqual(process.returncode, 125)

    def test_bad_image_does_not_run(self):
        process, _ = self.host(MOCK_IMAGE="sha256:other")
        self.assertEqual(process.returncode, 2)
        self.assertNotIn('"run"', self.calls.read_text())

    def test_unknown_option_does_not_run(self):
        process, _ = self.host(["--scenario", "normal"])
        self.assertEqual(process.returncode, 2)
        self.assertFalse(self.calls.exists())
        self.assertFalse(self.sudo_calls.exists())

    def test_duplicate_option_does_not_run(self):
        process, _ = self.host(["--tokens", "1", "--tokens", "8"])
        self.assertEqual(process.returncode, 2)
        self.assertFalse(self.calls.exists())

    def test_bad_range_does_not_run(self):
        process, _ = self.host(["--iterations", "0"])
        self.assertEqual(process.returncode, 2)
        self.assertFalse(self.calls.exists())

    def test_watchdog_option_ranges(self):
        for flag in ("--timeout-ms", "--init-timeout-ms"):
            for value in ("100", "60000", "120000"):
                with self.subTest(flag=flag, value=value):
                    for log in self.repo.glob("tp2-rccl-probe-*.log"):
                        log.unlink()
                    process, saved = self.host([flag, value])
                    self.assertEqual(process.returncode, 0, process.stderr + saved)
                    calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
                    run = [call for call in calls if call[0] == "run"][-1]
                    self.assertEqual(run[-2:], [flag, value])

    def test_invalid_watchdog_options_stop_before_authorization(self):
        for flag in ("--timeout-ms", "--init-timeout-ms"):
            for value in ("0", "99", "120001", "00100", "-1", "1.5", "abc", "1000000"):
                with self.subTest(flag=flag, value=value):
                    for log in self.repo.glob("tp2-rccl-probe-*.log"):
                        log.unlink()
                    process, _ = self.host([flag, value])
                    self.assertEqual(process.returncode, 2)
                    self.assertFalse(self.calls.exists())
                    self.assertFalse(self.sudo_calls.exists())

    def test_duplicate_init_timeout_stops_before_authorization(self):
        process, _ = self.host(["--init-timeout-ms", "60000", "--init-timeout-ms", "60000"])
        self.assertEqual(process.returncode, 2)
        self.assertFalse(self.calls.exists())
        self.assertFalse(self.sudo_calls.exists())

    def inner(self, args=(), **env):
        work = self.root / "inner-work"
        tmp = self.root / "inner-tmp"
        tmp.mkdir(exist_ok=True)
        work.mkdir(exist_ok=True)
        source = RUNNER.read_text().split("<<'RCCL_CONTAINER'\n", 1)[1].split("\nRCCL_CONTAINER\n", 1)[0]
        source = source.replace("/tmp", str(tmp)).replace("/work", str(work))
        # Keep the test's mocks first; do not execute an available HIP compiler.
        source = source.replace("export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH", ":")
        (self.root / "inner.sh").write_text(source)
        executable(self.bin / "python3", f'''#!/bin/bash
cat > {str(tmp / 'rccl-env')!r} <<'ENV'
export RCCL_LIBRARY={str(tmp / 'librccl.so')!r}
export RCCL_INCLUDE_DIR={str(tmp)!r}
export RCCL_HEADER={str(tmp / "rccl.h")!r}
export LD_LIBRARY_PATH={str(tmp)!r}
ENV
''')
        (tmp / "librccl.so").write_text("CPU-only fixture")
        fixture = self.root / "mock-probe"
        executable(fixture, '''#!/bin/bash
printf 'MOCK_PROBE_COMMAND'; printf ' %q' "$@"; printf '\\n'
full_suite=1
while (( $# )); do
 if [[ "$1" == --scenario ]]; then scenario=$2; fi
 if [[ "$1" == --tokens && "$2" != all ]] || [[ "$1" == --launch-order && "$2" != both ]]; then full_suite=0; fi
 shift
done
if [[ "$scenario" == normal && ${MOCK_NORMAL_EXIT:-0} != 0 ]]; then exit "$MOCK_NORMAL_EXIT"; fi
if [[ "$scenario" == delayed-rank && ${MOCK_DELAYED_EXIT:-0} != 0 ]]; then exit "$MOCK_DELAYED_EXIT"; fi
if [[ "$scenario" == missing-rank ]]; then
 [[ ${MOCK_ARMED:-1} == 1 ]] && echo 'FAULT_INJECTION_ARMED scenario=missing-rank launch_rank=0 omitted_or_delayed_rank=1'
 echo 'FAULT_GRAPH_LAUNCH_ENTER scenario=missing-rank rank=0'
 echo 'FAULT_PEER_OMITTED scenario=missing-rank rank=1'
 [[ ${MOCK_WATCHDOG_MARKER:-1} == 1 ]] && echo "WATCHDOG_TIMEOUT scenario=missing-rank stage=${MOCK_STAGE:-fault-launch} exit=70 teardown=cpu-test"
 [[ ${MOCK_FAULT_PASS:-0} == 1 ]] && echo 'RCCL_PREFLIGHT_PASS scenario=missing-rank'
 exit "${MOCK_MISSING_EXIT:-70}"
fi
[[ ${MOCK_EXACT_MARKER:-1} == 1 ]] && echo 'EXACT_GATE_PASS test=cpu'
[[ ${MOCK_LIFECYCLE_MARKER:-1} == 1 ]] && echo 'LIFECYCLE_PASS life=0 test=cpu'
[[ "$scenario" == delayed-rank && ${MOCK_DELAYED_MARKER:-1} == 1 ]] && echo 'DELAYED_RANK_PASS life=0 test=cpu timing_admitted=0'
[[ "$scenario" != normal ]] && full_suite=0
full_suite=${MOCK_FULL_SUITE:-$full_suite}
[[ ${MOCK_PASS_MARKER:-1} == 1 ]] && echo "RCCL_PREFLIGHT_PASS scenario=$scenario tokens=2 orders=2 lifecycles=2 full_suite=$full_suite layer_integration_admitted=0 timing_scope=transport_only"
exit 0
''')
        executable(self.bin / "hipcc", f'''#!/bin/bash
[[ "$1" == --version ]] && {{ echo 'CPU compiler fixture'; exit 0; }}
cp {str(fixture)!r} {str(tmp / 'tp2_rccl_transport')!r}
''')
        executable(self.bin / "ldd", "#!/bin/bash\necho 'fixture: all dependencies resolved'\n")
        real_tee = shutil.which("tee")
        executable(self.bin / "tee", f'''#!/bin/bash
{real_tee!r} "$@"
[[ "$1" == */normal.log && ${{MOCK_TEE_FAIL:-0}} == 1 ]] && exit 71
exit 0
''')
        process = subprocess.run(["bash", str(self.root / "inner.sh"), "5", "5", *args],
                                 env=dict(self.env, **env), text=True, capture_output=True, timeout=10)
        return process

    def test_inner_all_gates(self):
        process = self.inner()
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn(SUITE, process.stdout)

    def test_inner_init_budget_and_diagnostic_environment(self):
        process = self.inner(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="ALL", NCCL_DEBUG_FILE="/dev/stdout")
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        for value in ("NCCL_DEBUG=INFO", "NCCL_DEBUG_SUBSYS=ALL", "NCCL_DEBUG_FILE=/dev/stdout"):
            self.assertIn("RCCL_PROBE_DIAGNOSTIC_ENV " + value, process.stdout)
        self.assertIn("RCCL_PROBE_WATCHDOGS init_ms=60000 normal_command_ms=10000 fault_command_ms=10000 fault_outer_s=105", process.stdout)
        calls = [shlex.split(line)[1:] for line in process.stdout.splitlines() if line.startswith("MOCK_PROBE_COMMAND ")]
        self.assertEqual(len(calls), 3)
        self.assertEqual([call[call.index("--scenario") + 1] for call in calls], ["normal", "delayed-rank", "missing-rank"])
        for call in calls:
            self.assertEqual(call.count("--init-timeout-ms"), 1)
            self.assertEqual(call[call.index("--init-timeout-ms") + 1], "60000")
        for call in calls[1:]:
            self.assertEqual(call[call.index("--timeout-ms") + 1], "10000")
        for scenario in ("delayed-rank", "missing-rank"):
            self.assertIn(f"RCCL_PROBE_PROCESS_BEGIN scenario={scenario} bound_s=105", process.stdout)

    def test_inner_init_override_does_not_extend_fault_launch_watchdog(self):
        for value, bound in (("100", "46"), ("60001", "106"), ("120000", "165")):
            with self.subTest(value=value):
                process = self.inner(args=["--init-timeout-ms", value, "--timeout-ms", "120000"])
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertIn(f"init_ms={value} normal_command_ms=120000 fault_command_ms=10000 fault_outer_s={bound}", process.stdout)
                calls = [shlex.split(line)[1:] for line in process.stdout.splitlines() if line.startswith("MOCK_PROBE_COMMAND ")]
                self.assertEqual(len(calls), 3)
                for call in calls:
                    self.assertEqual(call.count("--init-timeout-ms"), 1)
                    self.assertEqual(call[call.index("--init-timeout-ms") + 1], value)
                self.assertEqual(calls[0][calls[0].index("--timeout-ms") + 1], "120000")
                for call in calls[1:]:
                    self.assertEqual(call[call.index("--timeout-ms") + 1], "10000")
                for scenario in ("delayed-rank", "missing-rank"):
                    self.assertIn(f"RCCL_PROBE_PROCESS_BEGIN scenario={scenario} bound_s={bound}", process.stdout)

    def test_inner_initialization_timeout_stops_after_one_attempt(self):
        process = self.inner(MOCK_NORMAL_EXIT="70")
        self.assertEqual(process.returncode, 70)
        self.assertEqual(process.stdout.count("MOCK_PROBE_COMMAND "), 1)
        self.assertNotIn("scenario=delayed-rank", process.stdout)
        self.assertNotIn("scenario=missing-rank", process.stdout)
        self.assertNotIn(SUITE, process.stdout)

    def test_inner_partial_scope_is_explicit(self):
        process = self.inner(args=["--tokens", "1"])
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn("full_suite=0", process.stdout)

    def test_inner_default_requires_full_suite(self):
        process = self.inner(MOCK_FULL_SUITE="0")
        self.assertEqual(process.returncode, 4)

    def test_inner_missing_exact_gate_fails(self):
        process = self.inner(MOCK_EXACT_MARKER="0")
        self.assertEqual(process.returncode, 4)

    def test_inner_missing_lifecycle_gate_fails(self):
        process = self.inner(MOCK_LIFECYCLE_MARKER="0")
        self.assertEqual(process.returncode, 4)

    def test_inner_missing_delayed_gate_fails(self):
        process = self.inner(MOCK_DELAYED_MARKER="0")
        self.assertEqual(process.returncode, 4)

    def test_inner_log_write_failure_fails(self):
        process = self.inner(MOCK_TEE_FAIL="1")
        self.assertEqual(process.returncode, 71)

    def test_inner_real_failure_beats_tee(self):
        process = self.inner(MOCK_NORMAL_EXIT="42")
        self.assertEqual(process.returncode, 42)
        self.assertNotIn(SUITE, process.stdout)

    def test_inner_missing_success_marker_fails(self):
        process = self.inner(MOCK_PASS_MARKER="0")
        self.assertEqual(process.returncode, 4)

    def test_inner_delayed_failure_fails(self):
        process = self.inner(MOCK_DELAYED_EXIT="43")
        self.assertEqual(process.returncode, 43)

    def test_inner_missing_rank_must_not_succeed(self):
        process = self.inner(MOCK_MISSING_EXIT="0")
        self.assertEqual(process.returncode, 4)

    def test_inner_missing_rank_wrong_error_fails(self):
        process = self.inner(MOCK_MISSING_EXIT="44")
        self.assertEqual(process.returncode, 44)

    def test_inner_missing_rank_requires_watchdog_marker(self):
        process = self.inner(MOCK_WATCHDOG_MARKER="0")
        self.assertEqual(process.returncode, 70)

    def test_inner_setup_timeout_does_not_count_as_fault_success(self):
        for stage in ("initialize", "warmup-exchange", "capture-and-instantiate", "fault-prepare"):
            with self.subTest(stage=stage):
                process = self.inner(MOCK_STAGE=stage)
                self.assertEqual(process.returncode, 70)
                self.assertNotIn("RCCL_PROBE_FAULT_GATE EXPECTED_FAILURE", process.stdout)
                self.assertNotIn(SUITE, process.stdout)

    def test_inner_fault_must_be_armed(self):
        process = self.inner(MOCK_ARMED="0")
        self.assertEqual(process.returncode, 70)

    def test_inner_fault_cannot_claim_pass(self):
        process = self.inner(MOCK_FAULT_PASS="1")
        self.assertEqual(process.returncode, 70)


if __name__ == "__main__":
    unittest.main(verbosity=2)
