#!/usr/bin/env python3
"""CPU-only tests for installed-dependency discovery and RCCL runner boundaries."""
from __future__ import annotations
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

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

    def fake_library(self, symbols=None, version=22703):
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("A host C compiler is needed to build the CPU-only fake shared library")
        symbols = PREFLIGHT.REQUIRED_SYMBOLS if symbols is None else symbols
        code = "\n".join(f"int {name}(void) {{ return 0; }}" for name in symbols if name != "ncclGetVersion")
        code += f"\nint ncclGetVersion(int *v) {{ *v = {version}; return 0; }}\n"
        source = self.root / "stub.c"
        source.write_text(code)
        library = self.root / "lib/librccl.so.1"
        subprocess.run([compiler, "-shared", "-fPIC", str(source), "-o", str(library)], check=True, capture_output=True)
        (self.root / "lib/librccl.so").symlink_to(library.name)
        return library

    def discover(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = PREFLIGHT.discover([self.root])
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
        executable(self.bin / "sudo", '#!/bin/bash\nexec "$@"\n')
        executable(self.bin / "docker", f'''#!/usr/bin/env python3
import json, os, sys, time
with open(os.environ["MOCK_CALLS"], "a") as f: f.write(json.dumps(sys.argv[1:]) + "\\n")
args = sys.argv[1:]
if args[:2] == ["image", "inspect"]:
 print(os.environ.get("MOCK_IMAGE", {IMAGE!r}))
elif args[0] == "run":
 script = sys.stdin.read()
 open(os.environ["MOCK_CALLS"] + ".container", "w").write(script)
 if os.environ.get("MOCK_SLEEP"): time.sleep(5)
 if os.environ.get("MOCK_SUITE", "1") == "1": print({SUITE!r})
 sys.exit(int(os.environ.get("MOCK_RUN_EXIT", "0")))
elif args[0] == "ps" and os.environ.get("MOCK_CLEANUP_FAIL"):
 print("still-running")
''')

    def host(self, args=(), **env):
        process = subprocess.run(["bash", str(self.repo / "tools" / RUNNER.name), *args],
                                 env=dict(self.env, **env), text=True, capture_output=True, timeout=12)
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
        subprocess.run(["bash", "-n", str(self.calls) + ".container"], check=True)

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

    def test_duplicate_option_does_not_run(self):
        process, _ = self.host(["--tokens", "1", "--tokens", "8"])
        self.assertEqual(process.returncode, 2)
        self.assertFalse(self.calls.exists())

    def test_bad_range_does_not_run(self):
        process, _ = self.host(["--iterations", "0"])
        self.assertEqual(process.returncode, 2)
        self.assertFalse(self.calls.exists())

    def inner(self, args=(), **env):
        work = self.root / "inner-work"
        tmp = self.root / "inner-tmp"
        tmp.mkdir()
        work.mkdir()
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
        process = self.inner(MOCK_STAGE="initialize")
        self.assertEqual(process.returncode, 70)

    def test_inner_fault_must_be_armed(self):
        process = self.inner(MOCK_ARMED="0")
        self.assertEqual(process.returncode, 70)

    def test_inner_fault_cannot_claim_pass(self):
        process = self.inner(MOCK_FAULT_PASS="1")
        self.assertEqual(process.returncode, 70)


if __name__ == "__main__":
    unittest.main(verbosity=2)
