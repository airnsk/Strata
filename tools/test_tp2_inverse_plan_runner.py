#!/usr/bin/env python3
"""CPU-only runner boundary tests. Docker, HIP and layer execution are mocked."""
from __future__ import annotations

import fcntl
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
RUNNER = ROOT / "tools/run_tp2_inverse_plan_mi50.sh"
IMAGE = "sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca"
SUITE = "INVERSE_PLAN_SUITE_PASS"
GATES = "INVERSE_PLAN_LAYER_GATES_PASS"


def executable(path, contents):
    path.write_text(contents)
    path.chmod(0o755)


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.repo = self.root / "repo"
        self.models = self.root / "models"
        (self.models / "pack").mkdir(parents=True)
        (self.models / "shard.gguf").write_text("CPU-only fixture")
        (self.models / "second.gguf").write_text("CPU-only fixture")
        for name in ("tools", "tests/hip", "deps/llama.cpp/ggml"):
            (self.repo / name).mkdir(parents=True)
        shutil.copy2(RUNNER, self.repo / "tools" / RUNNER.name)
        for name in ("CMakeLists.txt", "deps/llama.cpp/ggml/CMakeLists.txt", "tests/hip/tp2_gdn_layer.cpp",
                     "src/kernels/cuda/fused_gr.cu", "include/strata/kernels/fused_gr.hpp",
                     "src/core/tp_gdn_layer.cpp", "include/strata/core/tp_gdn_layer.hpp",
                     "src/kernels/cuda/iq_kernels.cu", "src/kernels/cuda/verify_kernels.cu",
                     "src/kernels/cuda/native_down_plan.cuh",
                     "include/strata/kernels/iq_kernels.hpp", "include/strata/kernels/native_down_plan.hpp",
                     "include/strata/kernels/verify_kernels.hpp", "tests/hip/native_down_plan_parity.cpp",
                     "tests/core/native_down_plan_test.cpp",
                     "src/kernels/cuda/tp_gdn_exchange.cu", "include/strata/kernels/tp_gdn_exchange.hpp",
                     "include/strata/kernels/tp_gdn_gather_layout.hpp", "tests/core/tp_gdn_gather_layout_test.cpp"):
            (self.repo / name).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / name).write_text("CPU-only fixture\n")
        shutil.copy2(ROOT / "tools/mi50_pipeline.py", self.repo / "tools/mi50_pipeline.py")
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=CPU Test", "-c",
                        "user.email=cpu@example.invalid", "commit", "-qm", "fixture", "--allow-empty"], check=True)
        self.calls = self.root / "docker.jsonl"
        self.sudo_calls = self.root / "sudo.jsonl"
        self.timeout_calls = self.root / "timeout.log"
        self.env = dict(os.environ, PATH=str(self.bin) + ":" + os.environ["PATH"],
                        MOCK_CALLS=str(self.calls),
                        MOCK_SUDO_CALLS=str(self.sudo_calls), MOCK_TIMEOUT_CALLS=str(self.timeout_calls))
        for key in ("TP2_TIMEOUT", "TP2_BUILD_TIMEOUT", "TP2_OUTER_TIMEOUT", "BUILD_JOBS"):
            self.env.pop(key, None)
        self.inputs = [str(self.models), "--pack", "/models/pack", "--gguf", "/models/shard.gguf"]
        executable(self.bin / "id", '''#!/bin/bash
case "$1" in
 -u) echo "${MOCK_UID:-1000}" ;;
 -g) echo "${MOCK_GID:-1000}" ;;
 -G) echo "${MOCK_GROUPS:-1000 27 44}" ;;
 *) exit 2 ;;
esac
''')
        executable(self.bin / "stat", '''#!/bin/bash
[[ ${MOCK_STAT_EXIT:-0} == 0 ]] || exit "$MOCK_STAT_EXIT"
printf '%s\\n' "${MOCK_DEVICE_GROUPS-109 44 0}"
''')
        executable(self.bin / "sudo", f'''#!{sys.executable}
import json, os, pathlib, sys, time
args = sys.argv[1:]
state = pathlib.Path(os.environ["MOCK_SUDO_CALLS"] + ".refreshed")
calls_file = pathlib.Path(os.environ["MOCK_CALLS"])
started = calls_file.exists() and any(json.loads(line)[0] == "run" for line in calls_file.read_text().splitlines())
expired = os.environ.get("MOCK_EXPIRE_AUTH") and started and not state.exists()
with open(os.environ["MOCK_SUDO_CALLS"], "a") as f: f.write(json.dumps(args) + "\\n")
if args == ["-v"]:
    if os.environ.get("MOCK_TIMED_OPERATION") or not all(os.isatty(fd) for fd in (0,1,2)): sys.exit(93)
    print("AUTH_TERMINAL_ONLY", flush=True)
    if expired: state.touch()
    time.sleep(float(os.environ.get("MOCK_AUTH_SLEEP", "0")))
    sys.exit(int(os.environ.get("MOCK_AUTH_EXIT", "0")))
if args == ["-n", "-v"]: sys.exit(1 if expired else int(os.environ.get("MOCK_AUTH_EXIT", "0")))
if args[:2] != ["-n", "docker"]: sys.exit(95)
if expired: sys.exit(1)
if args[2] in os.environ.get("MOCK_EXPIRE_ON", "").split(","): sys.exit(1)
os.execvp(args[1], args[1:])
''')
        self.real_timeout = shutil.which("timeout")
        executable(self.bin / "timeout", f'''#!/bin/bash
export MOCK_TIMED_OPERATION=1
printf '%s\\n' "$*" >> "$MOCK_TIMEOUT_CALLS"
if [[ " $* " == *" docker run "* && -n ${{MOCK_SLEEP:-}} ]]; then exec {shlex.quote(self.real_timeout)} 1s "${{@:4}}"; fi
if [[ " $* " == *" docker image inspect "* && -n ${{MOCK_IMAGE_TIMEOUT:-}} ]]; then exit "$MOCK_IMAGE_TIMEOUT"; fi
exec {shlex.quote(self.real_timeout)} "$@"
''')
        executable(self.bin / "docker", f'''#!{sys.executable}
import json, os, pathlib, sys, time
args = sys.argv[1:]
with open(os.environ["MOCK_CALLS"], "a") as f: f.write(json.dumps(args) + "\\n")
if args[:2] == ["image", "inspect"]:
    print(os.environ.get("MOCK_IMAGE", {IMAGE!r}))
elif args[0] == "run":
    pathlib.Path(os.environ["MOCK_CALLS"] + ".container").write_text(sys.stdin.read())
    print("INVERSE_PLAN_CONTAINER_ENTER", flush=True)
    if os.environ.get("MOCK_PROCESS_BEGIN"): print("INVERSE_PLAN_PROCESS_BEGIN phase=benchmark", flush=True)
    if os.environ.get("MOCK_SLEEP"): time.sleep(5)
    if os.environ.get("MOCK_SUITE", "1") == "1": print({GATES!r})
    sys.exit(int(os.environ.get("MOCK_RUN_EXIT", "0")))
elif args[0] == "ps" and os.environ.get("MOCK_CLEANUP_FAIL"):
    print("still-running")
''')

    def host(self, args=(), inputs=None, interactive=False, **env):
        for log in self.repo.glob("tp2-inverse-plan-*.log"):
            log.unlink()
        self.calls.unlink(missing_ok=True)
        self.sudo_calls.unlink(missing_ok=True)
        command = ["bash", str(self.repo / "tools" / RUNNER.name),
                   *(self.inputs if inputs is None else inputs), *args]
        kwargs = dict(env=dict(self.env, **env), text=True, capture_output=True, timeout=15)
        self.auth_output = ""
        if interactive:
            master, slave = pty.openpty()
            def terminal_session():
                os.setsid()
                fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
            try:
                result = subprocess.run(command, stdin=slave, preexec_fn=terminal_session, **kwargs)
                os.set_blocking(master, False)
                self.auth_output = os.read(master, 65536).decode()
            finally:
                os.close(slave)
                os.close(master)
        else:
            result = subprocess.run(command, start_new_session=True, **kwargs)
        logs = list(self.repo.glob("tp2-inverse-plan-*.log"))
        self.assertEqual(len(logs), 1)
        saved = logs[0].read_text()
        self.assertIn(f"TP2_INVERSE_PLAN_EXIT={result.returncode} ", saved)
        return result, saved

    def test_host_pinned_image_read_only_mounts_and_identity(self):
        result, saved = self.host(["--gguf", "/models/second.gguf", "--layer", "4", "--mode", "7"],
                                  MOCK_UID=str(os.getuid()), MOCK_GID=str(os.getgid()))
        self.assertEqual(result.returncode, 0, saved + result.stderr)
        run = next(call for call in records(self.calls) if call[0] == "run")
        self.assertEqual(sum(call[0] == "run" for call in records(self.calls)), 1)
        for option in ("--rm", "--read-only", "--device=/dev/kfd", "--device=/dev/dri"):
            self.assertIn(option, run)
        for flag, value in (("--pull", "never"), ("--network", "none"), ("--cap-drop", "ALL"),
                            ("--user", f"{os.getuid()}:{os.getgid()}")):
            self.assertEqual(run[run.index(flag) + 1], value)
        for flag in ("--privileged", "--cap-add", "--ipc", "--security-opt"):
            self.assertNotIn(flag, run)
        self.assertIn(IMAGE, run)
        self.assertEqual(run[run.index("--entrypoint")+1], "/usr/bin/timeout")
        self.assertEqual(run[run.index(IMAGE)+1:run.index(IMAGE)+5],
                         ["--signal=TERM", "--kill-after=15s", "2400s", "/bin/bash"])
        self.assertIn("--signal=TERM --kill-after=15s 2430s", self.timeout_calls.read_text())
        self.assertFalse(any("RCCL" in arg or "NCCL" in arg for arg in run))
        self.assertIn(f"type=bind,src={self.repo},dst=/work,readonly", run)
        self.assertIn(f"type=bind,src={self.models},dst=/models,readonly", run)
        self.assertTrue(all(run[i+1].endswith(",readonly") for i, v in enumerate(run) if v == "--mount"))
        groups = [run[i+1] for i, v in enumerate(run) if v == "--group-add"]
        self.assertEqual(len(groups), len(set(groups)))
        self.assertTrue({"44", "109", "27"}.issubset(groups))
        self.assertIn("verified_absent=", saved)
        subprocess.run(["bash", "-n", str(self.calls) + ".container"], check=True)

    def test_host_defaults_and_diagnostic_environment(self):
        result, saved = self.host()
        self.assertEqual(result.returncode, 0, saved)
        run = next(call for call in records(self.calls) if call[0] == "run")
        for value in ("HIP_VISIBLE_DEVICES=0,1", "STRATA_HC_PERSIST=0", "STRATA_GR_V3=0", "STRATA_GR_SPLIT=0"):
            self.assertIn(value, run)
        self.assertIn("runtime_s=1200 build_s=600 outer_s=2400", saved)
        self.assertEqual(run[run.index("-s")+2:][:3], ["1200", "600", "48"])

    def test_host_validation_rejects_before_authorization(self):
        cases = [(["--execution", "hybrid"], {}), (["--benchmark", "1"], {}),
                 (["--layer", "1"], {}), (["--layer", "47"], {}), (["--layer", "08"], {}),
                 (["--mode", "9"], {}), (["--rccl-library", "/other"], {}),
                 (["--hc-dispatch", "staged"], {}),
                 (["--hc-dispatch", "legacy", "--hc-dispatch", "production-check"], {}),
                 (["--bench-trials", "1"], {}), (["--bench-block-calls", "65"], {}),
                 (["--bench-warmup", "101"], {}), (["--bench-block-trials", "21"], {}),
                 (["--mode", "8", "--mode", "8"], {}), (["--layer"], {}),
                 ([], {"TP2_TIMEOUT": "0"}),
                 ([], {"TP2_BUILD_TIMEOUT": "86401"}), ([], {"TP2_OUTER_TIMEOUT": "01"}),
                 ([], {"BUILD_JOBS": "257"})]
        for args, env in cases:
            with self.subTest(args=args, env=env):
                result, saved = self.host(args, **env)
                self.assertEqual(result.returncode, 2, saved)
                self.assertFalse(self.calls.exists())
                self.assertFalse(self.sudo_calls.exists())

    def test_host_forwards_explicit_hc_policy_only(self):
        for policy in (None, "legacy", "production-check"):
            with self.subTest(policy=policy):
                result, saved = self.host([] if policy is None else ["--hc-dispatch", policy])
                self.assertEqual(result.returncode, 0, saved)
                run = next(call for call in records(self.calls) if call[0] == "run")
                if policy is None:
                    self.assertNotIn("--hc-dispatch", run)
                else:
                    self.assertEqual(run[run.index("--hc-dispatch")+1], policy)

    def test_host_rejects_invalid_model_inputs(self):
        cases = [[], [str(self.models)], ["relative", *self.inputs[1:]],
                 ["/", *self.inputs[1:]], [str(self.models), "--pack", "/models/shard.gguf", "--gguf", "/models/shard.gguf"],
                 [str(self.models), "--pack", "/models/pack", "--gguf", "/models/../shard.gguf"],
                 [str(self.models), "--pack", "/models/pack", "--gguf", "/other/shard.gguf"],
                 [str(self.models), "--pack", "/models/pack", "--gguf", "/models/missing"]]
        for inputs in cases:
            with self.subTest(inputs=inputs):
                result, saved = self.host(inputs=inputs)
                self.assertEqual(result.returncode, 2, saved)
                self.assertFalse(self.sudo_calls.exists())
                self.assertFalse(self.calls.exists())

    def test_host_preserves_logical_and_physical_root_aliases(self):
        logical = self.root / "logical-models"
        logical.symlink_to(self.models, target_is_directory=True)
        (self.models / "alias.gguf").symlink_to(logical / "shard.gguf")
        result, saved = self.host(inputs=[str(logical), "--pack", "/models/pack", "--gguf", "/models/alias.gguf"])
        self.assertEqual(result.returncode, 0, saved + result.stderr)
        run = next(call for call in records(self.calls) if call[0] == "run")
        for path in (logical, self.models):
            self.assertIn(f"type=bind,src={self.models},dst={path},readonly", run)

    def test_host_rejects_model_symlink_escape(self):
        outside = self.root / "outside.gguf"
        outside.write_text("CPU fixture")
        (self.models / "escape.gguf").symlink_to(outside)
        result, saved = self.host(inputs=[str(self.models), "--pack", "/models/pack", "--gguf", "/models/escape.gguf"])
        self.assertEqual(result.returncode, 2, saved)
        self.assertIn("escapes MODEL_ROOT", saved)
        self.assertFalse(self.sudo_calls.exists())

    def test_host_rejects_unavailable_gpu_identity_before_run(self):
        for env in ({"MOCK_STAT_EXIT": "1"}, {"MOCK_DEVICE_GROUPS": "render"},
                    {"MOCK_DEVICE_GROUPS": ""}, {"MOCK_GROUPS": "1000 invalid"}):
            with self.subTest(env=env):
                result, saved = self.host(**env)
                self.assertEqual(result.returncode, 2, saved)
                self.assertIn("stage=device-identity container_attempted=0", saved)
                self.assertFalse(any(call[0] == "run" for call in records(self.calls)))

    def test_auth_is_noninteractive_without_tty(self):
        result, saved = self.host()
        self.assertEqual(result.returncode, 0, saved)
        calls = records(self.sudo_calls)
        self.assertEqual(calls[0], ["-n", "-v"])
        self.assertTrue(all(call[0] == "-n" for call in calls))

    def test_auth_interactive_prompt_is_not_logged_or_timed(self):
        result, saved = self.host(interactive=True, MOCK_AUTH_SLEEP="1.1", TP2_OUTER_TIMEOUT="1")
        self.assertEqual(result.returncode, 0, saved)
        self.assertIn("AUTH_TERMINAL_ONLY", self.auth_output)
        self.assertNotIn("AUTH_TERMINAL_ONLY", saved + result.stdout + result.stderr)
        calls = records(self.sudo_calls)
        self.assertEqual(calls[0], ["-v"])
        self.assertTrue(all(call[:2] == ["-n", "docker"] or call == ["-n", "-v"] for call in calls[1:]))

    def test_auth_failure_stops_before_docker(self):
        result, saved = self.host(MOCK_AUTH_EXIT="1")
        self.assertEqual(result.returncode, 1, saved)
        self.assertIn("stage=authorization", saved)
        self.assertFalse(self.calls.exists())

    def test_root_never_uses_sudo(self):
        (self.bin / "sudo").unlink()
        result, saved = self.host(MOCK_UID="0")
        self.assertEqual(result.returncode, 0, saved)
        self.assertFalse(self.sudo_calls.exists())

    def test_expired_authorization_never_reprompts(self):
        result, saved = self.host(MOCK_EXPIRE_ON="run,rm,ps")
        self.assertEqual(result.returncode, 1, saved)
        self.assertIn("CLEANUP_UNVERIFIED", saved)
        self.assertTrue(all(call[0] == "-n" for call in records(self.sudo_calls)))

    def test_cleanup_expired_auth_can_reauthorize_only_on_tty(self):
        result, saved = self.host(interactive=True, MOCK_EXPIRE_AUTH="1")
        self.assertEqual(result.returncode, 0, saved + result.stderr)
        self.assertEqual(self.auth_output.count("AUTH_TERMINAL_ONLY"), 2)
        self.assertNotIn("AUTH_TERMINAL_ONLY", saved + result.stdout + result.stderr)
        self.assertIn(SUITE, saved)

    def test_cleanup_expired_auth_without_tty_fails_closed(self):
        result, saved = self.host(MOCK_EXPIRE_AUTH="1")
        self.assertEqual(result.returncode, 125, saved)
        self.assertIn("CLEANUP_UNVERIFIED", saved)
        self.assertNotIn(SUITE, saved)
        self.assertTrue(all(call[0] == "-n" for call in records(self.sudo_calls)))

    def test_image_mismatch_or_timeout_cannot_launch(self):
        for env, code in (({"MOCK_IMAGE": "other"}, 2), ({"MOCK_IMAGE_TIMEOUT": "124"}, 124)):
            with self.subTest(env=env):
                result, saved = self.host(**env)
                self.assertEqual(result.returncode, code, saved)
                self.assertNotIn("INVERSE_PLAN_CONTAINER_LAUNCH", saved)
                self.assertFalse(any(call[0] == "run" for call in records(self.calls)))

    def test_host_requires_suite_and_zero_exit(self):
        for env, code in (({"MOCK_SUITE": "0"}, 4), ({"MOCK_RUN_EXIT": "29"}, 29),
                          ({"MOCK_RUN_EXIT": "77"}, 77)):
            with self.subTest(env=env):
                result, saved = self.host(**env)
                self.assertEqual(result.returncode, code, saved)
                if code == 77: self.assertIn("SKIPPED is not a pass", saved)

    def test_outer_timeout_cleans_up_and_identifies_stage(self):
        for begun in ("", "1"):
            with self.subTest(begun=begun):
                result, saved = self.host(TP2_OUTER_TIMEOUT="1", MOCK_SLEEP="1", MOCK_PROCESS_BEGIN=begun)
                self.assertEqual(result.returncode, 124, saved)
                self.assertIn("verified_absent=", saved)
                if begun: self.assertIn("after real-layer launch", saved)
                else: self.assertIn("before observed real-layer launch", saved)

    def test_cleanup_must_be_independently_verified(self):
        result, saved = self.host(MOCK_CLEANUP_FAIL="1")
        self.assertEqual(result.returncode, 125, saved)
        self.assertIn("INVERSE_CONTAINER_CLEANUP_UNVERIFIED", saved)
        self.assertNotIn(SUITE, saved)

    def inner(self, args=(), **overrides):
        work, tmp = self.root / "inner-work", self.root / "inner-tmp"
        tmp.mkdir(exist_ok=True)
        work.mkdir(exist_ok=True)
        self.inner_calls = self.root / "inner.jsonl"
        self.inner_calls.unlink(missing_ok=True)
        source = RUNNER.read_text().split("<<'INVERSE_CONTAINER'\n", 1)[1].split("\nINVERSE_CONTAINER\n", 1)[0]
        source = source.replace("/tmp", str(tmp)).replace("/work", str(work))
        source = source.replace("export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH", ": # CPU mocks stay first")
        (self.root / "inner.sh").write_text(source)
        fixture = self.root / "mock-layer"
        executable(fixture, f'''#!{sys.executable}
import json, os, pathlib, sys
args=sys.argv[1:]
kind="kernel" if pathlib.Path(sys.argv[0]).name == "native_down_plan_parity" else "layer"
with open(os.environ["MOCK_INNER_CALLS"], "a") as f: f.write(json.dumps([kind, *args]) + "\\n")
if kind == "kernel":
    device=args[args.index("--device")+1]
    if os.environ.get("MOCK_KERNEL", "1") == "1": print("NATIVE_DOWN_PLAN_PARITY_PASS")
    code=int(os.environ.get("MOCK_KERNEL_EXIT", "0"))
    if os.environ.get("MOCK_SECOND_KERNEL_EXIT") and device == "1": code=int(os.environ["MOCK_SECOND_KERNEL_EXIT"])
    sys.exit(code)
hc_mode=os.environ.get("MOCK_HC_RECORD", "normal")
def hc_records():
    if hc_mode == "missing": return
    for device in (0, 1):
        if hc_mode == "missing-device" and device == 1: continue
        logged_device=0 if hc_mode == "duplicate-device" else (2 if hc_mode == "wrong-device" and device == 1 else device)
        policy="legacy" if hc_mode == "wrong-policy" else "production-check"
        name="plain" if hc_mode == "wrong-name" else "staged"
        checked=0 if hc_mode == "unchecked" else 1
        print(f"HC_DISPATCH_SELECTED policy={{policy}} device={{logged_device}} hc-variant=3 hc-variant-name={{name}} check-invoked={{checked}}")
    if hc_mode != "missing-init": print("HC_DISPATCH_INIT_PASS policy=production-check devices=2 before-layer-setup=1")
    if hc_mode == "duplicate-init": print("HC_DISPATCH_INIT_PASS policy=production-check devices=2 before-layer-setup=1")
if "--hc-dispatch" in args and args[args.index("--hc-dispatch")+1] == "production-check" and hc_mode != "late": hc_records()
if os.environ.get("MOCK_EARLY_TIMING"): print("BENCH_SAMPLE forbidden premature timing")
if os.environ.get("MOCK_BANKS", "1") == "1": print("INVERSE_PLAN_BANK_GATE_PASS T=1..8 owners=2 banks=2 fields=19")
if os.environ.get("MOCK_EXACT", "1") == "1": print("INVERSE_PLAN_EXACT_GATE_PASS prefix_cases=44 continuation_cases=44")
if os.environ.get("MOCK_PROFILE"): print("PROFILE_DIAGNOSTIC mode=hybrid T=1")
for baseline in ("tp-hybrid-captured", "single-gpu-captured"):
    for tokens in (1,2,4,5,8):
        for kind in ("BENCH_SUMMARY", "BLOCK_SUMMARY"):
            if os.environ.get("MOCK_MISSING_SUMMARY") == f"{{baseline}}:{{tokens}}:{{kind}}": continue
            candidate="wrong-mode" if os.environ.get("MOCK_WRONG_LABEL") else "tp-hybrid-inverse"
            print(f"{{kind}} baseline={{baseline}} candidate={{candidate}} mode=hybrid T={{tokens}} pairs=2")
if os.environ.get("MOCK_BENCH", "1") == "1": print("BENCH_GATE PASS test=CPU-mock")
if hc_mode == "late": hc_records()
sys.exit(int(os.environ.get("MOCK_NORMAL_EXIT", "0")))
''')
        executable(self.bin / "cmake", f'''#!{sys.executable}
import json, os, pathlib, shutil, sys
args=sys.argv[1:]
with open(os.environ["MOCK_INNER_CALLS"], "a") as f: f.write(json.dumps(["cmake", *args]) + "\\n")
if "--build" in args:
    if os.environ.get("MOCK_BUILD_EXIT"): sys.exit(int(os.environ["MOCK_BUILD_EXIT"]))
    root=pathlib.Path(args[args.index("--build")+1]); root.mkdir(parents=True, exist_ok=True)
    for name in args[args.index("--target")+1:args.index("--parallel")]:
        target=root / name
        if name in ("tp2_gdn_layer", "native_down_plan_parity"):
            shutil.copyfile({str(fixture)!r}, target); target.chmod(0o755); continue
        target.write_text("#!" + sys.executable + "\\nimport json, os, sys\\nwith open(os.environ['MOCK_INNER_CALLS'], 'a') as f: f.write(json.dumps(['host-gate', " + repr(name) + "]) + '\\\\n')\\nsys.exit(49 if os.environ.get('MOCK_HOST_GATE_FAIL') == " + repr(name) + " else 0)\\n")
        target.chmod(0o755)
elif os.environ.get("MOCK_CONFIGURE_EXIT"): sys.exit(int(os.environ["MOCK_CONFIGURE_EXIT"]))
''')
        executable(self.bin / "ninja", "#!/bin/bash\nexit 97\n")
        executable(self.bin / "hipcc", "#!/bin/bash\n[[ $1 == --version ]] || exit 98\necho 'CPU fixture only'\n")
        executable(self.bin / "rocm-smi", "#!/bin/bash\necho 'CPU mocked telemetry'\n")
        executable(self.bin / "ldd", "#!/bin/bash\necho \"${MOCK_LDD-fixture: all dependencies resolved}\"\n")
        real_tee = shutil.which("tee")
        executable(self.bin / "tee", f'''#!/bin/bash
{shlex.quote(real_tee)} "$@"
[[ "$1" == */inverse-plan-benchmark.log && ${{MOCK_TEE_FAIL:-0}} == 1 ]] && exit 71
[[ "$1" == */inverse-plan-kernel-0.log && ${{MOCK_KERNEL_TEE_FAIL:-0}} == 1 ]] && exit 72
exit 0
''')
        return subprocess.run(["bash", str(self.root / "inner.sh"), "5", "5", "2",
                               "--pack", "/models/pack", "--gguf", "/models/shard.gguf", *args],
                              env=dict(self.env, MOCK_INNER_CALLS=str(self.inner_calls), **overrides),
                              text=True, capture_output=True, timeout=15)

    def test_inner_gpu_gates_then_one_unprofiled_paired_benchmark(self):
        result = self.inner(["--bench-warmup", "2", "--bench-trials", "4", "--layer", "4", "--mode", "7"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(GATES, result.stdout)
        calls = records(self.inner_calls)
        executions = [call for call in calls if call[0] in ("kernel", "layer")]
        self.assertEqual(executions[:2], [["kernel", "--device", "0", "--peer-device", "1"],
                                          ["kernel", "--device", "1", "--peer-device", "0"]])
        self.assertEqual(len(executions), 3)
        layer = executions[-1]
        for flag, value in (("--execution", "hybrid-inverse"), ("--layer", "4"), ("--mode", "7"),
                            ("--bench-warmup", "2"), ("--bench-trials", "4")):
            self.assertEqual(layer[layer.index(flag)+1], value)
        self.assertIn("--benchmark", layer)
        self.assertNotIn("--profile-stages", layer)
        self.assertFalse(any("rccl" in arg.lower() for arg in layer))

    def test_inner_build_is_isolated_and_disables_rccl(self):
        result = self.inner()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        config, build = [call for call in records(self.inner_calls) if call[0] == "cmake"]
        for value in ("-DSTRATA_TP2_GDN_RCCL_BUILD=OFF", "-DSTRATA_TP2_GDN_BUILD=ON",
                      "-DSTRATA_HIP_GFX906=ON", "-DCMAKE_HIP_ARCHITECTURES=gfx906", "-DSTRATA_BUILD_TESTS=OFF"):
            self.assertIn(value, config)
        self.assertFalse(any("STRATA_RCCL_LIBRARY" in arg for arg in config))
        self.assertEqual(build[build.index("--target")+1:build.index("--parallel")],
                         ["tp2_shard_policy", "tp_layer_layout_test", "native_expert_call_policy",
                          "tp_gdn_weights_test", "tp_hc_layout_test", "tp_gdn_gather_layout_test",
                          "native_down_plan_test", "native_down_plan_parity", "tp2_gdn_layer"])
        self.assertEqual(len([call for call in records(self.inner_calls) if call[0] == "host-gate"]), 7)
        self.assertIn(str(self.root / "inner-tmp/build-tp2-inverse-plan"), build)

    def test_inner_production_hc_selection_is_required_before_timing(self):
        result = self.inner(["--hc-dispatch", "production-check"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        layer = next(call for call in records(self.inner_calls) if call[0] == "layer")
        self.assertEqual(layer[layer.index("--hc-dispatch")+1], "production-check")
        for mode in ("missing", "missing-device", "duplicate-device", "wrong-device", "wrong-policy",
                     "wrong-name", "unchecked", "missing-init", "duplicate-init", "late"):
            with self.subTest(mode=mode):
                result = self.inner(["--hc-dispatch", "production-check"], MOCK_HC_RECORD=mode)
                self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
                self.assertIn("HC production selection", result.stderr)
                self.assertNotIn(GATES, result.stdout)

    def test_inner_legacy_does_not_require_new_hc_records(self):
        for args in ([], ["--hc-dispatch", "legacy"]):
            result = self.inner(args, MOCK_HC_RECORD="missing")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_inner_build_failure_never_launches(self):
        for env, code in (({"MOCK_CONFIGURE_EXIT": "33"}, 33), ({"MOCK_BUILD_EXIT": "34"}, 34),
                          ({"MOCK_LDD": "libamdhip64.so => not found"}, 2),
                          ({"MOCK_HOST_GATE_FAIL": "tp_gdn_gather_layout_test"}, 49)):
            with self.subTest(env=env):
                result = self.inner(**env)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                self.assertFalse(any(call[0] in ("layer", "kernel") for call in records(self.inner_calls)))
                self.assertNotIn(GATES, result.stdout)

    def test_inner_kernel_failure_or_skip_prevents_timing(self):
        for env, code in (({"MOCK_KERNEL": "0"}, 4), ({"MOCK_KERNEL_EXIT": "42"}, 42),
                          ({"MOCK_KERNEL_EXIT": "77"}, 77), ({"MOCK_SECOND_KERNEL_EXIT": "43"}, 43),
                          ({"MOCK_KERNEL_TEE_FAIL": "1"}, 72)):
            with self.subTest(env=env):
                result = self.inner(**env)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                self.assertFalse(any(call[0] == "layer" for call in records(self.inner_calls)))
                self.assertNotIn(GATES, result.stdout)

    def test_inner_missing_any_admission_marker_fails(self):
        for key in ("MOCK_BANKS", "MOCK_EXACT", "MOCK_BENCH"):
            with self.subTest(key=key):
                result = self.inner(**{key: "0"})
                self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
                self.assertNotIn(GATES, result.stdout)

    def test_inner_partial_or_mislabeled_pairs_fail(self):
        for missing in ("tp-hybrid-captured:1:BENCH_SUMMARY", "tp-hybrid-captured:8:BLOCK_SUMMARY",
                        "single-gpu-captured:5:BENCH_SUMMARY", "single-gpu-captured:2:BLOCK_SUMMARY"):
            with self.subTest(missing=missing):
                result = self.inner(MOCK_MISSING_SUMMARY=missing)
                self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
                self.assertNotIn(GATES, result.stdout)
        result = self.inner(MOCK_WRONG_LABEL="1")
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)

    def test_inner_early_timing_or_instrumentation_is_rejected(self):
        for key in ("MOCK_EARLY_TIMING", "MOCK_PROFILE"):
            with self.subTest(key=key):
                result = self.inner(**{key: "1"})
                self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
                self.assertIn("timing precedes admission or instrumentation", result.stderr)
                self.assertNotIn(GATES, result.stdout)

    def test_inner_binary_failure_and_log_failure_propagate(self):
        for env, code in (({"MOCK_NORMAL_EXIT": "42"}, 42), ({"MOCK_NORMAL_EXIT": "77"}, 77),
                          ({"MOCK_TEE_FAIL": "1"}, 71)):
            with self.subTest(env=env):
                result = self.inner(**env)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                self.assertNotIn(GATES, result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
