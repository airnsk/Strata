#!/usr/bin/env python3
"""CPU-only baseline wrapper tests: Docker/HIP are mocked, no GPU execution."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_tp2_model_probe_mi50.sh"
IMAGE = "sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca"


def fixture():
    lines = []
    for half in ("attn", "ffn"):
        for part, kind, type_id, shape in (("norm","F32",0,"10240"),("down","BF16",30,"10240x320"),
                                          ("up","BF16",30,"320x10240"),("inject","BF16",30,"10240x4")):
            role = f"hc_{half}_{part}.weight"
            lines += [f"HC_GGUF_SELECTED layer=0 role={role} matches=1 shard-index=0 type={kind} type-id={type_id} shape={shape}",
                      f"HC_GGUF_COUNTS role={role} type={kind} type-id={type_id} layers=48 of=48",
                      f"HC_GGUF_COVERAGE role={role} missing-layers=0 ambiguous-layers=0 of=48"]
    for flag in ("STRATA_HC_PERSIST", "STRATA_HC_SPLIT", "STRATA_GR_FAST", "STRATA_GR_V3", "STRATA_GR_SPLIT",
                 "STRATA_GR_DOWN_MAX4", "STRATA_NO_MULTI_GR", "STRATA_TSUM", "STRATA_SM70_TABLE", "STRATA_HC_Q8", "STRATA_QFUSE"):
        lines.append(f"HC_DISPATCH_ENV {flag}=unset")
    lines += [f"HC_DISPATCH_SELECTED policy=production-check device={d} hc-variant=3 hc-variant-name=staged check-invoked=1" for d in (0,1)]
    lines.append("HC_DISPATCH_INIT_PASS policy=production-check devices=2 before-layer-setup=1")
    for mode in ("captured", "hybrid"):
        lines.append(f"CORRECTNESS_EXECUTION {mode} (reference=runtime)")
        for t in range(1,9):
            for keep in range(t+1): lines.append(f"CASE T={t} keep={keep} epoch=1")
        for t in range(1,9): lines.append(f"PROFILE_DIAGNOSTIC mode={mode} T={t} separate-from-correctness-and-timing")
    lines.append("PASS whole-GDN engineering parity cases=88 continuation-cases=88 failures=0; no full-model inference/quality/performance claim")
    for baseline in ("single-gpu-captured", "tp-row-captured"):
        for kind in ("BENCH", "BLOCK"):
            lines.append(f"{kind}_STUDY baseline={baseline} candidate=tp-hybrid-captured")
            lines.append("BENCH_CONFIG warmup_crossovers=3 measured_crossovers=4 pairs-per-crossover=2 order=ABBA/BAAB-alternating" if kind == "BENCH" else "BLOCK_METHOD calls=16 crossovers=4")
            for t in (1,2,4,5,8):
                common = f"baseline={baseline} candidate=tp-hybrid-captured mode=hybrid T={t}"
                for trial in (range(-3,4) if kind == "BENCH" else range(4)):
                    for leg in range(2):
                        first = "tp-hybrid-captured" if (trial+leg+(3 if kind == "BENCH" else 0)) % 2 else baseline
                        names = ("-propose-ms", "-commit-ms", "-total-ms") if kind == "BENCH" else ("-wall-ms", "-ms-per-call")
                        values = " ".join(arm+name+"=1.0" for arm in ("baseline","candidate") for name in names)
                        tag = "BENCH_WARMUP" if trial < 0 else kind+"_SAMPLE"
                        lines.append(f"{tag} {common} trial={trial} crossover-leg={leg} first={first} calls=16 {values}")
                    if trial >= 0:
                        suffix = "-mean-ms" if kind == "BENCH" else "-mean-ms-per-call"
                        lines.append(f"{kind}_CROSSOVER {common} trial={trial} baseline{suffix}=1.0 candidate{suffix}=1.0 crossed-ratio=1.0")
                suffix = "pairs=8" if kind == "BENCH" else "blocks-per-arm=8 calls-per-block=16 measured-intermediate-parity=not-downloaded shadow-every-step=passed final-block-parity=passed"
                names = ("-ms-median", "-ms-p10", "-ms-p90", "-propose-median", "-commit-median") if kind == "BENCH" else ("-ms-per-call-median",)
                values = " ".join(arm+name+"=1.0" for arm in ("baseline","candidate") for name in names)
                lines.append(f"{kind}_SUMMARY {common} {suffix} ratio-of-total-times=1.1 crossover-ratio-median=1.1 {values}")
    lines += ["BENCH_GATE PASS burst-pairs-every-result=checked block-shadow-every-step=checked measured-block-final-only=checked; no measured block intermediate downloads; one-layer wall times only", "BASELINE_PROCESS_EXIT=0"]
    return "\n".join(lines)+"\n"


class BaselineRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.repo = self.root / "repo"; self.bin = self.root / "bin"
        self.bin.mkdir(); (self.repo / "tools").mkdir(parents=True)
        shutil.copy2(RUNNER, self.repo / "tools" / RUNNER.name)
        shutil.copy2(ROOT / "tools/mi50_pipeline.py", self.repo / "tools/mi50_pipeline.py")
        for name in ("deps/llama.cpp/ggml/CMakeLists.txt", "tests/hip/tp2_gdn_layer.cpp", "src/core/tp_gdn_layer.cpp",
                     "src/kernels/cuda/fused_gr.cu", "include/strata/kernels/fused_gr.hpp", "include/strata/core/tp_gdn_layer.hpp"):
            p = self.repo / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("fixture\n")
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture", "--allow-empty"], check=True)
        self.physical = self.root / "physical"; (self.physical / "pack").mkdir(parents=True)
        (self.physical / "real.gguf").touch()
        self.models = self.root / "logical"; self.models.symlink_to(self.physical, target_is_directory=True)
        (self.physical / "alias.gguf").symlink_to(self.models / "real.gguf")
        self.output = self.root / "output"; self.output.write_text(fixture())
        self.calls = self.root / "calls"
        self.env = dict(os.environ, PATH=str(self.bin)+":"+os.environ["PATH"], MOCK_OUTPUT=str(self.output), MOCK_CALLS=str(self.calls))
        for key in ("TP2_TIMEOUT", "BUILD_JOBS"): self.env.pop(key, None)
        self.exe("sudo", '''#!/bin/bash
[[ "$1" == -n ]] && shift
[[ "$1" == -v ]] && exit "${MOCK_AUTH_EXIT:-0}"
exec "$@"
''')
        self.exe("id", '''#!/bin/bash
case "$1" in -u|-g) echo 1000;; *) /usr/bin/id "$@";; esac
''')
        self.exe("docker", f'''#!{sys.executable}
import json, os, pathlib, sys
args=sys.argv[1:]
with open(os.environ["MOCK_CALLS"], "a") as f: f.write(json.dumps(args)+"\\n")
if args[:2]==["image","inspect"]:
 print(os.environ.get("MOCK_IMAGE", "{IMAGE}")); sys.exit(0)
if any("cmake -S" in x for x in args):
 print("a"*64+"  /work/build-tp2-gdn/tp2_gdn_layer")
 sys.exit(int(os.environ.get("MOCK_BUILD_EXIT", "0")))
print(pathlib.Path(os.environ["MOCK_OUTPUT"]).read_text(), end="")
sys.exit(int(os.environ.get("MOCK_RUN_EXIT", "0")))
''')

    def exe(self, name, content):
        p=self.bin/name; p.write_text(content); p.chmod(0o755)

    def run_wrapper(self, extra=(), baseline=True, **env):
        args=["bash",str(self.repo/"tools"/RUNNER.name)]
        if baseline: args += ["--sustained-baseline"]
        args += [str(self.models),"--pack","/models/pack","--gguf","/models/alias.gguf",*extra]
        return subprocess.run(args, env=dict(self.env, **env), capture_output=True, text=True, timeout=15, start_new_session=True)

    def test_success_fixed_cli_alias_views_and_distinct_log(self):
        r=self.run_wrapper(); self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertIn("BASELINE_ADMISSION_PASS",r.stdout); self.assertIn("TP2_STAGED_BASELINE_EXIT=0",r.stdout)
        self.assertNotIn("PROBE_GATE",r.stdout)
        saved = next(self.repo.glob("tp2-staged-baseline-*.log")).read_text()
        self.assertIn("BASELINE_ADMISSION_PASS",saved)
        calls=[json.loads(x) for x in self.calls.read_text().splitlines()]
        runtime=calls[-1]
        for view in ("/models",str(self.models),str(self.physical)):
            self.assertIn(f"type=bind,src={self.physical},dst={view},readonly",runtime)
        for token in ("--benchmark","hybrid","production-check","--bench-block-calls","16","--bench-block-trials","4"):
            self.assertIn(token,runtime)
        self.assertNotIn("--model-probe",runtime)
        self.assertIn("--network",runtime); self.assertIn("none",runtime)

    def test_default_probe_cli_and_footer_remain_unchanged(self):
        self.output.write_text("PROBE_GATE PASS\nPROBE_PROCESS_EXIT=0\n")
        r=self.run_wrapper(baseline=False); self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertIn("TP2_MODEL_PROBE_EXIT=0",r.stdout)
        runtime=json.loads(self.calls.read_text().splitlines()[-1]); self.assertIn("--model-probe",runtime)
        self.assertNotIn("--benchmark",runtime)

    def test_reject_overrides_before_docker(self):
        for args in (("--hc-dispatch","legacy"),("--mode","7"),("--layer","2"),("--bench-trials","999"),
                     ("--bench-block-calls","64"),("--execution","flat"),("--sustained-baseline","1")):
            with self.subTest(args=args):
                r=self.run_wrapper(args); self.assertNotEqual(r.returncode,0)
                self.assertFalse(self.calls.exists())
        for env in ({"TP2_TIMEOUT":"1801"},{"BUILD_JOBS":"129"},{"TP2_TIMEOUT":"99999999999999999"}):
            r=self.run_wrapper(**env); self.assertNotEqual(r.returncode,0); self.assertFalse(self.calls.exists())

    def test_reject_escape_before_docker(self):
        (self.physical/"alias.gguf").unlink(); (self.physical/"alias.gguf").symlink_to(self.root/"outside")
        (self.root/"outside").touch(); r=self.run_wrapper()
        self.assertNotEqual(r.returncode,0); self.assertFalse(self.calls.exists())

    def test_admission_rejects_fallback_missing_duplicate_wrong_or_late_records(self):
        good=fixture()
        changes=[("hc-variant=3 hc-variant-name=staged","hc-variant=1 hc-variant-name=plain"),
                 ("continuation-cases=88","continuation-cases=44"),("shape=320x10240","shape=10240x320"),
                 ("calls-per-block=16","calls-per-block=8"),("ratio-of-total-times=1.1","ratio-of-total-times=nan"),
                 ("candidate=tp-hybrid-captured","candidate=tp-hybrid-inverse"),("BASELINE_PROCESS_EXIT=0","BASELINE_PROCESS_EXIT=1"),
                 ("baseline-ms-median=1.0", "baseline-ms-median=nan"),
                 ("baseline-ms-median=1.0", "baseline-ms-median=bad"),
                 ("baseline-ms-median=1.0", ""),
                 ("ratio-of-total-times=1.1", "ratio-of-total-times=1.1 ratio-of-total-times=1.2"),
                 ("baseline-total-ms=1.0", "baseline-total-ms=nan")]
        bad=[good.replace(a,b,1) for a,b in changes]
        for prefix in ("CASE ","BENCH_SUMMARY ","BLOCK_SUMMARY ","BENCH_SAMPLE ","BLOCK_SAMPLE ","HC_DISPATCH_SELECTED ","BENCH_GATE PASS"):
            line=next(x for x in good.splitlines() if x.startswith(prefix))
            bad += [good.replace(line+"\n","",1),good+line+"\n"]
        for marker in ("BENCH_GATE FAIL", "HC_DISPATCH_INIT_PASS policy=legacy devices=1 before-layer-setup=0", "BENCH_SUMMARY", "BLOCK_SAMPLE"):
            bad.append(good.replace("BENCH_GATE PASS", marker+"\nBENCH_GATE PASS", 1))
        bad.append(good+"BASELINE_PROCESS_EXIT=124\n")
        bad.append(good+"BENCH_WARMUP malformed\n")
        for prefix in ("CASE ", "PROFILE_DIAGNOSTIC ", "BLOCK_STUDY ", "CORRECTNESS_EXECUTION ", "BENCH_CONFIG ", "BENCH_CROSSOVER "):
            line=next(x for x in good.splitlines() if x.startswith(prefix))
            bad.append(good.replace(line+"\n", "", 1)+line+"\n")
            bad.append(good+line+"\n")
        sample=next(x for x in good.splitlines() if x.startswith("BENCH_SAMPLE "))
        bad.append(sample+"\n"+good.replace(sample+"\n","",1))
        for i,text in enumerate(bad):
            with self.subTest(case=i):
                self.output.write_text(text); r=self.run_wrapper()
                self.assertEqual(r.returncode,4,r.stdout+r.stderr)
                self.assertIn("TP2_STAGED_BASELINE_EXIT=4",r.stdout)

    def test_runtime_build_image_and_authorization_failures_never_pass(self):
        for env in ({"MOCK_RUN_EXIT":"124"},{"MOCK_BUILD_EXIT":"2"},{"MOCK_IMAGE":"sha256:bad"},{"MOCK_AUTH_EXIT":"1"}):
            with self.subTest(env=env):
                r=self.run_wrapper(**env); self.assertNotEqual(r.returncode,0)
                self.assertNotIn("BASELINE_ADMISSION_PASS",r.stdout)
                self.assertNotIn("TP2_STAGED_BASELINE_EXIT=0",r.stdout)


if __name__ == "__main__": unittest.main(verbosity=2)
