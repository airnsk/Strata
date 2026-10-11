#!/usr/bin/env python3
"""CPU-only tests of the actual harness parser/startup with CUDA/GR mocks.

These check control flow and provenance, never GPU parity or performance.
"""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/hip/tp2_gdn_layer.cpp"


class DispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++")
        if compiler is None:
            raise unittest.SkipTest("g++ is required for the CPU parser/startup test")
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        source = SOURCE.read_text()
        # Compile the production harness functions, not copies of their logic.
        functions = source[source.index("struct Options {"):source.index("void hc_source_metadata(")]
        mocks = r'''
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept>
namespace tp {
struct TpGdnRcclOptions { std::string library; int timeout_ms=1000,init_timeout_ms=1000; };
}
int device=7, calls[2]={0,0};
int cudaGetDevice(int* d) { *d=device; return 0; }
void ck(int code,const char* text) { if(code)throw std::runtime_error(text); }
void on(int d) { device=d; }
namespace k {
void fused_gr_check() {
    ++calls[device];std::printf("MOCK_CHECK device=%d\n",device);
    if(std::getenv("MOCK_CHECK_THROW")&&device==1)throw std::runtime_error("mock selector failure");
}
int fused_gr_variant() {
    if(std::getenv("MOCK_BAD_VARIANT"))return 4;
    if(calls[device])return device==0?3:1; // The selector may fall back per GPU.
    const char* value=std::getenv("STRATA_HC_SPLIT");
    return value&&value[0]=='1'?2:value&&value[0]=='2'?3:1;
}
}
'''
        driver = r'''
int main(int argc,char** argv) {
    try {
        const auto options=parse(argc,argv);
        std::printf("MOCK_OPTIONS policy=%s execution=%s model-probe=%d inverse-probe=%d\n",
            options.hc_dispatch.c_str(),options.execution.c_str(),int(options.model_probe),int(options.inverse_probe));
        hc_dispatch_startup(options);
        std::printf("MOCK_CAPTURE current-device=%d checks=%d,%d\n",device,calls[0],calls[1]);
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 2; }
}
'''
        path = root / "dispatch.cpp"
        path.write_text(mocks + functions + driver)
        cls.binary = root / "dispatch"
        subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(path), "-o", str(cls.binary)], check=True)

    def run_dispatch(self, args=(), **values):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("STRATA_", "MOCK_"))}
        env.update(values)
        return subprocess.run([str(self.binary), "--pack", "fixture", "--gguf", "fixture", *args],
                              env=env, text=True, capture_output=True, timeout=5)

    def test_default_and_explicit_legacy_keep_unchecked_selection(self):
        for args in ([], ["--hc-dispatch", "legacy"]):
            result = self.run_dispatch(args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("MOCK_CHECK", result.stdout)
            self.assertNotIn("HC_DISPATCH_INIT_PASS", result.stdout)
            self.assertEqual(result.stdout.count("hc-variant=1 hc-variant-name=plain check-invoked=0"), 2)
            self.assertIn("MOCK_CAPTURE current-device=7 checks=0,0", result.stdout)

    def test_legacy_preserves_explicit_hc_split_override(self):
        for override, variant, name in (("0", 1, "plain"), ("1", 2, "split"), ("2", 3, "staged")):
            result = self.run_dispatch(STRATA_HC_SPLIT=override)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("MOCK_CHECK", result.stdout)
            self.assertIn(f"hc-variant={variant} hc-variant-name={name} check-invoked=0", result.stdout)

    def test_production_checks_both_devices_before_admission_and_capture(self):
        result = self.run_dispatch(["--hc-dispatch", "production-check"], STRATA_GR_FAST="0", STRATA_NO_MULTI_GR="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        ordered = ["MOCK_CHECK device=0", "HC_DISPATCH_SELECTED policy=production-check device=0 hc-variant=3",
                   "MOCK_CHECK device=1", "HC_DISPATCH_SELECTED policy=production-check device=1 hc-variant=1",
                   "HC_DISPATCH_INIT_PASS policy=production-check devices=2 before-layer-setup=1",
                   "MOCK_CAPTURE current-device=7 checks=1,1"]
        positions = [result.stdout.index(item) for item in ordered]
        self.assertEqual(positions, sorted(positions))
        for key, value in (("STRATA_GR_FAST", "0"), ("STRATA_NO_MULTI_GR", "1"), ("STRATA_HC_SPLIT", "unset")):
            self.assertIn(f"HC_DISPATCH_ENV {key}={value}", result.stdout)

    def test_output_changing_flags_remain_refused_before_selection(self):
        for policy in ("legacy", "production-check"):
            for key in ("STRATA_GR_V3", "STRATA_GR_SPLIT"):
                result = self.run_dispatch(["--hc-dispatch", policy], **{key: "1"})
                self.assertEqual(result.returncode, 2)
                self.assertIn(f"requires {key}=0", result.stderr)
                self.assertNotIn("MOCK_CHECK", result.stdout)
                self.assertNotIn("MOCK_CAPTURE", result.stdout)

    def test_selector_failure_or_unknown_variant_cannot_admit_capture(self):
        for key in ("MOCK_CHECK_THROW", "MOCK_BAD_VARIANT"):
            result = self.run_dispatch(["--hc-dispatch", "production-check"], **{key: "1"})
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("HC_DISPATCH_INIT_PASS", result.stdout)
            self.assertNotIn("MOCK_CAPTURE", result.stdout)

    def test_policy_parser_and_existing_probe_modes(self):
        for args in (["--hc-dispatch", "staged"], ["--hc-dispatch"]):
            result = self.run_dispatch(args)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("MOCK_CHECK", result.stdout)
        for flag in ("--model-probe", "--model-probe-inverse"):
            result = self.run_dispatch([flag, "--hc-dispatch", "production-check"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("execution=hybrid model-probe=1", result.stdout)

    def test_real_main_initializes_before_all_capture_capable_branches(self):
        main = SOURCE.read_text().split("int main(int argc,char** argv) {", 1)[1]
        startup = main.index("hc_dispatch_startup(o);")
        for action in ("model_probe(o)", "rccl_fault_gate(o)", "flat_protocol_preflight(preflight)",
                       "full.load(", "tp::TpGdnLayer reference(", "prepare_captured("):
            self.assertLess(startup, main.index(action), action)

    def test_model_probe_wrapper_whitelist_preserves_legacy_and_accepts_option(self):
        source = (ROOT / "tools/run_tp2_model_probe_mi50.sh").read_text()
        validator = source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pack").mkdir()
            (root / "shard").touch()
            for policy, expected in ((None, 0), ("legacy", 0), ("production-check", 0), ("staged", 1)):
                args = [] if policy is None else ["--hc-dispatch", policy]
                result = subprocess.run(["python3", "-", str(ROOT), str(root), "--pack", "/models/pack",
                                         "--gguf", "/models/shard", *args], input=validator,
                                        text=True, capture_output=True, timeout=5)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_model_probe_wrapper_admission_uses_effective_variants_and_order(self):
        source = (ROOT / "tools/run_tp2_model_probe_mi50.sh").read_text()
        gate = source[source.index("hc_dispatch=legacy\n"):source.index("if (( RC == 77 ));")]
        selected = ["HC_DISPATCH_SELECTED policy=production-check device=0 hc-variant=3 hc-variant-name=staged check-invoked=1",
                    "HC_DISPATCH_SELECTED policy=production-check device=1 hc-variant=1 hc-variant-name=plain check-invoked=1"]
        init = "HC_DISPATCH_INIT_PASS policy=production-check devices=2 before-layer-setup=1"
        phase, passed = "PROBE_PHASE T=1", "PROBE_GATE PASS targeted-whole-layer-parity=checked"
        good = [*selected, init, phase, passed]
        cases = [(good, 0), ([selected[0], init, phase, passed], 4),
                 ([*selected, phase, passed], 4), ([phase, *selected, init, passed], 4),
                 ([*selected, init, init, phase, passed], 4),
                 ([*selected, selected[0], init, phase, passed], 4),
                 ([*selected, init, phase], 4)]
        for old, new in (("policy=production-check", "policy=legacy"), ("device=1", "device=2"),
                         ("device=1", "device=0"), ("hc-variant=1", "hc-variant=2"),
                         ("check-invoked=1", "check-invoked=0")):
            cases.append(([selected[0], selected[1].replace(old, new), init, phase, passed], 4))
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "probe.log"
            script = 'set -euo pipefail\nRC=0\n' + gate + '\nexit "$RC"\n'
            for lines, expected in cases:
                with self.subTest(lines=lines):
                    log.write_text("\n".join(lines) + "\n")
                    result = subprocess.run(["bash", "-c", script, "gate", "/models", "--pack", "/models/pack",
                                             "--hc-dispatch", "production-check"],
                                            env=dict(os.environ, LOG=str(log)), capture_output=True, text=True, timeout=5)
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
            log.write_text(phase + "\n")
            result = subprocess.run(["bash", "-c", script, "gate", "/models", "--pack", "/models/pack"],
                                    env=dict(os.environ, LOG=str(log)), capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
