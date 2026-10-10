"""CPU-only config and fail-closed log-gate tests; no Docker/model/service access."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("mi50_pipeline", Path(__file__).with_name("mi50_pipeline.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

GOOD = """strata generate: layer split: CUDA0 loads the dense weights of layers 0-24 only
strata generate: layer split: CUDA1 loads the dense weights of layers 25-47 only
strata generate: expert cache 12800 slots, 23.08 GiB of VRAM; policy is
strata generate: pre-filled 12800 of 12800 slots from the profile in 1.0 s; slot 0 verified
strata generate: layer split: CUDA1 runs layers 25-47, expert cache 11776 slots (23.75 GiB), 11776 of its 11776 profiled pairs; slot 0 verified
strata generate: layer split: CUDA0 runs layers 0-24
""" + ("strata verify: window up to 7 tokens, 80.8 MiB of device buffers (100% VRAM resident: zero-doorbell graph)\n" * 4) + """strata serve: --pipeline-windows 2: two verifiers per stage, the stages overlap across windows (decode too; the GDN snapshots beside the window)
"""


class Config(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for path in ["pack-iq3_s/tokenizer", "mtp/rt"]:
            (self.root / path).mkdir(parents=True)
        for path in [M.NATIVE, M.NATIVE.replace("00001-of", "00002-of"), M.DENSE2, "ple-fp8.gguf", "expert-profile.bin"]:
            (self.root / path).touch()
        self.base = M.defaults(self.root)

    def test_independent_and_preserves_model_inputs(self):
        before = copy.deepcopy(self.base)
        self.base.update({"exe": "/old/strata", "cwd": "/old", "api_key": "DO_NOT_COPY",
                          "before_load": "do not run", "mcp_servers": {"ignored": {}}})
        self.base["env"]["SECRET_TOKEN"] = "DO_NOT_COPY"
        cfg = M.make_config(self.base, self.root)
        text = json.dumps(cfg)
        self.assertNotIn("/old", text)
        self.assertNotIn("DO_NOT_COPY", text)
        self.assertNotIn("before_load", cfg)
        self.assertNotIn("--peer-device", cfg["args"])
        self.assertEqual(cfg["args"].count("--native-dense-gguf"), 2)
        self.assertEqual(cfg["args"].count("--trim-stage-weights"), 1)
        self.assertEqual(cfg["gpu"], [0, 1])
        self.assertEqual(cfg["sampling"], before["sampling"])
        self.assertEqual(cfg["env"]["STRATA_IQ_MT_MIN"], "1")
        self.assertEqual(cfg["env"]["STRATA_VERIFY_ALL_RESIDENT"], "1")
        self.assertEqual(cfg["exe"], "/work/build-mi50-pipeline/engine/strata")
        self.assertNotIn(str(self.root), text)
        self.assertEqual(self.base["args"], before["args"])

    def test_outside_and_missing_paths_rejected(self):
        self.base["args"][1] = "/etc"
        with self.assertRaises(ValueError):
            M.make_config(self.base, self.root)
        self.base["args"][1] = str(self.root / "missing")
        with self.assertRaises(ValueError):
            M.make_config(self.base, self.root)

    def test_unapproved_options_and_context_rejected(self):
        for key, value in [("--expert-profile-save", "/tmp/saved"), ("--batch", "2")]:
            base = copy.deepcopy(self.base)
            base["args"] += [key, value]
            with self.assertRaises(ValueError):
                M.make_config(base, self.root)
        self.base["args"][self.base["args"].index("--max-context") + 1] = "262144"
        with self.assertRaises(ValueError):
            M.make_config(self.base, self.root)

    def test_duplicate_singleton_rejected(self):
        self.base["args"] += ["--spec", "5"]
        with self.assertRaises(ValueError):
            M.make_config(self.base, self.root)

    def test_missing_same_stem_sibling_rejected(self):
        (self.root / M.NATIVE.replace("00001-of", "00002-of")).unlink()
        with self.assertRaisesRegex(ValueError, "same-stem"):
            M.make_config(self.base, self.root)

    def test_split_symlink_spelling_preserved(self):
        original = self.root / M.NATIVE
        original.unlink()
        (self.root / "blob.gguf").touch()
        original.symlink_to("blob.gguf")
        cfg = M.make_config(self.base, self.root)
        self.assertEqual(cfg["args"][cfg["args"].index("--native") + 1], "/models/" + M.NATIVE)

    def test_absolute_in_root_split_symlink_supported(self):
        original = self.root / M.NATIVE
        original.unlink()
        (self.root / "blob.gguf").touch()
        original.symlink_to(self.root / "blob.gguf")
        cfg = M.make_config(self.base, self.root)
        self.assertEqual(cfg["args"][cfg["args"].index("--native") + 1], "/models/" + M.NATIVE)
        self.assertIn(str(self.root), M.model_views(self.root))

    def test_protected_model_mount_refused(self):
        with self.assertRaisesRegex(ValueError, "obscure"):
            M.model_views(Path("/usr"))

    def test_symlinked_model_root_accepts_physical_and_logical_inputs(self):
        alias = self.root.with_name(self.root.name + "-alias")
        alias.symlink_to(self.root, target_is_directory=True)
        self.addCleanup(alias.unlink)
        for source in (self.base, M.defaults(alias)):
            cfg = M.make_config(source, alias)
            self.assertEqual(cfg["args"][cfg["args"].index("--native") + 1], "/models/" + M.NATIVE)


class Gate(unittest.TestCase):
    def test_exact_startup(self):
        self.assertEqual(M.startup_errors(GOOD), [])

    def test_partial_and_serial_refused(self):
        for replacement in [
            GOOD.replace("12800 slots", "12799 slots"),
            GOOD.replace("11776 of its", "11775 of its"),
            GOOD.replace(" (100% VRAM resident: zero-doorbell graph)", "", 1),
            GOOD.replace(" (decode too; the GDN snapshots beside the window)", ""),
            GOOD + "strata serve: --pipeline-windows: a request with repetition penalties decodes serially\n",
            GOOD + "strata generate: trying a smaller expert cache\n",
            GOOD + "strata verify: layers [0,25) have experts out of VRAM\n",
        ]:
            self.assertTrue(M.startup_errors(replacement))

    def test_accumulated_starts_are_not_a_pass(self):
        self.assertTrue(M.startup_errors(GOOD + GOOD))


class ServerGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / "engine.log"
        self.log.write_text(GOOD)
        self.proc = mock.Mock()
        parent = self

        class FakeEngine:
            def __init__(self, *args, **kwargs):
                self.log_path = str(parent.log)
                self.log_start = 0
                self.proc = parent.proc

        self.server = types.SimpleNamespace(StrataEngine=FakeEngine,
            api_key_of=lambda value: value, key_list=lambda value: [v for v in value.split(",") if v.strip()])
        for patch in [mock.patch.object(M, "load_server", return_value=self.server),
                      mock.patch.dict(M.os.environ, {"STRATA_API_KEY": "local-test-secret"}),
                      mock.patch.object(M.sys, "argv", []),
                      mock.patch.object(Path, "read_text", return_value='{"args": [], "tokenizer": "/models/tokenizer"}'),
                      mock.patch.object(Path, "is_file", return_value=True)]:
            patch.start()
            self.addCleanup(patch.stop)

    def test_pass_and_no_restart(self):
        def main():
            engine = self.server.StrataEngine()
            with self.assertRaisesRegex(RuntimeError, "restart is disabled"):
                engine.restart()
            return 0
        self.server.main = main
        self.assertEqual(M.serve(), 0)
        self.proc.terminate.assert_not_called()

    def test_bad_log_ends_new_engine(self):
        self.log.write_text("partial initialization")
        self.server.main = lambda: self.server.StrataEngine()
        with self.assertRaisesRegex(RuntimeError, "startup gate failed"):
            M.serve()
        self.proc.terminate.assert_called_once()

    def test_unreadable_log_ends_new_engine(self):
        self.log.unlink()
        self.server.main = lambda: self.server.StrataEngine()
        with self.assertRaises(OSError):
            M.serve()
        self.proc.terminate.assert_called_once()

    def test_commas_only_key_refused_before_start(self):
        self.server.main = mock.Mock()
        with mock.patch.dict(M.os.environ, {"STRATA_API_KEY": " , , "}):
            with self.assertRaisesRegex(ValueError, "usable API key"):
                M.serve()
        self.server.main.assert_not_called()


if __name__ == "__main__":
    unittest.main()
