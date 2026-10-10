#!/usr/bin/env python3
"""Prepare and gate a separate MI50 layer-pipeline experiment. No downloads or requests."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "build-mi50-pipeline" / "run"
NATIVE = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
DENSE2 = "Qwen3.8-Flash-Next-GSQ-RCO-Q8_0-PLE-00002-of-00002.gguf"
PATH_FLAGS = {"--pack", "--native", "--ple-gguf", "--expert-profile", "--mtp", "--native-dense-gguf"}
VALUE_FLAGS = PATH_FLAGS | {"--expert-cache", "--prefill", "--spec", "--spec-min-p", "--max-context",
                            "--kv", "--kv-resident", "--vram-reserve-mib", "--ple-io", "--pcie-frac",
                            "--peer-device"}
PERF_ENV = {"STRATA_PREFILL_TIMING", "STRATA_PREFILL_AUTO_MAX", "STRATA_PREFILL_GEMM_F16",
            "STRATA_FDOT2_SW", "STRATA_FDOT2_SW_VERBOSE", "STRATA_SH_STREAM", "STRATA_EXP_MODE"}


def model_views(model_root: Path) -> list[str]:
    roots = sorted({str(Path(os.path.abspath(model_root))), str(model_root.resolve(strict=True))})
    protected = ("/work", "/state", "/opt", "/usr", "/etc", "/dev", "/proc", "/sys", "/bin", "/sbin", "/lib", "/lib64")
    for root in roots:
        if root in ("/", "/tmp", "/var", "/home") or any(root == p or root.startswith(p + "/") for p in protected):
            raise ValueError("The model root would obscure a required container path; use a dedicated model directory")
        if any(c in root for c in ",\n\r"):
            raise ValueError("Model root paths cannot contain commas/newlines")
    return roots


def defaults(model_root: Path) -> dict:
    pairs = [
        ("--pack", "pack-iq3_s"), ("--native", NATIVE), ("--ple-gguf", "ple-fp8.gguf"),
        ("--expert-profile", "expert-profile.bin"), ("--expert-cache", "auto"), ("--prefill", "auto"),
        ("--spec", "5"), ("--spec-min-p", "0.5"), ("--mtp", "mtp/rt"),
        ("--max-context", "131072"), ("--kv", "int8"), ("--kv-resident", "32768"),
        ("--vram-reserve-mib", "700"), ("--ple-io", "direct"),
        ("--native-dense-gguf", NATIVE), ("--native-dense-gguf", DENSE2), ("--pcie-frac", "0"),
        ("--peer-device", "1"),
    ]
    return {
        "args": [x for flag, value in pairs for x in (flag, str(model_root / value) if flag in PATH_FLAGS else value)],
        "tokenizer": str(model_root / "pack-iq3_s/tokenizer"),
        "sampling": {"temperature": 0.1, "top_p": 0.95, "top_k": 20},
        "env": {"STRATA_PREFILL_TIMING": "1", "STRATA_PREFILL_AUTO_MAX": "16384",
                "STRATA_PREFILL_GEMM_F16": "1", "STRATA_FDOT2_SW": "1", "STRATA_FDOT2_SW_VERBOSE": "0",
                "STRATA_SH_STREAM": "0", "STRATA_EXP_MODE": "8"},
    }


def make_config(source: dict, model_root: Path) -> dict:
    """Narrowly support the reviewed b4 input; never carry executable paths or hooks."""
    model_root = Path(os.path.abspath(model_root))
    physical_root = model_root.resolve(strict=True)
    views = [Path(path) for path in model_views(model_root)]
    if not isinstance(source, dict) or not isinstance(source.get("env", {}), dict) or not isinstance(source.get("sampling", {}), dict):
        raise ValueError("Source config/env/sampling must be JSON objects")

    def model_path(value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("Model inputs must be absolute paths below MODEL_ROOT")
        try:
            path.resolve(strict=True).relative_to(physical_root)
            # Keep split GGUF basenames, including valid in-root symlinks. The engine
            # derives sibling names from this spelling rather than a symlink target.
            lexical = Path(os.path.abspath(path))
            matching = [root for root in views if lexical == root or root in lexical.parents]
            if not matching:
                raise ValueError("input is outside the known model-root views")
            rel = lexical.relative_to(max(matching, key=lambda root: len(root.parts)))
        except (ValueError, OSError) as exc:
            raise ValueError("A required model input is absent or outside MODEL_ROOT") from exc
        current = model_root
        for component in rel.parts:
            current /= component
            if current.is_symlink() and os.path.isabs(os.readlink(current)):
                target = Path(os.path.abspath(os.readlink(current)))
                if not any(target == root or root in target.parents for root in views):
                    raise ValueError("An absolute model alias escapes the supported model-root views; "
                                     "review that path without changing production files")
        return str(Path("/models") / rel)

    original = source.get("args")
    if not isinstance(original, list) or len(original) % 2:
        raise ValueError("Expected the reviewed b4 engine argument pairs")
    args, seen = [], {}
    for flag, value in zip(original[::2], original[1::2]):
        if flag not in VALUE_FLAGS or not isinstance(value, str):
            raise ValueError("Unsupported source engine option; review this configuration separately")
        if flag in seen and flag != "--native-dense-gguf":
            raise ValueError("Duplicate source engine option")
        seen[flag] = value
        if flag != "--peer-device":
            args.extend([flag, model_path(value) if flag in PATH_FLAGS else value])
    required = {"--expert-cache": "auto", "--spec": "5", "--spec-min-p": "0.5", "--max-context": "131072",
                "--kv": "int8", "--kv-resident": "32768", "--pcie-frac": "0"}
    if any(seen.get(k) != v for k, v in required.items()) or not PATH_FLAGS <= seen.keys():
        raise ValueError("Source differs from the reviewed b4 model/context/cache settings")
    native = Path(seen["--native"])
    split = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf", native.name)
    if split and 1 <= int(split[2]) <= int(split[3]):
        for index in range(1, int(split[3]) + 1):
            sibling = native.with_name(f"{split[1]}-{index:05d}-of-{int(split[3]):05d}.gguf")
            if not sibling.is_file():
                raise ValueError(f"The current engine requires missing same-stem native shard: {sibling.name}; "
                                 "the differently named PLE/dense shard does not replace it")
            model_path(str(sibling))
    args += ["--layer-split", "25", "--split-device", "1", "--trim-stage-weights",
             "--pipeline-windows", "2", "--adapt-every", "0"]
    env = {k: str(v) for k, v in source.get("env", {}).items() if k in PERF_ENV}
    env.update({"HIP_VISIBLE_DEVICES": "0,1", "STRATA_HC_PERSIST": "0", "STRATA_VERIFY_ALL_RESIDENT": "1",
                "STRATA_IQ_MT_MIN": "1", "STRATA_DECODE_TIMING": "1", "STRATA_PIPELINE_DEBUG": "1",
                "STRATA_PIPELINE_SWITCH": "/state/pipeline-switch.txt"})
    return {"exe": "/work/build-mi50-pipeline/engine/strata", "args": args, "cwd": "/state",
            "tokenizer": model_path(source["tokenizer"]), "model_name": "qwen3.8-flash-next-mi50-pipeline-k25",
            "log": "/state/engine.log", "backend": "hip", "lib_dirs": ["/opt/rocm/lib"],
            "port": 8001, "host": "0.0.0.0", "gpu": [0, 1], "gpu_order": "as_given", "layer_split": "25",
            "env": env, "sampling": copy.deepcopy(source.get("sampling", {}))}


def startup_errors(text: str) -> list[str]:
    errors = []
    required = {
        "primary range": "layer split: CUDA0 runs layers 0-24",
        "primary allocation": "expert cache 12800 slots",
        "primary fill": "pre-filled 12800 of 12800 slots",
        "later range/allocation": "CUDA1 runs layers 25-47, expert cache 11776 slots",
        "later fill": "11776 of its 11776 profiled pairs; slot 0 verified",
        "primary trim": "CUDA0 loads the dense weights of layers 0-24 only",
        "later trim": "CUDA1 loads the dense weights of layers 25-47 only",
        "pipeline": "--pipeline-windows 2: two verifiers per stage, the stages overlap across windows "
                    "(decode too; the GDN snapshots beside the window)",
    }
    errors += [name for name, marker in required.items() if marker not in text]
    windows = re.findall(r"^strata verify: window up to .*", text, flags=re.M)
    if len(windows) != 4 or any("(100% VRAM resident: zero-doorbell graph)" not in line for line in windows):
        errors.append("all four zero-doorbell verifiers")
    if re.search(r"--pipeline-windows .* is off:|decode stays serial|decodes serially|trying a smaller expert cache|"
                 r"shrinking the expert cache|have experts out of VRAM", text):
        errors.append("a fallback or cache shrink was reported")
    return errors


def load_server():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "serve"))
    from serve import server
    return server


def serve() -> int:
    if not os.environ.get("STRATA_API_KEY", "").strip():
        raise ValueError("STRATA_API_KEY is required before listening on 0.0.0.0")
    server = load_server()
    if not server.key_list(server.api_key_of(os.environ["STRATA_API_KEY"])):
        raise ValueError("A usable API key is required before loading a model")
    cfg = json.loads(Path("/state/config.json").read_text())
    for index, value in enumerate(cfg["args"]):
        if value in PATH_FLAGS and not Path(cfg["args"][index + 1]).exists():
            raise ValueError("An experimental model input is not readable inside the container")
    for name in ("vocab.json", "merges.txt", "token_type.json"):
        if not (Path(cfg["tokenizer"]) / name).is_file():
            raise ValueError("The local tokenizer is incomplete or unreadable inside the container")

    class GatedEngine(server.StrataEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            try:
                with open(self.log_path, encoding="utf-8", errors="replace") as stream:
                    stream.seek(self.log_start)
                    errors = startup_errors(stream.read())
                if errors:
                    raise RuntimeError("Experimental startup gate failed: " + ", ".join(errors))
            except BaseException:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except server.subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=10)
                raise
            print("MI50_STARTUP_GATE=PASS: full local residency and pipeline active; graph execution/parity untested.", flush=True)

        def restart(self, *args, **kwargs):
            raise RuntimeError("Automatic engine restart is disabled for this experiment; inspect the logs")

    server.StrataEngine = GatedEngine
    sys.argv = [str(ROOT / "serve/server.py"), "--engine", "strata", "--config", "/state/config.json",
                "--host", "0.0.0.0", "--port", "8001"]
    return server.main()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("model_root", type=Path)
    prep.add_argument("source_config", nargs="?", type=Path)
    sub.add_parser("dependencies")
    sub.add_parser("serve")
    check = sub.add_parser("check")
    check.add_argument("log", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            root = args.model_root.resolve(strict=True)
            if not args.model_root.is_absolute() or not root.is_dir() or any(c in str(root) for c in ",\n\r"):
                raise ValueError("MODEL_ROOT must be an existing absolute directory without commas/newlines")
            source = json.loads(args.source_config.read_text(encoding="utf-8-sig")) if args.source_config else defaults(root)
            cfg = make_config(source, args.model_root)
            if STATE.exists():
                raise ValueError("Experimental config already exists; reuse it with serve (prepare never overwrites it)")
            STATE.mkdir(parents=True, mode=0o700)
            (STATE / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
            (STATE / "models.path").write_text(str(root) + "\n")
            (STATE / "model-views.txt").write_text("\n".join(model_views(args.model_root)) + "\n")
            (STATE / "pipeline-switch.txt").write_text("pw=2\n")
            print("Prepared standalone config in build-mi50-pipeline/run; production config is unchanged.")
        elif args.action == "dependencies":
            import jinja2  # noqa: F401
            import regex  # noqa: F401
            import numpy  # noqa: F401
            load_server()
            import strata_tokenizer  # noqa: F401
            print("MI50_SERVER_IMPORTS=PASS (no model loaded, no packages installed)")
        elif args.action == "check":
            text = args.log.read_text(encoding="utf-8", errors="replace")
            if " engine started: " in text:
                text = text.rsplit(" engine started: ", 1)[1]
            errors = startup_errors(text)
            if errors:
                raise ValueError("Startup gate failed: " + ", ".join(errors))
            print("MI50_STARTUP_GATE=PASS; graph execution and correctness require a separate authorized request.")
        else:
            return serve()
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        # Never dump the source config, environment, or API key.
        print(f"MI50 helper: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
