#!/usr/bin/env python3
"""Discover an existing RCCL C API in the pinned image. Never installs anything."""
from __future__ import annotations

import argparse
import ctypes
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

REQUIRED_SYMBOLS = (
    "ncclGetVersion", "ncclGetUniqueId", "ncclGetErrorString", "ncclCommInitRank",
    "ncclCommDestroy", "ncclCommAbort", "ncclCommGetAsyncError",
    "ncclGroupStart", "ncclGroupEnd", "ncclSend", "ncclRecv",
)
HEADER_NAMES = ("include/rccl/rccl.h", "include/rccl.h", "include/nccl.h")
LIB_DIRS = ("lib", "lib64", "lib/x86_64-linux-gnu")
CHECK_VERSION_MARKER = "__STRATA_RCCL_CHECK_VERSION_V1__"


def installed_roots() -> list[Path]:
    patterns = (
        "/opt/rocm", "/opt/rocm-*", "/usr/local", "/usr",
        "/opt/venv/lib/python*/site-packages/torch",
        "/usr/local/lib/python*/site-packages/torch",
        "/usr/lib/python*/site-packages/torch",
        "/usr/lib/python*/dist-packages/torch",
    )
    return unique_paths(Path(p) for pattern in patterns for p in sorted(glob.glob(pattern)))


def unique_paths(paths):
    result, seen = [], set()
    for path in paths:
        if path.exists():
            resolved = path.resolve(strict=True)
            if resolved not in seen:
                result.append(resolved)
                seen.add(resolved)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_library(path: str) -> int:
    """Executed in a fresh subprocess so each candidate gets its own loader state."""
    library = ctypes.CDLL(path, mode=os.RTLD_NOW | os.RTLD_LOCAL)
    for symbol in REQUIRED_SYMBOLS:
        getattr(library, symbol)
    get_version = library.ncclGetVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    version = ctypes.c_int()
    status = get_version(ctypes.byref(version))
    if status != 0 or version.value <= 0:
        raise RuntimeError(f"ncclGetVersion status={status} version={version.value}")
    return version.value


def parse_checked_version(stdout: str) -> int:
    """Accept exactly one complete helper record, never native diagnostic numbers."""
    records = [line for line in stdout.splitlines() if CHECK_VERSION_MARKER in line]
    if len(records) != 1:
        raise ValueError("Expected exactly one RCCL check version record")
    match = re.fullmatch(re.escape(CHECK_VERSION_MARKER) + r"=([1-9][0-9]*)", records[0])
    if match is None:
        raise ValueError("Malformed RCCL check version record")
    return int(match.group(1))


def log_check_output(stdout, stderr):
    """Retain native diagnostics on success, failure, and timeout."""
    for name, output, stream in (("STDOUT", stdout, sys.stdout), ("STDERR", stderr, sys.stderr)):
        if output:
            if isinstance(output, bytes):  # TimeoutExpired may retain bytes in text mode.
                output = output.decode(errors="replace")
            print(f"RCCL_PREFLIGHT_CHECK_{name}_BEGIN\n" + output +
                  ("" if output.endswith("\n") else "\n") +
                  f"RCCL_PREFLIGHT_CHECK_{name}_END", file=stream, flush=True)


def discover(roots: list[Path]) -> dict[str, str | int]:
    headers = unique_paths(root / name for root in roots for name in HEADER_NAMES)
    # torch/csrc/cuda/nccl.h is a C++ wrapper, not the RCCL C API header.
    headers = [header for header in headers if all(
        re.search(r"\b" + symbol + r"\s*\(", header.read_text(errors="replace"))
        for symbol in ("ncclGetVersion", "ncclSend", "ncclRecv")
    )]
    if not headers:
        raise RuntimeError("No installed RCCL C API header (rccl.h/nccl.h), including Torch bundles. No packages were installed.")
    directories = unique_paths(root / suffix for root in roots for suffix in LIB_DIRS)
    libraries = unique_paths(
        path for pattern in ("librccl.so*", "libnccl.so*")
        for directory in directories for path in sorted(directory.glob(pattern))
        if path.is_file()
    )
    if not libraries:
        raise RuntimeError("No installed RCCL shared library, including Torch bundles. No packages were installed.")
    print("RCCL_PREFLIGHT_HEADER_CANDIDATES " + json.dumps([str(p) for p in headers]), flush=True)
    print("RCCL_PREFLIGHT_LIBRARY_CANDIDATES " + json.dumps([str(p) for p in libraries]), flush=True)
    errors = []
    for library in libraries:
        loader_dirs = unique_paths([library.parent, *directories])
        loader_path = ":".join(str(p) for p in loader_dirs)
        previous = os.environ.get("LD_LIBRARY_PATH", "")
        if previous:
            loader_path += ":" + previous
        env = dict(os.environ, LD_LIBRARY_PATH=loader_path)
        try:
            deps = subprocess.run(["ldd", str(library)], env=env, text=True,
                                  capture_output=True, timeout=20, check=True)
            if "not found" in deps.stdout or "not found" in deps.stderr:
                raise RuntimeError("unresolved dependency: " + deps.stdout.strip())
            checked = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--check-library", str(library)],
                env=env, text=True, capture_output=True, timeout=20, check=False)
            log_check_output(checked.stdout, checked.stderr)
            checked.check_returncode()
            version = parse_checked_version(checked.stdout)
        except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                log_check_output(exc.stdout, exc.stderr)
            detail = getattr(exc, "stderr", None) or str(exc)
            errors.append(f"{library}: {detail.strip()}")
            print(f"RCCL_PREFLIGHT_REJECT library={library} reason={detail.strip()}", flush=True)
            continue
        # Prefer a header from the same installation. The executable also records
        # its compiled header version; compilation/runtime remain separate gates.
        matching_roots = [root for root in roots if root == library or root in library.parents]
        matching_roots.sort(key=lambda root: len(root.parts), reverse=True)
        header = headers[0]
        for root in matching_roots:
            same_root = [p for p in headers if root in p.parents]
            if same_root:
                header = same_root[0]
                break
        include = header.parent  # permits <rccl.h> or <nccl.h>, including Torch.
        print(f"RCCL_PREFLIGHT_LIBRARY path={library} sha256={sha256(library)}", flush=True)
        print(f"RCCL_PREFLIGHT_HEADER path={header} sha256={sha256(header)}", flush=True)
        print(f"RCCL_PREFLIGHT_VERSION value={version}", flush=True)
        print("RCCL_PREFLIGHT_REQUIRED_SYMBOLS " + " ".join(REQUIRED_SYMBOLS), flush=True)
        print("RCCL_PREFLIGHT_DEPENDENCIES_BEGIN\n" + deps.stdout.rstrip() +
              "\nRCCL_PREFLIGHT_DEPENDENCIES_END", flush=True)
        return {"RCCL_LIBRARY": str(library), "RCCL_INCLUDE_DIR": str(include), "RCCL_HEADER": str(header),
                "LD_LIBRARY_PATH": loader_path, "RCCL_RUNTIME_VERSION": version}
    raise RuntimeError("No loadable installed RCCL C API library. No packages were installed.\n" + "\n".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-library", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.check_library:
            version = check_library(args.check_library)
            # Native libraries may write unterminated or buffered diagnostics.
            # Give our record its own line and flush before native exit handlers.
            print(f"\n{CHECK_VERSION_MARKER}={version}", flush=True)
            return 0
        if args.output is None:
            parser.error("--output is required")
        result = discover(installed_roots())
        args.output.write_text("".join(f"export {key}={shlex.quote(str(value))}\n" for key, value in result.items()))
        return 0
    except (OSError, RuntimeError, AttributeError) as exc:
        print(f"RCCL_PREFLIGHT_BLOCKED {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
