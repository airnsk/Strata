# Capture the first RCCL HIP host segfault

Run this once on the MI50 stand, with both GPUs idle:

```bash
bash tools/run_tp2_rccl_debug_mi50.sh
```

It saves `tp2-rccl-debug-<UTC timestamp>-<pid>.log` in the checkout. The original
full-suite runner is unchanged. No model is loaded.

## Scope

The diagnostic build retains the original `-O2` compilation, initialization,
persistent workers, and first warmup: T=8, bank 0, mode=sequence. Y is its first
seam, followed by output and FFN if execution reaches them. `--tokens 1` would
not change this original warmup, so the runner explicitly requests T=8.

A compile-time-only guard exits after one warmup and its existing verification.
It cannot proceed into graph capture, timing sweeps, a second lifecycle, delayed
rank, or missing rank. The ordinary probe build does not define this guard.
Host-only debug information uses `-Xarch_host -g`; device code does not receive
`-g`. Clang documents [host/device argument selection](https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang-Xarch_host).

## Debugger and permissions

The runner chooses an already-installed `rocgdb`, falling back to an installed
`gdb` only if `rocgdb` is absent. It does not install or download anything.
A CPU-only `/bin/true` debugger check must pass before the probe is built or run.
Missing tools, blocked ptrace, or unsupported commands stop the attempt.

The existing pinned image runs with a read-only root filesystem, network disabled,
a read-only checkout, temporary build files, and the same two GPU device mounts.
It drops all capabilities and adds only container-scoped `SYS_PTRACE`. Docker's
default seccomp and host security settings remain unchanged. There is no
`--privileged`, host PID namespace, seccomp-unconfined setting, or automatic
permission escalation. See [Docker's seccomp guidance](https://docs.docker.com/engine/security/seccomp/).
Sudo authentication happens before captured output and timed operations.

## Collected evidence and limits

The debugger runs in all-stop mode, without user init files, automatic script
loading, debuginfod downloads, or disabled ASLR. Its first SIGSEGV catchpoint
prints the faulting thread, `rip/rdi/rsp/rbp/rax`, nine instruction bytes, eight
stack words, the current stack, process mappings, loaded libraries, and up to
24 frames per thread. At the known leaf getter, the first stack word is the
immediate return-address slot. It then quits
and kills the inferior; it never continues the faulting instruction. Library
identity and mappings allow offline symbolization without uploading debug files
to the stand. See [GDB backtraces](https://sourceware.org/gdb/current/onlinedocs/gdb.html/Backtrace.html).

A capture-command error produces `capture_partial=1` and exit 87 while retaining
useful output. Optimized or stripped code can still leave some frames unresolved.
A successful warmup without SIGSEGV is reported separately. Neither outcome is
a full probe pass, performance result, or proof of the underlying cause. A
watchdog or another signal without the warmup completion marker is incomplete.

The container has a 180-second outer deadline, including setup and compilation;
the debugger attempt is separately limited to 120 seconds. The wrapper then
performs bounded removal and verifies that this attempt's container is absent.
No second probe attempt is made. After a crash or timeout, inspect the GPU state
before choosing whether to run anything else.

CPU-only checks:

```bash
bash -n tools/run_tp2_rccl_debug_mi50.sh
python3 tools/test_tp2_rccl_debug_runner.py
python3 tools/test_tp2_rccl_runner.py
```

The diagnostic runner tests mock Docker, HIP, and the debugger. This cloud
checkout has no HIP compiler, debugger, or GPU; real HIP compilation, ptrace, and
signal capture remain stand-only checks. The shared CPU oracle also passed its
811,008 word checks; that is not a GPU result.
