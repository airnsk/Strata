# Inspect the RCCL HIP host crash without another GPU run

For the October 10, 2026, 20:15:03 UTC stand crash, the kernel reported two host threads
at `0x701be2ec5634`, in the `libamdhip64.so.7.2.70204` mapping beginning at
`0x701be2ab0000` with length `0x489000`. The mapping-relative offset is `0x415634`.
That offset is **not necessarily the ELF address** accepted by `addr2line`.

From the same stand checkout, run:

```bash
bash tools/inspect_rccl_hip_crash.sh | tee hip-crash-inspection.txt
```

The inspector uses only the already-installed pinned image. Its container has no
GPU devices, network, or host mounts; its root filesystem is read-only. It reads
library bytes without loading the HIP runtime. It does not run the probe, install
anything, or change core settings. Interactive sudo authorization happens on the
terminal before timed operations; later sudo commands are noninteractive.

Output includes the resolved HIP library path, SHA-256, build ID when available,
program headers, and executable `PT_LOAD` candidates. For each candidate it uses
`load_bias = VMA_start - page_down(p_vaddr)` and `ELF_PC = IP - load_bias`.
The file offset is `p_offset + ELF_PC - p_vaddr`. A candidate is symbolized only if
its page-aligned executable extent matches the reported mapping length and its
nine fault-site bytes match `48 8b 47 20 48 83 c0 38 c3`. Multiple matching candidates
remain ambiguous; no candidate matching is a blocker, not permission to use the
raw offset. Missing debug symbols can leave line information unavailable.

The script also prints the host timezone and core pattern, and at most 20
`/var/crash` filenames, sizes, and modification times from 20:10–20:25 UTC, matching
the user's journal query. The filter is independent of the stand's local timezone.
It never reads Apport report or core contents. No matching filename does not
establish that no core exists.

The faulting instruction is `mov rax, [rdi + 0x20]`. Given the reported fault
address `0x43020`, `rdi = 0x43000` is the expected register value if these are the
correct fault-site bytes. The kernel record alone does not supply the register
set, runtime caller, stack, or the cause of that pointer value. Symbolization
names a candidate crash site; a saved core/backtrace and matching mapped-object
identity are needed to establish who called it. Do not conclude use-after-free,
RCCL incompatibility, or a graph bug from this record alone.

CPU-only validation:

```bash
bash -n tools/inspect_rccl_hip_crash.sh
python3 tools/test_inspect_rccl_hip_crash.py
```

These tests use synthetic ELF segments and mocked symbol tools. They validate
address arithmetic and output gates, not the pinned image or stand's actual
library. No Docker or HIP runtime was available in the cloud checkout used to
prepare this inspector; its real symbol output remains to be collected.
