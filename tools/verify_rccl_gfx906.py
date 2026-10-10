#!/usr/bin/env python3
"""CPU-only static gate for the isolated, pinned RCCL gfx906 build.

Never dlopen/execute RCCL or contact a GPU. LLVM utilities must already be present.
The source/image fields record the builder's required inputs, not a claim that
source provenance can be inferred from an ELF file. Supported fatbins contain one
complete Clang bundle (plain or compressed CCOB v2/v3); fail closed otherwise.
Format references: https://clang.llvm.org/docs/ClangOffloadBundler.html
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

SOURCE_COMMIT = '96a25b5fd6f73fba58c7d83eb57cf19a50230aa4'
BASE_IMAGE = 'sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca'
VERSION = '2.27.7'
KERNEL = 'ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)'
MAGIC = b'__CLANG_OFFLOAD_BUNDLE__'


class VerificationError(Exception):
    """Artifact could not be verified; do not produce a success manifest."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def contained_file(path, prefix):
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(prefix), f'File escapes isolated prefix: {path}')
    require(resolved.is_file(), f'Not a regular file: {path}')
    return resolved


def installed_files(prefix):
    prefix = prefix.resolve(strict=True)
    require(prefix.is_dir(), f'Not an install directory: {prefix}')
    # SONAME/development symlinks may all point at the same real library.
    candidates = list((prefix / 'lib').glob('librccl.so*'))
    require(candidates, 'Missing lib/librccl.so*')
    libraries = {contained_file(path, prefix) for path in candidates}
    require(len(libraries) == 1, 'Expected one unique real lib/librccl.so*')
    header = contained_file(prefix / 'include/rccl/rccl.h', prefix)
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', header.read_text(), flags=re.S)
    version = []
    for macro in ('NCCL_MAJOR', 'NCCL_MINOR', 'NCCL_PATCH'):
        values = re.findall(r'^\s*#\s*define\s+' + macro + r'\s+(\d+)\s*$', text, re.M)
        require(len(values) == 1, f'Missing/ambiguous integer {macro} in rccl.h')
        version.append(values[0])
    require('.'.join(version) == VERSION, f'Expected RCCL {VERSION}, found {".".join(version)}')
    return prefix, libraries.pop(), header


def find_tool(name):
    for directory in ('/opt/rocm/lib/llvm/bin', '/opt/rocm/llvm/bin'):
        preferred = Path(directory) / name
        if preferred.is_file() and os.access(preferred, os.X_OK):
            return str(preferred)
    found = shutil.which(name)
    require(found is not None, f'Missing required tool: {name} (ROCm LLVM bin or PATH)')
    return found


def run(executable, *arguments):
    try:
        result = subprocess.run([executable, *map(str, arguments)], check=False,
                                capture_output=True, text=True, timeout=300,
                                env=dict(os.environ, LC_ALL='C'))
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VerificationError(f'{Path(executable).name} failed: {exc}') from exc
    require(result.returncode == 0,
            f'{Path(executable).name} exited {result.returncode}: {result.stderr.strip()}')
    return result.stdout


def check_bundle_extent(path):
    """Reject unsupported/truncated bundles and hidden trailing payloads.

    LLVM does the decompression and target parsing. Checking the declared extent
    avoids silently verifying only the first of concatenated fatbin bundles.
    """
    size = path.stat().st_size
    with path.open('rb') as stream:
        head = stream.read(32)
        require(size >= 24, 'Missing or truncated .hip_fatbin')
        if head[:4] == b'CCOB':
            version = struct.unpack_from('<H', head, 4)[0]
            require(version in (2, 3), f'Unsupported compressed fatbin version: {version}')
            require(size >= (24 if version == 2 else 32), 'Truncated compressed fatbin header')
            total = struct.unpack_from('<I' if version == 2 else '<Q', head, 8)[0]
            require(total == size, 'Compressed fatbin extent mismatch/concatenated bundles')
            require(total > (24 if version == 2 else 32), 'Empty compressed fatbin')
            return None
        require(head.startswith(MAGIC), 'Unknown .hip_fatbin format')
        require(size >= len(MAGIC) + 8, 'Truncated plain fatbin header')
        count = struct.unpack_from('<Q', head, len(MAGIC))[0]
        require(0 < count <= 10000, 'Invalid offload bundle entry count')
        stream.seek(len(MAGIC) + 8)
        entries = []
        for _ in range(count):
            record = stream.read(24)
            require(len(record) == 24, 'Truncated offload entry header')
            offset, length, name_size = struct.unpack('<QQQ', record)
            require(0 < name_size <= 4096, 'Invalid offload target name length')
            name = stream.read(name_size)
            require(len(name) == name_size, 'Truncated offload target name')
            try:
                target = name.decode('ascii')
            except UnicodeError as exc:
                raise VerificationError('Non-ASCII offload target') from exc
            require(offset + length <= size, 'Offload entry exceeds fatbin extent')
            entries.append((offset, length, target))
        end_header = stream.tell()
        require(all(offset >= end_header for offset, _, _ in entries), 'Offload data overlaps header')
        ordered = sorted((offset, length) for offset, length, _ in entries if length)
        require(ordered and all(a + n <= b for (a, n), (b, _) in zip(ordered, ordered[1:])),
                'Overlapping/empty offload payloads')
        require(max(offset + length for offset, length, _ in entries) == size,
                'Unverified trailing payload/concatenated bundles')
        return [target for _, _, target in entries]


def parse_targets(output):
    targets = [line.strip() for line in output.splitlines() if line.strip()]
    require(targets, 'No offload targets found')
    require(len(set(targets)) == len(targets), 'Duplicate targets/concatenated fatbin bundles')
    gpu = []
    for target in targets:
        if re.fullmatch(r'host-[A-Za-z0-9_.-]+', target):
            continue
        require(re.fullmatch(r'hip(?:v[0-9]+)?-amdgcn-amd-amdhsa--gfx906(?::(?:sramecc|xnack)[+-])*', target),
                f'Unexpected GPU/offload target (gfx906 only): {target}')
        features = target.split(':')[1:]
        require(len({feature[:-1] for feature in features}) == len(features),
                f'Duplicate target feature: {target}')
        gpu.append(target)
    require(gpu, 'No gfx906 GPU target found')
    return sorted(targets), sorted(gpu)


def device_elf(path):
    with path.open('rb') as stream:
        header = stream.read(64)
    require(len(header) == 64 and header[:6] == b'\x7fELF\x02\x01',
            'Unbundled device object is not little-endian ELF64')
    require(header[7] == 64, 'Device ELF OSABI is not AMDGPU HSA')
    require(header[8] in (2, 3, 4), 'Unsupported HSA code-object ABI (expected V4/V5/V6)')
    require(header[6] == 1 and struct.unpack_from('<I', header, 20)[0] == 1,
            'Invalid ELF header version')
    require(struct.unpack_from('<H', header, 16)[0] == 3,
            'Device ELF must be a linked ET_DYN HSA codeobject')
    require(struct.unpack_from('<H', header, 18)[0] == 224,
            'Unbundled object is not AMDGPU (host symbols do not count)')
    flags = struct.unpack_from('<I', header, 48)[0]
    # https://llvm.org/docs/AMDGPUUsage.html: EF_AMDGPU_MACH mask / gfx906.
    require(flags & 0xff == 0x2f, 'AMDGPU ELF machine flags are not gfx906')
    return {'elf_machine': 224, 'elf_flags': hex(flags),
            'elf_osabi': header[7], 'code_object_version': header[8] + 2}


def symbols(output):
    result = {}
    table = None
    for line in output.splitlines():
        match = re.match(r"Symbol table '([^']+)'", line)
        if match:
            table = match.group(1)
            continue
        match = re.match(r'^\s*(\d+):\s+([0-9a-fA-F]+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*(.*)$', line)
        if match:
            require(table is not None, 'Symbol row without symbol-table heading')
            number, value, size, kind, bind, visibility, index, name = match.groups()
            key = (table, number)
            require(key not in result, 'Duplicate ELF symbol table row')
            result[key] = (value, int(size), kind, bind, visibility, index, name)
    require(result, 'Could not read ELF symbols')
    return result


def kernel_symbols(raw_output, demangled_output):
    raw, demangled = symbols(raw_output), symbols(demangled_output)
    require(raw.keys() == demangled.keys(), 'Inconsistent raw/demangled symbol tables')
    for key in raw:
        require(raw[key][:-1] == demangled[key][:-1], 'Inconsistent raw/demangled symbol row')
    for key, entry in demangled.items():
        _, size, kind, _, _, index, name = entry
        if name != KERNEL or kind != 'FUNC' or not index.isdecimal() or int(index) == 0 or size <= 0:
            continue
        mangled = raw[key][-1]
        descriptor = mangled + '.kd'
        if any(row[2] == 'OBJECT' and row[1] > 0 and row[5].isdecimal()
               and int(row[5]) > 0 and row[-1] == descriptor for row in raw.values()):
            return {'generic_kernel': name, 'generic_kernel_mangled': mangled,
                    'kernel_descriptor': descriptor}
    raise VerificationError('Missing defined device FUNC ' + KERNEL + ' with defined OBJECT .kd descriptor')


def verify(prefix):
    prefix, library, header = installed_files(prefix)
    library_hash, header_hash = sha256(library), sha256(header)
    objcopy, bundler, readelf = (find_tool(name) for name in
                               ('llvm-objcopy', 'clang-offload-bundler', 'llvm-readelf'))
    with tempfile.TemporaryDirectory(prefix='rccl-gfx906-static-') as scratch:
        temp = Path(scratch)
        fatbin = temp / 'rccl.hip_fatbin'
        # An explicit output keeps llvm-objcopy from rewriting the input library.
        run(objcopy, '--dump-section', f'.hip_fatbin={fatbin}', library, temp / 'host-copy.so')
        require(fatbin.is_file(), 'llvm-objcopy did not extract .hip_fatbin')
        declared = check_bundle_extent(fatbin)
        targets, gpu_targets = parse_targets(run(bundler, '-type=o', f'-input={fatbin}', '-list'))
        if declared is not None:
            require(sorted(declared) == targets, 'Bundler target list differs from complete fatbin header')
        objects = []
        for number, target in enumerate(gpu_targets):
            codeobject = temp / f'device-{number}.co'
            run(bundler, '-type=o', f'-input={fatbin}', f'-output={codeobject}',
                f'-targets={target}', '-unbundle')
            require(codeobject.is_file(), f'Unbundler did not create codeobject: {target}')
            elf_identity = device_elf(codeobject)
            raw = run(readelf, '--symbols', '--wide', codeobject)
            demangled = run(readelf, '--symbols', '--wide', '--demangle', codeobject)
            found = kernel_symbols(raw, demangled)
            objects.append({'target': target, 'sha256': sha256(codeobject),
                            **elf_identity, **found})
        fatbin_hash = sha256(fatbin)
    require(installed_files(prefix) == (prefix, library, header),
            'Installed library/header selection changed during verification')
    require(sha256(library) == library_hash and sha256(header) == header_hash,
            'Installed library/header changed during verification')
    return {'schema_version': 1, 'source_commit': SOURCE_COMMIT, 'base_image': BASE_IMAGE,
            'rccl_version': VERSION, 'verification': 'static-only; no GPU execution',
            'library': {'path': library.relative_to(prefix).as_posix(), 'sha256': library_hash},
            'header': {'path': header.relative_to(prefix).as_posix(), 'sha256': header_hash},
            'fatbin_sha256': fatbin_hash, 'targets': targets, 'codeobjects': objects}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-manifest', type=Path,
                        help='Require exact stable identity match before writing output')
    args = parser.parse_args(argv)
    try:
        # Never allow the manifest output to overwrite files being verified.
        prefix, library, header = installed_files(args.prefix)
        require(args.output.resolve() not in (library, header), 'Output would overwrite an artifact')
        manifest = verify(prefix)
        if args.check_manifest:
            expected = json.loads(args.check_manifest.read_text())
            require(expected == manifest, 'Artifact identity does not match --check-manifest')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', dir=args.output.parent, prefix='.rccl-manifest-',
                                         delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(manifest, stream, indent=2, sort_keys=True)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, args.output)
            finally:
                temporary.unlink(missing_ok=True)
        print(f'RCCL_GFX906_STATIC_VERIFIED manifest={args.output} gpu_tested=0')
        return 0
    except (VerificationError, OSError, ValueError) as exc:
        print(f'RCCL_GFX906_STATIC_REJECTED: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
