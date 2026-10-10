#!/usr/bin/env python3
"""CPU-only verifier tests; mock LLVM output, never use HIP, Docker or a GPU."""
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import verify_rccl_gfx906 as verifier

TARGET = 'hipv4-amdgcn-amd-amdhsa--gfx906'
HOST = 'host-x86_64-unknown-linux-gnu-'
MANGLED = '_Z23ncclDevKernel_Generic_424ncclDevKernelArgsStorageILm4096EE'


def elf(machine=224, flags=0x2f, osabi=64, abi=3, elf_type=3):
    data = bytearray(64)
    data[:9] = b'\x7fELF\x02\x01\x01' + bytes([osabi, abi])
    struct.pack_into('<H', data, 16, elf_type)
    struct.pack_into('<I', data, 20, 1)
    struct.pack_into('<H', data, 18, machine)
    struct.pack_into('<I', data, 48, flags)
    return bytes(data)


def bundle(targets=(HOST, TARGET)):
    header_size = len(verifier.MAGIC) + 8 + sum(24 + len(t) for t in targets)
    data = bytearray(verifier.MAGIC + struct.pack('<Q', len(targets)))
    position = header_size
    payload = bytearray()
    for target in targets:
        content = b'' if target.startswith('host-') else elf()
        data.extend(struct.pack('<QQQ', position, len(content), len(target)))
        data.extend(target.encode())
        payload.extend(content)
        position += len(content)
    return bytes(data + payload)


def symbol_table(demangled=False, function_type='FUNC', function_index='7',
                 descriptor=True, descriptor_index='8', function_name=None):
    name = function_name or (verifier.KERNEL if demangled else MANGLED)
    result = "Symbol table '.symtab' contains 3 entries:\n"
    result += '   Num:    Value          Size Type    Bind   Vis      Ndx Name\n'
    result += '     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND \n'
    result += f'     1: 0000000000000100   320 {function_type} GLOBAL DEFAULT {function_index} {name}\n'
    if descriptor:
        result += f'     2: 0000000000000200    64 OBJECT GLOBAL DEFAULT {descriptor_index} {MANGLED}.kd\n'
    return result


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefix = self.root / 'install'
        (self.prefix / 'lib').mkdir(parents=True)
        (self.prefix / 'include/rccl').mkdir(parents=True)
        self.library = self.prefix / 'lib/librccl.so.1.0.0'
        self.library.write_bytes(b'RCCL host fixture')
        (self.prefix / 'lib/librccl.so').symlink_to(self.library.name)
        (self.prefix / 'lib/librccl.so.1').symlink_to(self.library.name)
        self.header = self.prefix / 'include/rccl/rccl.h'
        self.header.write_text('#define NCCL_MAJOR 2\n#define NCCL_MINOR 27\n#define NCCL_PATCH 7\n')
        self.commands = []
        self.target_output = HOST + '\n' + TARGET + '\n'
        self.fatbin = bundle()
        self.codeobject = elf()
        self.symbol_options = {}

    def fake_run(self, tool, *args):
        args = tuple(map(str, args))
        self.commands.append((tool, args))
        if tool == 'llvm-objcopy':
            self.assertEqual(args[0], '--dump-section')
            self.assertEqual(args[2], str(self.library))
            self.assertNotEqual(args[3], str(self.library))
            Path(args[1].split('=', 1)[1]).write_bytes(self.fatbin)
            return ''
        if tool == 'clang-offload-bundler':
            self.assertIn('-type=o', args)
            if '-list' in args:
                return self.target_output
            self.assertIn('-unbundle', args)
            self.assertIn('-targets=' + TARGET, args)
            Path(next(a.split('=', 1)[1] for a in args if a.startswith('-output='))).write_bytes(self.codeobject)
            return ''
        self.assertEqual(tool, 'llvm-readelf')
        self.assertIn('--symbols', args)
        self.assertIn('--wide', args)
        return symbol_table('--demangle' in args, **self.symbol_options)

    def verify(self):
        with mock.patch.object(verifier, 'find_tool', side_effect=lambda name: name), \
             mock.patch.object(verifier, 'run', side_effect=self.fake_run):
            return verifier.verify(self.prefix)

    def cli(self, *extra):
        with mock.patch.object(verifier, 'find_tool', side_effect=lambda name: name), \
             mock.patch.object(verifier, 'run', side_effect=self.fake_run), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return verifier.main(['--prefix', str(self.prefix), '--output', str(self.root / 'manifest.json'), *extra])

    def test_good_install_manifest_and_identity_check(self):
        self.assertEqual(self.cli(), 0)
        path = self.root / 'manifest.json'
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest['targets'], sorted([HOST, TARGET]))
        self.assertEqual(manifest['rccl_version'], '2.27.7')
        self.assertEqual(manifest['library']['path'], 'lib/librccl.so.1.0.0')
        self.assertEqual(manifest['library']['sha256'], verifier.sha256(self.library))
        self.assertEqual(manifest['header']['sha256'], verifier.sha256(self.header))
        self.assertEqual(manifest['source_commit'], verifier.SOURCE_COMMIT)
        self.assertEqual(manifest['base_image'], verifier.BASE_IMAGE)
        self.assertEqual(manifest['codeobjects'][0]['generic_kernel'], verifier.KERNEL)
        self.assertEqual(manifest['codeobjects'][0]['kernel_descriptor'], MANGLED + '.kd')
        self.assertEqual(manifest['codeobjects'][0]['elf_flags'], '0x2f')
        self.assertEqual(self.cli('--check-manifest', str(path)), 0)
        self.library.write_bytes(b'changed host library')
        saved = path.read_bytes()
        self.assertEqual(self.cli('--check-manifest', str(path)), 1)
        self.assertEqual(path.read_bytes(), saved)

    def test_reject_multiple_real_libraries(self):
        (self.prefix / 'lib/librccl.so.extra').write_bytes(b'another')
        with self.assertRaisesRegex(verifier.VerificationError, 'one unique'):
            self.verify()

    def test_reject_escape_symlink(self):
        self.library.unlink()
        outside = self.root / 'outside.so'
        outside.write_bytes(b'outside')
        self.library.symlink_to(outside)
        with self.assertRaisesRegex(verifier.VerificationError, 'escapes'):
            self.verify()

    def test_reject_wrong_or_duplicate_version(self):
        for header in ('#define NCCL_MAJOR 2\n#define NCCL_MINOR 27\n#define NCCL_PATCH 6\n',
                       self.header.read_text() + '#define NCCL_PATCH 7\n'):
            with self.subTest(header=header):
                self.header.write_text(header)
                self.assertEqual(self.cli(), 1)
                self.assertFalse((self.root / 'manifest.json').exists())

    def test_reject_mixed_host_only_duplicate_and_unknown_targets(self):
        for output in (TARGET + '\n' + TARGET.replace('gfx906', 'gfx908'), HOST,
                       TARGET + '\n' + TARGET, TARGET + '\nopenmp-amdgcn-amd-amdhsa--gfx906', ''):
            with self.subTest(output=output), self.assertRaises(verifier.VerificationError):
                verifier.parse_targets(output)

    def test_allow_features_but_not_arch_prefix_collision(self):
        featured = TARGET + ':sramecc+:xnack-'
        self.assertEqual(verifier.parse_targets(HOST + '\n' + featured)[1], [featured])
        for target in (TARGET + '0', TARGET + ':xnack+:xnack-', TARGET + ':anything+'):
            with self.subTest(target=target), self.assertRaises(verifier.VerificationError):
                verifier.parse_targets(target)

    def test_reject_hidden_or_truncated_bundle(self):
        for content in (self.fatbin[:-1], self.fatbin + bundle(), b'unknown' * 10):
            with self.subTest(size=len(content)):
                self.fatbin = content
                self.assertEqual(self.cli(), 1)

    def test_compressed_v2_v3_extent(self):
        path = self.root / 'fatbin'
        for version in (2, 3):
            header = (struct.pack('<4sHHIIQ', b'CCOB', version, 1, 25, 64, 0) if version == 2 else
                      struct.pack('<4sHHQQQ', b'CCOB', version, 1, 33, 64, 0))
            path.write_bytes(header + b'x')
            self.assertIsNone(verifier.check_bundle_extent(path))
            path.write_bytes(path.read_bytes() + b'garbage')
            with self.assertRaisesRegex(verifier.VerificationError, 'extent'):
                verifier.check_bundle_extent(path)

    def test_reject_target_list_not_matching_plain_bundle(self):
        self.target_output = TARGET + '\n'
        with self.assertRaisesRegex(verifier.VerificationError, 'differs'):
            self.verify()

    def test_reject_host_elf_or_other_amdgpu_architecture(self):
        for codeobject in (elf(machine=62), elf(flags=0x30), elf(osabi=0), elf(abi=0),
                           elf(abi=255), elf(elf_type=1), b'not an ELF'):
            with self.subTest(codeobject=codeobject):
                self.codeobject = codeobject
                self.assertEqual(self.cli(), 1)

    def test_reject_undefined_host_data_stub_or_missing_descriptor(self):
        for options in ({'function_index': 'UND'}, {'function_type': 'OBJECT'},
                        {'function_name': '__device_stub__' + verifier.KERNEL},
                        {'descriptor': False}, {'descriptor_index': 'UND'}):
            with self.subTest(options=options):
                self.symbol_options = options
                self.assertEqual(self.cli(), 1)

    def test_reject_artifact_mutation_during_verification(self):
        original = self.fake_run

        def mutate(tool, *args):
            result = original(tool, *args)
            if tool == 'llvm-readelf':
                self.library.write_bytes(b'changed during verification')
            return result

        with mock.patch.object(verifier, 'find_tool', side_effect=lambda name: name), \
             mock.patch.object(verifier, 'run', side_effect=mutate):
            with self.assertRaisesRegex(verifier.VerificationError, 'changed during verification'):
                verifier.verify(self.prefix)

    def test_reject_raw_demangled_table_mismatch(self):
        with self.assertRaisesRegex(verifier.VerificationError, 'Inconsistent'):
            verifier.kernel_symbols(symbol_table(), symbol_table(True).replace('320 FUNC', '321 FUNC'))

    def test_reject_failed_extraction_without_manifest(self):
        original = self.fake_run

        def omit_output(tool, *args):
            if tool == 'llvm-objcopy':
                return ''
            return original(tool, *args)

        with mock.patch.object(verifier, 'find_tool', side_effect=lambda name: name), \
             mock.patch.object(verifier, 'run', side_effect=omit_output):
            with self.assertRaisesRegex(verifier.VerificationError, 'did not extract'):
                verifier.verify(self.prefix)

    def test_missing_dependency_fails_closed(self):
        with mock.patch.object(verifier.shutil, 'which', return_value=None), \
             mock.patch.object(verifier.Path, 'is_file', return_value=False), \
             self.assertRaisesRegex(verifier.VerificationError, 'Missing required tool'):
            verifier.find_tool('llvm-readelf')

    def test_subprocess_failure_and_timeout(self):
        with mock.patch.object(verifier.subprocess, 'run', return_value=mock.Mock(returncode=1, stderr='bad input')):
            with self.assertRaisesRegex(verifier.VerificationError, 'exited 1'):
                verifier.run('llvm-readelf', 'fixture')
        with mock.patch.object(verifier.subprocess, 'run', side_effect=verifier.subprocess.TimeoutExpired('tool', 300)):
            with self.assertRaisesRegex(verifier.VerificationError, 'failed'):
                verifier.run('llvm-readelf', 'fixture')

    def test_output_cannot_overwrite_artifact(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(verifier.main(['--prefix', str(self.prefix), '--output', str(self.library)]), 1)
        self.assertEqual(self.library.read_bytes(), b'RCCL host fixture')


if __name__ == '__main__':
    unittest.main()
