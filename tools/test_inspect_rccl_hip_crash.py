#!/usr/bin/env python3
"""CPU-only checks for the crash inspector's ELF address mapping; no Docker/GPU."""
import contextlib
import io
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SCRIPT = Path(__file__).with_name('inspect_rccl_hip_crash.sh')
SOURCE = SCRIPT.read_text().split("<<'PY'\n", 1)[1].rsplit('\nPY\n', 1)[0]
EXPECTED = bytes.fromhex('48 8b 47 20 48 83 c0 38 c3')


class InspectorTests(unittest.TestCase):
    def run_fixture(self, *, matches=True, span_matches=True, multiple=False):
        with tempfile.TemporaryDirectory() as directory:
            lib = Path(directory) / 'libamdhip64.so.7.2.70204'
            # Executable segment starts inside a page; its virtual address and
            # file offset differ. Raw ip-vma is deliberately the wrong ELF PC.
            segments = [(0x1345, 0x2345)]
            if multiple:
                segments.append((0x501345, 0x902345))
            size = 0x488cbb if span_matches else 0x487cbb
            with lib.open('wb') as f:
                ident = b'\x7fELF\x02\x01' + b'\0' * 10
                f.write(struct.pack('<16sHHIQQQIHHHHHH', ident, 3, 62, 1, 0, 64,
                                    0, 0, 64, 56, len(segments), 0, 0, 0))
                for offset, vaddr in segments:
                    f.write(struct.pack('<IIQQQQQQ', 1, 5, offset, vaddr, 0,
                                        size, size, 0x1000))
                for offset, vaddr in segments:
                    pc = vaddr // 4096 * 4096 + 0x415634
                    f.seek(offset + pc - vaddr)
                    f.write(EXPECTED if matches else bytes(len(EXPECTED)))
            source = SOURCE.replace('/opt/rocm-7.2.4/lib/libamdhip64.so.7', str(lib))
            output, calls, error = io.StringIO(), [], None

            def run(args, **kwargs):
                calls.append(args)
                text = '00417600 T fixture_function\n00418000 T next_function\n' if 'nm' in args[0] else ''
                return SimpleNamespace(stdout=text, stderr='', returncode=0)

            with contextlib.redirect_stdout(output), mock.patch('shutil.which', side_effect=lambda name: name), \
                    mock.patch('subprocess.run', side_effect=run), mock.patch('os.sysconf', return_value=4096):
                try:
                    exec(compile(source, str(SCRIPT), 'exec'), {})
                except SystemExit as exc:
                    error = str(exc)
            return output.getvalue(), calls, error

    def test_page_aligned_load_bias_not_raw_delta(self):
        output, calls, error = self.run_fixture()
        self.assertIsNone(error)
        self.assertIn('delta=0x415634', output)
        self.assertIn('elf_ip=0x417634 file_offset=0x416634', output)
        self.assertIn('span_match=1', output)
        self.assertIn('bytes_match=1', output)
        addr_calls = [args for args in calls if 'addr2line' in args[0]]
        self.assertEqual(addr_calls[0][-1], '0x417634')
        self.assertNotEqual(addr_calls[0][-1], '0x415634')
        self.assertIn('do not establish the runtime caller', output)

    def test_mismatched_bytes_are_never_symbolized(self):
        output, calls, error = self.run_fixture(matches=False)
        self.assertIn('No candidate matches', error)
        self.assertIn('HIP_INSPECT_REJECT', output)
        self.assertFalse(any('addr2line' in args[0] or 'objdump' in args[0] for args in calls))

    def test_different_vma_span_is_visible(self):
        output, calls, error = self.run_fixture(span_matches=False)
        self.assertIn('No candidate matches', error)
        self.assertIn('span_match=0', output)
        self.assertIn('bytes_match=1 mapping_match=0', output)
        self.assertFalse(any('addr2line' in args[0] for args in calls))

    def test_multiple_matches_remain_candidates(self):
        output, calls, error = self.run_fixture(multiple=True)
        self.assertIsNone(error)
        self.assertIn('HIP_INSPECT_MATCHING_CANDIDATES 2', output)
        self.assertEqual(len([c for c in calls if 'addr2line' in c[0]]), 2)

    def test_shell_syntax_help_and_container_limits(self):
        subprocess.run(['bash', '-n', str(SCRIPT)], check=True)
        result = subprocess.run(['bash', str(SCRIPT), '--help'], check=True, text=True, capture_output=True)
        self.assertIn('No downloads, GPU devices, probe rerun', result.stdout)
        shell = SCRIPT.read_text().split("<<'PY'\n", 1)[0]
        launch = shell.split('120s "${PRIV[@]}" docker run', 1)[1]
        for option in ('--pull never', '--network none', '--read-only', '--cap-drop ALL'):
            self.assertIn(option, launch)
        for option in ('--device', '--privileged', '--mount', 'no-new-privileges'):
            self.assertNotIn(option, launch)
        self.assertIn("-newermt '2026-10-10 20:10:00 UTC'", shell)
        self.assertIn("! -newermt '2026-10-10 20:25:00 UTC'", shell)
        self.assertIn('utc_window=2026-10-10T20:10:00Z..20:25:00Z', shell)
        self.assertIn('PRIV=(sudo -n)', shell)
        self.assertNotIn('timeout sudo -v', shell)


if __name__ == '__main__':
    unittest.main()
