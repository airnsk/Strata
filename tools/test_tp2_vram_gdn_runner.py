#!/usr/bin/env python3
"""CPU-only admission and runner tests. No GPU, Docker daemon or model is used."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
import time

import tp2_vram_gdn_admission as admission

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'tools/run_tp2_vram_gdn_mi50.sh'
ADMISSION = ROOT / 'tools/tp2_vram_gdn_admission.py'
CPP = (ROOT / 'tests/hip/tp2_gdn_layer.cpp').read_text()
SOURCE = RUNNER.read_text()
INNER = SOURCE.split("<<'VRAM_GDN_CONTAINER'\n", 1)[1].split('\nVRAM_GDN_CONTAINER\n', 1)[0]
SUITE = 'VRAM_GDN_SUITE_PASS'


def cpp_format(prefix, *args):
    """Use real executable format strings, rather than a mock-only contract."""
    formats = re.findall(r'std::printf\("((?:[^"\\]|\\.)*)"', CPP)
    selected = [s for s in formats if s.startswith(prefix)]
    if len(selected) != 1:
        raise AssertionError('ambiguous/missing actual C++ format: ' + prefix)
    fmt = bytes(selected[0], 'utf8').decode('unicode_escape').replace('%zu', '%d').replace('%llu', '%d').replace('%016llx', '%016x')
    return (fmt % args).rstrip('\n')


def startup():
    lines = []
    for half in ('attn', 'ffn'):
        for part, kind, type_id, shape in (('norm', 'F32', 0, '10240'), ('down', 'BF16', 30, '10240x320'),
                                          ('up', 'BF16', 30, '320x10240'), ('inject', 'BF16', 30, '10240x4')):
            role = f'hc_{half}_{part}.weight'
            lines += [cpp_format('HC_GGUF_SELECTED ', 0, role, 1) + f' shard-index=0 type={kind} type-id={type_id} shape={shape}',
                      cpp_format('HC_GGUF_COUNTS ', role, kind, type_id, 48), cpp_format('HC_GGUF_COVERAGE ', role, 0, 0)]
    for d in (0, 1):
        lines.append(cpp_format('VRAM_LAYER_GPU ', d, d, f'0000:0{d+3}:00.0', 'gfx906:sramecc+:xnack-', 70200, 70200))
    for flag in admission.ENV:
        value = '0' if flag in {'STRATA_HC_PERSIST', 'STRATA_GR_V3', 'STRATA_GR_SPLIT'} else 'unset'
        lines.append(cpp_format('HC_DISPATCH_ENV ', flag, value))
    for d in (0, 1):
        lines.append(cpp_format('HC_DISPATCH_SELECTED ', 'production-check', d, 3, 'staged', 1))
    lines.append(cpp_format('HC_DISPATCH_INIT_PASS '))
    return lines


def fault(scenario, first):
    lines = startup()
    lines.append(cpp_format('VRAM_LAYER_FAULT_ARMED ', scenario, first, 1-first))
    if scenario == 'missing-rank':
        lines += ['PASS VRAM missing peer rejects proposal', 'PASS VRAM failed snapshot unavailable',
                  cpp_format('VRAM_LAYER_FAULT_PASS ', first, 1-first)]
    else:
        lines += ['VRAM_LAYER_EXACT_OWNER rank=0', 'PASS VRAM_LAYER exact residual n=1/1 different=0',
                  'VRAM_LAYER_EXACT_OWNER rank=1', cpp_format('VRAM_LAYER_DELAY_PASS ', first, 1-first, 20)]
    lines += [cpp_format('VRAM_LAYER_RECOVERY_PASS ', scenario, first), cpp_format('VRAM_LAYER_FAULT_GATE_PASS ', scenario, first)]
    return lines


def study(family, baseline):
    result = [cpp_format(family + '_STUDY ', baseline, admission.CANDIDATE)]
    if family == 'BENCH':
        result.append(cpp_format('BENCH_CONFIG ', 3, 4))
    else:
        result.append(cpp_format('BLOCK_METHOD ', 16, 4, 16))
    for t in admission.TOKENS:
        result.append(cpp_format('VRAM_LAYER_TRANSPORT '))
        if family == 'BLOCK':
            result.append(cpp_format('VRAM_MATCHED_FIXTURE family=block ', baseline, admission.CANDIDATE, t, 16, t, 11, 22))
            for step in range(16):
                result.append(cpp_format('BLOCK_SHADOW ', 'hybrid-vram', t, step))
        for trial in range(-3 if family == 'BENCH' else 0, 4):
            if family == 'BENCH': result.append(cpp_format('VRAM_MATCHED_FIXTURE family=burst ', baseline, admission.CANDIDATE, t, trial, t*100+trial, 11, 22))
            for leg in (0, 1):
                first = admission.CANDIDATE if (trial+leg+(3 if family == 'BENCH' else 0)) % 2 else baseline
                if family == 'BENCH':
                    result.append(cpp_format('BENCH_%s ', 'WARMUP' if trial < 0 else 'SAMPLE', baseline, admission.CANDIDATE,
                                             'hybrid-vram', t, trial, leg, first, 1., 1., 2., 1., 1., 2.))
                else:
                    result.append(cpp_format('BLOCK_SAMPLE ', baseline, admission.CANDIDATE, 'hybrid-vram', t, trial, leg, first, 16, 32., 32., 2., 2.))
            if trial >= 0:
                result.append(cpp_format(family + '_CROSSOVER ', baseline, admission.CANDIDATE, 'hybrid-vram', t, trial, 2., 2., 1.))
        if family == 'BENCH':
            result.append(cpp_format('BENCH_SUMMARY ', baseline, admission.CANDIDATE, 'hybrid-vram', t, 8,
                                     2., 2., 2., 2., 2., 2., 1., 1., 1., 1., 1., 1.))
        else:
            result.append(cpp_format('BLOCK_SUMMARY ', baseline, admission.CANDIDATE, 'hybrid-vram', t, 8, 16, 2., 2., 1., 1.))
    return result


def normal():
    lines = startup()
    lines += [cpp_format('VRAM_LAYER_LIFECYCLE_PASS ', life) for life in (0, 1)]
    for first in (0, 1):
        for t in range(1, 9):
            for bank in (0, 1):
                lines += [cpp_format('VRAM_LAYER_BANK_CASE ', first, t, bank), 'VRAM_LAYER_EXACT_OWNER rank=0',
                          'PASS VRAM_LAYER exact residual n=1/1 different=0', 'VRAM_LAYER_EXACT_OWNER rank=1']
    lines.append(cpp_format('VRAM_LAYER_BANK_GATE_PASS '))
    epoch = 0
    for mode in ('hybrid', 'hybrid-vram'):
        lines += [f'CORRECTNESS_EXECUTION {mode} (reference=runtime)']
        for t in range(1, 9):
            for keep in range(t+1):
                epoch += 2
                lines += [f'CASE T={t} keep={keep} epoch={epoch}', 'PASS continuation legacy GDN n=1 ref_rms=1 max=0 rms=0 limits=1/1']
    lines += [admission.WHOLE, cpp_format('VRAM_LAYER_EXACT_GATE_PASS ', 44, 44)]
    for baseline in admission.BASELINES:
        for family in ('BENCH', 'BLOCK'):
            lines += study(family, baseline)
    lines.append(cpp_format('BENCH_GATE PASS'))
    return lines


def container_log(source_sha='a'*64):
    lines = ['VRAM_GDN_CONTAINER_ENTER', 'VRAM_GDN_CONTAINER_SOURCE_SHA256 ' + source_sha,
             'VRAM_GDN_IDLE_PASS stage=before-build cards=2 utilization=0 kfd_processes=0']
    lines += ['VRAM_GDN_HOST_GATE_PASS target=' + target for target in admission.HOST_GATES]
    lines += ['VRAM_GDN_BINARY_SHA256 ' + 'b'*64, 'VRAM_GDN_BUILD_GATE PASS']
    for scenario in admission.SCENARIOS:
        lines += [f'VRAM_GDN_IDLE_PASS stage={scenario} cards=2 utilization=0 kfd_processes=0',
                  f'VRAM_GDN_PROCESS_BEGIN scenario={scenario} bound_s=1200']
        if scenario == 'normal': lines += normal()
        else:
            name, first = scenario.rsplit('-', 1)
            lines += fault(name, int(first))
        lines += [f'VRAM_GDN_PROCESS_EXIT scenario={scenario} code=0']
    lines += ['VRAM_GDN_CONTAINER_COMPLETE fault_processes=4 normal_processes=1']
    return lines


def host_log():
    return ['VRAM_GDN_AUTH_EXIT code=0', 'VRAM_GDN_RESOLVED_IMAGE ' + admission.IMAGE,
            'VRAM_GDN_HOST_SOURCE_SHA256 ' + 'a'*64, *container_log(),
            'VRAM_GDN_CONTAINER_CLEANUP verified_absent=test-container']


def executable(path, content):
    path.write_text(content)
    path.chmod(0o755)


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid = normal()
        cls.host = host_log()

    def reject(self, lines):
        with self.assertRaises(admission.AdmissionError):
            admission.validate_normal(lines)

    def test_real_format_fixture_passes_complete_normal_and_host(self):
        admission.validate_normal(self.valid)
        admission.validate_host(self.host, 'test-container')

    def test_all_four_fault_process_contracts(self):
        for name in ('delayed-rank', 'missing-rank'):
            for first in (0, 1):
                lines = fault(name, first)
                admission.validate_fault(lines, name, first)
                for index, line in enumerate(lines):
                    if line.startswith(('VRAM_LAYER_FAULT_', 'VRAM_LAYER_DELAY_', 'VRAM_LAYER_RECOVERY_')):
                        with self.subTest(name=name, first=first, omitted=line):
                            with self.assertRaises(admission.AdmissionError):
                                admission.validate_fault(lines[:index]+lines[index+1:], name, first)
                with self.assertRaises(admission.AdmissionError):
                    admission.validate_fault(lines+['BENCH_SAMPLE fake=1'], name, first)
                with self.assertRaises(admission.AdmissionError):
                    admission.validate_fault(lines, name, 1-first)

    def test_every_machine_gate_missing_duplicate_and_malformed_rejected(self):
        prefixes = ('HC_DISPATCH_SELECTED ', 'HC_DISPATCH_INIT_PASS ', 'VRAM_LAYER_LIFECYCLE_PASS ',
                    'VRAM_LAYER_BANK_CASE ', 'VRAM_LAYER_BANK_GATE_PASS ', 'VRAM_LAYER_EXACT_GATE_PASS ',
                    'BENCH_GATE ', 'BENCH_STUDY ', 'BLOCK_STUDY ', 'VRAM_MATCHED_FIXTURE ')
        for i, line in enumerate(self.valid):
            if line.startswith(prefixes):
                with self.subTest(line=line):
                    self.reject(self.valid[:i]+self.valid[i+1:])
                    self.reject(self.valid[:i]+[line]+self.valid[i:])
                    self.reject(self.valid[:i]+[line+' truncated']+self.valid[i+1:])

    def test_every_timing_sample_crossover_summary_missing_or_duplicate_rejected(self):
        for i, line in enumerate(self.valid):
            if line.startswith(('BENCH_WARMUP ', 'BENCH_SAMPLE ', 'BLOCK_SHADOW ', 'BLOCK_SAMPLE ',
                                'BENCH_CROSSOVER ', 'BLOCK_CROSSOVER ', 'BENCH_SUMMARY ', 'BLOCK_SUMMARY ')):
                with self.subTest(line=line):
                    self.reject(self.valid[:i]+self.valid[i+1:])
                    self.reject(self.valid[:i]+[line]+self.valid[i:])

    def test_failure_skip_partial_profile_and_fault_cannot_admit_timings(self):
        for extra in ('FAIL numerical', 'SKIPPED no peer', 'VRAM_LAYER_WATCHDOG code=70', 'tp2_gdn_layer: bad',
                      'ERROR GPU reset', 'PROBE_GATE PASS', 'PROFILE_DIAGNOSTIC mode=hybrid-vram',
                      'VRAM_LAYER_FAULT_GATE_PASS scenario=missing-rank first=0 exact_recovery=1 timing_admitted=0', ' BENCH_SAMPLE malformed'):
            with self.subTest(extra=extra): self.reject(self.valid[:-1]+[extra]+self.valid[-1:])
        self.reject(fault('missing-rank', 0))
        self.reject(self.valid[:-1])
        self.reject(self.valid+['extra'])

    def test_late_gates_and_samples_outside_phase_rejected(self):
        for prefix in ('HC_DISPATCH_SELECTED ', 'VRAM_LAYER_LIFECYCLE_PASS ', 'VRAM_LAYER_BANK_GATE_PASS ', 'VRAM_LAYER_EXACT_GATE_PASS '):
            i = next(i for i, line in enumerate(self.valid) if line.startswith(prefix))
            moved = self.valid[:i]+self.valid[i+1:-1]+[self.valid[i]]+self.valid[-1:]
            self.reject(moved)
        sample = next(line for line in self.valid if line.startswith('BENCH_SAMPLE '))
        self.reject([sample]+self.valid)
        self.reject(self.valid+[sample])

    def test_bad_numeric_fields_counts_modes_and_order(self):
        cases = [('baseline-total-ms=2.000000', 'baseline-total-ms='+value) for value in ('nan', 'inf', '-inf', '0', '-1', 'garbage')]
        cases += [('crossed-ratio=1.000000', 'crossed-ratio=nan'), ('pairs=8', 'pairs=7'),
                  ('calls-per-block=16', 'calls-per-block=15'), ('mode=hybrid-vram', 'mode=hybrid'),
                  ('candidate=tp-hybrid-vram', 'candidate=tp-hybrid-captured'),
                  ('first=tp-hybrid-vram', 'first=tp-hybrid-captured'),
                  ('baseline-wall-ms=32.000000', 'baseline-wall-ms=0'),
                  ('trial=0 crossover-leg=0', 'trial=0 crossover-leg=00'),
                  ('baseline-total-ms=2.000000', 'baseline-total-ms=2.000000 baseline-total-ms=2.000000')]
        for old, new in cases:
            with self.subTest(old=old, new=new):
                changed = '\n'.join(self.valid).replace(old, new, 1).splitlines()
                self.assertNotEqual(changed, self.valid)
                self.reject(changed)

    def test_inventory_and_staged_environment_fail_closed(self):
        for old, new in (('hc-variant=3 hc-variant-name=staged', 'hc-variant=1 hc-variant-name=plain'),
                         ('arch=gfx906', 'arch=gfx908'), ('pci=0000:04:00.0', 'pci=0000:03:00.0'),
                         ('runtime=70200', 'runtime=0'), ('STRATA_HC_SPLIT=unset', 'STRATA_HC_SPLIT=1'),
                         ('role=hc_attn_norm.weight', 'role=hc_attn_bad.weight')):
            with self.subTest(old=old): self.reject('\n'.join(self.valid).replace(old, new, 1).splitlines())

    def test_host_revalidates_boundaries_exits_completion_and_cleanup(self):
        prefixes = ('VRAM_GDN_IDLE_PASS ', 'VRAM_GDN_PROCESS_BEGIN ', 'VRAM_GDN_PROCESS_EXIT ',
                    'VRAM_GDN_CONTAINER_COMPLETE ', 'VRAM_GDN_CONTAINER_CLEANUP ')
        for i, line in enumerate(self.host):
            if line.startswith(prefixes):
                for lines in (self.host[:i]+self.host[i+1:], self.host[:i]+[line]+self.host[i:]):
                    with self.subTest(line=line):
                        with self.assertRaises(admission.AdmissionError): admission.validate_host(lines, 'test-container')
        for bad in ('VRAM_GDN_SETUP_FAILURE bad', 'BENCH_GATE PASS', 'VRAM_LAYER_LIFECYCLE_PASS life=0 orders=2 tokens=1,8 banks=2'):
            with self.assertRaises(admission.AdmissionError): admission.validate_host(self.host+[bad], 'test-container')

    def test_line_endings_and_idle_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); proc = root/'proc'; proc.mkdir(); path = root/'log'
            for raw in (b'partial', b'bad\r\n', b'bad\0\n', b'bad\xff\n'):
                path.write_bytes(raw)
                with self.assertRaises(ValueError): admission.read_log(path)
            good = {'card3': {'GPU use (%)': '0'}, 'card5': {'GPU use (%)': '0.0'}}
            path.write_text(json.dumps(good)); admission.validate_idle(path, proc)
            bad = [[], {}, {'card3': {}}, {'card3': {'GPU use (%)': 0}, 'card5': {'GPU use (%)': True}},
                   {'card3': {'GPU use (%)': 1}, 'card5': {'GPU use (%)': 0}},
                   {'cardA': {'GPU use (%)': 0}, 'card5': {'GPU use (%)': 0}}]
            for obj in bad:
                path.write_text(json.dumps(obj))
                with self.assertRaises(admission.AdmissionError): admission.validate_idle(path, proc)
            path.write_text('{"card3":{"GPU use (%)":0},"card3":{"GPU use (%)":0}}')
            with self.assertRaises(admission.AdmissionError): admission.validate_idle(path, proc)
            path.write_text(json.dumps(good)); (proc/'123').mkdir()
            with self.assertRaises(admission.AdmissionError): admission.validate_idle(path, proc)
            with self.assertRaises(admission.AdmissionError): admission.validate_idle(path, root/'missing')

    def test_cache_config_reuses_only_unchanged_compiler_recipe_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cache = root/'cache'; compiler = root/'compiler'; recipe = root/'recipe'; cmake = root/'CMakeLists.txt'
            for path in (compiler, recipe, cmake): path.write_text('version one')
            def config():
                out = io.StringIO()
                with contextlib.redirect_stdout(out): admission.cache_config(cache, admission.IMAGE, compiler, recipe, cmake)
                return Path(out.getvalue().strip())
            first = config(); (first/'sentinel.o').write_text('cached object')
            self.assertEqual(config(), first)
            self.assertEqual((first/'sentinel.o').read_text(), 'cached object')
            compiler.write_text('version two'); second = config(); self.assertNotEqual(first, second)
            recipe.write_text('flags two'); third = config(); self.assertNotEqual(second, third)
            cmake.write_text('cmake two'); fourth = config(); self.assertNotEqual(third, fourth)
            with self.assertRaises(admission.AdmissionError): admission.cache_config(cache, 'wrong image', compiler, recipe, cmake)
            (fourth/'bad').symlink_to('/etc/passwd')
            with self.assertRaises(admission.AdmissionError): config()

    def test_changed_source_preserved_timestamp_cannot_reuse_cached_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source'; source.mkdir(); build = root/'build'; build.mkdir(mode=0o700)
            for name in admission.SOURCE_FILES:
                path = source/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('original')
            binary = build/'tp2_gdn_layer'; binary.write_text('mock executable')
            def prepare():
                out = io.StringIO()
                with contextlib.redirect_stdout(out): admission.cache_prepare(build, source, admission.source_manifest(source))
                return out.getvalue().strip()
            def commit(): admission.cache_commit(build, source, admission.source_manifest(source))
            self.assertEqual(prepare(), 'clean'); commit()
            self.assertEqual(prepare(), 'incremental'); commit()
            # Normal one-file edit after the build retains the object cache.
            edited = source/'src/core/tp_gdn_layer.cpp'; time.sleep(0.002); edited.write_text('ordinary fix')
            self.assertEqual(prepare(), 'incremental'); commit()
            old = edited.stat(); edited.write_text('restored different bytes'); os.utime(edited, ns=(old.st_atime_ns, old.st_mtime_ns))
            self.assertEqual(prepare(), 'clean')
            # No commit simulates an interrupted rebuild; the next run cleans.
            self.assertEqual(prepare(), 'clean'); commit()
            old_ns = edited.stat().st_mtime_ns
            built_ns = json.loads((build/'.vram-built.json').read_text())['built_ns']
            edited.write_text('intermediate restored timestamp')
            halfway = (old_ns + built_ns)//2
            self.assertGreater(halfway, old_ns)
            os.utime(edited, ns=(halfway, halfway))
            self.assertEqual(prepare(), 'clean'); commit()
            binary.write_text('unexpected changed executable')
            with self.assertRaises(admission.AdmissionError): prepare()

    def test_fixture_hash_labels_consistency_and_order_fail_closed(self):
        i = next(i for i, line in enumerate(self.valid) if line.startswith('VRAM_MATCHED_FIXTURE'))
        line = self.valid[i]
        for old, new in (('same-bytes-both-arms=1', 'same-bytes-both-arms=0'),
                         ('state-fnv1a64=000000000000000b', 'state-fnv1a64=000000000000000c'),
                         ('residual-fnv1a64=', 'residual-fnv1a64=x'), ('family=burst', 'family=block')):
            changed = list(self.valid); changed[i] = line.replace(old, new)
            self.reject(changed)
        # A valid hash record from a later trial cannot admit an earlier sample.
        changed = list(self.valid); changed.pop(i); changed.insert(-1, line); self.reject(changed)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.bin = self.root/'bin'; self.bin.mkdir()
        self.repo = self.root/'repo'; self.repo.mkdir(); self.models = self.root/'models'
        (self.models/'pack').mkdir(parents=True); (self.models/'shard.gguf').write_text('fixture')
        for name in (*admission.SOURCE_FILES, 'deps/llama.cpp/ggml/CMakeLists.txt'):
            path = self.repo/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('// CPU fixture\n')
        for name in (RUNNER.name, ADMISSION.name, 'mi50_pipeline.py'):
            shutil.copy2(ROOT/'tools'/name, self.repo/'tools'/name)
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        subprocess.run(['git', '-C', str(self.repo), '-c', 'user.name=CPU Test', '-c', 'user.email=cpu@example.invalid',
                        'commit', '-qm', 'fixture', '--allow-empty'], check=True)
        self.calls = self.root/'docker.jsonl'; self.sudo = self.root/'sudo.log'; self.timeouts = self.root/'timeouts.log'
        self.fixture = self.root/'fixture'; self.fixture.write_text('\n'.join(container_log())+'\n')
        self.inputs = [str(self.models), '--pack', '/models/pack', '--gguf', '/models/shard.gguf']
        self.env = dict(os.environ, PATH=str(self.bin)+':'+os.environ['PATH'], MOCK_CALLS=str(self.calls),
                        MOCK_SUDO=str(self.sudo), MOCK_TIMEOUTS=str(self.timeouts), MOCK_FIXTURE=str(self.fixture))
        for key in ('BUILD_JOBS', 'TP2_TIMEOUT', 'TP2_BUILD_TIMEOUT', 'TP2_OUTER_TIMEOUT', 'TP2_VRAM_GDN_CACHE'): self.env.pop(key, None)
        executable(self.bin/'id', '#!/bin/bash\ncase "$1" in -u) echo "${MOCK_UID:-1000}" ;; -g) echo 1000 ;; -G) echo "1000 27 44" ;; *) exit 2 ;; esac\n')
        executable(self.bin/'stat', '#!/bin/bash\n[[ ${MOCK_STAT_EXIT:-0} == 0 ]] || exit "$MOCK_STAT_EXIT"\nprintf "%s\\n" "109 44 0"\n')
        executable(self.bin/'sudo', '''#!/bin/bash
printf '%s\\n' "$*" >> "$MOCK_SUDO"
if [[ "$*" == '-n -v' ]]; then exit "${MOCK_AUTH_EXIT:-0}"; fi
[[ "$1 $2" == '-n docker' ]] || exit 93
shift; exec "$@"
''')
        timeout = shutil.which('timeout')
        executable(self.bin/'timeout', f'''#!/bin/bash
printf '%s\\n' "$*" >> "$MOCK_TIMEOUTS"
if [[ " $* " == *' docker run '* && -n ${{MOCK_TIMEOUT:-}} ]]; then exit "$MOCK_TIMEOUT"; fi
exec {shlex.quote(timeout)} "$@"
''')
        executable(self.bin/'docker', f'''#!{sys.executable}
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['MOCK_CALLS'], 'a') as f: f.write(json.dumps(args)+'\\n')
if args[:2] == ['image', 'inspect']:
    print(os.environ.get('MOCK_IMAGE', {admission.IMAGE!r}))
elif args[0] == 'run':
    pathlib.Path(os.environ['MOCK_CALLS']+'.container').write_text(sys.stdin.read())
    fixture = pathlib.Path(os.environ['MOCK_FIXTURE']).read_text()
    start = args.index('-s')+2
    sha = args[start+3]
    print(fixture.replace('a'*64, sha), end='')
    sys.exit(int(os.environ.get('MOCK_RUN_EXIT', '0')))
elif args[0] == 'ps' and os.environ.get('MOCK_CLEANUP_FAIL'):
    print('still-running')
''')

    def host(self, args=(), inputs=None, **env):
        for log in self.repo.glob('tp2-vram-gdn-*.log'): log.unlink()
        self.calls.unlink(missing_ok=True); self.sudo.unlink(missing_ok=True)
        result = subprocess.run(['bash', str(self.repo/'tools'/RUNNER.name), *(self.inputs if inputs is None else inputs), *args],
                                env=dict(self.env, **env), capture_output=True, text=True, timeout=15, start_new_session=True)
        logs = list(self.repo.glob('tp2-vram-gdn-*.log')); self.assertEqual(len(logs), 1)
        saved = logs[0].read_text(); self.assertIn(f'TP2_VRAM_GDN_EXIT={result.returncode} ', saved)
        return result, saved

    def docker_calls(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_host_success_pinned_isolated_bounded_private_build(self):
        result, saved = self.host()
        self.assertEqual(result.returncode, 0, result.stderr+saved[-2000:])
        self.assertIn(SUITE, saved)
        calls = self.docker_calls(); runs = [c for c in calls if c[0] == 'run']; self.assertEqual(len(runs), 1)
        run = runs[0]
        for flag, value in (('--pull', 'never'), ('--network', 'none'), ('--cap-drop', 'ALL'), ('--user', '1000:1000')):
            self.assertEqual(run[run.index(flag)+1], value)
        for flag in ('--privileged', '--cap-add', '--security-opt', '--ipc'): self.assertNotIn(flag, run)
        for flag in ('--read-only', '--rm', '--device=/dev/kfd', '--device=/dev/dri'): self.assertIn(flag, run)
        self.assertIn(admission.IMAGE, run)
        mounts = [run[i+1] for i,v in enumerate(run) if v == '--mount']
        self.assertEqual([m for m in mounts if not m.endswith(',readonly')], [f'type=bind,src={self.repo}/build-tp2-gdn-vram,dst=/build-cache'])
        self.assertEqual(run[run.index('-s')+2:run.index('-s')+5], ['1200', '600', '48'])
        inner = Path(str(self.calls)+'.container').read_text()
        subprocess.run(['bash', '-n'], input=inner, text=True, check=True)
        self.assertIn('-DSTRATA_TP2_GDN_VRAM_BUILD=ON', inner)
        self.assertIn('-DSTRATA_TP2_GDN_RCCL_BUILD=OFF', inner)
        self.assertIn('-B "$build"', inner)
        self.assertIn('cache-config', inner)
        self.assertIn('flock -n', inner)
        self.assertIn('run_case normal 0 normal', inner)
        self.assertNotIn('--model-probe', inner)
        self.assertNotIn('--profile-stages', inner)
        self.assertTrue(any(c[0] == 'rm' for c in calls)); self.assertTrue(any(c[0] == 'ps' for c in calls))

    def test_overrides_invalid_paths_and_jobs_rejected_before_authorization(self):
        cases = [(['--execution', 'hybrid'], {}), (['--layer', '1'], {}), (['--mode', '7'], {}),
                 (['--benchmark', '1'], {}), (['--bench-trials', '8'], {}), (['--hc-dispatch', 'legacy'], {}),
                 (['--pack', '/models/pack'], {}), (['--gguf'], {}), ([], {'BUILD_JOBS': '49'}), ([], {'TP2_VRAM_GDN_CACHE': '../escape'}),
                 ([], {'BUILD_JOBS': '0'}), ([], {'TP2_TIMEOUT': '0'}), ([], {'TP2_OUTER_TIMEOUT': '01'})]
        for args, env in cases:
            with self.subTest(args=args, env=env):
                result, saved = self.host(args, **env)
                self.assertEqual(result.returncode, 2, saved)
                self.assertFalse(self.calls.exists()); self.assertFalse(self.sudo.exists())
        (self.models/'escape.gguf').symlink_to('/etc/passwd')
        result, saved = self.host(inputs=[str(self.models), '--pack', '/models/pack', '--gguf', '/models/escape.gguf'])
        self.assertEqual(result.returncode, 2, saved)
        self.assertFalse(self.calls.exists())

    def test_auth_image_process_timeout_cleanup_and_partial_logs_fail(self):
        for env, expected in (({'MOCK_AUTH_EXIT': '7'}, 7), ({'MOCK_IMAGE': 'wrong'}, 2),
                              ({'MOCK_STAT_EXIT': '8'}, 2), ({'MOCK_RUN_EXIT': '77'}, 77),
                              ({'MOCK_TIMEOUT': '124'}, 124), ({'MOCK_CLEANUP_FAIL': '1'}, 125)):
            with self.subTest(env=env):
                result, saved = self.host(**env)
                self.assertEqual(result.returncode, expected, saved)
                self.assertNotIn(SUITE, saved)
        self.fixture.write_text('VRAM_GDN_CONTAINER_COMPLETE fault_processes=4 normal_processes=1\n')
        result, saved = self.host(); self.assertEqual(result.returncode, 4, saved); self.assertNotIn(SUITE, saved)

    def test_bad_fault_precedes_normal_and_complete_host_rejects_tampering(self):
        original = self.fixture.read_text()
        for old, new in (('first=0 target=1 rejected=1', 'first=0 target=1 rejected=0'),
                         ('hc-variant=3 hc-variant-name=staged', 'hc-variant=2 hc-variant-name=split'),
                         ('baseline-total-ms=2.000000', 'baseline-total-ms=nan'),
                         ('VRAM_GDN_BUILD_GATE PASS', 'VRAM_GDN_BUILD_GATE MAYBE')):
            self.fixture.write_text(original.replace(old, new, 1))
            result, saved = self.host(); self.assertEqual(result.returncode, 4, saved); self.assertNotIn(SUITE, saved)

    def test_model_views_reuse_safe_existing_absolute_aliases(self):
        (self.models/'alias').symlink_to(self.models/'pack')
        result, saved = self.host(inputs=[str(self.models), '--pack', '/models/alias', '--gguf', '/models/shard.gguf'])
        self.assertEqual(result.returncode, 0, saved[-2000:])
        run = next(c for c in self.docker_calls() if c[0] == 'run')
        self.assertIn(f'type=bind,src={self.models},dst={self.models},readonly', run)

    def test_cache_off_symlink_owner_and_config_safety(self):
        result, saved = self.host(TP2_VRAM_GDN_CACHE='off')
        self.assertEqual(result.returncode, 0, saved[-2000:])
        run = next(c for c in self.docker_calls() if c[0] == 'run')
        self.assertTrue(all(run[i+1].endswith(',readonly') for i,v in enumerate(run) if v == '--mount'))
        (self.repo/'unsafe').symlink_to(self.root, target_is_directory=True)
        result, saved = self.host(TP2_VRAM_GDN_CACHE='unsafe')
        self.assertEqual(result.returncode, 4, saved[-2000:]); self.assertNotIn(SUITE, saved)
        self.assertFalse(any(c[0] == 'run' for c in self.docker_calls()))
        path = self.root/'cache'; admission.cache_directory(path)
        (path/'.lock').symlink_to(self.root/'nope')
        with self.assertRaises(admission.AdmissionError): admission.cache_directory(path)

    def inner(self, **extra):
        fixtures = self.root/'processes'; fixtures.mkdir(exist_ok=True)
        for scenario in admission.SCENARIOS:
            if scenario == 'normal': lines = normal()
            else:
                name, first = scenario.rsplit('-', 1); lines = fault(name, int(first))
            (fixtures/scenario).write_text('\n'.join(lines)+'\n')
        tmp = self.root/'inner-tmp'; tmp.mkdir(exist_ok=True)
        proc = self.root/'proc'; proc.mkdir(exist_ok=True)
        cache = self.root/'inner-cache'
        script = INNER.replace('/tmp', str(tmp)).replace('/work', str(self.repo)).replace('/build-cache', str(cache)).replace('/sys/class/kfd/kfd/proc', str(proc))
        executable(self.bin/'hipcc', '#!/bin/bash\necho "CPU compiler fixture"\n')
        executable(self.bin/'ninja', '#!/bin/bash\nexit 0\n')
        executable(self.bin/'rocm-smi', '''#!/bin/bash
[[ ${MOCK_TELEMETRY_EXIT:-0} == 0 ]] || exit "$MOCK_TELEMETRY_EXIT"
if [[ " $* " == *' --json '* ]]; then
  echo '{"card3":{"GPU use (%)":"0"},"card5":{"GPU use (%)":"0"}}'
else echo 'CPU-only mocked read-only telemetry'; fi
''')
        executable(self.bin/'ldd', '#!/bin/bash\necho "libfixture.so => /fixture"\n')
        binary = f'''#!{sys.executable}
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['MOCK_BINARY_CALLS'], 'a') as out: out.write(json.dumps(args)+'\\n')
scenario=args[args.index('--vram-scenario')+1]; first=args[args.index('--vram-launch-first')+1]
label=scenario if scenario=='normal' else scenario+'-'+first
text=pathlib.Path(os.environ['MOCK_PROCESSES'], label).read_text()
if label == os.environ.get('MOCK_BAD_SCENARIO'): text=text.replace('timing_admitted=0', 'timing_admitted=1')
print(text, end='')
if label == os.environ.get('MOCK_EXIT_SCENARIO'): sys.exit(int(os.environ.get('MOCK_CASE_EXIT', '77')))
'''
        binary_template = self.root/'binary-template'; binary_template.write_text(binary)
        executable(self.bin/'cmake', f'''#!{sys.executable}
import os, pathlib, shutil, sys
args = sys.argv[1:]
if os.environ.get('MOCK_BUILD_EXIT'): sys.exit(int(os.environ['MOCK_BUILD_EXIT']))
if '--build' not in args: sys.exit(0)
build=pathlib.Path(args[args.index('--build')+1])
for name in {admission.HOST_GATES!r}:
    (build/name).write_text('#!/bin/bash\\nexit 0\\n'); (build/name).chmod(0o755)
shutil.copy2(os.environ['MOCK_BINARY_TEMPLATE'], build/'tp2_gdn_layer'); (build/'tp2_gdn_layer').chmod(0o755)
''')
        calls = self.root/'binary.jsonl'; calls.unlink(missing_ok=True)
        env = dict(self.env, MOCK_BINARY_CALLS=str(calls), MOCK_BINARY_TEMPLATE=str(binary_template), MOCK_PROCESSES=str(fixtures), **extra)
        result = subprocess.run(['bash', '-s', '--', '5', '5', '3', admission.source_manifest(self.repo), 'off', admission.IMAGE, '--pack', '/models/pack', '--gguf', '/models/shard.gguf'],
                                input=script, env=env, capture_output=True, text=True, timeout=20)
        records = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return result, records

    def test_actual_inner_shell_runs_four_faults_then_single_fixed_benchmark(self):
        result, calls = self.inner()
        self.assertEqual(result.returncode, 0, result.stderr+result.stdout[-2000:])
        self.assertEqual(len(calls), 5)
        self.assertEqual([c[c.index('--vram-scenario')+1] for c in calls], ['delayed-rank', 'delayed-rank', 'missing-rank', 'missing-rank', 'normal'])
        self.assertEqual([c[c.index('--vram-launch-first')+1] for c in calls], ['0', '1', '0', '1', '0'])
        for i, call in enumerate(calls):
            for flag, value in (('--execution', 'hybrid-vram'), ('--layer', '0'), ('--mode', '8'), ('--hc-dispatch', 'production-check')):
                self.assertEqual(call[call.index(flag)+1], value)
            self.assertEqual('--benchmark' in call, i == 4)
        self.assertIn('VRAM_GDN_CONTAINER_COMPLETE', result.stdout)

    def test_inner_build_telemetry_fault_and_process_errors_stop_before_timing(self):
        for env, code, count in (({'MOCK_BUILD_EXIT':'9'}, 9, 0), ({'MOCK_TELEMETRY_EXIT':'8'}, 8, 0),
                                  ({'MOCK_BAD_SCENARIO':'missing-rank-0'}, 4, 3),
                                  ({'MOCK_EXIT_SCENARIO':'delayed-rank-0', 'MOCK_CASE_EXIT':'77'}, 77, 1)):
            with self.subTest(env=env):
                result, calls = self.inner(**env)
                self.assertEqual(result.returncode, code, result.stderr+result.stdout[-2000:])
                self.assertEqual(len(calls), count)
                self.assertFalse(any('--benchmark' in c for c in calls))
                self.assertNotIn('VRAM_GDN_CONTAINER_COMPLETE', result.stdout)

    def test_shell_and_compiler_options_are_fixed_and_no_external_changes(self):
        subprocess.run(['bash', '-n', str(RUNNER)], check=True)
        subprocess.run(['bash', '-n'], input=INNER, text=True, check=True)
        for text in ('--execution hybrid-vram --layer 0 --mode 8 --hc-dispatch production-check',
                     '--benchmark --bench-warmup 3 --bench-trials 4 --bench-block-calls 16 --bench-block-trials 4',
                     'for scenario in delayed-rank missing-rank', 'for first in 0 1',
                     'idle_gate before-build', 'idle_gate "$label"', '--parallel "$1"'):
            self.assertIn(text, INNER)
        for forbidden in ('sudo ', 'apt-get ', 'systemctl ', '--setpower', '--setsclk', '--setmclk', 'git clone', 'curl '):
            self.assertNotIn(forbidden, INNER)


if __name__ == '__main__':
    unittest.main()
