#!/usr/bin/env python3
"""Fail-closed CPU log admission for the opt-in, one-layer MI50 VRAM study.

These checks admit a complete run, never generation throughput. The executable
owns numerical oracles; this module checks their coverage, ordering and timing
records, including the durable host transcript after verified cleanup.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import stat
import time

IMAGE = 'sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca'
CANDIDATE = 'tp-hybrid-vram'
BASELINES = ('tp-hybrid-captured', 'single-gpu-captured')
TOKENS = (1, 2, 4, 5, 8)
SCENARIOS = ('delayed-rank-0', 'delayed-rank-1', 'missing-rank-0', 'missing-rank-1', 'normal')
WHOLE = 'PASS whole-GDN engineering parity cases=88 continuation-cases=88 failures=0; no full-model inference/quality/performance claim'
EXACT = 'VRAM_LAYER_EXACT_GATE_PASS prefix_cases=44 continuation_cases=44 T=1..8 orders=2 banks=2 old_hybrid_tolerance_bits=0 original_full_gates=unchanged'
FINAL = 'BENCH_GATE PASS burst-pairs-every-result=checked block-shadow-every-step=checked measured-block-final-only=checked; no measured block intermediate downloads; one-layer wall times only'
INIT = 'HC_DISPATCH_INIT_PASS policy=production-check devices=2 before-layer-setup=1'
HOST_GATES = ('tp2_shard_policy', 'tp_layer_layout_test', 'native_expert_call_policy',
              'tp_gdn_weights_test', 'tp_hc_layout_test', 'tp_gdn_gather_layout_test')
ENV = ('STRATA_HC_PERSIST', 'STRATA_HC_SPLIT', 'STRATA_GR_FAST', 'STRATA_GR_V3',
       'STRATA_GR_SPLIT', 'STRATA_GR_DOWN_MAX4', 'STRATA_NO_MULTI_GR', 'STRATA_TSUM',
       'STRATA_SM70_TABLE', 'STRATA_HC_Q8', 'STRATA_QFUSE')
SOURCE_FILES = ('CMakeLists.txt', 'tests/hip/tp2_gdn_layer.cpp',
                'src/core/tp_gdn_layer.cpp', 'src/core/tp_gdn_vram.cpp',
                'src/core/tp_gdn_vram.hpp', 'src/core/tp_gdn_rccl_workers.hpp',
                'src/kernels/cuda/tp_vram_rendezvous.cu',
                'tools/mi50_pipeline.py', 'tools/run_tp2_vram_gdn_mi50.sh',
                'tools/tp2_vram_gdn_admission.py')


class AdmissionError(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise AdmissionError(reason)


def read_log(path):
    raw = Path(path).read_bytes()
    require(raw.endswith(b'\n') and b'\r' not in raw and b'\0' not in raw, 'partial/invalid log encoding')
    return raw.decode('utf-8', errors='strict').splitlines()


def one(lines, text):
    require(lines.count(text) == 1, 'missing/duplicate ' + text)
    return lines.index(text)


def records(lines, kind):
    result = []
    for i, line in enumerate(lines):
        if line.startswith(kind):
            require(line.startswith(kind + ' ') and bool(line[len(kind):].strip()), 'malformed ' + kind)
            result.append((i, fields(line)))
    return result


def fields(line):
    pairs = [item.split('=', 1) for item in line.split()[1:] if '=' in item]
    require(all(k and v for k, v in pairs), 'empty record field')
    require(len(pairs) == len({k for k, _ in pairs}), 'duplicate record field')
    return dict(pairs)


def strict_fields(line, expected):
    f = fields(line)
    require(set(f) == set(expected), 'missing/extra fields: ' + line.split()[0])
    require(all(f[k] == str(v) for k, v in expected.items()), 'wrong record values: ' + line.split()[0])


def positive(f, names):
    for name in names:
        try:
            value = float(f.get(name, 'nan'))
        except (TypeError, ValueError):
            raise AdmissionError('malformed numeric field ' + name) from None
        require(math.isfinite(value) and value > 0, 'invalid/missing numeric field ' + name)


def forbidden(line):
    # Explanatory lowercase "failure" in a successful negative-test contract is
    # allowed. Actual errors, skips, watchdogs and unrelated execution are not.
    token = line.split(' ', 1)[0]
    require(line == line.lstrip() or not line.lstrip().startswith(('BENCH', 'BLOCK', 'VRAM_', 'HC_DISPATCH_', 'HC_GGUF_', 'CORRECTNESS_EXECUTION', 'CASE ')), 'indented/malformed marker')
    require(not re.search(r'(?:^|[_\s])(?:FAIL|FAILED|FAILURE|ERROR|SKIP|SKIPPED|WATCHDOG)(?:\b|_)', line),
            'failure/skip/watchdog output')
    require(not line.startswith(('tp2_gdn_layer:', 'Traceback', 'terminate called',
                                 'Segmentation fault', 'Aborted', 'hipError', 'CUDA error')),
            'runtime error output')
    require(not re.search(r'(?i)(?:^|[ _])(?:error|fatal|exception)(?:[: ]|$)', line), 'error output')
    require(not token.startswith(('PROBE_', 'CAL_', 'PROFILE_', 'CAPTURE_PROFILE', 'RCCL_')),
            'unrelated/instrumented output')


def startup(lines):
    for line in lines:
        forbidden(line)
    init = one(lines, INIT)
    require(len(records(lines, 'HC_DISPATCH_INIT_PASS')) == 1, 'conflicting HC initialization')
    selected = records(lines, 'HC_DISPATCH_SELECTED')
    require(len(selected) == 2, 'expected two HC selections')
    for device in (0, 1):
        pos = one(lines, f'HC_DISPATCH_SELECTED policy=production-check device={device} hc-variant=3 hc-variant-name=staged check-invoked=1')
        require(pos < init, 'HC selection after initialization')
    env = records(lines, 'HC_DISPATCH_ENV')
    require(len(env) == len(ENV), 'incomplete HC environment manifest')
    for flag in ENV:
        matches = [(i, f) for i, f in env if set(f) == {flag}]
        require(len(matches) == 1 and matches[0][0] < init, 'HC environment field/order ' + flag)
        expected = '0' if flag in {'STRATA_HC_PERSIST', 'STRATA_GR_V3', 'STRATA_GR_SPLIT'} else 'unset'
        require(matches[0][1][flag] == expected, 'uncontrolled HC environment ' + flag)
    for kind in ('HC_GGUF_SELECTED', 'HC_GGUF_COUNTS', 'HC_GGUF_COVERAGE'):
        require(len(records(lines, kind)) == 8, 'incomplete/duplicate ' + kind)
    for half in ('attn', 'ffn'):
        for part, kind, type_id, shape in (('norm', 'F32', 0, '10240'), ('down', 'BF16', 30, '10240x320'),
                                           ('up', 'BF16', 30, '320x10240'), ('inject', 'BF16', 30, '10240x4')):
            role = f'hc_{half}_{part}.weight'
            found = [(i, f) for i, f in records(lines, 'HC_GGUF_SELECTED') if f.get('role') == role]
            require(len(found) == 1, 'HC source role ' + role)
            i, f = found[0]
            require(i < init and set(f) == {'layer', 'role', 'matches', 'shard-index', 'type', 'type-id', 'shape'},
                    'HC source field/order ' + role)
            require(re.fullmatch(r'0|[1-9][0-9]*', f['shard-index']) is not None, 'invalid shard index')
            require(all(f[k] == str(v) for k, v in {'layer': 0, 'matches': 1, 'type': kind, 'type-id': type_id, 'shape': shape}.items()),
                    'HC source type/shape ' + role)
            require(one(lines, f'HC_GGUF_COUNTS role={role} type={kind} type-id={type_id} layers=48 of=48') < init,
                    'late HC counts')
            require(one(lines, f'HC_GGUF_COVERAGE role={role} missing-layers=0 ambiguous-layers=0 of=48') < init,
                    'late HC coverage')
    gpus, pci = {}, set()
    for i, line in enumerate(lines):
        if not line.startswith('VRAM_LAYER_GPU'):
            continue
        m = re.fullmatch(r'VRAM_LAYER_GPU rank=([01]) device=([01]) pci=([0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]) arch=gfx906(?::[^ ]+)? runtime=[1-9][0-9]* driver=[1-9][0-9]*', line)
        require(m is not None and i < init, 'malformed/late MI50 identity')
        rank, device = int(m[1]), int(m[2])
        require(rank == device and rank not in gpus and m[3].lower() not in pci, 'wrong/duplicate GPU identity')
        gpus[rank] = device
        pci.add(m[3].lower())
    require(set(gpus) == {0, 1}, 'missing MI50 identity')
    return init


def lifecycle(life):
    return f'VRAM_LAYER_LIFECYCLE_PASS life={life} orders=2 tokens=1,8 banks=2'


TRANSPORT = 'VRAM_LAYER_TRANSPORT one-complete-proposal-graph-per-rank; Y=raw-mapped-gather output=raw-row-gather FFN=rank0-then-rank1-fadd_rn; no-host-phase-joins; persistent-worker-dispatch-and-completion-in-wall; final-completion-witness-D2H-inside-wall; replicated-original-staged-HC; no-new-quantization; protocol-epochs-not-reset-between-replays'


def vram_gates(lines, init):
    result = []
    for i, line in enumerate(lines):
        if not line.startswith('VRAM_LAYER_') or line.startswith('VRAM_LAYER_GPU '):
            continue
        require(i > init, 'VRAM gate before HC admission')
        if line.startswith('VRAM_LAYER_EXACT_OWNER'):
            require(line in {'VRAM_LAYER_EXACT_OWNER rank=0', 'VRAM_LAYER_EXACT_OWNER rank=1'}, 'invalid exact owner')
        elif line.startswith('VRAM_LAYER_TRANSPORT'):
            require(line == TRANSPORT, 'malformed transport contract')
        else:
            result.append((i, line))
    return result


def validate_fault(lines, scenario, first):
    init = startup(lines)
    require(scenario in {'delayed-rank', 'missing-rank'} and first in (0, 1), 'wrong fault scenario')
    require(not any(line.startswith(('BENCH', 'BLOCK', 'CORRECTNESS_EXECUTION', 'CASE ', 'PASS whole-GDN')) for line in lines),
            'fault subprocess contains timing/correctness benchmark output')
    # The exact marker contract is deliberately closed to unrelated or duplicate
    # VRAM gates, including a normal-process success pasted into a fault log.
    expected = [f'VRAM_LAYER_FAULT_ARMED scenario={scenario} first={first} target={1-first}']
    if scenario == 'delayed-rank':
        expected += [f'VRAM_LAYER_DELAY_PASS first={first} delayed={1-first} delay_ms=20 observed_polls=1 exact_continuation=1 both_drained=1 timing_admitted=0']
    else:
        expected += [f'VRAM_LAYER_FAULT_PASS scenario=missing-rank first={first} target={1-first} rejected=1 sticky=1 both_drained=1 timing_admitted=0']
    expected += [f'VRAM_LAYER_RECOVERY_PASS scenario={scenario} first={first} prefix=1 continuation=2 owners=2 banks=2 exact=1 released=1',
                 f'VRAM_LAYER_FAULT_GATE_PASS scenario={scenario} first={first} exact_recovery=1 timing_admitted=0']
    actual = vram_gates(lines, init)
    require([line for _, line in actual] == expected, 'incomplete/duplicate/malformed fault gates')
    require(init < actual[0][0] and lines[-1] == expected[-1], 'fault completion order')


def validate_normal(lines):
    init = startup(lines)
    whole = one(lines, WHOLE)
    require(sum('whole-GDN engineering parity' in line for line in lines) == 1, 'conflicting whole-layer result')
    exact = one(lines, EXACT)
    gate = one(lines, FINAL)
    require(init < whole < exact < gate and lines[-1] == FINAL, 'normal gate order/partial completion')
    banks = [f'VRAM_LAYER_BANK_CASE first={first} T={t} bank={bank} capacity=8' for first in (0, 1) for t in range(1, 9) for bank in (0, 1)]
    bank_gate = 'VRAM_LAYER_BANK_GATE_PASS T=1..8 orders=2 owners=2 banks=2 fields=19 capacity=8 inputs=changing'
    expected_vram = [lifecycle(0), lifecycle(1), *banks, bank_gate, EXACT]
    vram = vram_gates(lines, init)
    require([line for _, line in vram] == expected_vram, 'incomplete/duplicate/malformed lifecycle/exact gates')
    require(init < vram[0][0] < vram[1][0] < one(lines, bank_gate) < whole, 'lifecycle/bank gate order')
    headers = [(i, line) for i, line in enumerate(lines) if line.startswith('CORRECTNESS_EXECUTION')]
    require([line for _, line in headers] == ['CORRECTNESS_EXECUTION hybrid (reference=runtime)',
                                            'CORRECTNESS_EXECUTION hybrid-vram (reference=runtime)'], 'correctness mode/count/order')
    cases, mode = Counter(), None
    for i, line in enumerate(lines):
        if line.startswith('CORRECTNESS_EXECUTION'):
            require(one(lines, bank_gate) < i < whole, 'correctness header order')
            mode = line.split()[1]
        if line.startswith('CASE'):
            m = re.fullmatch(r'CASE T=([1-8]) keep=([0-8]) epoch=([1-9][0-9]*)', line)
            require(m is not None and mode is not None and init < i < whole, 'malformed/late prefix case')
            t, keep = int(m[1]), int(m[2])
            require(keep <= t, 'invalid keep prefix')
            cases[(mode, t, keep)] += 1
    require(cases == Counter({(m, t, k): 1 for m in ('hybrid', 'hybrid-vram') for t in range(1, 9) for k in range(t + 1)}),
            'incomplete/duplicate 88 prefix cases')
    # Study records form four closed blocks. A correct-looking sample cannot be
    # moved ahead of its own configuration or after that study's summary.
    study, family, completed = None, None, set()
    bucket = {}
    allowed = {'BENCH_STUDY', 'BENCH_SCOPE', 'BENCH_METHOD', 'BENCH_CONTRACT', 'BENCH_LIMIT', 'BENCH_CONFIG',
               'BENCH_WARMUP', 'BENCH_SAMPLE', 'BENCH_CROSSOVER', 'BENCH_SUMMARY', 'BENCH_GATE',
               'BLOCK_STUDY', 'BLOCK_SCOPE', 'BLOCK_METHOD', 'BLOCK_SHADOW', 'BLOCK_SAMPLE', 'BLOCK_CROSSOVER', 'BLOCK_SUMMARY'}
    for i, line in enumerate(lines):
        if not line.startswith(('BENCH', 'BLOCK')):
            continue
        token = line.split(' ', 1)[0]
        require(token in allowed and line.startswith(token + ' ') and bool(line[len(token):].strip()), 'malformed/unknown timing marker')
        if token == 'BENCH_GATE':
            require(line == FINAL, 'malformed/duplicate final benchmark gate')
            continue
        require(exact < i < gate, 'timing before correctness or after final gate')
        if token.endswith('_STUDY'):
            f = fields(line)
            strict_fields(line, {'baseline': f.get('baseline'), 'candidate': CANDIDATE})
            require(f['baseline'] in BASELINES, 'unknown baseline')
            if study is not None:
                _validate_study(bucket, family, study)
            family, study = token.split('_')[0], f['baseline']
            require((family, study) not in completed, 'duplicate study')
            completed.add((family, study))
            bucket = {}
        else:
            require(study is not None and token.startswith(family + '_'), 'timing outside its study')
            bucket.setdefault(token, []).append((i, line, fields(line)))
    require(study is not None, 'missing studies')
    _validate_study(bucket, family, study)
    validate_fixtures(lines, exact, gate)
    require(completed == {(family, baseline) for family in ('BENCH', 'BLOCK') for baseline in BASELINES}, 'incomplete studies')


def validate_fixtures(lines, exact, gate):
    counts, state_hashes, conv_hashes, inputs = Counter(), set(), set(), {}
    for pos, f in records(lines, 'VRAM_MATCHED_FIXTURE'):
        family = f.get('family')
        require(family in {'burst', 'block'} and exact < pos < gate, 'wrong/late fixture family')
        field = 'trial' if family == 'burst' else 'calls'
        require(set(f) == {'family', 'baseline', 'candidate', 'T', field, 'residual-fnv1a64', 'state-fnv1a64', 'conv-fnv1a64', 'same-bytes-both-arms'}, 'malformed fixture fields')
        require(len(lines[pos].split()) == len(f)+1, 'malformed fixture record')
        require(f['baseline'] in BASELINES and f['candidate'] == CANDIDATE and f['same-bytes-both-arms'] == '1', 'mismatched fixture identity')
        for name in ('residual-fnv1a64', 'state-fnv1a64', 'conv-fnv1a64'):
            require(re.fullmatch(r'[0-9a-f]{16}', f[name]) is not None, 'malformed fixture hash')
        if family == 'block': require(f['calls'] == '16', 'fixture block call count')
        state_hashes.add(f['state-fnv1a64']); conv_hashes.add(f['conv-fnv1a64'])
        key = (family, f['T'], f[field])
        previous = inputs.setdefault(key, f['residual-fnv1a64'])
        require(previous == f['residual-fnv1a64'], 'cross-study input bytes differ')
        counts[(family, f['baseline'], f['T'], f[field])] += 1
        kind = 'BENCH' if family == 'burst' else 'BLOCK'
        study = one(lines, f'{kind}_STUDY baseline={f["baseline"]} candidate={CANDIDATE}')
        require(study < pos, 'fixture before study')
        matches = []
        for i, sample in records(lines, kind + ('_WARMUP' if family == 'burst' and f['trial'].startswith('-') else '_SAMPLE')):
            if sample.get('baseline') == f['baseline'] and sample.get('T') == f['T'] and (family == 'block' or sample.get('trial') == f['trial']):
                matches.append(i)
        require(matches and pos < min(matches), 'fixture after matching sample')
        if family == 'block':
            shadow = [i for i, value in records(lines, 'BLOCK_SHADOW') if i > study and value.get('T') == f['T']]
            require(shadow and pos < min(shadow), 'fixture after block shadow')
    expected = {('burst', b, str(t), str(trial)): 1 for b in BASELINES for t in TOKENS for trial in range(-3, 4)}
    expected.update({('block', b, str(t), '16'): 1 for b in BASELINES for t in TOKENS})
    require(counts == Counter(expected), 'incomplete/duplicate matched fixture manifest')
    require(len(state_hashes) == len(conv_hashes) == 1, 'initial-state bytes differ between studies')


def _validate_study(bucket, family, baseline):
    config = family + ('_CONFIG' if family == 'BENCH' else '_METHOD')
    require(len(bucket.get(config, [])) == 1, 'missing/duplicate study config')
    config_pos, _, f = bucket[config][0]
    expected_config = {'warmup_crossovers': '3', 'measured_crossovers': '4', 'pairs-per-crossover': '2', 'order': 'ABBA/BAAB-alternating'} if family == 'BENCH' else {'calls': '16', 'crossovers': '4'}
    require(all(f.get(k) == v for k, v in expected_config.items()), 'wrong fixed study config')
    ident = {'baseline': baseline, 'candidate': CANDIDATE, 'mode': 'hybrid-vram'}
    for suffix in (['WARMUP'] if family == 'BENCH' else ['SHADOW']) + ['SAMPLE', 'CROSSOVER', 'SUMMARY']:
        kind = family + '_' + suffix
        found = bucket.get(kind, [])
        counts = Counter()
        for pos, line, f in found:
            require(pos > config_pos, 'measurement before study configuration')
            if suffix == 'SHADOW':
                expected_fields = {'mode', 'T', 'step'}
                require(f.get('mode') == 'hybrid-vram', 'shadow mode')
                key = (f.get('T'), f.get('step'))
            else:
                require(all(f.get(k) == v for k, v in ident.items()), 'mislabeled timing record')
                expected_fields = set(ident) | {'T'}
                key = (f.get('T'),)
            if suffix in ('SAMPLE', 'WARMUP'):
                expected_fields |= {'trial', 'crossover-leg', 'first'}
                require(re.fullmatch(r'-?[0-9]+', f.get('trial', '')) is not None and f.get('crossover-leg') in {'0', '1'}, 'invalid trial/leg')
                trial, leg = int(f['trial']), int(f['crossover-leg'])
                expected_first = CANDIDATE if (trial + leg + (3 if family == 'BENCH' else 0)) % 2 else baseline
                require(f.get('first') == expected_first, 'wrong crossover order')
                if family == 'BENCH':
                    numeric = {arm + suffix for arm in ('baseline', 'candidate') for suffix in ('-propose-ms', '-commit-ms', '-total-ms')}
                else:
                    expected_fields.add('calls')
                    require(f.get('calls') == '16', 'wrong block call count')
                    numeric = {arm + suffix for arm in ('baseline', 'candidate') for suffix in ('-wall-ms', '-ms-per-call')}
                key += (f['trial'], f['crossover-leg'])
            elif suffix == 'CROSSOVER':
                expected_fields.add('trial')
                tail = '-mean-ms' if family == 'BENCH' else '-mean-ms-per-call'
                numeric = {'baseline' + tail, 'candidate' + tail, 'crossed-ratio'}
                key += (f.get('trial'),)
            elif suffix == 'SUMMARY':
                numeric = {'ratio-of-total-times', 'crossover-ratio-median'}
                if family == 'BENCH':
                    expected_fields.add('pairs')
                    require(f.get('pairs') == '8', 'wrong burst pair count')
                    numeric |= {arm + tail for arm in ('baseline', 'candidate') for tail in ('-ms-median', '-ms-p10', '-ms-p90', '-propose-median', '-commit-median')}
                else:
                    contract = {'blocks-per-arm': '8', 'calls-per-block': '16', 'shadow-every-step': 'passed', 'final-block-parity': 'passed', 'measured-intermediate-parity': 'not-downloaded'}
                    expected_fields |= set(contract)
                    require(all(f.get(k) == v for k, v in contract.items()), 'wrong sustained parity contract')
                    numeric |= {'baseline-ms-per-call-median', 'candidate-ms-per-call-median'}
            else:
                numeric = set()
            require(set(f) == expected_fields | numeric, 'missing/extra timing fields ' + kind)
            # Strict machine records cannot hide stray prose after valid fields.
            require(len(line.split()) == len(f) + 1, 'malformed timing fields ' + kind)
            positive(f, numeric)
            counts[key] += 1
        if suffix == 'SHADOW':
            expected = {(str(t), str(j)): 1 for t in TOKENS for j in range(16)}
        elif suffix in ('SAMPLE', 'WARMUP'):
            trials = range(-3, 0) if suffix == 'WARMUP' else range(4)
            expected = {(str(t), str(trial), str(leg)): 1 for t in TOKENS for trial in trials for leg in range(2)}
        elif suffix == 'CROSSOVER':
            expected = {(str(t), str(trial)): 1 for t in TOKENS for trial in range(4)}
        else:
            expected = {(str(t),): 1 for t in TOKENS}
        require(counts == Counter(expected), 'incomplete/duplicate ' + kind)
    # Each shape is admitted only after its warmup/shadow, samples and crossover
    # records; this also refuses plausible summaries pasted before measurements.
    for t in TOKENS:
        by_suffix = {suffix: [(p, f) for p, _, f in bucket.get(family + '_' + suffix, []) if f.get('T') == str(t)]
                     for suffix in ('WARMUP', 'SHADOW', 'SAMPLE', 'CROSSOVER', 'SUMMARY')}
        pre = by_suffix['WARMUP' if family == 'BENCH' else 'SHADOW']
        require(max(p for p, _ in pre) < min(p for p, _ in by_suffix['SAMPLE']), 'warmup/shadow after measured sample')
        for trial in range(4):
            sample = [p for p, f in by_suffix['SAMPLE'] if f['trial'] == str(trial)]
            cross = [p for p, f in by_suffix['CROSSOVER'] if f['trial'] == str(trial)]
            require(max(sample) < cross[0] < by_suffix['SUMMARY'][0][0], 'crossover/summary order')


def unique_json(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate telemetry key')
        result[key] = value
    return result


def validate_idle(path, kfd_root):
    root = Path(kfd_root)
    require(root.is_dir() and os.access(root, os.R_OK | os.X_OK), 'KFD process visibility unavailable')
    require(not list(root.iterdir()), 'KFD processes present')
    data = json.loads(Path(path).read_text(), object_pairs_hook=unique_json)
    require(isinstance(data, dict), 'unknown GPU telemetry')
    cards = {key: value for key, value in data.items() if key.startswith('card')}
    require(len(cards) == 2 and all(re.fullmatch(r'card(?:0|[1-9][0-9]*)', key) for key in cards), 'expected exactly two DRM GPU records')
    for name, card in cards.items():
        require(isinstance(card, dict), 'malformed card record')
        value = card.get('GPU use (%)')
        require(not isinstance(value, bool) and str(value).strip() in {'0', '0.0'}, 'nonzero/unknown GPU utilization: ' + name)


def source_inputs(root):
    root = Path(root)
    paths = {root / name for name in SOURCE_FILES}
    # Bind-mounted inputs are immutable from the container. Hash all local
    # compilation inputs as well, to catch a concurrent host checkout edit.
    for tree in ('src', 'include', 'tests/core', 'cmake', 'third_party/ggml', 'deps/llama.cpp/ggml'):
        directory = root / tree
        if directory.exists():
            require(not directory.is_symlink(), 'symlinked source tree')
            paths.update(directory.rglob('*'))
    result = {}
    for path in sorted(paths):
        require(not path.is_symlink(), 'symlinked source: ' + str(path))
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        require('\n' not in relative and '\r' not in relative, 'invalid source name')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[relative] = {'sha256': digest, 'mtime_ns': path.stat().st_mtime_ns}
    return result


def inputs_digest(inputs):
    return hashlib.sha256(''.join(value['sha256'] + '  ' + name + '\n' for name, value in sorted(inputs.items())).encode()).hexdigest()


def source_manifest(root, verbose=False):
    inputs = source_inputs(root)
    if verbose:
        for name, value in sorted(inputs.items()):
            print('VRAM_GDN_SOURCE_FILE path=' + name + ' sha256=' + value['sha256'], file=sys.stderr)
    return inputs_digest(inputs)


def validate_host(lines, container):
    for line in lines:
        forbidden(line)
    def prefix(kind):
        found = [(i, x) for i, x in enumerate(lines) if x.startswith(kind)]
        require(len(found) == 1, 'missing/duplicate ' + kind)
        return found[0]
    one(lines, 'VRAM_GDN_AUTH_EXIT code=0')
    prefix('VRAM_GDN_AUTH_EXIT')
    prefix('VRAM_GDN_RESOLVED_IMAGE')
    prefix('VRAM_GDN_BUILD_GATE')
    require(len(records(lines, 'VRAM_GDN_HOST_GATE_PASS')) == len(HOST_GATES), 'duplicate/malformed host gate')
    one(lines, 'VRAM_GDN_RESOLVED_IMAGE ' + IMAGE)
    build = one(lines, 'VRAM_GDN_BUILD_GATE PASS')
    for name in HOST_GATES:
        require(one(lines, 'VRAM_GDN_HOST_GATE_PASS target=' + name) < build, 'late host gate')
    before = one(lines, 'VRAM_GDN_IDLE_PASS stage=before-build cards=2 utilization=0 kfd_processes=0')
    require(before < build, 'build before idle admission')
    _, host_sha = prefix('VRAM_GDN_HOST_SOURCE_SHA256')
    _, child_sha = prefix('VRAM_GDN_CONTAINER_SOURCE_SHA256')
    require(re.fullmatch(r'VRAM_GDN_HOST_SOURCE_SHA256 [0-9a-f]{64}', host_sha) is not None and
            child_sha == host_sha.replace('HOST_SOURCE', 'CONTAINER_SOURCE'), 'source identity mismatch')
    _, binary = prefix('VRAM_GDN_BINARY_SHA256')
    require(re.fullmatch(r'VRAM_GDN_BINARY_SHA256 [0-9a-f]{64}', binary) is not None, 'missing built binary identity')
    previous, included = build, set()
    for scenario in SCENARIOS:
        begin, begin_line = prefix('VRAM_GDN_PROCESS_BEGIN scenario=' + scenario)
        require(re.fullmatch('VRAM_GDN_PROCESS_BEGIN scenario=' + scenario + r' bound_s=[1-9][0-9]*', begin_line) is not None, 'invalid process bound')
        end = one(lines, f'VRAM_GDN_PROCESS_EXIT scenario={scenario} code=0')
        require(len(records(lines, 'VRAM_GDN_PROCESS_EXIT scenario=' + scenario)) == 1, 'conflicting process exit')
        idle = one(lines, f'VRAM_GDN_IDLE_PASS stage={scenario} cards=2 utilization=0 kfd_processes=0')
        require(previous < idle < begin < end, 'process/idle order')
        segment = lines[begin + 1:end]
        if scenario == 'normal':
            validate_normal(segment)
        else:
            fault, first = scenario.rsplit('-', 1)
            validate_fault(segment, fault, int(first))
        included.update(range(begin + 1, end))
        previous = end
    require(len(records(lines, 'VRAM_GDN_PROCESS_BEGIN')) == 5 and len(records(lines, 'VRAM_GDN_PROCESS_EXIT')) == 5,
            'unexpected process')
    require(len(records(lines, 'VRAM_GDN_IDLE_PASS')) == 6, 'unexpected idle gate')
    for i, line in enumerate(lines):
        if i not in included:
            require(not line.startswith(('BENCH', 'BLOCK', 'VRAM_LAYER_', 'HC_DISPATCH_', 'HC_GGUF_', 'CORRECTNESS_EXECUTION', 'CASE ', 'VRAM_MATCHED_FIXTURE')),
                    'child output outside process')
    complete = one(lines, 'VRAM_GDN_CONTAINER_COMPLETE fault_processes=4 normal_processes=1')
    cleanup = one(lines, 'VRAM_GDN_CONTAINER_CLEANUP verified_absent=' + container)
    require(len(records(lines, 'VRAM_GDN_CONTAINER_COMPLETE')) == 1 and len(records(lines, 'VRAM_GDN_CONTAINER_CLEANUP')) == 1,
            'conflicting completion/cleanup')
    require(previous < complete < cleanup, 'completion/cleanup order')


def private_directory(path):
    path = Path(path)
    if not path.exists() and not path.is_symlink():
        path.mkdir(mode=0o700)
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700,
            'cache must be an owned private directory mode 0700, never a symlink')
    return path


def cache_directory(path):
    root = private_directory(path)
    lock = root / '.lock'
    if lock.exists() or lock.is_symlink():
        info = lock.lstat()
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1,
                'unsafe cache lock')


def cache_config(path, image, compiler, recipe, cmake):
    root = private_directory(path)
    require(image == IMAGE, 'cache image mismatch')
    config = hashlib.sha256(image.encode() + b'\0' + Path(compiler).read_bytes() + b'\0' +
                            Path(recipe).read_bytes() + b'\0' + Path(cmake).read_bytes()).hexdigest()
    build = private_directory(root / config)
    # Reconfigure every invocation; only the dedicated same-config objects may
    # be reused. Never accept cache paths that escape this private build tree.
    for file in build.rglob('*'):
        info = file.lstat()
        require(info.st_uid == os.getuid(), 'cache entry owned by another user')
        if file.is_symlink():
            require(not os.path.isabs(os.readlink(file)) and file.resolve().is_relative_to(build), 'unsafe cache symlink')
        elif file.is_file():
            require(info.st_nlink == 1, 'hardlinked cache artifact')
        else:
            require(file.is_dir(), 'nonregular cache entry')
    print(build)


def cache_prepare(build, root, expected):
    build = private_directory(build)
    inputs = source_inputs(root)
    require(inputs_digest(inputs) == expected, 'source changed before cache admission')
    marker, pending = build / '.vram-built.json', build / '.vram-pending.json'
    clean = True
    if marker.exists():
        old = json.loads(marker.read_text(), object_pairs_hook=unique_json)
        require(set(old) == {'inputs', 'built_ns', 'binary_sha256'} and isinstance(old['inputs'], dict) and
                isinstance(old['built_ns'], int) and old['built_ns'] > 0, 'invalid cache build manifest')
        binary = build / 'tp2_gdn_layer'
        require(binary.is_file() and hashlib.sha256(binary.read_bytes()).hexdigest() == old['binary_sha256'], 'cached binary identity changed')
        clean = False
        for name in set(inputs) | set(old['inputs']):
            current, prior = inputs.get(name), old['inputs'].get(name)
            if current is None:
                clean = True
            elif prior is None or current['sha256'] != prior.get('sha256'):
                # Normal edits made after the last successful build use Ninja's
                # ordinary depfile incrementality. Restored/older timestamps may
                # hide changed bytes from Ninja, so only that case forces clean.
                if current['mtime_ns'] <= old['built_ns'] or (prior and current['mtime_ns'] <= prior.get('mtime_ns', 0)):
                    clean = True
    marker.unlink(missing_ok=True)
    pending.write_text(json.dumps({'inputs': inputs, 'expected': expected}, sort_keys=True)+'\n')
    print('clean' if clean else 'incremental')


def cache_commit(build, root, expected):
    build = private_directory(build)
    pending = build / '.vram-pending.json'
    snapshot = json.loads(pending.read_text(), object_pairs_hook=unique_json)
    require(snapshot['expected'] == expected and source_manifest(root) == expected, 'source changed during build')
    binary = build / 'tp2_gdn_layer'
    payload = {'inputs': snapshot['inputs'], 'built_ns': time.time_ns(),
               'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
    # No prior success marker survives a failed/interrupted build. This commit
    # record is published only after configure, compile and every host gate pass.
    pending.write_text(json.dumps(payload, sort_keys=True)+'\n')
    pending.replace(build / '.vram-built.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for mode in ('normal', 'host'):
        p = sub.add_parser(mode)
        p.add_argument('log')
        if mode == 'host': p.add_argument('container')
    p = sub.add_parser('fault'); p.add_argument('log'); p.add_argument('scenario'); p.add_argument('first', type=int)
    p = sub.add_parser('idle'); p.add_argument('json'); p.add_argument('kfd_root')
    for mode in ('cache-prepare', 'cache-commit'):
        p = sub.add_parser(mode); p.add_argument('build'); p.add_argument('root'); p.add_argument('expected')
    p = sub.add_parser('cache'); p.add_argument('path')
    p = sub.add_parser('cache-config'); p.add_argument('path'); p.add_argument('image'); p.add_argument('compiler'); p.add_argument('recipe'); p.add_argument('cmake')
    p = sub.add_parser('manifest'); p.add_argument('root'); p.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'manifest': print(source_manifest(args.root, args.verbose))
        elif args.command == 'cache': cache_directory(args.path)
        elif args.command == 'cache-prepare': cache_prepare(args.build, args.root, args.expected)
        elif args.command == 'cache-commit': cache_commit(args.build, args.root, args.expected)
        elif args.command == 'cache-config': cache_config(args.path, args.image, args.compiler, args.recipe, args.cmake)
        elif args.command == 'idle': validate_idle(args.json, args.kfd_root)
        elif args.command == 'normal': validate_normal(read_log(args.log))
        elif args.command == 'fault': validate_fault(read_log(args.log), args.scenario, args.first)
        else: validate_host(read_log(args.log), args.container)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        print('VRAM_GDN_ADMISSION_FAIL ' + str(error), file=sys.stderr)
        return 4
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
