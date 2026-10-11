#!/usr/bin/env bash
# Bounded, opt-in phase calibration. No production services/settings are changed.
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: bash tools/run_tp2_model_probe_mi50.sh MODEL_ROOT --pack /models/PACK --gguf /models/SHARD [--gguf /models/SHARD ...]
Uses only the already installed, pinned MI50 image; sudo docker supports the stand's Snap Docker.
Runs --model-probe: hybrid local-hidden FFN plus output-row attention, T=1/8 only.
Five targeted prefix cases and one-token continuations precede paired fine-phase probes.
Defaults: --layer 0 --mode 8 --bench-warmup 1 --bench-trials 4.
Optional: --hc-dispatch legacy|production-check (default legacy).
production-check selects HC on each GPU using the production startup check before captures.
Optional leading --sustained-baseline: fixed staged whole-layer benchmark, T=1,2,4,5,8.
Baseline accepts only pack/GGUF and optional layer=0, mode=8, hc-dispatch=production-check.
Baseline uses warmup=3, trials=4, block-calls=16, block-trials=4; no benchmark overrides.
Optional: BUILD_JOBS=48 TP2_TIMEOUT=600. Runtime timeout excludes compilation.
Reuse the verified model root, pack and GGUF shards from the existing stand run.
Both GPUs must be idle. No downloads, service stops, clock or power changes.
EOF
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
BASELINE=0
LABEL=PROBE
LOG_KIND=model-probe
if [[ ${1:-} == --sustained-baseline ]]; then
  BASELINE=1; LABEL=BASELINE; LOG_KIND=staged-baseline; shift
fi
DOCKER=(sudo docker)
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG="$ROOT/tp2-$LOG_KIND-$RUN_ID.log"
IMAGE=sha256:1947f7b9ea3514f137b3cabe5dc4f48ddabeb9c7bac78b4959a03b7dc7fcf4ca
main() {
  [[ $# -ge 3 ]] || { usage; return 2; }
  local models=$1; shift
  local jobs=${BUILD_JOBS:-48} deadline=${TP2_TIMEOUT:-600}
  [[ "$jobs" =~ ^[1-9][0-9]*$ && "$deadline" =~ ^[1-9][0-9]*$ ]] || {
    echo 'BUILD_JOBS and TP2_TIMEOUT must be positive integers.' >&2; return 2;
  }
  [[ -f "$ROOT/deps/llama.cpp/ggml/CMakeLists.txt" ]] || {
    echo 'Missing deps/llama.cpp: reuse the pinned dependency from the existing stand build.' >&2; return 2;
  }
  [[ "$models" == /* && -d "$models" ]] || { echo 'MODEL_ROOT must be an existing absolute directory.' >&2; return 2; }
  if (( BASELINE )) && (( ${#jobs} > 3 || ${#deadline} > 4 )); then
    echo 'Baseline bounds: BUILD_JOBS=1..128, TP2_TIMEOUT=1..1800.' >&2; return 2
  fi
  if (( BASELINE )) && (( 10#$jobs > 128 || 10#$deadline > 1800 )); then
    echo 'Baseline bounds: BUILD_JOBS=1..128, TP2_TIMEOUT=1..1800.' >&2; return 2
  fi
  # Reuse the existing pipeline runner's reviewed logical/physical model-root
  # view validation. Preserve in-root absolute aliases without broader mounts.
  local views_text
  views_text=$(STRATA_TP2_SUSTAINED_BASELINE="$BASELINE" python3 -B - "$ROOT" "$models" "$@" <<'PY'
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "tools"))
from mi50_pipeline import model_views
root = Path(sys.argv[2])
physical = root.resolve(strict=True)
views = [Path(p) for p in model_views(root)]
args = sys.argv[3:]
allowed = {"--pack", "--gguf", "--layer", "--mode", "--bench-warmup", "--bench-trials", "--hc-dispatch"}
if len(args) % 2:
    raise SystemExit("Expected option/value pairs; the runner adds --model-probe itself")
baseline = os.environ.get("STRATA_TP2_SUSTAINED_BASELINE") == "1"
if baseline:
    allowed = {"--pack", "--gguf", "--layer", "--mode", "--hc-dispatch"}
seen = set()
for flag, value in zip(args[::2], args[1::2]):
    if flag not in allowed or (flag in seen and flag != "--gguf"):
        raise SystemExit("Unsupported or repeated option: " + flag)
    seen.add(flag)
    if baseline and flag in {"--layer", "--mode", "--hc-dispatch"}:
        expected = {"--layer": "0", "--mode": "8", "--hc-dispatch": "production-check"}[flag]
        if value != expected:
            raise SystemExit("Baseline requires " + flag + "=" + expected)
    if flag == "--hc-dispatch" and value not in {"legacy", "production-check"}:
        raise SystemExit("HC dispatch must be legacy or production-check")
    if flag not in {"--pack", "--gguf"}:
        continue
    path = Path(value)
    if not path.is_absolute():
        raise SystemExit("Model input must be absolute: " + value)
    if path == Path("/models") or Path("/models") in path.parents:
        path = root / path.relative_to("/models")
    elif not any(path == view or view in path.parents for view in views):
        raise SystemExit("Model input is outside MODEL_ROOT views: " + value)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(physical)
    except (OSError, ValueError) as exc:
        raise SystemExit("Model input is absent or escapes MODEL_ROOT: " + value) from exc
    if flag == "--pack" and not resolved.is_dir():
        raise SystemExit("--pack must name a directory")
    if flag == "--gguf" and not resolved.is_file():
        raise SystemExit("--gguf must name a file")
    # Resolve absolute alias destinations only through the same known root
    # views. An out-of-view detour may not exist in the restricted container.
    lexical = Path(os.path.abspath(path))
    matching = [v for v in views if lexical == v or v in lexical.parents]
    base = max(matching, key=lambda p: len(p.parts))
    current = base
    for component in lexical.relative_to(base).parts:
        current /= component
        if current.is_symlink() and os.path.isabs(os.readlink(current)):
            target = Path(os.path.abspath(os.readlink(current)))
            if not any(target == v or v in target.parents for v in views):
                raise SystemExit("Absolute model alias escapes supported views: " + value)
if not {"--pack", "--gguf"} <= seen:
    raise SystemExit("Both --pack and --gguf are required")
print(physical)
for view in views:
    print(view)
PY
  )
  local views=() mounts=() view
  mapfile -t views <<< "$views_text"
  models=${views[0]}
  mounts=(--mount "type=bind,src=$models,dst=/models,readonly")
  for view in "${views[@]:1}"; do
    [[ "$view" == /models ]] || mounts+=(--mount "type=bind,src=$models,dst=$view,readonly")
  done
  if [[ ${PREFLIGHT_ONLY:-0} == 1 ]]; then return 0; fi
  local run_args=(--model-probe "$@")
  if (( BASELINE )); then
    run_args=(--execution hybrid --benchmark --layer 0 --mode 8 --hc-dispatch production-check
      --bench-warmup 3 --bench-trials 4 --bench-block-calls 16 --bench-block-trials 4)
    while (( $# )); do
      if [[ "$1" == --pack || "$1" == --gguf ]]; then run_args+=("$1" "$2"); fi
      shift 2
    done
  fi
  local head image_id
  head=$(git -C "$ROOT" rev-parse HEAD)
  printf "${LABEL}_RUN id=%s head=%s image=%s\n" "$RUN_ID" "$head" "$IMAGE"
  printf "${LABEL}_STAND repo=%q model-root=%q log=%q\n" "$ROOT" "$models" "$LOG"
  printf '%s_ARGUMENTS' "$LABEL"; printf ' %q' "${run_args[@]}"; printf '\n'
  echo 'Both MI50 cards must be idle. No service is stopped and no power/clock setting is changed.'
  printf '%s_TRACKED_DIFF_SHA256 ' "$LABEL"; git -C "$ROOT" diff --binary HEAD -- | sha256sum
  git -C "$ROOT" status --short
  sha256sum "$ROOT/tests/hip/tp2_gdn_layer.cpp" "$ROOT/src/core/tp_gdn_layer.cpp" \
    "$ROOT/src/kernels/cuda/fused_gr.cu" "$ROOT/include/strata/kernels/fused_gr.hpp" \
    "$ROOT/include/strata/core/tp_gdn_layer.hpp" "$ROOT/tools/run_tp2_model_probe_mi50.sh"
  image_id=$("${DOCKER[@]}" image inspect --format '{{.Id}}' "$IMAGE")
  printf "${LABEL}_RESOLVED_IMAGE %s\n" "$image_id"
  [[ "$image_id" == "$IMAGE" ]] || { echo 'Installed image ID differs from the pinned stand image.' >&2; return 2; }
  local common=(--rm --pull never --network none --read-only --cap-drop ALL
    --tmpfs /tmp:rw,exec,nosuid,size=8g -e HOME=/tmp
    --workdir /work --entrypoint /bin/bash)
  # Keep the existing incremental build directory, but build only the probe
  # executable and small host-side seam/layout gates. No legacy GPU suite.
  "${DOCKER[@]}" run "${common[@]}" --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$ROOT,dst=/work" "$IMAGE" -c '
set -euo pipefail
export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH
baseline_cmake=()
if [[ ${2:-0} == 1 ]]; then baseline_cmake=(-DSTRATA_TP2_GDN_RCCL_BUILD=OFF); fi
cmake -S /work -B /work/build-tp2-gdn -G Ninja "${baseline_cmake[@]}" \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_HIP_GFX906=ON \
  -DSTRATA_HC_PERSIST_BUILD=OFF -DSTRATA_TP2_BUILD=OFF -DSTRATA_TP2_GDN_BUILD=ON \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_DIR=/work/deps/llama.cpp -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
cmake --build /work/build-tp2-gdn --target tp2_shard_policy tp_layer_layout_test native_expert_call_policy tp_gdn_weights_test tp_hc_layout_test tp2_gdn_layer --parallel "$1"
for test in tp2_shard_policy tp_layer_layout_test native_expert_call_policy tp_gdn_weights_test tp_hc_layout_test; do
  "/work/build-tp2-gdn/$test"
done
sha256sum /work/build-tp2-gdn/tp2_gdn_layer
' build "$jobs" "$BASELINE"
  echo "${LABEL}_BUILD_GATE PASS"
  "${DOCKER[@]}" run "${common[@]}" --device=/dev/kfd --device=/dev/dri \
    -e HIP_VISIBLE_DEVICES=0,1 -e STRATA_HC_PERSIST=0 -e STRATA_GR_V3=0 -e STRATA_GR_SPLIT=0 \
    --mount "type=bind,src=$ROOT,dst=/work,readonly" "${mounts[@]}" \
    "$IMAGE" -c '
set -euo pipefail
export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH
label=$1; deadline=$2; shift 2
telemetry() {
  date -u +GPU_TELEMETRY_%Y-%m-%dT%H:%M:%SZ
  if command -v rocm-smi >/dev/null 2>&1; then
    timeout 15s rocm-smi --showproductname --showuse --showclocks --showpower --showtemp || true
  else
    echo "rocm-smi unavailable; clock/power/utilization telemetry unavailable"
  fi
}
telemetry
set +e
timeout --signal=TERM --kill-after=10s "${deadline}s" /work/build-tp2-gdn/tp2_gdn_layer "$@"
rc=$?
set -e
telemetry
printf "%s_PROCESS_EXIT=%s\n" "$label" "$rc"
exit "$rc"
' run "$LABEL" "$deadline" "${run_args[@]}"
}
# Capture every build/runtime/preflight stdout and stderr line together. Read
# the actual main exit, never tee's status as the test result. Preserve failures
# from either process and put the OUTER status in the saved log itself.
authorize_docker() {
  if (( $(id -u) == 0 )); then DOCKER=(docker); return 0; fi
  DOCKER=(sudo -n docker)
  local auth_tty auth_rc=0
  # Same terminal-only authorization pattern as the inverse-plan runner.
  if { exec {auth_tty}<>/dev/tty; } 2>/dev/null; then
    printf 'Checking sudo authorization outside timed operations.\n' >&"$auth_tty"
    sudo -v <&"$auth_tty" >&"$auth_tty" 2>&1 || auth_rc=$?
    exec {auth_tty}>&-
  else
    timeout --signal=TERM --kill-after=3s 20s sudo -n -v || auth_rc=$?
  fi
  return "$auth_rc"
}
set +e
if (( BASELINE )); then
  (set -e; PREFLIGHT_ONLY=1 main "$@") 2>&1 | tee "$LOG"
  STATUS=("${PIPESTATUS[@]}"); RC=${STATUS[0]}
  if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
  if (( RC == 0 )); then
    authorize_docker; RC=$?
    printf 'BASELINE_AUTH_EXIT=%s\n' "$RC" | tee -a "$LOG"
    STATUS=("${PIPESTATUS[@]}")
    if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
  fi
  if (( RC == 0 )); then
    (set -e; main "$@") 2>&1 | tee -a "$LOG"
    STATUS=("${PIPESTATUS[@]}"); RC=${STATUS[0]}
    if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
  fi
else
  (set -e; main "$@") 2>&1 | tee "$LOG"
  STATUS=("${PIPESTATUS[@]}"); RC=${STATUS[0]}
  if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
fi
hc_dispatch=legacy
args=("$@")
for ((i=1; i+1<${#args[@]}; i+=2)); do
  if [[ ${args[$i]} == --hc-dispatch ]]; then hc_dispatch=${args[$((i+1))]}; fi
done
if (( RC == 0 && ! ${BASELINE:-0} )) && [[ "$hc_dispatch" == production-check ]]; then
  # Checked selection must precede phase samples, independently of the
  # targeted numerical gate. This does not assert a full-model result.
  awk '
    /^HC_DISPATCH_SELECTED/ {
      if ($0 !~ /^HC_DISPATCH_SELECTED policy=production-check device=[01] hc-variant=[123] hc-variant-name=(plain|split|staged) check-invoked=1$/) bad=1
      if (($4=="hc-variant=1" && $5!="hc-variant-name=plain") ||
          ($4=="hc-variant=2" && $5!="hc-variant-name=split") ||
          ($4=="hc-variant=3" && $5!="hc-variant-name=staged")) bad=1
      if (++selected[$3]!=1 || initialized) bad=1
    }
    /^HC_DISPATCH_INIT_PASS/ {
      if ($0!="HC_DISPATCH_INIT_PASS policy=production-check devices=2 before-layer-setup=1" ||
          selected["device=0"]!=1 || selected["device=1"]!=1 || initialized++) bad=1
    }
    /^PROBE_PHASE([[:space:]]|$)/ { if (!initialized) bad=1 }
    /^PROBE_GATE PASS([[:space:]]|$)/ { gate=1 }
    END { exit (bad || initialized!=1 || selected["device=0"]!=1 || selected["device=1"]!=1 || !gate) }
  ' "$LOG" || { echo 'PROBE_ADMISSION_FAIL HC production selection missing, malformed or late; final probe gate required' | tee -a "$LOG"; RC=4; }
fi
if (( RC == 0 && ${BASELINE:-0} )); then
  python3 -B - "$LOG" "$ROOT" "$IMAGE" <<'BASELINE_PY' 2>&1 | tee -a "$LOG"
import collections
import hashlib
import math
from pathlib import Path
import re
import subprocess
import sys

log, root, image = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
lines = log.read_text().splitlines()
def require(condition, message):
    if not condition:
        raise SystemExit("BASELINE_ADMISSION_FAIL " + message)
def one(text):
    require(lines.count(text) == 1, "missing/duplicate " + text)
    return lines.index(text)
def fields(line):
    pairs = [item.split("=", 1) for item in line.split() if "=" in item]
    require(len({key for key, _ in pairs}) == len(pairs), "duplicate record field")
    return dict(pairs)
def positive(f, names):
    for name in names:
        try:
            value = float(f.get(name, "nan"))
        except ValueError:
            require(False, "malformed numeric field " + name)
        require(math.isfinite(value) and value > 0, "invalid/missing numeric field " + name)

# Do not let malformed/truncated known markers hide beside valid records.
for prefix in ("BASELINE_RUN", "HC_DISPATCH_SELECTED", "HC_DISPATCH_ENV", "HC_GGUF_SELECTED",
               "HC_GGUF_COUNTS", "HC_GGUF_COVERAGE", "CORRECTNESS_EXECUTION", "CASE", "PROFILE_DIAGNOSTIC",
               "BENCH_STUDY", "BLOCK_STUDY", "BENCH_CONFIG", "BLOCK_METHOD", "BENCH_WARMUP",
               "BENCH_SAMPLE", "BLOCK_SAMPLE", "BENCH_CROSSOVER", "BLOCK_CROSSOVER",
               "BENCH_SUMMARY", "BLOCK_SUMMARY"):
    for line in lines:
        if line.startswith(prefix):
            require(line.startswith(prefix + " ") and bool(line[len(prefix):].strip()), "malformed marker " + prefix)
for prefix in ("HC_DISPATCH_INIT_PASS", "BENCH_GATE"):
    require(sum(line.startswith(prefix) for line in lines) == 1, "duplicate/conflicting marker " + prefix)

head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
runs = [x for x in lines if x.startswith("BASELINE_RUN ")]
require(len(runs) == 1 and fields(runs[0]).get("head") == head and
        fields(runs[0]).get("image") == image, "source/image identity mismatch")
one("BASELINE_RESOLVED_IMAGE " + image)
source_files = ["tests/hip/tp2_gdn_layer.cpp", "src/core/tp_gdn_layer.cpp",
                "src/kernels/cuda/fused_gr.cu", "include/strata/kernels/fused_gr.hpp",
                "include/strata/core/tp_gdn_layer.hpp", "tools/run_tp2_model_probe_mi50.sh"]
for name in source_files:
    path = root / name
    one(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + str(path))
require(sum(bool(re.fullmatch(r"[0-9a-f]{64}  /work/build-tp2-gdn/tp2_gdn_layer", x)) for x in lines) == 1,
        "missing/duplicate built executable hash")
require(sum(bool(re.fullmatch(r"BASELINE_TRACKED_DIFF_SHA256 [0-9a-f]{64}  -", x)) for x in lines) == 1,
        "missing tracked source diff hash")
build = one("BASELINE_BUILD_GATE PASS")
init = one("HC_DISPATCH_INIT_PASS policy=production-check devices=2 before-layer-setup=1")
selected = [i for i, x in enumerate(lines) if x.startswith("HC_DISPATCH_SELECTED")]
require(len(selected) == 2, "expected exactly two selector records")
for device in (0, 1):
    pos = one(f"HC_DISPATCH_SELECTED policy=production-check device={device} hc-variant=3 hc-variant-name=staged check-invoked=1")
    require(build < pos < init, "selector order")
for flag in ("STRATA_HC_PERSIST", "STRATA_HC_SPLIT", "STRATA_GR_FAST", "STRATA_GR_V3", "STRATA_GR_SPLIT",
             "STRATA_GR_DOWN_MAX4", "STRATA_NO_MULTI_GR", "STRATA_TSUM", "STRATA_SM70_TABLE", "STRATA_HC_Q8", "STRATA_QFUSE"):
    matches = [x for x in lines if x.startswith("HC_DISPATCH_ENV " + flag + "=")]
    require(len(matches) == 1, "HC environment manifest " + flag)
for half in ("attn", "ffn"):
    for part, kind, type_id, shape in (("norm", "F32", 0, "10240"), ("down", "BF16", 30, "10240x320"),
                                       ("up", "BF16", 30, "320x10240"), ("inject", "BF16", 30, "10240x4")):
        role = f"hc_{half}_{part}.weight"
        records = [x for x in lines if x.startswith("HC_GGUF_SELECTED ") and fields(x).get("role") == role]
        require(len(records) == 1, "HC source manifest " + role)
        f = fields(records[0])
        require(all(f.get(k) == v for k, v in {"layer":"0", "matches":"1", "type":kind,
                    "type-id":str(type_id), "shape":shape}.items()), "HC source shape/type " + role)
        one(f"HC_GGUF_COUNTS role={role} type={kind} type-id={type_id} layers=48 of=48")
        one(f"HC_GGUF_COVERAGE role={role} missing-layers=0 ambiguous-layers=0 of=48")
whole = one("PASS whole-GDN engineering parity cases=88 continuation-cases=88 failures=0; no full-model inference/quality/performance claim")
require(init < whole, "correctness precedes selector")
require(not any(x.startswith(("FAIL ", "PROBE_PHASE ", "PROBE_GATE ")) for x in lines), "unexpected failure/probe output")
gate = one("BENCH_GATE PASS burst-pairs-every-result=checked block-shadow-every-step=checked measured-block-final-only=checked; no measured block intermediate downloads; one-layer wall times only")
exits = [x for x in lines if x.startswith("BASELINE_PROCESS_EXIT")]
require(exits == ["BASELINE_PROCESS_EXIT=0"], "expected exactly one zero inner exit")
require(gate < one("BASELINE_PROCESS_EXIT=0"), "inner exit precedes final gate")
headers = [x for x in lines if x.startswith("CORRECTNESS_EXECUTION ")]
require(headers == ["CORRECTNESS_EXECUTION captured (reference=runtime)", "CORRECTNESS_EXECUTION hybrid (reference=runtime)"],
        "correctness header identity/count/order")
for i, line in enumerate(lines):
    if line.startswith(("BENCH_", "BLOCK_")) and not line.startswith("BENCH_GATE "):
        require(whole < i < gate, "benchmark record outside admitted timing phase")
mode = None
cases = collections.Counter()
profiles = collections.Counter()
for index, line in enumerate(lines):
    if line.startswith("CORRECTNESS_EXECUTION "):
        mode = line.split()[1]
        require(init < index < whole and mode in {"captured", "hybrid"}, "correctness mode/order")
    if line.startswith("CASE "):
        require(init < index < whole, "prefix case outside correctness phase")
        f = fields(line); cases[(mode, int(f["T"]), int(f["keep"]))] += 1
    if line.startswith("PROFILE_DIAGNOSTIC "):
        require(init < index < whole, "profile outside correctness phase")
        f = fields(line); require(f.get("mode") == mode, "profile mode/header mismatch")
        profiles[(f["mode"], int(f["T"]))] += 1
expected_cases = collections.Counter({(m,t,k):1 for m in ("captured","hybrid") for t in range(1,9) for k in range(t+1)})
require(cases == expected_cases, "incomplete/duplicate 88 prefix cases")
require(profiles == collections.Counter({(m,t):1 for m in ("captured","hybrid") for t in range(1,9)}),
        "incomplete separate diagnostic profiles")
expected = {(b, "tp-hybrid-captured", "hybrid", str(t)) for b in ("single-gpu-captured", "tp-row-captured") for t in (1,2,4,5,8)}
for kind in ("BENCH", "BLOCK"):
    studies = [x for x in lines if x.startswith(kind + "_STUDY ")]
    require(len(studies) == 2, "study count " + kind)
    for baseline in ("single-gpu-captured", "tp-row-captured"):
        require(one(f"{kind}_STUDY baseline={baseline} candidate=tp-hybrid-captured") > whole, "study before admission")
for kind, values in (("BENCH_CONFIG", {"warmup_crossovers":"3", "measured_crossovers":"4", "pairs-per-crossover":"2", "order":"ABBA/BAAB-alternating"}),
                     ("BLOCK_METHOD", {"calls":"16", "crossovers":"4"})):
    records = [fields(x) for x in lines if x.startswith(kind + " ")]
    require(len(records) == 2 and all(all(f.get(k) == v for k,v in values.items()) for f in records), "fixed configuration " + kind)
for kind in ("BENCH_SUMMARY", "BLOCK_SUMMARY"):
    records = [(i,fields(x)) for i,x in enumerate(lines) if x.startswith(kind + " ")]
    keys = collections.Counter(tuple(f.get(k) for k in ("baseline","candidate","mode","T")) for _,f in records)
    require(keys == collections.Counter({key:1 for key in expected}), "incomplete/duplicate " + kind)
    for index, f in records:
        require(index > whole, "timing before whole-layer admission")
        positive(f, ("ratio-of-total-times", "crossover-ratio-median"))
        if kind == "BENCH_SUMMARY":
            require(f.get("pairs") == "8", "burst crossover count")
            positive(f, [arm + suffix for arm in ("baseline", "candidate") for suffix in
                        ("-ms-median", "-ms-p10", "-ms-p90", "-propose-median", "-commit-median")])
        else:
            positive(f, ("baseline-ms-per-call-median", "candidate-ms-per-call-median"))
            require(all(f.get(k) == v for k,v in {"blocks-per-arm":"8", "calls-per-block":"16",
                        "shadow-every-step":"passed", "final-block-parity":"passed",
                        "measured-intermediate-parity":"not-downloaded"}.items()), "sustained count/parity contract")
for kind in ("BENCH_SAMPLE", "BLOCK_SAMPLE", "BENCH_WARMUP"):
    samples = collections.Counter()
    for i,line in enumerate(lines):
        if line.startswith(kind + " "):
            f = fields(line)
            require(i > whole, "samples precede admission")
            trial, leg = int(f["trial"]), int(f["crossover-leg"])
            tp_first = (trial + leg + (3 if kind.startswith("BENCH_") else 0)) % 2
            require(f.get("first") == f.get("candidate" if tp_first else "baseline"), "crossover order")
            if kind.startswith("BENCH_"):
                positive(f, [arm + suffix for arm in ("baseline", "candidate") for suffix in
                            ("-propose-ms", "-commit-ms", "-total-ms")])
            else:
                require(f.get("calls") == "16", "sample block calls")
                positive(f, [arm + suffix for arm in ("baseline", "candidate") for suffix in ("-wall-ms", "-ms-per-call")])
            samples[tuple(f.get(k) for k in ("baseline","candidate","mode","T","trial","crossover-leg"))] += 1
    trials = range(-3,0) if kind == "BENCH_WARMUP" else range(4)
    require(samples == collections.Counter({key+(str(trial),str(leg)):1 for key in expected for trial in trials for leg in range(2)}),
            "incomplete/duplicate " + kind)
for kind in ("BENCH_CROSSOVER", "BLOCK_CROSSOVER"):
    records = [fields(x) for x in lines if x.startswith(kind + " ")]
    keys = collections.Counter(tuple(f.get(k) for k in ("baseline","candidate","mode","T","trial")) for f in records)
    require(keys == collections.Counter({key+(str(trial),):1 for key in expected for trial in range(4)}), "crossover manifest " + kind)
    suffix = "-mean-ms" if kind == "BENCH_CROSSOVER" else "-mean-ms-per-call"
    for f in records:
        positive(f, ("baseline"+suffix, "candidate"+suffix, "crossed-ratio"))
print("BASELINE_ADMISSION_PASS staged=3,3 prefix-cases=88 continuation-cases=88 studies=2 shapes=1,2,4,5,8")
BASELINE_PY
  GATE_STATUS=("${PIPESTATUS[@]}")
  if (( GATE_STATUS[0] != 0 )); then RC=4;
  elif (( GATE_STATUS[1] != 0 )); then RC=${GATE_STATUS[1]}; fi
fi
if (( RC == 77 )); then echo 'SKIPPED is not a pass.' | tee -a "$LOG"; fi
if (( RC == 124 || RC == 137 )); then echo 'Timeout/kill: inspect GPU state before another run.' | tee -a "$LOG"; fi
if (( BASELINE )); then exit_label=TP2_STAGED_BASELINE; else exit_label=TP2_MODEL_PROBE; fi
printf '%s_EXIT=%s LOG=%s\n' "$exit_label" "$RC" "$LOG" | tee -a "$LOG"
FOOTER_STATUS=("${PIPESTATUS[@]}")
if (( RC == 0 && FOOTER_STATUS[1] != 0 )); then RC=${FOOTER_STATUS[1]}; fi
exit "$RC"
