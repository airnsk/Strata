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
Optional: BUILD_JOBS=48 TP2_TIMEOUT=600. Runtime timeout excludes compilation.
Reuse the verified model root, pack and GGUF shards from the existing stand run.
Both GPUs must be idle. No downloads, service stops, clock or power changes.
EOF
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG="$ROOT/tp2-model-probe-$RUN_ID.log"
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
  # Reuse the existing pipeline runner's reviewed logical/physical model-root
  # view validation. Preserve in-root absolute aliases without broader mounts.
  local views_text
  views_text=$(python3 -B - "$ROOT" "$models" "$@" <<'PY'
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.argv[1]) / "tools"))
from mi50_pipeline import model_views
root = Path(sys.argv[2])
physical = root.resolve(strict=True)
views = [Path(p) for p in model_views(root)]
args = sys.argv[3:]
allowed = {"--pack", "--gguf", "--layer", "--mode", "--bench-warmup", "--bench-trials"}
if len(args) % 2:
    raise SystemExit("Expected option/value pairs; the runner adds --model-probe itself")
seen = set()
for flag, value in zip(args[::2], args[1::2]):
    if flag not in allowed or (flag in seen and flag != "--gguf"):
        raise SystemExit("Unsupported or repeated option: " + flag)
    seen.add(flag)
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
  local head image_id
  head=$(git -C "$ROOT" rev-parse HEAD)
  printf 'PROBE_RUN id=%s head=%s image=%s\n' "$RUN_ID" "$head" "$IMAGE"
  printf 'PROBE_STAND repo=%q model-root=%q log=%q\n' "$ROOT" "$models" "$LOG"
  printf 'PROBE_ARGUMENTS'; printf ' %q' --model-probe "$@"; printf '\n'
  echo 'Both MI50 cards must be idle. No service is stopped and no power/clock setting is changed.'
  printf 'PROBE_TRACKED_DIFF_SHA256 '; git -C "$ROOT" diff --binary HEAD -- | sha256sum
  git -C "$ROOT" status --short
  sha256sum "$ROOT/tests/hip/tp2_gdn_layer.cpp" "$ROOT/src/core/tp_gdn_layer.cpp" \
    "$ROOT/include/strata/core/tp_gdn_layer.hpp" "$ROOT/tools/run_tp2_model_probe_mi50.sh"
  image_id=$(sudo docker image inspect --format '{{.Id}}' "$IMAGE")
  printf 'PROBE_RESOLVED_IMAGE %s\n' "$image_id"
  [[ "$image_id" == "$IMAGE" ]] || { echo 'Installed image ID differs from the pinned stand image.' >&2; return 2; }
  local common=(--rm --pull never --network none --read-only --cap-drop ALL
    --tmpfs /tmp:rw,exec,nosuid,size=8g -e HOME=/tmp
    --workdir /work --entrypoint /bin/bash)
  # Keep the existing incremental build directory, but build only the probe
  # executable and small host-side seam/layout gates. No legacy GPU suite.
  sudo docker run "${common[@]}" --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$ROOT,dst=/work" "$IMAGE" -c '
set -euo pipefail
export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH
cmake -S /work -B /work/build-tp2-gdn -G Ninja \
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
' build "$jobs"
  echo 'PROBE_BUILD_GATE PASS'
  sudo docker run "${common[@]}" --device=/dev/kfd --device=/dev/dri \
    -e HIP_VISIBLE_DEVICES=0,1 -e STRATA_HC_PERSIST=0 -e STRATA_GR_V3=0 -e STRATA_GR_SPLIT=0 \
    --mount "type=bind,src=$ROOT,dst=/work,readonly" "${mounts[@]}" \
    "$IMAGE" -c '
set -euo pipefail
export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH
deadline=$1; shift
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
timeout --signal=TERM --kill-after=10s "${deadline}s" /work/build-tp2-gdn/tp2_gdn_layer --model-probe "$@"
rc=$?
set -e
telemetry
printf "PROBE_PROCESS_EXIT=%s\n" "$rc"
exit "$rc"
' probe "$deadline" "$@"
}
# Capture every build/runtime/preflight stdout and stderr line together. Read
# the actual main exit, never tee's status as the test result. Preserve failures
# from either process and put the OUTER status in the saved log itself.
set +e
(set -e; main "$@") 2>&1 | tee "$LOG"
STATUS=("${PIPESTATUS[@]}")
RC=${STATUS[0]}
if (( RC == 0 && STATUS[1] != 0 )); then RC=${STATUS[1]}; fi
if (( RC == 77 )); then echo 'SKIPPED is not a pass.' | tee -a "$LOG"; fi
if (( RC == 124 || RC == 137 )); then echo 'Timeout/kill: inspect GPU state before another run.' | tee -a "$LOG"; fi
printf 'TP2_MODEL_PROBE_EXIT=%s LOG=%s\n' "$RC" "$LOG" | tee -a "$LOG"
FOOTER_STATUS=("${PIPESTATUS[@]}")
if (( RC == 0 && FOOTER_STATUS[1] != 0 )); then RC=${FOOTER_STATUS[1]}; fi
exit "$RC"
