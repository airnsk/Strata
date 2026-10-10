#!/usr/bin/env bash
# Opt-in exact real-weight HC retiling gate and whole-HC A/B timing.
# No server/config/driver changes; benchmarking never skips correctness.
set -euo pipefail
if [[ $# -lt 2 || "$1" == --help ]]; then
  echo 'Usage: bash tools/run_hc_retiled_mi50.sh EXISTING_IMAGE MODEL_ROOT --pack /models/PACK --gguf /models/SHARD [--gguf /models/SHARD]'
  echo 'Layer0 real BF16 HC only; no column or whole-layer gate. GPUs 0/1 measured sequentially.'
  echo 'Exact all-T1..8 intermediates/residual gates, then whole-HC captured original/candidate ABBA timings T1/2/4/5/8.'
  echo 'Optional: --trials 8 --batch 32 --warmup 16; BUILD_JOBS=48 TP2_TIMEOUT=1200.'
  echo 'Both GPUs must be idle. Existing image only, model mount read-only, no clock/power changes.'
  exit 0
fi
IMAGE=$1
TP2_MODELS_DIR=$2
shift 2
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
JOBS=${BUILD_JOBS:-48}
DEADLINE=${TP2_TIMEOUT:-1200}
[[ "$JOBS" =~ ^[1-9][0-9]*$ && "$DEADLINE" =~ ^[1-9][0-9]*$ ]] || {
  echo 'BUILD_JOBS and TP2_TIMEOUT must be positive integers.' >&2; exit 2;
}
[[ -f "$ROOT/deps/llama.cpp/ggml/CMakeLists.txt" ]] || {
  echo 'Missing deps/llama.cpp: use the pinned dependency from the build instructions.' >&2; exit 2;
}
MODEL_MOUNT=()
if [[ -z ${TP2_MODELS_DIR:-} ]]; then echo "MODEL_ROOT is required" >&2; exit 2; fi
if [[ -n ${TP2_MODELS_DIR:-} ]]; then
  [[ "$TP2_MODELS_DIR" == /* && -d "$TP2_MODELS_DIR" && "$TP2_MODELS_DIR" != *,* ]] || {
    echo 'TP2_MODELS_DIR must be an existing absolute directory without commas.' >&2; exit 2;
  }
  MODEL_MOUNT=(--mount "type=bind,src=$TP2_MODELS_DIR,dst=/models,readonly")
  # Preserve absolute in-root model symlinks without exposing any other directory.
  if [[ "$TP2_MODELS_DIR" != /models ]]; then
    MODEL_MOUNT+=(--mount "type=bind,src=$TP2_MODELS_DIR,dst=$TP2_MODELS_DIR,readonly")
  fi
fi
LOG="$ROOT/hc-retiled-$(date -u +%Y%m%dT%H%M%SZ).log"
COMMON=(--rm --pull never --network none --read-only --cap-drop ALL
  --tmpfs /tmp:rw,exec,nosuid,size=8g -e HOME=/tmp
  --workdir /work --entrypoint /bin/bash)
{
echo 'Both MI50 cards must be idle. This script does not stop any service.'
printf 'Source commit: '; git -C "$ROOT" rev-parse HEAD
if ! git -C "$ROOT" diff --quiet HEAD --; then
  echo 'Source contains tracked modifications; record them with the result.'
  printf 'Tracked source diff SHA256: '; git -C "$ROOT" diff --binary HEAD -- | sha256sum
fi
printf 'Focused source SHA256: '; sha256sum "$ROOT/tests/hip/hc_retiled_bench.cpp" "$ROOT/src/kernels/cuda/fused_gr.cu" "$ROOT/include/strata/kernels/fused_gr.hpp" "$ROOT/tools/run_hc_retiled_mi50.sh"
printf 'Test arguments:'; printf ' %q' "$@"; printf '\n'
echo "Image: $IMAGE"
echo "Log: $LOG"
} | tee "$LOG"
sudo docker image inspect --format '{{.Id}}' "$IMAGE" | tee -a "$LOG"

set +e
sudo docker run "${COMMON[@]}" --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$ROOT,dst=/work" "$IMAGE" -c '
set -euo pipefail
export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH
cmake -S /work -B /work/build-hc-retiled -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_HIP_GFX906=ON \
  -DSTRATA_HC_PERSIST_BUILD=OFF -DSTRATA_TP2_BUILD=OFF -DSTRATA_TP2_GDN_BUILD=ON \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_DIR=/work/deps/llama.cpp -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
cmake --build /work/build-hc-retiled --target hc_retiled_bench --parallel "$1"
' build "$JOBS" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e
if (( RC != 0 )); then echo "Build/CPU test failed: $RC. Log: $LOG"; exit "$RC"; fi

set +e
sudo docker run "${COMMON[@]}" --device=/dev/kfd --device=/dev/dri \
  -e HIP_VISIBLE_DEVICES=0,1 -e STRATA_HC_PERSIST=0 -e STRATA_GR_V3=0 -e STRATA_GR_SPLIT=0 \
  --mount "type=bind,src=$ROOT,dst=/work,readonly" "${MODEL_MOUNT[@]}" \
  "$IMAGE" -c '
set -euo pipefail
deadline=$1; shift
# Read-only telemetry is outside the test and every timed interval. It records
# external load/clocks rather than silently changing power/clock settings.
telemetry() {
  date -u +GPU_TELEMETRY_%Y-%m-%dT%H:%M:%SZ
  if command -v rocm-smi >/dev/null 2>&1; then
    timeout 15s rocm-smi --showproductname --showuse --showclocks --showpower --showtemp || true
  else
    echo "rocm-smi unavailable; no clock/power/utilization telemetry"
  fi
}
telemetry
set +e
timeout --signal=TERM --kill-after=10s "${deadline}s" /work/build-hc-retiled/hc_retiled_bench "$@"
rc=$?
set -e
telemetry
exit "$rc"
' run "$DEADLINE" "$@" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e
printf 'HC_RETILED_EXIT=%s LOG=%s\n' "$RC" "$LOG"
if (( RC == 77 )); then echo 'SKIPPED is not a pass.'; fi
if (( RC == 124 || RC == 137 )); then echo 'Timeout/kill: inspect GPU state before another run.'; fi
exit "$RC"
