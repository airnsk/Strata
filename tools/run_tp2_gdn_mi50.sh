#!/usr/bin/env bash
# Opt-in real-weight GDN-layer correctness and optional wall-time A/B gate.
# No server/config/driver changes; benchmarking never skips correctness.
set -euo pipefail
if [[ $# -lt 2 || "$1" == --help ]]; then
  echo 'Usage: bash tools/run_tp2_gdn_mi50.sh EXISTING_IMAGE MODEL_ROOT --pack /models/PACK --gguf /models/SHARD [--gguf /models/SHARD ...] [--layer 0]'
  echo 'This checks one non-PLE GDN layer, not generation speed. Layer 1 is PLE and is refused.'
  echo 'Optional: --execution runtime|consolidated|captured|flat|flat-hc|all (default runtime)'
  echo 'Optional: --benchmark [--bench-warmup 3] [--bench-trials 12] [--bench-block-calls 16] [--bench-block-trials 4]'
  echo 'Benchmark defaults to captured, flat and flat-hc; --execution all explicitly includes all five.'
  echo 'Flat modes require the integrated model-free and full-layer timeout/abort gates before benchmarking.'
  echo 'Paired trials use identical input in AB and BA order; trials count complete crossovers.'
  echo 'Sustained blocks add per-step shadow parity and measured final-block parity (no measured intermediate downloads).'
  echo 'Optional --profile-flat uses separate diagnostic graphs; timing always uses unprofiled graphs.'
  echo 'Wall timing includes complete propose_device + commit(T); reset/uploads/capture/snapshots are outside.'
  echo 'Optional: BUILD_JOBS=48 TP2_TIMEOUT=1200'
  echo 'MODEL_ROOT is mounted read-only as /models. Both GPUs must be idle.'
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
LOG="$ROOT/tp2-gdn-$(date -u +%Y%m%dT%H%M%SZ).log"
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
cmake -S /work -B /work/build-tp2-gdn -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_HIP_GFX906=ON \
  -DSTRATA_HC_PERSIST_BUILD=OFF -DSTRATA_TP2_BUILD=OFF -DSTRATA_TP2_GDN_BUILD=ON \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_DIR=/work/deps/llama.cpp -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
cmake --build /work/build-tp2-gdn --target tp2_shard_policy tp_layer_layout_test native_expert_call_policy tp_gdn_weights_test tp_hc_layout_test tp2_gdn_layer --parallel "$1"
/work/build-tp2-gdn/tp2_shard_policy
/work/build-tp2-gdn/tp_layer_layout_test
/work/build-tp2-gdn/native_expert_call_policy
/work/build-tp2-gdn/tp_gdn_weights_test
/work/build-tp2-gdn/tp_hc_layout_test
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
timeout --signal=TERM --kill-after=10s "${deadline}s" /work/build-tp2-gdn/tp2_gdn_layer "$@"
rc=$?
set -e
telemetry
exit "$rc"
' run "$DEADLINE" "$@" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e
printf 'TP2_GDN_EXIT=%s LOG=%s\n' "$RC" "$LOG"
if (( RC == 77 )); then echo 'SKIPPED is not a pass.'; fi
if (( RC == 124 || RC == 137 )); then echo 'Timeout/kill: inspect GPU state before another run.'; fi
exit "$RC"
