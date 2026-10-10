#!/usr/bin/env bash
# Isolated, model-free TP2 gate. No services or host ROCm installation are changed.
set -euo pipefail
if [[ $# -lt 1 || "$1" == --help ]]; then
  echo 'Usage: bash tools/run_tp2_mi50.sh EXISTING_DOCKER_IMAGE [tp2_ffn arguments]'
  echo 'Example: bash tools/run_tp2_mi50.sh sha256:... --iters 2'
  echo 'Optional: BUILD_JOBS=48 TP2_TIMEOUT=1200 TP2_MODELS_DIR=/absolute/model/directory'
  echo 'With TP2_MODELS_DIR, the directory is mounted read-only as /models.'
  exit 0
fi
IMAGE=$1
shift
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
if [[ -n ${TP2_MODELS_DIR:-} ]]; then
  [[ "$TP2_MODELS_DIR" == /* && -d "$TP2_MODELS_DIR" && "$TP2_MODELS_DIR" != *,* ]] || {
    echo 'TP2_MODELS_DIR must be an existing absolute directory without commas.' >&2; exit 2;
  }
  MODEL_MOUNT=(--mount "type=bind,src=$TP2_MODELS_DIR,dst=/models,readonly")
fi
LOG="$ROOT/tp2-$(date -u +%Y%m%dT%H%M%SZ).log"
COMMON=(--rm --pull never --network none --read-only --cap-drop ALL
  --tmpfs /tmp:rw,exec,nosuid,size=8g -e HOME=/tmp
  --workdir /work --entrypoint /bin/bash)
{
echo 'Both MI50 cards must be idle. This script does not stop any service.'
printf 'Source commit: '; git -C "$ROOT" rev-parse HEAD
if ! git -C "$ROOT" diff --quiet HEAD --; then
  echo 'Source contains tracked modifications; record them with the result.'
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
cmake -S /work -B /work/build-tp2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_HIP_GFX906=ON \
  -DSTRATA_HC_PERSIST_BUILD=OFF -DSTRATA_TP2_BUILD=ON \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_DIR=/work/deps/llama.cpp -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
cmake --build /work/build-tp2 --target tp2_shard_policy tp_layer_layout_test tp2_ffn --parallel "$1"
/work/build-tp2/tp2_shard_policy
/work/build-tp2/tp_layer_layout_test
' build "$JOBS" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e
if (( RC != 0 )); then echo "Build/CPU test failed: $RC. Log: $LOG"; exit "$RC"; fi

set +e
sudo docker run "${COMMON[@]}" --device=/dev/kfd --device=/dev/dri \
  -e HIP_VISIBLE_DEVICES=0,1 -e STRATA_HC_PERSIST=0 \
  --mount "type=bind,src=$ROOT,dst=/work,readonly" "${MODEL_MOUNT[@]}" \
  "$IMAGE" -c '
set -euo pipefail
deadline=$1; shift
exec timeout --signal=TERM --kill-after=10s "${deadline}s" /work/build-tp2/tp2_ffn "$@"
' run "$DEADLINE" "$@" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e
printf 'TP2_EXIT=%s LOG=%s\n' "$RC" "$LOG"
if (( RC == 77 )); then echo 'SKIPPED is not a pass.'; fi
if (( RC == 124 || RC == 137 )); then echo 'Timeout/kill: inspect GPU state before another run.'; fi
exit "$RC"
