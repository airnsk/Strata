#!/usr/bin/env bash
# Explicit build, prepare, serve. Never downloads, modifies production, or sends requests.
set -euo pipefail
umask 077
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BUILD="$ROOT/build-mi50-pipeline"
STATE="$BUILD/run"
usage() {
  echo 'Usage: bash tools/run_mi50_pipeline.sh build EXISTING_IMAGE'
  echo '       bash tools/run_mi50_pipeline.sh prepare MODEL_ROOT [SOURCE_CONFIG]'
  echo '       bash tools/run_mi50_pipeline.sh serve EXISTING_IMAGE'
  echo '       bash tools/run_mi50_pipeline.sh check'
  echo 'Build and prepare are CPU-only. Serve requires both GPUs idle and an API key.'
}
[[ $# -ge 1 ]] || { usage; exit 2; }
ACTION=$1; shift
case "$ACTION" in
  prepare) exec python3 "$ROOT/tools/mi50_pipeline.py" prepare "$@" ;;
  check) [[ $# == 0 ]] || { usage; exit 2; }; exec python3 "$ROOT/tools/mi50_pipeline.py" check "$STATE/engine.log" ;;
  --help|-h) usage; exit 0 ;;
  build|serve) [[ $# == 1 ]] || { usage; exit 2; } ;;
  *) usage; exit 2 ;;
esac
[[ "$ROOT" != *,* && "$ROOT" != *$'\n'* ]] || { echo 'Checkout path cannot contain commas/newlines.' >&2; exit 2; }
IMAGE=$1
command -v flock >/dev/null || { echo 'The Linux flock command is required.' >&2; exit 2; }
mkdir -p "$BUILD"
exec 9>"$BUILD/experiment.lock"
flock -n 9 || { echo 'Another build/serve helper holds this experiment lock.' >&2; exit 2; }
DOCKER=(sudo docker)
if [[ $(id -u) == 0 ]]; then DOCKER=(docker); fi
IMAGE_ID=$("${DOCKER[@]}" image inspect --format '{{.Id}}' "$IMAGE")
[[ "$IMAGE_ID" == sha256:* ]] || { echo 'Existing image could not be identified.' >&2; exit 2; }
COMMON=(--rm --pull never --read-only --cap-drop ALL --user "$(id -u):$(id -g)"
        --workdir /work --entrypoint /bin/bash -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1)
PIN=3cf03257f219afbe7334045ff7c6a06ac68c627d
DEP="$ROOT/deps/llama.cpp"
[[ -f "$DEP/ggml/CMakeLists.txt" && "$(git -C "$DEP" rev-parse --show-toplevel 2>/dev/null)" == "$DEP" && \
   "$(git -C "$DEP" rev-parse HEAD)" == "$PIN" && -z "$(git -C "$DEP" status --porcelain --untracked-files=no)" ]] || {
  echo "Need a clean pinned deps/llama.cpp checkout at $PIN; no dependency is fetched or changed." >&2; exit 2;
}
source_id() { { git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" diff --binary HEAD; printf '%s\n' "$PIN";
  sha256sum "$ROOT/tools/run_mi50_pipeline.sh" "$ROOT/tools/mi50_pipeline.py"; } | sha256sum | cut -d' ' -f1; }
if [[ "$ACTION" == build ]]; then
  [[ -f "$ROOT/deps/llama.cpp/ggml/CMakeLists.txt" ]] || {
    echo 'Missing pinned deps/llama.cpp; this helper does not fetch dependencies.' >&2; exit 2;
  }
  JOBS=${BUILD_JOBS:-48}
  [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo 'BUILD_JOBS must be positive.' >&2; exit 2; }
  mkdir -p "$BUILD"
  rm -f "$BUILD/build-id.txt"
  BEFORE=$(source_id)
  "${DOCKER[@]}" run "${COMMON[@]}" --network none --tmpfs /tmp:rw,exec,nosuid,size=8g \
    --mount "type=bind,src=$ROOT,dst=/work,readonly" \
    --mount "type=bind,src=$BUILD,dst=/work/build-mi50-pipeline" "$IMAGE_ID" -c '
set -euo pipefail
export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH
python3 /work/tools/mi50_pipeline.py dependencies
echo "ROCBLAS_TENSILE_LIBPATH=${ROCBLAS_TENSILE_LIBPATH:-<unset>}"
for dir in /opt/rocm/lib/rocblas/library /opt/gfx906-tensile; do
  if [[ -d "$dir" ]]; then
    echo "Tensile inventory: $dir"
    find "$dir" -maxdepth 2 -type f -iname "*gfx906*" -print | sed -n "1,30p"
  fi
done
/opt/rocm/lib/llvm/bin/clang --version | sed -n "1p"
cmake -S /work -B /work/build-mi50-pipeline/engine -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_HC_PERSIST_BUILD=OFF -DSTRATA_TP2_BUILD=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_GGML_DIR=/work/deps/llama.cpp -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
cmake --build /work/build-mi50-pipeline/engine --target strata --parallel "$1"
echo "Actual Strata dynamic-library resolution in this image:"
LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  ldd /work/build-mi50-pipeline/engine/strata > /work/build-mi50-pipeline/linked-libraries.txt
cat /work/build-mi50-pipeline/linked-libraries.txt
if grep -q "not found" /work/build-mi50-pipeline/linked-libraries.txt; then
  echo "An engine library is unresolved; do not load the model."; exit 2
fi
LD_LIBRARY_PATH=/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
  /work/build-mi50-pipeline/engine/strata --help > /work/build-mi50-pipeline/engine-help.txt 2>&1
for flag in --layer-split --split-device --trim-stage-weights --pipeline-windows --adapt-every; do
  grep -q -- "$flag" /work/build-mi50-pipeline/engine-help.txt || { echo "Missing engine flag: $flag"; exit 2; }
done
echo MI50_BUILD_AND_FLAGS=PASS
echo "CPU-only: the linked hipBLAS/rocBLAS gfx906 GEMM path is NOT validated by this build."
' build "$JOBS" 2>&1 | tee "$BUILD/build.log"
  [[ "$BEFORE" == "$(source_id)" ]] || { echo 'Source changed during build; rebuild before serving.' >&2; exit 2; }
  printf '%s\n' "$IMAGE_ID" "$BEFORE" "$(sha256sum "$BUILD/engine/strata" | cut -d' ' -f1)" > "$BUILD/build-id.txt"
  echo 'Build checked in the selected image. No GPU/model was started.'
  exit 0
fi
[[ -f "$BUILD/build-id.txt" && -f "$STATE/config.json" && -f "$STATE/models.path" && -f "$STATE/model-views.txt" ]] || {
  echo 'Run build, then prepare before serve.' >&2; exit 2;
}
mapfile -t BUILT < "$BUILD/build-id.txt"
[[ ${BUILT[0]} == "$IMAGE_ID" && ${BUILT[1]} == "$(source_id)" && \
   ${BUILT[2]} == "$(sha256sum "$BUILD/engine/strata" | cut -d' ' -f1)" ]] || {
  echo 'Image/source/binary differs from the checked build; rebuild before serving.' >&2; exit 2;
}
MODEL_ROOT=$(cat "$STATE/models.path")
[[ "$MODEL_ROOT" == /* && -d "$MODEL_ROOT" && "$MODEL_ROOT" != *,* && "$MODEL_ROOT" != *$'\n'* ]] || {
  echo 'Prepared model directory is no longer valid.' >&2; exit 2;
}
MODEL_MOUNTS=(--mount "type=bind,src=$MODEL_ROOT,dst=/models,readonly")
while IFS= read -r view; do
  [[ "$view" == /* && "$view" != *,* && "$view" != *$'\r'* ]] || {
    echo 'Invalid prepared model-root view.' >&2; exit 2;
  }
  case "$view" in
    /|/tmp|/var|/home|/work|/work/*|/state|/state/*|/opt|/opt/*|/usr|/usr/*|/etc|/etc/*|/dev|/dev/*|/proc|/proc/*|/sys|/sys/*|/bin|/bin/*|/sbin|/sbin/*|/lib|/lib/*|/lib64|/lib64/*)
      echo 'Prepared model-root view would obscure container runtime files.' >&2; exit 2 ;;
  esac
  [[ "$view" == /models ]] || MODEL_MOUNTS+=(--mount "type=bind,src=$MODEL_ROOT,dst=$view,readonly")
done < "$STATE/model-views.txt"
[[ -c /dev/kfd && -d /dev/dri ]] || { echo 'HIP device nodes are unavailable.' >&2; exit 2; }
[[ ${MI50_BLAS_REVIEWED:-0} == 1 ]] || {
  echo 'Review build.log / linked-libraries.txt first: Torch GEMM success does not validate this Strata BLAS stack.' >&2
  echo 'Set MI50_BLAS_REVIEWED=1 only after that library/runtime review approves a model start.' >&2; exit 2;
}
echo 'Both MI50s must be idle. This helper never stops production or changes its service/configuration.'
if [[ ${MI50_GPUS_IDLE_CONFIRMED:-0} != 1 ]]; then
  [[ -t 0 ]] || { echo 'Set MI50_GPUS_IDLE_CONFIRMED=1 only after checking both GPUs are idle.' >&2; exit 2; }
  read -r -p 'Have you stopped conflicting GPU work yourself and checked both cards are idle? [yes/no] ' ANSWER
  [[ "$ANSWER" == yes ]] || exit 2
fi
if [[ -z ${STRATA_API_KEY:-} ]]; then
  [[ -t 0 ]] || { echo 'Supply STRATA_API_KEY locally; do not paste it into chat.' >&2; exit 2; }
  read -r -s -p 'API key for the experimental server (hidden): ' STRATA_API_KEY
  printf '\n'
fi
[[ -n "$STRATA_API_KEY" && "$STRATA_API_KEY" != *$'\n'* && "$STRATA_API_KEY" != *$'\r'* ]] || {
  echo 'A nonempty single-line API key is required.' >&2; exit 2;
}
export STRATA_API_KEY
if [[ $(id -u) != 0 ]]; then DOCKER=(sudo --preserve-env=STRATA_API_KEY docker); fi
GROUPS_ARGS=()
while read -r group; do GROUPS_ARGS+=(--group-add "$group"); done < <(
  stat -c '%g' /dev/kfd /dev/dri/renderD* | sort -u
)
if [[ -e "$STATE/engine.log" || -e "$STATE/server.log" ]]; then
  ARCHIVE="$STATE/archive/$(date -u +%Y%m%dT%H%M%SZ)-$$"
  mkdir -p "$ARCHIVE"
  for log in engine.log server.log; do [[ ! -e "$STATE/$log" ]] || mv -- "$STATE/$log" "$ARCHIVE/$log"; done
  cp -- "$STATE/config.json" "$STATE/pipeline-switch.txt" "$BUILD/build-id.txt" "$ARCHIVE/"
  echo "Previous experimental logs retained in $ARCHIVE"
fi
echo 'Starting experimental API on 0.0.0.0:8001, with API-key authentication. Use a trusted network.'
echo 'No inference request is sent. Ctrl+C stops this foreground container.'
"${DOCKER[@]}" run "${COMMON[@]}" --network bridge --publish 0.0.0.0:8001:8001 \
  --device=/dev/kfd --device=/dev/dri "${GROUPS_ARGS[@]}" --ulimit memlock=-1:-1 --shm-size=1g \
  --tmpfs /tmp:rw,exec,nosuid,size=2g -e HIP_VISIBLE_DEVICES=0,1 -e STRATA_API_KEY \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --mount "type=bind,src=$ROOT,dst=/work,readonly" \
  "${MODEL_MOUNTS[@]}" \
  --mount "type=bind,src=$STATE,dst=/state" "$IMAGE_ID" -c '
set -euo pipefail
export PATH=/opt/venv/bin:/opt/rocm/bin:$PATH
exec python3 /work/tools/mi50_pipeline.py serve
' 2>&1 | tee "$STATE/server.log"
