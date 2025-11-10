#!/usr/bin/env bash
set -euo pipefail

# Rewritten host launcher: build the Docker image from this repo and run train.py
# Behavior:
# - Builds image tagged $IMAGE if it doesn't exist
# - Mounts repo, results, data and hf_cache directories into /workspace
# - Rewrites --data-dir or --data-path provided as host paths to /workspace/data
# - Runs `python -u train.py` inside the container as the invoking host UID

# image tag to build/use
IMAGE=${IMAGE:-meanflow:local}
ROOT="$(cd "$(dirname "$0")" && pwd)"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Docker image $IMAGE not found. Building image from $ROOT..."
  docker build -t "$IMAGE" "$ROOT"
fi

# Collect args and drop a leading bare -- if present
ARGS=("$@")
if [ ${#ARGS[@]} -gt 0 ] && [ "${ARGS[0]}" = "--" ]; then
  ARGS=("${ARGS[@]:1}")
fi

# Rewrite --data-dir / --data-dir= / --data-path / --data-path=
HOST_DATA_DIR=""
for i in "${!ARGS[@]}"; do
  a="${ARGS[$i]}"
  case "$a" in
    --data-dir)
      if [ $((i+1)) -lt ${#ARGS[@]} ]; then
        val="${ARGS[$((i+1))]}"
        if [[ "$val" != /workspace/* ]]; then
          HOST_DATA_DIR="$val"
          ARGS[$((i+1))]="/workspace/data"
        fi
      fi
      ;;
    --data-dir=*)
      val="${a#--data-dir=}"
      if [[ "$val" != /workspace/* ]]; then
        HOST_DATA_DIR="$val"
        ARGS[$i]="--data-dir=/workspace/data"
      fi
      ;;
    --data-path)
      if [ $((i+1)) -lt ${#ARGS[@]} ]; then
        val="${ARGS[$((i+1))]}"
        if [[ "$val" != /workspace/* ]]; then
          HOST_DATA_DIR="$val"
          ARGS[$((i+1))]="/workspace/data"
        fi
      fi
      ;;
    --data-path=*)
      val="${a#--data-path=}"
      if [[ "$val" != /workspace/* ]]; then
        HOST_DATA_DIR="$val"
        ARGS[$i]="--data-path=/workspace/data"
      fi
      ;;
  esac
done

# Mount locations
: ${DATA_DIR:=${HOST_DATA_DIR:-$ROOT/data}}
: ${RESULTS_DIR:=${ROOT}/results}
: ${HF_CACHE:=${ROOT}/hf_cache}

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$HF_CACHE"

# Forward common env vars (example)
ENV_ARGS=()
for var in WANDB_API_KEY NCCL_DEBUG; do
  if [ -n "${!var-}" ]; then
    ENV_ARGS+=("-e" "$var=${!var}")
  fi
done

USER_ARG=("--user" "$(id -u):$(id -g)")

docker run --gpus=all -it --rm "${USER_ARG[@]}" \
  -v "$ROOT":/workspace \
  -v "$RESULTS_DIR":/workspace/results \
  -v "$DATA_DIR":/workspace/data \
  -v "$HF_CACHE":/workspace/hf_cache \
  -w /workspace \
  "${ENV_ARGS[@]}" \
  "$IMAGE" \
  python -u train.py "${ARGS[@]}"
