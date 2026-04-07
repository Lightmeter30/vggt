#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_DIR="$ROOT_DIR/training"
CONFIG_NAME="${CONFIG_NAME:-local_finetune}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-my_vggt_relocation}"
CO3D_DIR="$ROOT_DIR/dataset/co3d"
CO3D_ANNO_LOCAL_DIR="$ROOT_DIR/dataset/co3d-anno-local"
CKPT_PATH="$ROOT_DIR/ckpt/model.pt"
PREPARE_LOCAL_ANNO="${PREPARE_LOCAL_ANNO:-1}"

GPU_IDS="${1:-${GPU_IDS:-0}}"
MASTER_PORT="${2:-${MASTER_PORT:-29601}}"

if [[ $# -ge 3 ]]; then
  NPROC_PER_NODE="$3"
else
  IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
  NPROC_PER_NODE="${#GPU_ARRAY[@]}"
fi

if [[ ! -d "$CO3D_DIR" ]]; then
  echo "Missing dataset directory: $CO3D_DIR" >&2
  exit 1
fi

if [[ ! -e "$CKPT_PATH" ]]; then
  echo "Missing checkpoint: $CKPT_PATH" >&2
  exit 1
fi

if [[ "$PREPARE_LOCAL_ANNO" == "1" || ! -f "$CO3D_ANNO_LOCAL_DIR/summary.json" ]]; then
  echo "[local-finetune] regenerating local Co3D annotations..."
  python3 "$ROOT_DIR/training/data/preprocess/generate_local_co3d_annotations.py" \
    --co3d_dir "$CO3D_DIR" \
    --output_dir "$CO3D_ANNO_LOCAL_DIR"
fi

echo "[local-finetune] config       : $CONFIG_NAME"
echo "[local-finetune] conda env    : $CONDA_ENV_NAME"
echo "[local-finetune] gpu ids      : $GPU_IDS"
echo "[local-finetune] nproc/node   : $NPROC_PER_NODE"
echo "[local-finetune] master port  : $MASTER_PORT"
echo "[local-finetune] dataset      : $CO3D_DIR"
echo "[local-finetune] annotations  : $CO3D_ANNO_LOCAL_DIR"
echo "[local-finetune] checkpoint   : $CKPT_PATH"

cd "$TRAIN_DIR"

exec conda run -n "$CONDA_ENV_NAME" env \
  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  PYTHONPATH="$ROOT_DIR" \
  torchrun \
  --master_port "$MASTER_PORT" \
  --nproc_per_node="$NPROC_PER_NODE" \
  launch.py \
  --config "$CONFIG_NAME"
