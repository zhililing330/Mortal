#!/usr/bin/env bash
set -euo pipefail

# Kaggle bootstrap for local/offline Mortal training.
#
# Expected Kaggle inputs:
#   /kaggle/input/mortal-core/libriichi.so
#   /kaggle/input/mortal-models/baseline.pth
#   /kaggle/input/mortal-models/mortal_pretrain_restart.pth
#   /kaggle/input/mortal-models/best_pretrain_restart.pth        optional
#   /kaggle/input/mortal-1v3-data/**/*.json.gz                   optional training data
#
# Usage in a Kaggle notebook:
#   !git clone https://github.com/zhililing330/Mortal.git /kaggle/working/Mortal
#   !cd /kaggle/working/Mortal && bash kaggle_init.sh
#   !cd /kaggle/working/Mortal/mortal && MORTAL_CFG=config.kaggle.toml python train.py

REPO_ROOT="${REPO_ROOT:-/kaggle/working/Mortal}"
MORTAL_DIR="$REPO_ROOT/mortal"

CORE_DIR="${CORE_DIR:-/kaggle/input/mortal-core}"
MODELS_DIR="${MODELS_DIR:-/kaggle/input/mortal-models}"

LIBRIICHI_SO="${LIBRIICHI_SO:-$CORE_DIR/libriichi.so}"
BASELINE_PTH="${BASELINE_PTH:-$MODELS_DIR/baseline.pth}"
PRETRAIN_PTH="${PRETRAIN_PTH:-$MODELS_DIR/mortal_pretrain_restart.pth}"
BEST_PTH="${BEST_PTH:-$MODELS_DIR/best_pretrain_restart.pth}"

echo "[kaggle] repo: $REPO_ROOT"
cd "$REPO_ROOT"

echo "[kaggle] installing Python dependencies"
python -m pip install -q --upgrade pip
python -m pip install -q toml tqdm tensorboard

echo "[kaggle] preparing runtime directories"
mkdir -p "$MORTAL_DIR/local_models" "$MORTAL_DIR/local_logs"
printf 'mortal\n' > "$MORTAL_DIR/player_names_mortal.txt"

echo "[kaggle] installing prebuilt libriichi.so"
if [[ ! -f "$LIBRIICHI_SO" ]]; then
    echo "[kaggle][error] missing $LIBRIICHI_SO"
    echo "Upload the Kaggle dataset that contains your compiled Linux libriichi.so,"
    echo "or set LIBRIICHI_SO=/path/to/libriichi.so before running this script."
    exit 1
fi
cp -f "$LIBRIICHI_SO" "$MORTAL_DIR/libriichi.so"

echo "[kaggle] installing model checkpoints"
if [[ ! -f "$BASELINE_PTH" ]]; then
    echo "[kaggle][error] missing $BASELINE_PTH"
    exit 1
fi
if [[ ! -f "$PRETRAIN_PTH" ]]; then
    echo "[kaggle][error] missing $PRETRAIN_PTH"
    exit 1
fi
cp -f "$BASELINE_PTH" "$MORTAL_DIR/local_models/baseline.pth"
cp -f "$PRETRAIN_PTH" "$MORTAL_DIR/local_models/mortal_pretrain_restart.pth"
if [[ -f "$BEST_PTH" ]]; then
    cp -f "$BEST_PTH" "$MORTAL_DIR/local_models/best_pretrain_restart.pth"
else
    cp -f "$PRETRAIN_PTH" "$MORTAL_DIR/local_models/best_pretrain_restart.pth"
fi

echo "[kaggle] checking Python imports"
cd "$MORTAL_DIR"
python - <<'PY'
import torch
import libriichi
from libriichi.consts import ACTION_SPACE, obs_shape

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("libriichi ok:", ACTION_SPACE, obs_shape(4))
PY

echo "[kaggle] ready"
echo "Run:"
echo "  cd $MORTAL_DIR"
echo "  MORTAL_CFG=config.kaggle.toml python train.py"
