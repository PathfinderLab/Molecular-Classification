#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"
: "${TCGA_FEATURE_DIR:?Set TCGA_FEATURE_DIR to a folder of slide-ID.h5 UNI features}"
: "${DGIST_FEATURE_DIR:?Set DGIST_FEATURE_DIR to a folder of slide-ID.h5 UNI features}"
for model in transmil_mba acmil; do
  "$python_bin" -u "$repo_dir/train.py" --model "$model" --tcga-features "$TCGA_FEATURE_DIR"
  "$python_bin" -u "$repo_dir/evaluate.py" --model "$model" --tcga-features "$TCGA_FEATURE_DIR" --dgist-features "$DGIST_FEATURE_DIR"
done
