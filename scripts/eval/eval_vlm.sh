# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# Check if enough arguments are provided
LOG_PATH=$1
FULL_MODEL_PATH=$2
shift 1
ARGS=("$@")
export MASTER_PORT=10042

IFS=' ' read -r -a DATASETS <<< "$DATASETS_STR"

for DATASET in "${DATASETS[@]}"; do
    bash eval/vlm/evaluate.sh \
        "$DATASET" \
        --out-dir "$LOG_PATH/$DATASET" \
        --model-path "$FULL_MODEL_PATH" 
done