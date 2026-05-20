# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

# Set proxy and API key
export OPENAI_API_KEY=''
echo $PYTHONPATH
export GPUS=8

DATASETS=("mathvista-testmini-online")
output_path="results/rep/mathvista"

model_path="no_need"


DATASETS_STR="${DATASETS[*]}"
export DATASETS_STR

bash scripts/eval/eval_vlm.sh \
    $output_path \
    $model_path