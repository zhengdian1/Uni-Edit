# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

echo $PYTHONPATH
export GPUS=8

# DATASETS=("mmmu-val","mme","mathvista-testmini","mmbench-dev-en","mmvp",)
DATASETS=("mmvp")

output_path="results/mmvp"

model_path="your/path/to/Uni-Edit/Uni-Edit-BAGEL"


DATASETS_STR="${DATASETS[*]}"
export DATASETS_STR

bash scripts/eval/eval_vlm.sh \
    $output_path \
    $model_path