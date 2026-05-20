# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

export OPENAI_API_KEY=''

GPUS=8

output_path="results/rise"
model_path="your/path/to/Uni-Edit/Uni-Edit-BAGEL"

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12348 \
    eval/gen/gen_images_mp_rise.py \
    --output_dir $output_path/rise \
    --metadata_file eval/gen/rise/data/datav2_total_w_subtask.json \
    --max_latent_size 64 \
    --model-path $model_path \
    --think

python eval/gen/rise/gpt_eval.py \
    --data eval/gen/rise/data/datav2_total_w_subtask.json \
    --input eval/gen/rise/data \
    --output $output_path/rise