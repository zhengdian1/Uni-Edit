# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

export OPENAI_API_KEY=''

output_path="results/imgedit"
model_path="your/path/to/Uni-Edit/Uni-Edit-BAGEL"
GPUS=8

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=22345 \
    eval/gen/gen_images_mp_imgedit.py \
    --output_dir $output_path \
    --metadata_file eval/gen/imgedit/Benchmark/singleturn/singleturn.json \
    --max_latent_size 64 \
    --model-path $model_path


python eval/gen/imgedit/basic_bench.py \
    --result_img_folder $output_path \
    --edit_json eval/gen/imgedit/Benchmark/singleturn/singleturn.json \
    --origin_img_root eval/gen/imgedit/Benchmark/singleturn \
    --num_processes 4 \
    --prompts_json eval/gen/imgedit/Benchmark/singleturn/judge_prompt.json

# summarize score
python eval/gen/imgedit/step1_get_avgscore.py \
    --result_json $output_path/result.json \
    --average_score_json $output_path/average_score.json

python eval/gen/imgedit/step2_typescore.py \
    --average_score_json  $output_path/average_score.json \
    --edit_json eval/gen/imgedit/Benchmark/singleturn/singleturn.json \
    --typescore_json $output_path/typescore.json