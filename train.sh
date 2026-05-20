# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# --------------------------sft--------------------------

torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file data/configs/example.yaml \
  --model_path ByteDance-Seed/BAGEL-7B-MoT \
  --llm_path Qwen/Qwen2.5-7B-Instruct \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from ByteDance-Seed/BAGEL-7B-MoT \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 1 \
  --freeze_vit True \
  --visual_gen True \
  --save_every 2500 \
  --total_steps 500000 \
  --vit_cond_dropout_prob 0.0 \
  --vae_cond_dropout_prob 1 \
  --results_dir OUTPUT_FILE \
  --checkpoint_dir OUTPUT_FILE/checkpoints \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240