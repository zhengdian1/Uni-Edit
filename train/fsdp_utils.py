# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file

from modeling.bagel.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.bagel.qwen2_navit import (
    Qwen2DecoderLayer, 
    Qwen2MoEDecoderLayer, 
    Qwen2MoTDecoderLayer,
)
from modeling.bagel.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[]):
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2DecoderLayer,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
                SiglipEncoderLayer,
                SiglipVisionTransformer,
                MLPconnector,
                TimestepEmbedder,
                PositionEmbedding,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
    )

import gc
class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir, 
        train_steps, 
        model, 
        ema_model, 
        optimizer, 
        scheduler, 
        data_status,
        logger, 
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        if ema_model is not None:
            with FSDP.state_dict_type(
                ema_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                ema_state_dict = ema_model.state_dict()
                if dist.get_rank() == 0:
                    save_file(ema_state_dict, os.path.join(save_path, "ema.safetensors"))
        
            del ema_state_dict
            gc.collect()
            dist.barrier() 


        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            model_state_dict = model.state_dict()
            if dist.get_rank() == 0:
                save_file(model_state_dict, os.path.join(save_path, "model.safetensors"))
        
        del model_state_dict
        gc.collect()
        dist.barrier() 

        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_save_path = os.path.join(
                save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                torch.save(optimizer.state_dict(), optimizer_save_path)
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                if dist.get_rank() < fsdp_config.num_shard:
                    torch.save(optimizer.state_dict(), optimizer_save_path)
            else:
                raise NotImplementedError

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if dist.get_rank() == 0 and data_status is not None:
            torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        dist.barrier()
        return

    # @staticmethod
    # def fsdp_save_ckpt(
    #     ckpt_dir, 
    #     train_steps, 
    #     model, 
    #     ema_model, 
    #     optimizer, 
    #     scheduler, 
    #     data_status,
    #     logger, 
    #     fsdp_config,
    # ):
    #     save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
    #     # 所有 Rank 都会尝试创建目录，exist_ok=True 保证安全
    #     os.makedirs(save_path, exist_ok=True)
        
    #     if dist.get_rank() == 0:
    #         logger.info(f"Saving checkpoint to {save_path}.")
    #     rank = dist.get_rank()

    #     if ema_model is not None:
    #         with FSDP.state_dict_type(
    #             ema_model,
    #             StateDictType.SHARDED_STATE_DICT,
    #             ShardedStateDictConfig(offload_to_cpu=False),
    #         ):
    #             ema_state_dict = ema_model.state_dict()
    #             save_name = f"ema-shard-{rank:05d}.safetensors"
    #             # for k, v in ema_state_dict.items():
    #             #     ema_state_dict[k] = v.contiguous()
    #             # save_file(ema_state_dict, os.path.join(save_path, save_name))
    #             clean_ema_dict = {k: v.detach().cpu() for k, v in ema_state_dict.items()}
    #             save_file(clean_ema_dict, os.path.join(save_path, save_name))
        
    #         del ema_state_dict
    #         del clean_ema_dict
    #         gc.collect()
    #         dist.barrier() 

    #     with FSDP.state_dict_type(
    #         model,
    #         StateDictType.SHARDED_STATE_DICT,
    #         ShardedStateDictConfig(offload_to_cpu=False),
    #     ):
    #         model_state_dict = model.state_dict()
    #         save_name = f"model-shard-{rank:05d}.safetensors"
    #         # for k, v in model_state_dict.items():
    #         #     model_state_dict[k] = v.contiguous()
    #         # save_file(model_state_dict, os.path.join(save_path, save_name))
    #         clean_model_dict = {k: v.detach().cpu() for k, v in model_state_dict.items()}
    #         save_file(clean_model_dict, os.path.join(save_path, save_name))
        
    #     del model_state_dict
    #     del clean_model_dict
    #     gc.collect()
    #     dist.barrier() 

    #     # === 3. Optimizer 等状态存储 (保持原样) ===
    #     with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
    #         if fsdp_config.sharding_strategy == "FULL_SHARD":
    #             shard_index = dist.get_rank()
    #             total_shards = dist.get_world_size()
    #         elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
    #             shard_index = dist.get_rank() % fsdp_config.num_shard
    #             total_shards = fsdp_config.num_shard
    #         else:
    #             raise NotImplementedError

    #         optimizer_save_path = os.path.join(
    #             save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
    #         )
    #         if fsdp_config.sharding_strategy == "FULL_SHARD":
    #             torch.save(optimizer.state_dict(), optimizer_save_path)
    #         elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
    #             if dist.get_rank() < fsdp_config.num_shard:
    #                 torch.save(optimizer.state_dict(), optimizer_save_path)
    #         else:
    #             raise NotImplementedError

    #     if dist.get_rank() == 0 and scheduler is not None:
    #         torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

    #     if dist.get_rank() == 0 and data_status is not None:
    #         torch.save(data_status, os.path.join(save_path, "data_status.pt"))

    #     dist.barrier()
    #     return

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
            else:
                model_state_dict_path = os.path.join(resume_from, f"model.safetensors")
            model_state_dict = load_file(model_state_dict_path, device="cpu")
            # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
            # which makes it easier to adapt to different resolutions.
            model_state_dict.pop('latent_pos_embed.pos_embed')
            model_state_dict.pop('vit_pos_embed.pos_embed')
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(msg)
            del model_state_dict

            if ema_model is not None:
                ema_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
                if not os.path.exists(ema_state_dict_path):
                    logger.info(f"replicaing ema model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
                # which makes it easier to adapt to different resolutions.
                ema_state_dict.pop('latent_pos_embed.pos_embed')
                ema_state_dict.pop('vit_pos_embed.pos_embed')
                msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                logger.info(msg)
                del ema_state_dict
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_ckpt_shared(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            rank = dist.get_rank()
            logger.info(f"[Rank {rank}] Loading sharded checkpoint from {resume_from}.")
            
            # 定义一个内部函数来处理加载逻辑，避免代码重复
            def load_shard_to_model(target_model, filename_prefix):
                shard_name = f"{filename_prefix}-shard-{rank:05d}.safetensors"
                shard_path = os.path.join(resume_from, shard_name)
                
                if not os.path.exists(shard_path):
                    # 如果分片不存在，可能是想从单文件恢复？或者是文件缺失
                    # 这里假设你已经全面切换到切片存储
                    raise FileNotFoundError(f"Shard file not found: {shard_path}")

                # 2. 加载分片到 CPU
                shard_state_dict = load_file(shard_path, device="cpu")

                # 3. 处理不需要的参数 (pos_embed)
                # 使用 pop(key, None) 避免 key 不在当前分片时报错
                shard_state_dict.pop('latent_pos_embed.pos_embed', None)
                shard_state_dict.pop('vit_pos_embed.pos_embed', None)

                # 4. 关键：在 SHARDED_STATE_DICT 上下文中加载
                # 这样 FSDP 才知道传入的 state_dict 只是当前卡的一部分
                with FSDP.state_dict_type(target_model, StateDictType.SHARDED_STATE_DICT):
                    msg = target_model.load_state_dict(shard_state_dict, strict=False)
                    logger.info(f"[Rank {rank}] Loaded {filename_prefix}: {msg}")
                
                del shard_state_dict
                gc.collect()

            # === 加载主模型 ===
            # 如果 resume_from_ema=True，则去读 ema-shard-xxxxx 文件加载给 model
            target_prefix = "ema" if resume_from_ema else "model"
            load_shard_to_model(model, target_prefix)

            # === 加载 EMA 模型 ===
            if ema_model is not None:
                # 检查 EMA 分片是否存在
                ema_shard_name = f"ema-shard-{rank:05d}.safetensors"
                ema_shard_path = os.path.join(resume_from, ema_shard_name)
                
                if os.path.exists(ema_shard_path):
                    load_shard_to_model(ema_model, "ema")
                else:
                    logger.info(f"[Rank {rank}] EMA shard missing, replicating from model shard.")
                    # 如果 EMA 文件不存在，就用 model 的权重初始化 EMA
                    # 注意：如果上面 resume_from_ema=True，这里其实是把 EMA 权重又加载回 EMA，逻辑也是通的
                    load_shard_to_model(ema_model, "model")

        else:
            logger.info(f"Training from scratch.")
        
        # 记得同步一下，确保所有卡都加载完了再继续
        dist.barrier()
        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2DecoderLayer, 
        SiglipEncoderLayer, 
        MLPconnector, 
        Qwen2MoEDecoderLayer, 
        Qwen2MoTDecoderLayer
    )
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)
