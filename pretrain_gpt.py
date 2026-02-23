# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import math
import os
import torch
from functools import partial
from contextlib import nullcontext
import inspect

from typing import List, Optional, Tuple, Union

from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)


stimer = StragglerDetector()


def tokens_to_packed_seq_params(input_ids, eod_token, orig_seq_len, qkv_format='thd', cu_seqlens_padded=None):
    """
    Compute PackedSeqParams from input tokens using EOD token boundaries.

    Args:
        input_ids: Input token IDs, shape assumed flattened (1, tokens) or (tokens)
        eod_token: End-of-Document token ID (from tokenizer.eod)
        orig_seq_len: Original sequence length for fixed boundaries
        qkv_format: QKV format - 'sbhd' (default) or 'thd' (for CP with padding)
        cu_seqlens_padded: Optional padded cumulative lengths for context parallelism

    Returns:
        PackedSeqParams with cu_seqlens respecting both EOD and orig_seq_len boundaries
    """
    from megatron.core.packed_seq_params import PackedSeqParams

    # Create boundaries at fixed intervals (based on orig_seq_len)
    # Find EOD token positions (+1 to mark position AFTER eod)
    # Concatenate and sort to get all boundaries (fixed + EOD)
    cu_seq, _ = torch.sort(torch.cat((
        torch.arange(0, input_ids.size(-1) + orig_seq_len, orig_seq_len, device=input_ids.device, dtype=torch.int32),
        (input_ids.flatten() == eod_token).nonzero()[:, 0].int() + 1,
    )))

    # Compute max sequence length between boundaries
    max_len = (cu_seq[1:] - cu_seq[:-1]).max()

    return PackedSeqParams(
        cu_seqlens_q=cu_seq,
        cu_seqlens_kv=cu_seq,
        max_seqlen_q=max_len,
        max_seqlen_kv=max_len,
        qkv_format=qkv_format,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        num_experts=args.num_experts, 
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm, 
                        multi_latent_attention=args.multi_latent_attention,
                        attn_layernorm=args.attn_layernorm,
                        mlp_layernorm=args.mlp_layernorm,
                        qknorm_impl=args.qknorm_impl,
                        post_layer_norm=args.post_layer_norm,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)

        # Logic to handle additional vocab size for multimodal models
        if hasattr(args, "base_vocab_size") and args.base_vocab_size != args.padded_vocab_size:
            args.total_multimodal_vocab_size = args.padded_vocab_size
            if args.extend_model_vocab:
                args.padded_vocab_size = args.base_padded_vocab_size

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling,
                final_layernorm=args.final_layernorm,
                input_embeddings_multiplier=args.input_embeddings_multiplier,
            )

    print_rank_0("Built model:")
    print_rank_0(model)
    print_rank_0("Config:")
    print_rank_0(config)

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


# define spiky loss as a variation of 20% or more
SPIKY_LOSS_PERC = 0.2


def get_current_image_weight(args):
    """Compute image weight for the current training step.

    Returns a static weight when --image-weight-decay is not enabled,
    otherwise computes a decay from image_weight_max to image_weight_min
    using the selected schedule (logistic, cosine, or linear).
    """
    if not args.image_weight_decay:
        return args.image_weight

    step = getattr(args, 'curr_iteration', 0)
    max_w = args.image_weight_max
    min_w = args.image_weight_min
    start_step = args.image_weight_decay_start_step
    end_step = args.image_weight_decay_end_step
    schedule = args.image_weight_decay_schedule

    if step <= start_step:
        return max_w
    if step >= end_step:
        return min_w

    # Normalize step within [start_step, end_step] to [0, 1]
    t = (step - start_step) / (end_step - start_step)

    if schedule == 'cosine':
        cosine_decay = 0.5 * (1 + math.cos(t * math.pi))
        return min_w + (max_w - min_w) * cosine_decay
    elif schedule == 'linear':
        return max_w - (max_w - min_w) * t
    else:
        raise ValueError(f"Unknown image weight decay schedule: {schedule}")


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, labels: torch.Tensor = None, assistant_mask: torch.Tensor = None, current_image_weight: float = None):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        labels (torch.Tensor): The labels for image/text loss separation (optional)
        assistant_mask (torch.Tensor): Mask identifying assistant tokens for SFT (optional)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(rerun_state_machine.is_spiky_loss, threshold=SPIKY_LOSS_PERC),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)

    # --- Per-modality token losses and separate assistant loss in SFT---
    stats_dict = {'lm loss': (reporting_loss[0], reporting_loss[1])}

    if labels is not None and hasattr(args, 'base_vocab_size'):
        losses_flat = losses.detach().reshape(-1)
        labels_flat = labels.reshape(-1)
        loss_mask_flat = loss_mask.reshape(-1)

        modalities = [('text', 0, args.base_vocab_size)]
        omnimodal_config = getattr(args, 'omnimodal_config', None)
        if omnimodal_config is not None:
            for modality in omnimodal_config.get('modalities', []):
                modalities.append((modality['name'], modality['offset'], modality['vocab_size']))

        sum_list = []
        weighted_count_list = []
        raw_count_list = []
        for name, offset, vocab_size in modalities:
            in_range = (labels_flat >= offset) & (labels_flat < offset + vocab_size)
            in_range_f = in_range.float()
            weights = loss_mask_flat * in_range_f
            sum_list.append(torch.sum(losses_flat * weights))
            weighted_count_list.append(torch.sum(weights))
            raw_count_list.append(torch.sum(in_range_f))

        modality_stats = torch.stack((
            torch.stack(sum_list),
            torch.stack(weighted_count_list),
            torch.stack(raw_count_list),
        ), dim=1)
        if args.context_parallel_size > 1:
            torch.distributed.all_reduce(modality_stats, group=mpu.get_context_parallel_group())
        torch.distributed.all_reduce(modality_stats, group=mpu.get_data_parallel_group())

        for i, (name, _, _) in enumerate(modalities):
            # sum / weighted_count = per-token avg raw CE (image_weight cancels out)
            stats_dict[f'{name}_token_loss'] = (modality_stats[i, 0], modality_stats[i, 1])
            # sum / raw_count = weighted avg contribution per token (reflects image_weight)
            stats_dict[f'{name}_weighted_loss'] = (modality_stats[i, 0], modality_stats[i, 2])

        # Log separate assistant loss for SFT training
        if args.sft and assistant_mask is not None:
            # Use assistant_mask to identify assistant response tokens
            assistant_mask_flat = assistant_mask.view(-1).float()
            assert (assistant_mask_flat >= 0).all(), f"assistant_mask has negative values: min={assistant_mask_flat.min()}"
            assert (losses_flat >= 0).all(), f"CE losses have negative values: min={losses_flat.min()}"
            assistant_loss = torch.cat([
                torch.sum(losses_flat * assistant_mask_flat).view(1),
                assistant_mask_flat.sum().view(1)
            ])
            if args.context_parallel_size > 1:
                torch.distributed.all_reduce(assistant_loss, group=mpu.get_context_parallel_group())
            reporting_assistant_loss = assistant_loss.clone().detach()
            torch.distributed.all_reduce(reporting_assistant_loss, group=mpu.get_data_parallel_group())
            stats_dict['assistant_loss'] = (reporting_assistant_loss[0], reporting_assistant_loss[1])

    if args.log_image_weight:
        stats_dict['image-weight'] = current_image_weight

    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        stats_dict,
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids, assistant_mask = get_batch(
            data_iterator)

        # Compute packed sequence parameters if enabled
        packed_seq_params = None
        if args.use_packed_seq_params:
            # Reshape tensors from [B, S] to [1, B*S] for THD format
            tokens = tokens.view(1, -1)  # [1, B*S]
            labels = labels.view(1, -1)  # [1, B*S]
            loss_mask = loss_mask.view(1, -1)  # [1, B*S]
            position_ids = position_ids.view(1, -1)  # [1, B*S]
            if args.sft and assistant_mask is not None:
                assistant_mask = assistant_mask.view(1, -1) # [1, B*S]
            # Note: attention_mask not needed in THD format with packed_seq_params

            tokenizer = get_tokenizer()

            # Hardcoded to 'thd' format for now
            # TODO: Add cu_seqlens_padded support for context parallelism when needed
            qkv_format = 'thd'

            packed_seq_params = tokens_to_packed_seq_params(
                tokens,
                eod_token=tokenizer.eod,
                orig_seq_len=args.seq_length,
                qkv_format=qkv_format,
                cu_seqlens_padded=None  # TODO: Compute for CP support
            )
    timers('batch-generator').stop()

    # Apply dynamic image weight decay (overrides static weight set in dataset)
    current_image_weight = get_current_image_weight(args)
    if args.image_weight_decay:
        if labels is not None and hasattr(args, 'vision_token_offset'):
            image_mask = (labels >= args.vision_token_offset) & (
                labels < args.vision_token_offset + args.vision_vocab_size
            )
            loss_mask[image_mask] = current_image_weight

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels, packed_seq_params=packed_seq_params)

    return output_tensor, partial(loss_func, loss_mask, labels=labels,
                                  assistant_mask=assistant_mask,
                                  current_image_weight=current_image_weight)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    # Double sequence length if loading loss masks from disk (dataset stores tokens + loss_mask concatenated)
    sequence_length = args.seq_length
    if args.sft and args.sft_load_loss_mask:
        sequence_length = args.seq_length * 2
        print_rank_0(f"> SFT: Loading loss masks from disk, doubling dataset sequence_length to {sequence_length} "
                     f"(model will see {args.seq_length} tokens)")

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=sequence_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
        goldfish_loss=args.goldfish_loss,
        goldfish_k=args.goldfish_k,
        goldfish_h=args.goldfish_h,
        sft_mask_special_tokens=args.sft_mask_special_tokens,
        sft_plw=args.sft_plw,
        sft_pack_samples=args.sft_pack_samples,
        sft_equalize_sample_loss=args.sft_equalize_sample_loss,
        sft_load_loss_mask=args.sft_load_loss_mask,
        skip_margin_samples=args.data_skip_margin_samples,
        image_weight=args.image_weight,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        from megatron.core.datasets.sft_dataset import SFTIndexedDataset
        dataset_type = SFTIndexedDataset
    elif args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
