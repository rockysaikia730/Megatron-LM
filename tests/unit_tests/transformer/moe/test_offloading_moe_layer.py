# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils
from megatron.core.transformer.moe.experts import (
    OffloadingExpertsMLP
)

from megatron.core.transformer.moe.experts_util import (
    grouped_swiglu_mlp,
    MergedSwiGLU,
)

from megatron.core.transformer.moe.experts_offloading_util import (
    offloading_grouped_swiglu_mlp,
    OffloadingExpertsGroupedSwiMLP,
    StreamManager,
)



class TestOffloadingMoELayerBF16:
    """Test MoE layer with FP16 precision."""

    def setup_method(self, method):
        pass

    @pytest.mark.parametrize("num_moe_experts", [32])
    def test_offloading_experts_bf16_forward(
        self, num_moe_experts, 
    ):
        """Test MoE layer forward and backward pass with fp16 params and inputs."""
        # _set_random_seed(seed_=123, data_parallel_random_init=False)

        hidden_size = 2048
        moe_ffn_hidden_size = 1024

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_moe_experts=num_moe_experts,
            num_attention_heads=16,
            use_cpu_initialization=True,
            perform_initialization=False,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            add_bias_linear=False,
            fp16=False,
            params_dtype=torch.bfloat16,

            gated_linear_unit=True,
            moe_offloading_num_chunks=4,
            moe_offloading_num_stages=2,
            # moe_offloading_chunk_size=4,
        )

        offloading_expert = OffloadingExpertsMLP(
            num_moe_experts,
            transformer_config
        )

        tokens_per_expert = torch.randint(
            2047, 2048, (num_moe_experts,), 
            device="cpu"
        )

        hidden_states = torch.randn(
            tokens_per_expert.sum().item(),
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        permuted_probs = torch.rand(
            tokens_per_expert.sum().item(),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        # Forward pass
        # output, _ = offloading_expert(
        #     hidden_states,
        #     tokens_per_expert,
        #     permuted_probs,
        # )

        wait, warmup, active = 1, 1, 2
        num_steps = wait + warmup + active
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=1, skip_first=1
            ),
        ) as prof:
            for _ in range(num_steps):
                output, _ = offloading_expert(
                    hidden_states,
                    tokens_per_expert,
                    permuted_probs,
                )
                prof.step()
        prof.export_chrome_trace(f"test.json")


    def test_offloading_moe_forward_backward(
        self, num_moe_experts, profile=False
    ):
        """Test MoE layer forward and backward pass with fp16 params and inputs."""

        micro_batch_size = 2
        sequence_length = 4096
        hidden_size = 7168
        moe_ffn_hidden_size = 2048
        moe_offloading_num_chunks = 4
        moe_offloading_num_stages = 2
        moe_offloading_chunk_size = num_moe_experts // moe_offloading_num_chunks

        tokens_per_expert = torch.randint(
            511, 512, (num_moe_experts,), 
            device="cpu"
        )
        tokens_per_expert_ref = tokens_per_expert.detach().clone()

        hidden_states = torch.randn(
            tokens_per_expert.sum().item(),
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        hidden_states_ref = hidden_states.detach().clone()

        permuted_probs = torch.rand(
            tokens_per_expert.sum().item(),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        permuted_probs_ref = permuted_probs.detach().clone()

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_moe_experts=num_moe_experts,
            num_attention_heads=16,
            use_cpu_initialization=True,
            perform_initialization=False,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            add_bias_linear=False,
            fp16=False,
            params_dtype=torch.bfloat16,

            gated_linear_unit=True,
            moe_offloading_num_chunks=moe_offloading_num_chunks,
            moe_offloading_num_stages=moe_offloading_num_stages,
            moe_offloading_chunk_size=moe_offloading_chunk_size,
            gradient_accumulation_fusion=True,
        )

        cpu_w1 = [
            torch.nn.Parameter(
                torch.randn(
                    hidden_size, moe_ffn_hidden_size*2, device="cpu", dtype=torch.bfloat16, requires_grad=True,
                    pin_memory=True
                )
            ) for _ in range(num_moe_experts)
        ]
        cpu_w2 = [
            torch.nn.Parameter(
                torch.randn(
                    moe_ffn_hidden_size, hidden_size, device="cpu", dtype=torch.bfloat16, requires_grad=True,
                    pin_memory=True
                )
            ) for _ in range(num_moe_experts)
        ]

        # main grad
        for i in range(num_moe_experts):
            cpu_w1[i].main_grad = torch.zeros_like(cpu_w1[i], device="cuda", dtype=torch.float32)
            cpu_w2[i].main_grad = torch.zeros_like(cpu_w2[i], device="cuda", dtype=torch.float32)

        gpu_w1_buffer = [
            [torch.empty(
                hidden_size, moe_ffn_hidden_size*2, device="cuda", dtype=torch.bfloat16, requires_grad=False
            ) for _ in range(moe_offloading_chunk_size)] for _ in range(moe_offloading_num_stages)
        ]
        gpu_w2_buffer = [
            [torch.empty(
                 moe_ffn_hidden_size, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=False
            ) for _ in range(moe_offloading_chunk_size)] for _ in range(moe_offloading_num_stages)
        ]

        # reference
        gpu_w1 = [
            torch.nn.Parameter(
                cpu_w1[i].detach().clone().cuda(), requires_grad=True
            ) for i in range(num_moe_experts)
        ]
        gpu_w2 = [
            torch.nn.Parameter(
                cpu_w2[i].detach().clone().cuda(), requires_grad=True
            ) for i in range(num_moe_experts)
        ]
        for i in range(num_moe_experts):
            gpu_w1[i].main_grad = torch.zeros_like(gpu_w1[i], device="cuda", dtype=torch.float32)
            gpu_w2[i].main_grad = torch.zeros_like(gpu_w2[i], device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()

        stream_manager = StreamManager(moe_offloading_num_stages)

        if not profile:
            hidden_states_ref_list = list(torch.split(hidden_states_ref, tokens_per_expert_ref.tolist(), dim=0))
            output_ref_list = []
            for i in range(num_moe_experts):
                output_ref_list.append(
                    torch.mm(hidden_states_ref_list[i], gpu_w1[i])
                )
            outputs_fc1 = torch.cat(output_ref_list, dim=0)
            outputs_act = MergedSwiGLU.apply(outputs_fc1, permuted_probs_ref.unsqueeze(-1))
            outputs_act_list = list(torch.split(outputs_act, tokens_per_expert_ref.tolist(), dim=0))
            output_ref_list = []
            for i in range(num_moe_experts):
                output_ref_list.append(
                    torch.mm(outputs_act_list[i], gpu_w2[i])
                )
            output_ref = torch.cat(output_ref_list, dim=0)
            output_ref.sum().backward()
            torch.cuda.synchronize()

        if profile:
            wait, warmup, active = 1, 5, 2
            num_steps = wait + warmup + active
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=wait, warmup=warmup, active=active, repeat=1, skip_first=1
                ),
            ) as prof:
                for _ in range(num_steps):
                    output = offloading_grouped_swiglu_mlp(
                        cpu_w1,
                        cpu_w2,
                        gpu_w1_buffer,
                        gpu_w2_buffer,
                        hidden_states,
                        tokens_per_expert,
                        num_moe_experts,
                        permuted_probs,
                        None,
                        stream_manager,
                        transformer_config
                    )
                    torch.cuda.synchronize()
                    output.sum().backward()
                    torch.cuda.synchronize()
                    prof.step()
            prof.export_chrome_trace(f"e2e.json")
        else:
            output = offloading_grouped_swiglu_mlp(
                cpu_w1,
                cpu_w2,
                gpu_w1_buffer,
                gpu_w2_buffer,
                hidden_states,
                tokens_per_expert,
                num_moe_experts,
                permuted_probs,
                None,
                stream_manager,
                transformer_config
            )
            output.sum().backward()
            torch.cuda.synchronize()


        if not profile:
            def diff_tensor_norm(tensor1, tensor2):
                return torch.norm(tensor1.to(torch.float) - tensor2.to(torch.float)).item() / torch.norm(tensor2.to(torch.float)).item()

            assert output.shape == output_ref.shape, f"Output shape mismatch: {output.shape} vs {output_ref.shape}"
            print(diff_tensor_norm(output.cuda(), output_ref.cuda()))
            print(output.cuda())
            print(output_ref.cuda())
            assert torch.allclose(output.cuda(), output_ref.cuda(), atol=1e-3)

            for i in range(num_moe_experts):
                print(diff_tensor_norm(cpu_w1[i].main_grad, gpu_w1[i].grad))
                print(diff_tensor_norm(cpu_w2[i].main_grad, gpu_w2[i].grad))

    def test_offloading_moe_layer(
        self,
        num_moe_experts, 
        moe_token_dispatcher_type, 
        tp_size, 
        ep_size,
    ):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        hidden_size = 2048
        sequence_length = 4096
        micro_batch_size = 2

        moe_ffn_hidden_size = 1024
        moe_offloading_num_chunks = 4
        moe_offloading_num_stages = 2
        moe_offloading_chunk_size = num_moe_experts // moe_offloading_num_chunks

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=32,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=8,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=True,  # Use SequentialMLP for fp16 test
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            add_bias_linear=False,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            fp16=False,

            bf16=True,
            params_dtype=torch.bfloat16,
            gated_linear_unit=True,
            gradient_accumulation_fusion=True,
            activation_func=torch.nn.functional.silu,

            moe_use_legacy_grouped_gemm=True,
            moe_use_offloading_experts=True,
            moe_offloading_num_chunks=moe_offloading_num_chunks,
            moe_offloading_num_stages=moe_offloading_num_stages,
            moe_offloading_chunk_size=moe_offloading_chunk_size,
        )

        transformer_layer_spec = get_gpt_layer_local_spec(
            num_experts=num_moe_experts, 
            moe_grouped_gemm=True,
            moe_use_offloading_experts=True,
        )

        moe_layer = MoELayer(
            transformer_config, transformer_layer_spec.submodules.mlp.submodules
        )
        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        moe_layer.router.set_layer_number(1)

        output, _ = moe_layer(hidden_states)

        



        

if __name__ == "__main__":
    TestOffloadingMoELayerBF16().test_offloading_moe_forward_backward(
        num_moe_experts=64, profile=True
    )
    
    
    