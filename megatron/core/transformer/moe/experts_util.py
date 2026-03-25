
from __future__ import annotations
import torch
import collections

from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from transformer_engine.pytorch import (
        Float8BlockQuantizer,
    )
    from transformer_engine.pytorch.constants import TE_DType
    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None
        

class MergedSwiGLU(torch.autograd.Function):
    """Re-implementation of Silu
    """

    @classmethod
    @torch.compile()
    def call_forward(
        cls,
        input_tensor: torch.Tensor,
        probs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """forward with optional probability scaling for SwiGLU activation. 
        If `probs` is provided, it will be used to scale the output of the SwiGLU activation, 
        otherwise it will compute the standard SwiGLU activation without scaling.

        Args:
            input_tensor (torch.Tensor): input tensor to the activation function
            probs (torch.Tensor | None, optional): Defaults to None.

        Returns:
            torch.Tensor: activation output
        """
        if probs is not None:
            return MergedSwiGLU.call_forward_silu_probs(input_tensor, probs)
        else:
            return MergedSwiGLU.call_forward_silu(input_tensor)

    @classmethod
    @torch.compile()
    def call_forward_silu(
        cls,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """forward pass for SwiGLU activation without probability scaling.

        Args:
            input_tensor (torch.Tensor): input tensor to the activation function

        Returns:
            torch.Tensor: activation output
        """
        a, b = input_tensor.chunk(2, dim=-1)
        return (torch.nn.functional.silu(a) * b).to(input_tensor.dtype)
    
    @classmethod
    @torch.compile()
    def call_forward_silu_probs(
        cls,
        input_tensor: torch.Tensor,
        probs: torch.Tensor
    ) -> torch.Tensor:
        """actual forward function with probability.

        Args:
            input_tensor (torch.Tensor): input tensor to the activation function
            probs (torch.Tensor): probability derived from router

        Returns:
            torch.Tensor: activation output
        """
        a, b = input_tensor.chunk(2, dim=-1)
        return ((torch.nn.functional.silu(a) * b) * probs).to(input_tensor.dtype)
    
    @classmethod
    @torch.compile()
    def call_backward(
        cls,
        grad_output: torch.Tensor,
        input_tensor: torch.Tensor,
        probs: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor | None, torch.Tensor | None]:
        """backward function for SwiGLU activation with optional probability scaling.

        Args:
            grad_output (torch.Tensor): gradient of the output from the activation function
            input_tensor (torch.Tensor): input tensor to the activation function
            probs (torch.Tensor | None, optional): Defaults to None.
        """
        if probs is not None:
            return MergedSwiGLU.call_backward_silu_probs(grad_output, input_tensor, probs)
        else:
            return MergedSwiGLU.call_backward_silu(grad_output, input_tensor)
    
    @classmethod
    @torch.compile()
    def call_backward_silu(
        cls,
        grad_output: torch.Tensor,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """actual backward function without probability.

        Args:
            grad_output (torch.Tensor): gradient of the output from the activation function
            input_tensor (torch.Tensor): input tensor to the activation function

        Returns:
            torch.Tensor: gradient of the input tensor
        """
        a, b = input_tensor.chunk(2, dim=-1)
        sigmoid_a = torch.sigmoid(a)
        ones = torch.ones(sigmoid_a.shape, device=sigmoid_a.device, dtype=sigmoid_a.dtype)
        grad_a = grad_output * (sigmoid_a + a * sigmoid_a * (ones - sigmoid_a)) * b
        grad_b = grad_output * torch.nn.functional.silu(a)
        return torch.cat([grad_a, grad_b], dim=-1)
    
    @classmethod
    @torch.compile()
    def call_backward_silu_probs(
        cls,
        grad_output: torch.Tensor,
        input_tensor: torch.Tensor,
        probs: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """actual backward function with probability. 
        It computes the gradient of the input tensor and the probability.

        Args:
            grad_output (torch.Tensor): gradient of the output from the activation function
            input_tensor (torch.Tensor): input tensor to the activation function
            probs (torch.Tensor): probability derived from router
        """
        input_grad = MergedSwiGLU.call_backward_silu(
            grad_output * probs, 
            input_tensor
        )
        weights_grad = MergedSwiGLU.call_forward_silu(
            input_tensor
        ) * grad_output.to(probs.dtype)
        weights_grad = torch.sum(
            weights_grad, dim=-1
        )

        return input_grad.to(input_tensor.dtype) if input_grad is not None else None, \
        weights_grad.to(probs.dtype) if weights_grad is not None else None
    
    @staticmethod
    def forward(
        ctx,
        *args,
    ):
        args_q = collections.deque(args)
        input_tensor: torch.Tensor = args_q.popleft()
        probs: torch.Tensor | None = args_q.popleft()

        ctx.save_for_backward(input_tensor, probs)
        return MergedSwiGLU.call_forward(input_tensor, probs)
    
    @staticmethod
    def backward(
        ctx, 
        *grad_outputs
    ):
        grad_y: torch.Tensor = grad_outputs[0]
        (x, probs) = ctx.saved_tensors
        input_grad, prob_grad = MergedSwiGLU.call_backward(
            grad_y, x, probs
        )
        if prob_grad is not None:
            prob_grad = prob_grad.unsqueeze(-1)
        return input_grad, prob_grad
        

def release(t: torch.Tensor):
    """Helper function to release tensors that are no longer needed to save memory.
    """
    t.untyped_storage().resize_(0)

class GroupedSwiMLP(torch.autograd.Function):
    @classmethod
    def call_forward_a(
        cls,
        w1: torch.nn.Parameter,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """First linear projection in forward pass.

        Args:
            w1 (torch.nn.Parameter): weight parameter for the first linear layer
            permuted_local_hidden_states (torch.Tensor): input hidden states
            tokens_per_expert (torch.Tensor): number of tokens assigned to each expert

        Returns:
            torch.Tensor: output of the first linear layer
        """
        fc1_output = grouped_gemm.grouped_gemm.backend.gmm(
            permuted_local_hidden_states, 
            w1, 
            tokens_per_expert, 
            trans_a=False, 
            trans_b=False,
        )

        return fc1_output
    
    @classmethod
    def call_forward_y(
        cls,
        w2: torch.nn.Parameter,
        a: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Activation and second linear projection in forward pass.

        Args:
            w2 (torch.nn.Parameter): weight parameter for the second linear layer
            a (torch.Tensor): output of the first linear layer
            tokens_per_expert (torch.Tensor): number of tokens assigned to each expert
            permuted_probs (torch.Tensor): probability derived from router

        Returns:
            tuple[torch.Tensor, torch.Tensor]
        """
        s = MergedSwiGLU.call_forward(
            a, permuted_probs.unsqueeze(-1)
        )
        fc2_output = grouped_gemm.grouped_gemm.backend.gmm(
            s, 
            w2, 
            tokens_per_expert, 
            trans_a=False,
            trans_b=False,
        )
        return fc2_output, s
        

    @classmethod
    def call_backward_grad_a(
        cls,
        grad_y: torch.Tensor,
        a: torch.Tensor,
        w2: torch.nn.Parameter,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the gradient of the input to the activation function in the backward pass.

        Args:
            grad_y (torch.Tensor): gradient of the output
            a (torch.Tensor): input to the activation function
            w2 (torch.nn.Parameter): weight parameter for the second linear layer
            tokens_per_expert (torch.Tensor): number of tokens assigned to each expert
            permuted_probs (torch.Tensor): probability derived from router
        """
        grad_s = grouped_gemm.grouped_gemm.backend.gmm(
            grad_y, 
            w2, 
            tokens_per_expert,
            trans_a=False,
            trans_b=True,
        )
        return MergedSwiGLU.call_backward(grad_s, a, permuted_probs.unsqueeze(-1))

    @classmethod
    def call_backward_grad_x(
        cls,
        grad_a: torch.Tensor,
        w1: torch.nn.Parameter,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the gradient of the input to the first linear layer in the backward pass.

        Args:
            grad_a (torch.Tensor): gradient of the input to the activation function
            w1 (torch.nn.Parameter): weight parameter for the first linear layer
            tokens_per_expert (torch.Tensor): number of tokens assigned to each expert
        """
        grad_x = grouped_gemm.grouped_gemm.backend.gmm(
            grad_a,
            w1,
            tokens_per_expert,
            trans_a=False,
            trans_b=True,
        )

        return grad_x

    @classmethod
    def call_backward_grad_w2(
        cls,
        grad_y: torch.Tensor,
        a: torch.Tensor,
        w2: torch.nn.Parameter,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        fuse_gradient_accumulation: bool = False,
    ) -> torch.Tensor:
        """Calculate the gradient of the weight parameter for the second linear layer in the backward pass.
        Note: For now fuse_gradient_accumulation is not supported.
        Args:
            grad_y (torch.Tensor): gradient of the output
            a (torch.Tensor): input to the activation function
            w2 (torch.nn.Parameter): weight parameter for the second linear layer
            tokens_per_expert (torch.Tensor): number of tokens assigned to each expert
            permuted_probs (torch.Tensor): probability derived from router
            fuse_gradient_accumulation (bool, optional): Fuse gradient accumulation in gemm. Defaults to False.

        Returns:
            torch.Tensor: gradient of the weight parameter for the second linear layer
        """
        s = MergedSwiGLU.call_forward(a, permuted_probs.unsqueeze(-1))
        grad_w2 = grouped_gemm.grouped_gemm.backend.gmm(
            s, 
            grad_y,
            tokens_per_expert,
            trans_a=True,
            trans_b=False,
            c = None if not fuse_gradient_accumulation else w2.main_grad
        )
        return grad_w2

    @classmethod
    def call_backward_grad_w1(
        cls,
        grad_a: torch.Tensor,
        x: torch.Tensor,
        w1: torch.nn.Parameter,
        tokens_per_expert: torch.Tensor,
        fuse_gradient_accumulation: bool = False,
    ) -> torch.Tensor:
        """Calculate the gradient of the weight parameter for the first linear layer in the backward pass.
        Note: For now fuse_gradient_accumulation is not supported.

        Args:
            grad_a (torch.Tensor): gradient of the input to the activation function
            x (torch.Tensor): input to the first linear layer
            w1 (torch.nn.Parameter): weight parameter for the first linear layer
            tokens_per_expert (torch.Tensor): number of tokens assigned to each expert
            fuse_gradient_accumulation (bool, optional): Fuse gradient accumulation in gemm. Defaults to False.

        Returns:
            torch.Tensor: gradient of the weight parameter for the first linear layer
        """
        grad_w1 = grouped_gemm.grouped_gemm.backend.gmm(
            x,
            grad_a,
            tokens_per_expert,
            trans_a=True,
            trans_b=False,
            c = None if not fuse_gradient_accumulation else w1.main_grad
        )
        return grad_w1


    @staticmethod
    def forward(
        ctx,
        *args, 
        **kwargs
    ):
        if len(args) < 6:
            raise ValueError(f"Insufficient arguments for forward pass of GroupedSwiMLP. Expected at least 6, got {len(args)}")
        
        w1: torch.nn.Parameter = args[0]
        w2: torch.nn.Parameter = args[1]
        permuted_local_hidden_states: torch.Tensor = args[2]
        tokens_per_expert: torch.Tensor = args[3]
        permuted_probs: torch.Tensor = args[4]
        config: TransformerConfig = args[5]

        # mlp1
        a = GroupedSwiMLP.call_forward_a(
            w1, permuted_local_hidden_states, tokens_per_expert
        )

        # act + mlp2
        y, _ = GroupedSwiMLP.call_forward_y(
            w2, a, tokens_per_expert, permuted_probs
        )

        # context saving
        ctx.w1 = w1
        ctx.w2 = w2
        ctx.tokens_per_expert = tokens_per_expert
        ctx.config = config

        activation_recompute = (
            config.recompute_granularity == 'selective'
            and "moe_act" in config.recompute_modules
        )
        ctx.activation_recompute = activation_recompute
        if config.fp8_activation:
            if HAVE_TE:
                quantizer = Float8BlockQuantizer(
                    fp8_dtype=TE_DType[torch.float8_e4m3fn],
                    rowwise=True,
                    columnwise=False,
                    amax_epsilon=0.0,
                    force_pow_2_scales=True,
                    block_scaling_dim=1,
                )
                qx = quantizer.make_empty(
                    permuted_local_hidden_states.shape, 
                    dtype=permuted_local_hidden_states.dtype, 
                    device=permuted_local_hidden_states.device, 
                    requires_grad=False
                )
                qx = quantizer.update_quantized(
                    permuted_local_hidden_states, qx
                )
                release(permuted_local_hidden_states)
                

                if activation_recompute:
                    ctx.qx = qx
                    ctx.qa = None
                    ctx.save_for_backward(
                        None, None, permuted_probs
                    )
                    release(a)
                else:
                    qa = quantizer.make_empty(
                        a.shape, 
                        dtype=a.dtype, 
                        device=a.device, 
                        requires_grad=False
                    )
                    qa = quantizer.update_quantized(
                        a, qa
                    )
                    ctx.qx = qx
                    ctx.qa = qa
                    ctx.save_for_backward(
                        None, None, permuted_probs
                    )
                    release(a)
        else:
            if activation_recompute:
                ctx.save_for_backward(
                    permuted_local_hidden_states, None, permuted_probs
                )
                release(a)
            else:
                ctx.save_for_backward(
                    permuted_local_hidden_states, a, permuted_probs
                )

        return y, None

    @staticmethod
    def backward(
        ctx, 
        *grad_outputs
    ):
        config: TransformerConfig = ctx.config
        tokens_per_expert: torch.Tensor = ctx.tokens_per_expert
        w1: torch.nn.Parameter = ctx.w1
        w2: torch.nn.Parameter = ctx.w2
        (x, a, probs) = ctx.saved_tensors

        # rematerialize activation if needed
        # NOTE: fp8 tensors have to be manually released after dequantization
        if config.fp8_activation:
            x = ctx.qx.dequantize()
            release(ctx.qx)
            if not ctx.activation_recompute:
                a = ctx.qa.dequantize()
                release(ctx.qa)
            else:
                a = GroupedSwiMLP.call_forward_a(
                    w1, x, tokens_per_expert
                )

        grad_y = grad_outputs[0].contiguous()

        # backward computation
        grad_a, grad_probs = GroupedSwiMLP.call_backward_grad_a(
            grad_y, 
            a, 
            w2, 
            tokens_per_expert,
            probs,
        )

        grad_x = None if grad_a is None else GroupedSwiMLP.call_backward_grad_x(
            grad_a,
            w1,
            tokens_per_expert,
        )

        grad_w2 = GroupedSwiMLP.call_backward_grad_w2(
            grad_y, 
            a,
            w2,
            tokens_per_expert, 
            probs
        )

        grad_w1 = None if grad_a is None else GroupedSwiMLP.call_backward_grad_w1(
            grad_a, 
            x, 
            w1,
            tokens_per_expert
        )

        return grad_w1, grad_w2, grad_x, None, grad_probs, None
    
def grouped_swiglu_mlp(
    w1: torch.nn.Parameter,
    w2: torch.nn.Parameter,
    permuted_local_hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    permuted_probs: torch.Tensor,
    config: TransformerConfig,
) -> torch.Tensor:
    """Autograd function for Grouped SwiGLU MLP.

    Args:
        w1 (torch.nn.Parameter): weight parameter for the first linear layer
        w2 (torch.nn.Parameter): weight parameter for the second linear layer
        permuted_local_hidden_states (torch.Tensor): input hidden states
        tokens_per_expert (torch.Tensor): number of tokens assigned to each expert
        permuted_probs (torch.Tensor): probability derived from router
        config (TransformerConfig): transformer configuration

    Returns:
        torch.Tensor: output of the MLP
    """
    output, _ = GroupedSwiMLP.apply(
        w1, 
        w2, 
        permuted_local_hidden_states,
        tokens_per_expert,
        permuted_probs,
        config,
    )

    return output