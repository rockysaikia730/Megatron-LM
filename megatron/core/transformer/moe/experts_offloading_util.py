from __future__ import annotations
import torch
import collections
import queue

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

from megatron.core.transformer.moe.experts_util import ExpertsWgradScheduler, MergedSwiGLU


class StreamManager:
    '''
    StreamManager manages CUDA streams and events for offloading experts.
    It provides methods to get H2D streams and compute streams, and to synchronize between them.
    '''
    _instance = None

    def __init__(self, num_h2d_streams):
        self.num_compute_streams = 4
        self.num_h2d_streams = num_h2d_streams
        self.h2d_streams = [torch.cuda.Stream() for _ in range(num_h2d_streams)]
        # self.h2d_streams.append(torch.cuda.default_stream())
        self.compute_stream_events = [torch.cuda.Event() for _ in range(self.num_compute_streams)]
        self.compute_streams = [torch.cuda.Stream() for _ in range(self.num_compute_streams)]
        self.compute_cuda_streams = [stream.cuda_stream for stream in self.compute_streams]

    @classmethod
    def get_instance(cls, num_h2d_streams=2):
        if cls._instance is None:
            cls._instance = StreamManager(num_h2d_streams)
        return cls._instance

    def get_h2d_stream(self, idx) -> torch.cuda.Stream:
        return self.h2d_streams[idx]
    
    def get_compute_streams(self) -> list[int]:
        return self.compute_cuda_streams
    
    def default_stream_wait_compute_streams(self):
        default_stream = torch.cuda.default_stream()
        for i in range(self.num_compute_streams):
            self.compute_streams[i].record_event(self.compute_stream_events[i])
        for i in range(self.num_compute_streams):
            default_stream.wait_event(self.compute_stream_events[i])
    
    def compute_streams_wait_default_stream(self):
        default_stream = torch.cuda.default_stream()
        default_stream.record_event(self.compute_stream_events[0])
        for i in range(self.num_compute_streams):
            self.compute_streams[i].wait_event(self.compute_stream_events[0])
    
    def h2d_stream_wait_compute_streams(self, idx):
        h2d_stream = self.get_h2d_stream(idx)
        for i in range(self.num_compute_streams):
            self.compute_streams[i].record_event(self.compute_stream_events[i])
        for i in range(self.num_compute_streams):
            h2d_stream.wait_event(self.compute_stream_events[i])

    def compute_streams_wait_h2d_stream(self, idx):
        h2d_stream = self.get_h2d_stream(idx)
        h2d_stream.record_event(self.compute_stream_events[0])
        for i in range(self.num_compute_streams):
            self.compute_streams[i].wait_event(self.compute_stream_events[0])


class OffloadingExpertsGroupedSwiMLP(torch.autograd.Function):
    '''
    Autograd function for Offloading Experts Grouped SwiGLU MLP. 
    The forward and backward pass are implemented with chunk-level interleaving of GPU computation and CPU-GPU communication 
    to hide the data transfer latency of expert weights.
    '''

    @classmethod
    def call_forward_a(
        cls,
        cpu_w1: list[torch.nn.Parameter],
        gpu_w1_buffers: list[list[torch.Tensor]],
        permuted_local_hidden_states: list[torch.Tensor],
        hidden_state_per_chunk: list[torch.Tensor],
        total_token_num_per_chunk: list[int],
        tokens_per_expert_chunks: list[torch.Tensor],
        stream_manager: StreamManager,
        config: TransformerConfig,
    ):
        # allocate output buffer for the first linear layer
        fc1_output = torch.empty(
            permuted_local_hidden_states.shape[0],
            config.moe_ffn_hidden_size * (2 if config.gated_linear_unit else 1),
            device=permuted_local_hidden_states.device,
            dtype=permuted_local_hidden_states.dtype,
        )
        fc1_output_per_chunk = list(torch.split(fc1_output, total_token_num_per_chunk))

        # prefetch the first chunk of expert weights to GPU
        curr_buffer_metadata = cls.prefetch_expert_weights(0, cpu_w1, gpu_w1_buffers, stream_manager, config)

        # fc1 chunk-level interleaving computation
        for chunk_idx in range(config.moe_offloading_num_chunks):
            # prefetch the next chunk of expert weights to GPU buffer
            if chunk_idx + 1 < config.moe_offloading_num_chunks:
                next_buffer_metadata = cls.prefetch_expert_weights(chunk_idx + 1, cpu_w1, gpu_w1_buffers, stream_manager, config)

            # computation on the current GPU buffer
            experts_chunk = gpu_w1_buffers[curr_buffer_metadata[0]]
            hidden_states_chunk = hidden_state_per_chunk[chunk_idx]
            fc1_output_chunk = fc1_output_per_chunk[chunk_idx]
            tokens_per_expert_chunk = tokens_per_expert_chunks[chunk_idx]

            # NOTE: underlying groupedgemm is has sync problem
            # wait for the current chunk of weights to be ready on GPU
            stream_manager.compute_streams_wait_default_stream()
            stream_manager.compute_streams_wait_h2d_stream(curr_buffer_metadata[1])
            grouped_gemm.grouped_gemm.backend.gmmfwd(
                batch_sizes=tokens_per_expert_chunk,
                a=hidden_states_chunk,
                b=experts_chunk,
                c=fc1_output_chunk,
                compute_streams=stream_manager.get_compute_streams(),
                trans_a=False,
                trans_b=False,
            )
            stream_manager.h2d_stream_wait_compute_streams(curr_buffer_metadata[1])
            stream_manager.default_stream_wait_compute_streams()

            # update current buffer metadata
            curr_buffer_metadata = next_buffer_metadata if chunk_idx + 1 < config.moe_offloading_num_chunks else None

        return fc1_output
    
    @classmethod
    def call_forward_y(
        cls,
        cpu_w2: list[torch.nn.Parameter],
        gpu_w2_buffers: list[list[torch.Tensor]],
        permuted_local_hidden_states: torch.Tensor,
        fc1_output: torch.Tensor,
        total_token_num_per_chunk: list[int],
        tokens_per_expert_chunks: list[torch.Tensor],
        permuted_probs: torch.Tensor,
        stream_manager: StreamManager,
        config: TransformerConfig,
    ):
        # prefetch the first chunk of expert weights to GPU
        curr_buffer_metadata = cls.prefetch_expert_weights(0, cpu_w2, gpu_w2_buffers, stream_manager, config)

        s = MergedSwiGLU.call_forward(
            fc1_output,
            permuted_probs.unsqueeze(-1)
        )

        s_per_chunk = list(torch.split(s, total_token_num_per_chunk))

        # fc2 chunk-level interleaving computation
        fc2_output = torch.empty_like(permuted_local_hidden_states)
        fc2_output_per_chunk = list(torch.split(fc2_output, total_token_num_per_chunk))

        for chunk_idx in range(config.moe_offloading_num_chunks):
            # prefetch the next chunk of expert weights to GPU buffer
            if chunk_idx + 1 < config.moe_offloading_num_chunks:
                next_buffer_metadata = cls.prefetch_expert_weights(chunk_idx + 1, cpu_w2, gpu_w2_buffers, stream_manager, config)

            # computation on the current GPU buffer
            experts_chunk = gpu_w2_buffers[curr_buffer_metadata[0]]
            s_chunk = s_per_chunk[chunk_idx]
            fc2_output_chunk = fc2_output_per_chunk[chunk_idx]
            tokens_per_expert_chunk = tokens_per_expert_chunks[chunk_idx]

            # NOTE: underlying groupedgemm is multi-stream
            stream_manager.compute_streams_wait_default_stream()
            stream_manager.compute_streams_wait_h2d_stream(curr_buffer_metadata[1])
            grouped_gemm.grouped_gemm.backend.gmmfwd(
                batch_sizes=tokens_per_expert_chunk,
                a=s_chunk,
                b=experts_chunk,
                c=fc2_output_chunk,
                compute_streams=stream_manager.get_compute_streams(),
                trans_a=False,
                trans_b=False,
            )
            stream_manager.h2d_stream_wait_compute_streams(curr_buffer_metadata[1])
            stream_manager.default_stream_wait_compute_streams()

            # update current buffer metadata
            curr_buffer_metadata = next_buffer_metadata if chunk_idx + 1 < config.moe_offloading_num_chunks else None
        
        return fc2_output, s
    
    @classmethod
    def call_backward_grad_a(
        cls,
        grad_y: torch.Tensor,
        a: torch.Tensor,
        cpu_w2: list[torch.nn.Parameter],
        gpu_w2_buffers: list[torch.Tensor],
        total_token_num_per_chunk: list[int],
        tokens_per_expert_chunks: list[torch.Tensor],
        permuted_probs: torch.Tensor,
        stream_manager: StreamManager,
        config: TransformerConfig,
    ):
        grad_y_per_chunk = list(torch.split(grad_y, total_token_num_per_chunk))
        grad_s = torch.empty(
            grad_y.shape[0],
            config.moe_ffn_hidden_size,
            device=grad_y.device,
            dtype=grad_y.dtype,
        )
        grad_s_per_chunk = list(torch.split(grad_s, total_token_num_per_chunk))

        # prefetch the first chunk of expert weights to GPU
        curr_buffer_metadata = cls.prefetch_expert_weights(0, cpu_w2, gpu_w2_buffers, stream_manager, config)

        for chunk_idx in range(config.moe_offloading_num_chunks):
            # prefetch the next chunk of expert weights to GPU buffer
            if chunk_idx + 1 < config.moe_offloading_num_chunks:
                next_buffer_metadata = cls.prefetch_expert_weights(chunk_idx + 1, cpu_w2, gpu_w2_buffers, stream_manager, config)

            # computation on the current GPU buffer
            experts_chunk = gpu_w2_buffers[curr_buffer_metadata[0]]
            grad_y_chunk = grad_y_per_chunk[chunk_idx]
            grad_s_chunk = grad_s_per_chunk[chunk_idx]
            tokens_per_expert_chunk = tokens_per_expert_chunks[chunk_idx]

            stream_manager.compute_streams_wait_default_stream()
            stream_manager.compute_streams_wait_h2d_stream(curr_buffer_metadata[1])
            grouped_gemm.grouped_gemm.backend.gmmfwd(
                batch_sizes=tokens_per_expert_chunk,
                a=grad_y_chunk,
                b=experts_chunk,
                c=grad_s_chunk,
                compute_streams=stream_manager.get_compute_streams(),
                trans_a=False,
                trans_b=True,
            )
            stream_manager.h2d_stream_wait_compute_streams(curr_buffer_metadata[1])
            stream_manager.default_stream_wait_compute_streams()

            # update current buffer metadata
            curr_buffer_metadata = next_buffer_metadata if chunk_idx + 1 < config.moe_offloading_num_chunks else None

        return MergedSwiGLU.call_backward(grad_s, a, permuted_probs.unsqueeze(-1))
    
    @classmethod
    def call_backward_grad_x(
        cls,
        grad_a: torch.Tensor,
        cpu_w1: list[torch.nn.Parameter],
        gpu_w1_buffers: list[torch.Tensor],
        total_token_num_per_chunk: list[int],
        tokens_per_expert_chunks: list[torch.Tensor],
        stream_manager: StreamManager,
        config: TransformerConfig,
    ):
        grad_a_per_chunk = list(torch.split(grad_a, total_token_num_per_chunk))
        grad_x = torch.empty(
            grad_a.shape[0],
            config.hidden_size,
            device=grad_a.device,
            dtype=grad_a.dtype,
        )
        grad_x_per_chunk = list(torch.split(grad_x, total_token_num_per_chunk))

        # prefetch the first chunk of expert weights to GPU
        curr_buffer_metadata = cls.prefetch_expert_weights(0, cpu_w1, gpu_w1_buffers, stream_manager, config)

        for chunk_idx in range(config.moe_offloading_num_chunks):            
            # prefetch the next chunk of expert weights to GPU buffer
            if chunk_idx + 1 < config.moe_offloading_num_chunks:
                next_buffer_metadata = cls.prefetch_expert_weights(chunk_idx + 1, cpu_w1, gpu_w1_buffers, stream_manager, config)

            # computation on the current GPU buffer
            experts_chunk = gpu_w1_buffers[curr_buffer_metadata[0]]
            grad_a_chunk = grad_a_per_chunk[chunk_idx]
            grad_x_chunk = grad_x_per_chunk[chunk_idx]
            tokens_per_expert_chunk = tokens_per_expert_chunks[chunk_idx]

            stream_manager.compute_streams_wait_default_stream()
            stream_manager.compute_streams_wait_h2d_stream(curr_buffer_metadata[1])
            grouped_gemm.grouped_gemm.backend.gmmfwd(
                batch_sizes=tokens_per_expert_chunk,
                a=grad_a_chunk,
                b=experts_chunk,
                c=grad_x_chunk,
                compute_streams=stream_manager.get_compute_streams(),
                trans_a=False,
                trans_b=True,
            )
            stream_manager.h2d_stream_wait_compute_streams(curr_buffer_metadata[1])
            stream_manager.default_stream_wait_compute_streams()
            
            # update current buffer metadata
            curr_buffer_metadata = next_buffer_metadata if chunk_idx + 1 < config.moe_offloading_num_chunks else None

        return grad_x
    
    @staticmethod
    def _wgrad_post_process(
        w: list[torch.nn.Parameter],
        wgrad_output: list[torch.Tensor],
        fuse_gradient_accumulation: bool,
    ):
        # handle ddp
        for i in range(len(w)):
            if fuse_gradient_accumulation:
                w[i].grad_added_to_main_grad = True
            else:
                w[i].grad_added_to_main_grad = False
                w[i].grad = wgrad_output[i].view(w[i].shape)
    
    @classmethod
    def call_backward_grad_w2(
        cls,
        grad_y: torch.Tensor,
        a: torch.Tensor,
        cpu_w2: list[torch.nn.Parameter],
        gpu_w2_buffers: list[torch.Tensor],
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
        stream_manager: StreamManager,
        config: TransformerConfig,
        delay_wgrad_compute: bool = False,
        fuse_gradient_accumulation: bool = False,
    ):
        s = MergedSwiGLU.call_forward(a, permuted_probs.unsqueeze(-1))
        
        wgrad_output = None
        alpha = 1.0
        beta = 0.0
        if fuse_gradient_accumulation:
            wgrad_output = [w.main_grad for w in cpu_w2]
            beta = 1.0
        else:
            wgrad_output = [torch.empty(
                w.shape, 
                device=grad_y.device, 
                dtype=w.dtype
            ) for w in cpu_w2]
            beta = 0.0

        # compute wgrad immediately if not delay_wgrad_compute or wgrad_scheduler is None
        stream_manager.compute_streams_wait_default_stream()
        grad_w2 = grouped_gemm.grouped_gemm.backend.gmmbwd(
            s, 
            grad_y,
            tokens_per_expert,
            trans_a=True,
            trans_b=False,
            compute_streams=stream_manager.get_compute_streams(),
            c = wgrad_output,
            alpha = alpha,
            beta = beta,
        )
        stream_manager.default_stream_wait_compute_streams()

        OffloadingExpertsGroupedSwiMLP._wgrad_post_process(cpu_w2, wgrad_output, fuse_gradient_accumulation)
        return grad_w2
    
    @classmethod
    def call_backward_grad_w1(
        cls,
        grad_a: torch.Tensor,
        x: torch.Tensor,
        cpu_w1: list[torch.nn.Parameter],
        tokens_per_expert: torch.Tensor,
        stream_manager: StreamManager,
        wgrad_scheduler: ExpertsWgradScheduler = None,
        delay_wgrad_compute: bool = False,
        fuse_gradient_accumulation: bool = False,
    ) -> list[torch.Tensor]:
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
        wgrad_output = None
        alpha = 1.0
        beta = 0.0
        if fuse_gradient_accumulation:
            wgrad_output = [w.main_grad for w in cpu_w1]
            beta = 1.0
        else:
            wgrad_output = [torch.empty(
                w.shape, 
                device=grad_a.device, 
                dtype=w.dtype
            ) for w in cpu_w1]
            beta = 0.0

        
        # compute wgrad immediately if not delay_wgrad_compute or wgrad_scheduler is None
        stream_manager.compute_streams_wait_default_stream()
        grad_w1 = grouped_gemm.grouped_gemm.backend.gmmbwd(
            x,
            grad_a,
            tokens_per_expert,
            compute_streams=stream_manager.get_compute_streams(),
            trans_a=True,
            trans_b=False,
            c = wgrad_output,
            alpha = alpha,
            beta = beta,
        )
        stream_manager.default_stream_wait_compute_streams()

        # post process wgrad for ddp
        OffloadingExpertsGroupedSwiMLP._wgrad_post_process(cpu_w1, grad_w1, fuse_gradient_accumulation)
        return grad_w1

    @classmethod
    def prefetch_expert_weights(
        cls, 
        chunk_idx: int,
        cpu_weights: list[torch.nn.Parameter],
        gpu_buffers: list[list[torch.Tensor]],
        stream_manager: StreamManager,
        config: TransformerConfig,
    ) -> tuple:
        h2d_stream_idx = chunk_idx % config.moe_offloading_num_stages
        gpu_buffer_idx = h2d_stream_idx
        h2d_stream = stream_manager.get_h2d_stream(h2d_stream_idx)
        
        with torch.cuda.stream(h2d_stream):
            experts_idx_start = chunk_idx * config.moe_offloading_chunk_size
            experts_idx_end = (chunk_idx + 1) * config.moe_offloading_chunk_size
            cpu_weights_slice = cpu_weights[experts_idx_start:experts_idx_end]
            buf = gpu_buffers[gpu_buffer_idx]

            # NOTE: batched H2D copy is used to reduce cpu overhead
            assert len(cpu_weights_slice) == len(buf), \
                f"Number of weights in CPU slice {len(cpu_weights_slice)} does not match number of GPU buffers {len(buf)}"
            grouped_gemm.grouped_gemm.backend.batched_h2d_async(
                cpu_weights_slice,
                buf,
                h2d_stream.cuda_stream
            )

            # NOTE: fallback to non-batched copy if the above async copy has issues
            # for idx in range(experts_idx_start, experts_idx_end):
            #      buf[idx - experts_idx_start].copy_(cpu_weights[idx].data, non_blocking=True)
        
        return (gpu_buffer_idx, h2d_stream_idx, experts_idx_start, experts_idx_end)
    
    @staticmethod
    def forward(
        ctx,
        *args, 
        **kwargs
    ):
        if len(args) < 9:
            raise ValueError(f"Insufficient arguments for forward pass of GroupedSwiMLP. Expected at least 9, got {len(args)}")
        
        cpu_weights: list[torch.nn.Parameter] = args[:-9]
        cpu_w1: list[torch.nn.Parameter] = cpu_weights[:len(cpu_weights)//2]
        cpu_w2: list[torch.nn.Parameter] = cpu_weights[len(cpu_weights)//2:]
        gpu_w1_buffers: list[torch.Tensor] = args[-9]
        gpu_w2_buffers: list[torch.Tensor] = args[-8]
        permuted_local_hidden_states: torch.Tensor = args[-7]
        tokens_per_expert: torch.Tensor = args[-6]
        num_local_experts: int = args[-5]
        permuted_probs: torch.Tensor = args[-4]
        expert_wgrad_scheduler: ExpertsWgradScheduler = args[-3]
        stream_manager: StreamManager = args[-2]
        config: TransformerConfig = args[-1]

        # split hidden states, outputs and token_per_experts into chunks for each expert chunks
        tokens_per_expert_chunks = torch.split(tokens_per_expert, config.moe_offloading_chunk_size)
        total_token_num_per_chunk = [chunk.sum().item() for chunk in tokens_per_expert_chunks]
        hidden_state_per_chunk = list(torch.split(permuted_local_hidden_states, total_token_num_per_chunk))

        # forward for the first linear layer
        fc1_output = OffloadingExpertsGroupedSwiMLP.call_forward_a(
            cpu_w1,
            gpu_w1_buffers,
            permuted_local_hidden_states,
            hidden_state_per_chunk,
            total_token_num_per_chunk,
            tokens_per_expert_chunks,
            stream_manager,
            config,
        )

        # activation and forward for the second linear layer
        y, _ = OffloadingExpertsGroupedSwiMLP.call_forward_y(
            cpu_w2,
            gpu_w2_buffers,
            permuted_local_hidden_states,
            fc1_output,
            total_token_num_per_chunk,
            tokens_per_expert_chunks,
            permuted_probs,
            stream_manager,
            config,
        )

        # context saving for backward
        ctx.tokens_per_expert_chunks = tokens_per_expert_chunks
        ctx.total_token_num_per_chunk = total_token_num_per_chunk
        ctx.expert_wgrad_scheduler = expert_wgrad_scheduler
        ctx.cpu_w1 = cpu_w1
        ctx.cpu_w2 = cpu_w2
        ctx.gpu_w1_buffers = gpu_w1_buffers
        ctx.gpu_w2_buffers = gpu_w2_buffers
        ctx.stream_manager = stream_manager
        ctx.config = config
        ctx.save_for_backward(permuted_local_hidden_states, fc1_output, tokens_per_expert, permuted_probs)

        # TODO: for now activation recomputation and activation quantization are not supported

        return y, None


    @staticmethod
    def backward(
        ctx, 
        *grad_outputs
    ):
        config: TransformerConfig = ctx.config
        cpu_w1: list[torch.nn.Parameter] = ctx.cpu_w1
        cpu_w2: list[torch.nn.Parameter] = ctx.cpu_w2
        gpu_w1_buffers: list[torch.Tensor] = ctx.gpu_w1_buffers
        gpu_w2_buffers: list[torch.Tensor] = ctx.gpu_w2_buffers
        stream_manager: StreamManager = ctx.stream_manager
        expert_wgrad_scheduler: ExpertsWgradScheduler = ctx.expert_wgrad_scheduler
        total_token_num_per_chunk: list[int] = ctx.total_token_num_per_chunk
        tokens_per_expert_chunks: tuple[torch.Tensor] = ctx.tokens_per_expert_chunks
        permuted_local_hidden_states, fc1_output, tokens_per_expert, permuted_probs = ctx.saved_tensors

        grad_y = grad_outputs[0].contiguous()

        # backward computation
        grad_a, grad_probs = OffloadingExpertsGroupedSwiMLP.call_backward_grad_a(
            grad_y,
            fc1_output,
            cpu_w2,
            gpu_w2_buffers,
            total_token_num_per_chunk,
            tokens_per_expert_chunks,
            permuted_probs,
            stream_manager,
            config,
        )

        grad_x = None if grad_a is None else OffloadingExpertsGroupedSwiMLP.call_backward_grad_x(
            grad_a,
            cpu_w1,
            gpu_w1_buffers,
            total_token_num_per_chunk,
            tokens_per_expert_chunks,
            stream_manager,
            config,
        )

        grad_w2 = OffloadingExpertsGroupedSwiMLP.call_backward_grad_w2(
            grad_y,
            fc1_output,
            cpu_w2,
            gpu_w2_buffers,
            tokens_per_expert,
            permuted_probs,
            stream_manager,
            config,
            config.delay_wgrad_compute,
            config.gradient_accumulation_fusion,
        )

        grad_w1 = None if grad_a is None else OffloadingExpertsGroupedSwiMLP.call_backward_grad_w1(
            grad_a,
            permuted_local_hidden_states,
            cpu_w1,
            tokens_per_expert,
            stream_manager,
            expert_wgrad_scheduler,
            config.delay_wgrad_compute,
            config.gradient_accumulation_fusion,
        )

        # NOTE: for weights on CPU, gradients are directly accumulated into main_grad
        # to avoid graph breaking
        grad_w1_ret = [None for _ in cpu_w1]
        grad_w2_ret = [None for _ in cpu_w2]

        return *grad_w1_ret, *grad_w2_ret, None, None, grad_x, None, None, grad_probs, None, None, None




def offloading_grouped_swiglu_mlp(
    cpu_w1: list[torch.nn.Parameter],
    cpu_w2: list[torch.nn.Parameter],
    gpu_w1_buffers: list[torch.Tensor],
    gpu_w2_buffers: list[torch.Tensor],
    permuted_local_hidden_states: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_local_experts: int,
    permuted_probs: torch.Tensor,
    expert_wgrad_scheduler: ExpertsWgradScheduler,
    stream_manager: StreamManager,
    config: TransformerConfig,
) -> torch.Tensor:
    """Autograd function for Offloading Experts Grouped SwiGLU MLP.

    Args:
        cpu_w1 (list[torch.nn.Parameter]): CPU weight parameters for the first linear layer
        cpu_w2 (list[torch.nn.Parameter]): CPU weight parameters for the second linear layer
        gpu_w1_buffers (list[torch.Tensor]): GPU buffers for w1 weights
        gpu_w2_buffers (list[torch.Tensor]): GPU buffers for w2 weights
        permuted_local_hidden_states (torch.Tensor): input hidden states
        tokens_per_expert (torch.Tensor): number of tokens assigned to each expert
        num_local_experts (int): number of local experts
        permuted_probs (torch.Tensor): probability derived from router
        expert_wgrad_scheduler (ExpertsWgradScheduler): scheduler for expert weight gradients
        stream_manager (StreamManager): manager for CUDA streams
        config (TransformerConfig): transformer configuration

    Returns:
        torch.Tensor: output of the MLP
    """
    output, _ = OffloadingExpertsGroupedSwiMLP.apply(
        *cpu_w1,
        *cpu_w2,
        gpu_w1_buffers,
        gpu_w2_buffers,
        permuted_local_hidden_states,
        tokens_per_expert,
        num_local_experts,
        permuted_probs,
        expert_wgrad_scheduler,
        stream_manager,
        config,
    )

    return output