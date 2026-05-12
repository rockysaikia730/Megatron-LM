# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Multi-codebook output heads for RVQ-based audio prediction.

Provides K parallel output heads, one per RVQ codebook, for training a
decoder-LM (e.g., Llama3) to predict HCodec-1.0 audio tokens alongside text.
The K heads share the hidden state with the text output head and are computed
in parallel; modality routing (text vs audio, acoustic vs semantic) is handled
by the data pipeline via per-position loss masks.

Parallelism support:
  - TP: each head is a ColumnParallelLinear, sharding the audio vocab across
    the TP group. `gather_output=False` so the loss can use
    `vocab_parallel_cross_entropy` without materializing the full vocab.
  - PP: instantiated only on the last pipeline stage (caller gates on
    `post_process=True`). VPP-safe because gating is at construction time.
  - CP: no internal CP handling needed; the hidden state arriving at the head
    is already sequence-sharded by upstream blocks, and CP-aware loss
    reduction happens in the loss function on each head independently.
  - DP: parameters are standard nn.Parameter, so they are discovered and
    sharded by the distributed optimizer automatically.
"""

from typing import Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class MultiCodebookOutputHead(MegatronModule):
    """Parallel output heads for K audio RVQ codebooks.

    Each head is a ColumnParallelLinear projecting hidden_states -> audio_vocab.
    All K heads consume the same hidden state and are computed in parallel.

    The returned tensor stacks all K heads along a new codebook dimension so the
    downstream loss can vectorize cross-entropy across codebooks instead of
    looping in Python.

    Args:
        config: TransformerConfig (uses hidden_size, init_method, etc.).
        num_codebooks: Number of audio codebooks K (e.g., 4 for HCodec-1.0).
        codebook_vocab_size: Size of each codebook vocab (e.g., 1024).
        parallel_output: If True, keep logits sharded along vocab for use with
            vocab_parallel_cross_entropy. If False, all-gather to full vocab.
        pg_collection: Process group collection providing TP/PP/CP groups.
    """

    def __init__(
        self,
        config: TransformerConfig,
        num_codebooks: int,
        codebook_vocab_size: int,
        parallel_output: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=config)

        if num_codebooks <= 0:
            raise ValueError(
                f"num_codebooks must be positive, got {num_codebooks}"
            )
        if codebook_vocab_size <= 0:
            raise ValueError(
                f"codebook_vocab_size must be positive, got {codebook_vocab_size}"
            )

        self.num_codebooks = num_codebooks
        self.codebook_vocab_size = codebook_vocab_size
        self.parallel_output = parallel_output

        tp_group = pg_collection.tp if pg_collection is not None else None

        self.heads = torch.nn.ModuleList(
            [
                tensor_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    codebook_vocab_size,
                    config=config,
                    init_method=config.init_method,
                    bias=False,
                    skip_bias_add=False,
                    gather_output=not parallel_output,
                    skip_weight_param_allocation=False,
                    tp_group=tp_group,
                )
                for _ in range(num_codebooks)
            ]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Compute logits for each audio codebook.

        Args:
            hidden_states: Tensor of shape [S, B, H] (Megatron's internal
                seq-first layout, matching the input to the text output head).

        Returns:
            audio_logits: Tensor of shape [S, B, K, V_audio_local], where
                V_audio_local = codebook_vocab_size // TP when parallel_output
                is True, else codebook_vocab_size. Stacking along dim=2 lets
                the loss vectorize cross-entropy across codebooks.
        """
        per_head_logits = []
        for head in self.heads:
            logits, _ = head(hidden_states)
            per_head_logits.append(logits)
        return torch.stack(per_head_logits, dim=2)
