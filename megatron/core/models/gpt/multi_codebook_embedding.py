# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Multi-codebook input embedding for RVQ audio tokens.

K independent embedding tables, one per RVQ codebook layer. At each audio position, the K codebook
tokens are looked up in their respective tables and the K resulting vectors are summed into a single 
H-dim transformer input. Same architecture as described in UniTok-Audio Section 3.2.2 (https://arxiv.org/pdf/2510.26372)
"""

from typing import Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class MultiCodebookInputEmbedding(MegatronModule):
    """K parallel embedding tables for audio RVQ codebooks, summed per position.

    Each table is a `VocabParallelEmbedding(vocab_size=audio_codebook_size,
    embedding_dim=hidden_size)`. For each input position with K codebook token
    IDs, the K lookups are summed elementwise to produce one H-dim vector.

    Args:
        config: TransformerConfig (uses hidden_size, init_method, etc.).
        num_codebooks: Number of audio codebooks K (e.g., 4 for HCodec-1.0).
        codebook_vocab_size: Size of each codebook vocab (e.g., 1024).
        pg_collection: Process group collection providing TP/PP/CP groups.
    """

    def __init__(
        self,
        config: TransformerConfig,
        num_codebooks: int,
        codebook_vocab_size: int,
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

        tp_group = pg_collection.tp if pg_collection is not None else None

        init_method = getattr(
            config, 'embedding_init_method', config.init_method
        )

        self.tables = torch.nn.ModuleList(
            [
                tensor_parallel.VocabParallelEmbedding(
                    num_embeddings=codebook_vocab_size,
                    embedding_dim=config.hidden_size,
                    init_method=init_method,
                    reduce_scatter_embeddings=False,
                    config=config,
                    tp_group=tp_group,
                )
                for _ in range(num_codebooks)
            ]
        )

    def forward(self, audio_token_ids: Tensor) -> Tensor:
        """Sum embeddings for K codebooks at every position.

        Args:
            audio_token_ids: Long tensor of shape [B, S, K]. Position (b, s, k)
                holds the codebook-k token ID at sequence position s. Values
                must be in [0, codebook_vocab_size). Positions that are not
                audio should still be filled with valid IDs (e.g., 0); the
                downstream `modality_mask` decides whether this embedding is
                used at all.

        Returns:
            Tensor of shape [B, S, H], the elementwise sum of K table lookups.
        """
        # VocabParallelEmbedding returns [B, S, H] for input [B, S].
        out = self.tables[0](audio_token_ids[..., 0])
        for k in range(1, self.num_codebooks):
            out = out + self.tables[k](audio_token_ids[..., k])
        return out
