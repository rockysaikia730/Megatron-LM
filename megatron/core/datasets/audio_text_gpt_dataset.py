# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPT-style dataset that returns the multi-codebook delay-pattern tensors.

This thin wrapper reuses the parent ``GPTDataset`` machinery for everything
expensive (mmaped .bin/.idx access, document-sample-shuffle indices, sequence
packing) and only overrides ``__getitem__`` to:

  1. Pull the raw document text (parent's ``_query_document_sample_shuffle_indices``).
  2. Re-parse it with ``build_delay_pattern_sample`` to extract the audio span
     between the configured markers, apply the MusicGen delay shift, and emit
     the seven per-position tensors the multi-codebook GPT model expects.

The parent's text-only path (``tokens / labels / loss_mask / position_ids``)
is replaced by the collator output, which also returns ``modality_mask /
audio_tokens / audio_labels / audio_loss_mask``. All eight tensors share the
same sequence length, so downstream batch broadcasting, CP slicing, and
packing logic apply uniformly.

Activation: pass ``--multi-codebook-data`` plus marker ids matching whatever
the preprocessor wrote (``tools/audio/preprocess_fleurs_hcodec.py``).
"""

from typing import Dict, Optional

import torch

from megatron.core.datasets.delay_pattern import build_delay_pattern_sample
from megatron.core.datasets.gpt_dataset import GPTDataset


class AudioTextGPTDataset(GPTDataset):
    """GPTDataset variant that produces multi-codebook delay-pattern tensors.

    Construction follows the standard ``MegatronDataset`` pattern; the only
    extra state lives on ``self.config`` (see ``GPTDatasetConfig`` fields
    ``audio_start_id / audio_end_id / num_audio_codebooks / audio_pad_token_id``).
    """

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        # Pull the raw .bin/.idx slice exactly as the parent would (handles None
        # for batch-padding sequences, document-sample-shuffle indexing, etc.).
        if idx is None:
            text, _ = self._query_document_sample_shuffle_indices(0)
        else:
            text, _ = self._query_document_sample_shuffle_indices(idx)

        # ``text`` already has length ``seq_length + add_extra_token_to_sequence``;
        # the collator handles the trailing token by treating it as the target
        # of the last position, so we trim to seq_length for the input view.
        text_list = text.tolist()
        seq_length = self.config.sequence_length

        sample = build_delay_pattern_sample(
            flat_tokens=text_list,
            seq_length=seq_length,
            num_codebooks=int(self.config.num_audio_codebooks),
            audio_start_id=int(self.config.audio_start_id),
            audio_end_id=int(self.config.audio_end_id),
            audio_pad_id=int(self.config.audio_pad_token_id),
        )

        # Pre-convert to tensors with the dtypes the training loop expects.
        # GPTDataset emits int64 for tokens/labels/position_ids, float32 for
        # the loss masks, bool for the modality mask; we mirror that so the
        # broadcast paths don't need per-key dtype handling.
        out: Dict[str, torch.Tensor] = {
            "tokens": torch.tensor(sample["tokens"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
            "loss_mask": torch.tensor(sample["loss_mask"], dtype=torch.float32),
            "position_ids": torch.tensor(sample["position_ids"], dtype=torch.long),
            "modality_mask": torch.tensor(sample["modality_mask"], dtype=torch.bool),
            "audio_tokens": torch.tensor(sample["audio_tokens"], dtype=torch.long),
            "audio_labels": torch.tensor(sample["audio_labels"], dtype=torch.long),
            "audio_loss_mask": torch.tensor(sample["audio_loss_mask"], dtype=torch.float32),
        }

        # Batch-padding sequence: parent zeroes loss_mask; we do the same and
        # additionally zero the audio loss mask so no head trains on padding.
        if idx is None:
            out["loss_mask"].zero_()
            out["audio_loss_mask"].zero_()

        return out
