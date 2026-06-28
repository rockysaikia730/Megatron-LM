# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPT-style dataset that returns the multi-codebook delay-pattern tensors.

Overrides ``__getitem__`` to:

  1. Pull the raw document text.
  2. Re-parse it with ``build_delay_pattern_sample`` to extract the audio span
     between the configured markers, apply the delay shift, and emit
     the seven per-position tensors the multi-codebook GPT model expects.
"""

from typing import Dict, Optional

import torch

from megatron.core.datasets.delay_pattern import build_delay_pattern_sample
from megatron.core.datasets.flatten_pattern import build_flatten_pattern_sample
from megatron.core.datasets.gpt_dataset import GPTDataset


class AudioTextGPTDataset(GPTDataset):
    """GPTDataset variant that produces multi-codebook delay-pattern tensors."""

    def _lazy_load_indexes(self) -> None:
        """Build our per-rank doc permutation."""
        if getattr(self, "_doc_order", None) is not None:
            return

        import numpy as _np

        num_docs = int(self.dataset.sequence_lengths.shape[0])
        if num_docs == 0:
            raise RuntimeError(
                "AudioTextGPTDataset: underlying IndexedDataset has 0 documents"
            )
        rng = _np.random.RandomState(int(self.config.random_seed))
        self._doc_order = rng.permutation(num_docs).astype(_np.int64)
        self._num_docs = num_docs

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        self._lazy_load_indexes()

        if idx is None:
            doc_idx = int(self._doc_order[0])
        else:
            doc_idx = int(self._doc_order[idx % self._num_docs])

        text = self.dataset.get(doc_idx)
        text_list = text.tolist()
        seq_length = self.config.sequence_length

        audio_pattern = getattr(self.config, "audio_pattern", "delay")
        if audio_pattern == "flatten":
            return self._getitem_flatten(text_list, seq_length, idx)

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

    def _getitem_flatten(
        self, text_list: list, seq_length: int, idx: Optional[int]
    ) -> Dict[str, torch.Tensor]:
        """Flatten + 3D-RoPE sample (unified vocab & single head)."""
        sample = build_flatten_pattern_sample(
            flat_tokens=text_list,
            seq_length=seq_length,
            num_codebooks=int(self.config.num_audio_codebooks),
            audio_start_id=int(self.config.audio_start_id),
            audio_end_id=int(self.config.audio_end_id),
            audio_vocab_base=int(self.config.audio_vocab_base),
            audio_codebook_size=int(self.config.audio_codebook_size),
            num_streams=int(self.config.audio_num_streams),
            stream_order=str(self.config.audio_stream_order),
        )

        out: Dict[str, torch.Tensor] = {
            "tokens": torch.tensor(sample["tokens"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
            "loss_mask": torch.tensor(sample["loss_mask"], dtype=torch.float32),
            # [3, S]: (time, depth, stream); collated to [B, 3, S].
            "position_ids": torch.tensor(sample["position_ids"], dtype=torch.long),
            "modality_mask": torch.tensor(sample["modality_mask"], dtype=torch.bool),
        }
        if idx is None:
            out["loss_mask"].zero_()
        return out
