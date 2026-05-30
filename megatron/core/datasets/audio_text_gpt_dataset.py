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

    def _lazy_load_indexes(self) -> None:
        """Trigger the parent's lazy index build, then ignore its outputs.

        The parent's shuffle_index/sample_index are designed for sequence
        packing -- ``shuffle_index[idx]`` indexes into ``sample_index``, not
        into ``document_index``, so naive composition gives wrong doc ids.
        We don't need any of that: we want one whole document per sample.

        But we DO still need to trigger the parent's index-build code path
        once, because that's also what materialises the on-disk caches and
        decides ``num_samples``. We call its query method on idx=0 in a
        try/except so any window-slicing failure (e.g. mid-frame audio span)
        is swallowed -- we just want the side effect of populating
        ``self.document_index`` etc.

        After this, we shuffle independently per-rank using a numpy RNG
        seeded by ``config.random_seed`` so doc ordering is deterministic.
        """
        if getattr(self, "_doc_order", None) is not None:
            return

        import numpy as _np

        # Force the parent's lazy mmap by calling the query path once. We
        # don't care about the returned tokens; we care about the side
        # effect of self.{shuffle,sample,document}_index being populated.
        try:
            self._query_document_sample_shuffle_indices(0)
        except Exception:
            # The window-slicing path may itself trip on a mid-audio cut;
            # ignore -- the index mmap happens before the slice.
            pass

        # Build our own per-epoch deterministic shuffle over the *unique*
        # documents (not the parent's sample_index slots).
        num_docs = int(self.dataset.sequence_lengths.shape[0])
        rng = _np.random.RandomState(int(self.config.random_seed))
        self._doc_order = rng.permutation(num_docs).astype(_np.int64)
        self._num_docs = num_docs

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        # IMPORTANT: do NOT use the parent's `_query_document_sample_shuffle_indices`.
        # That returns a fixed-length window cut from the *concatenated* token
        # stream, which would slice audio spans mid-frame and break the
        # `<audio_start> ... <audio_end>` invariant the delay collator relies on.
        #
        # For FLEURS-style data each document is ~520 tokens after delay
        # expansion -- well under seq_length -- so loading whole documents
        # wastes a little compute on padding but keeps the audio span intact.
        self._lazy_load_indexes()

        if idx is None:
            doc_idx = int(self._doc_order[0])
        else:
            # Cycle through the per-epoch shuffled order. Multiple epochs use
            # the same shuffle -- good enough for a PoC; if epoch-distinct
            # shuffles matter later we can salt the RNG with idx // num_docs.
            doc_idx = int(self._doc_order[idx % self._num_docs])

        text = self.dataset.get(doc_idx)
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
