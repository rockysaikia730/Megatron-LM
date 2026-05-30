#!/usr/bin/env python3
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Preprocess a paired text+audio dataset into a Megatron IndexedDataset.

Each output document is a single flat int32 sequence laid out as

    [ <text_tokens...>  <audio_start_id>  <audio_codes_flat...>  <audio_end_id> ]

where ``<audio_codes_flat>`` is the row-major flattening of the per-sample
``[T_50, K]`` codebook matrix produced by HCodec-1.5 in fixed-rate mode
(acoustic and semantic 25Hz streams interleaved frame-by-frame to 50Hz). At
training time the multi-codebook GPT consumes this document via
``megatron/core/datasets/delay_pattern.py`` which performs the delay shift,
attaches modality masks, and produces the per-position tensors the model sees.

Marker IDs live in the padded-vocab "slack" range that the swissai tokenizer
already pads to (``make-vocab-size-divisible-by 128``). Two reserved IDs are
appropriated here so we don't need to surgery the tokenizer for the PoC; for
the production run we'll move to a proper tokenizer extension (see
``tools/checkpoint/extend_megatron_vocab.py``).

This script lives in the *audio-tokenizer* container (HCodec deps), not the
Megatron training container; the only Megatron dependency is the
``IndexedDatasetBuilder`` writer, which is pure-Python + numpy.

Example
-------
    python preprocess_fleurs_hcodec.py \\
        --hcodec-repo /users/$USER/benchmark-audio-tokenizer/src \\
        --fleurs-dir /capstor/store/cscs/swissai/infra01/audio-datasets/benchmark/fleurs_cache/en_us \\
        --output-prefix /iopsstor/scratch/cscs/$USER/datasets/fleurs_en_us_hcodec/train \\
        --tokenizer alehc/swissai-tokenizer \\
        --audio-start-id 131080 --audio-end-id 131081 \\
        --max-samples 500
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("preprocess_fleurs")


# --------------------------------------------------------------------------- #
# HCodec output -> [T_50, K] interleaved frame matrix
# --------------------------------------------------------------------------- #

def hcodec_to_interleaved(merged_codes: torch.Tensor) -> torch.Tensor:
    """Turn HCodecWrapper.encode output into a flat-row ``[T_50, K]`` matrix.

    HCodecWrapper returns ``[B, 2, K, T_25]`` where dim 1 stacks (acoustic,
    semantic) and dim 3 is 25Hz. The downstream model treats audio as a single
    interleaved 50Hz stream with K codebook columns. Interleaving order:
    acoustic frame t, semantic frame t, acoustic frame t+1, semantic t+1, ...

    Args:
        merged_codes: ``[B, 2, K, T_25]`` integer codebook ids from HCodec.

    Returns:
        ``[T_50, K]`` long tensor on CPU. T_50 = 2 * T_25.
    """
    if merged_codes.dim() != 4 or merged_codes.shape[1] != 2:
        raise ValueError(
            f"expected [B, 2, K, T_25] from HCodec, got {tuple(merged_codes.shape)}"
        )
    if merged_codes.shape[0] != 1:
        raise ValueError(
            "preprocessor handles one sample at a time; got batch "
            f"{merged_codes.shape[0]}"
        )
    acoustic = merged_codes[0, 0]                       # [K, T_25]
    semantic = merged_codes[0, 1]                       # [K, T_25]
    # Stack the two streams on a new last axis, then flatten that axis with the
    # time axis: positions become (t=0,a), (t=0,s), (t=1,a), (t=1,s), ...
    stacked = torch.stack([acoustic, semantic], dim=-1)  # [K, T_25, 2]
    K, T_25, _ = stacked.shape
    interleaved = stacked.reshape(K, T_25 * 2)           # [K, T_50]
    return interleaved.transpose(0, 1).contiguous().cpu().long()  # [T_50, K]


# --------------------------------------------------------------------------- #
# Document assembly
# --------------------------------------------------------------------------- #

def build_document(
    text_ids: list,
    audio_TK: torch.Tensor,
    audio_start_id: int,
    audio_end_id: int,
    audio_codebook_size: int,
) -> np.ndarray:
    """Assemble one flat int32 document.

    Layout (one document)::

        [ text_ids... , audio_start_id , audio_flat... , audio_end_id ]

    ``audio_flat`` is the row-major flattening of ``audio_TK`` (frame 0's K
    codebook ids, then frame 1's, ...). The training-side delay collator parses
    this back into ``[T_50, K]`` using the marker IDs.

    Args:
        text_ids: text-vocab token ids for the transcript.
        audio_TK: ``[T_50, K]`` long tensor of codebook ids.
        audio_start_id / audio_end_id: text-vocab marker ids.
        audio_codebook_size: V_a, used for the bounds assertion only.

    Returns:
        1-D int32 numpy array of length ``len(text_ids) + 2 + T_50 * K``.
    """
    # Guard against id values that would clobber the text-vocab markers when
    # the collator parses the audio span. Codes must be in [0, V_a).
    audio_flat = audio_TK.flatten().numpy()
    if audio_flat.size:
        lo, hi = int(audio_flat.min()), int(audio_flat.max())
        if lo < 0 or hi >= audio_codebook_size:
            raise ValueError(
                f"audio code out of range: [{lo}, {hi}] "
                f"not subset of [0, {audio_codebook_size})"
            )

    parts = [
        np.asarray(text_ids, dtype=np.int32),
        np.asarray([audio_start_id], dtype=np.int32),
        audio_flat.astype(np.int32, copy=False),
        np.asarray([audio_end_id], dtype=np.int32),
    ]
    return np.concatenate(parts)


# --------------------------------------------------------------------------- #
# FLEURS iterator (HuggingFace datasets cache layout)
# --------------------------------------------------------------------------- #

def iter_fleurs(
    fleurs_dir: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Iterable[Tuple[np.ndarray, int, str]]:
    """Yield ``(audio_array_mono_float32, sample_rate, transcript)`` triples.

    Uses HuggingFace ``datasets.load_from_disk`` if the directory is in HF cache
    format. FLEURS exposes a ``raw_transcription`` field plus ``audio.array`` /
    ``audio.sampling_rate``.
    """
    from datasets import load_from_disk

    ds = load_from_disk(fleurs_dir)
    if split in ds:
        ds = ds[split]
    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    logger.info(f"FLEURS: yielding {n} samples from {fleurs_dir} ({split=})")
    for i in range(n):
        sample = ds[i]
        audio = sample["audio"]
        transcript = sample.get("raw_transcription") or sample.get("transcription") or ""
        yield np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"]), transcript


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hcodec-repo", required=True,
                    help="Path containing audio_tokenizers.implementations.hcodec.")
    ap.add_argument("--megatron-repo", required=True,
                    help="Path to the Megatron-LM checkout (for IndexedDatasetBuilder).")
    ap.add_argument("--fleurs-dir", required=True,
                    help="FLEURS language subset dir (HF datasets cache).")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output-prefix", required=True,
                    help="Output path prefix; .bin and .idx are written here.")
    ap.add_argument("--tokenizer", default="alehc/swissai-tokenizer",
                    help="HF tokenizer model id or local path.")
    ap.add_argument("--audio-start-id", type=int, required=True,
                    help="Text-vocab id reserved for <audio_start>.")
    ap.add_argument("--audio-end-id", type=int, required=True,
                    help="Text-vocab id reserved for <audio_end>.")
    ap.add_argument("--audio-codebook-size", type=int, default=1024,
                    help="V_a, the per-codebook vocab (HCodec=1024).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap on samples processed; omit for full split.")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    # Imports gated on CLI paths so the script reports a clear error if either
    # repo is missing, instead of an unhelpful traceback at the top.
    sys.path.insert(0, args.hcodec_repo)
    sys.path.insert(0, args.megatron_repo)
    from audio_tokenizers.implementations.hcodec import HCodecWrapper
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
    from transformers import AutoTokenizer

    # --- text tokenizer ---
    logger.info(f"loading text tokenizer: {args.tokenizer}")
    text_tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(text_tok)
    if not (vocab_size <= args.audio_start_id and vocab_size <= args.audio_end_id):
        logger.warning(
            f"marker ids ({args.audio_start_id}, {args.audio_end_id}) overlap "
            f"the live text vocab (size {vocab_size}); choose ids in the "
            "padded-vocab slack range to avoid collisions"
        )

    # --- audio tokenizer ---
    logger.info("loading HCodec-1.5 (fixed-rate mode assumed via patched config)")
    hcodec = HCodecWrapper(device=args.device)
    hcodec._load_model()  # explicit, since HCodecWrapper defers in some versions

    # --- writer ---
    out_prefix = args.output_prefix
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    bin_path = out_prefix + ".bin"
    idx_path = out_prefix + ".idx"
    builder = IndexedDatasetBuilder(bin_path, dtype=np.int32, multimodal=False)
    logger.info(f"writing -> {bin_path} / {idx_path}")

    n_docs = 0
    n_tokens = 0
    t0 = time.time()

    for audio_np, sr, transcript in iter_fleurs(args.fleurs_dir, args.split, args.max_samples):
        # Skip empty / malformed entries quietly so one bad row doesn't kill the run.
        if audio_np.size == 0 or not transcript:
            continue

        # Run HCodec; encode() resamples internally and returns merged codes
        # shaped [B=1, 2, K, T_25] in fixed-rate mode.
        audio_t = torch.from_numpy(audio_np).unsqueeze(0)  # [1, T]
        with torch.no_grad():
            merged_codes, _ = hcodec.encode(audio_t, sr=sr)
        audio_TK = hcodec_to_interleaved(merged_codes)     # [T_50, K]

        text_ids = text_tok.encode(transcript, add_special_tokens=True)
        doc = build_document(
            text_ids=text_ids,
            audio_TK=audio_TK,
            audio_start_id=args.audio_start_id,
            audio_end_id=args.audio_end_id,
            audio_codebook_size=args.audio_codebook_size,
        )

        # add_item + end_document = "one document containing one sequence".
        # IndexedDataset's per-document granularity (rather than per-sequence)
        # is what the GPTDataset shuffles over, so this is the right unit.
        builder.add_item(torch.from_numpy(doc))
        builder.end_document()

        n_docs += 1
        n_tokens += int(doc.size)
        if n_docs % args.log_every == 0:
            dt = time.time() - t0
            logger.info(
                f"docs={n_docs}  tokens={n_tokens:,}  "
                f"{n_docs/dt:.1f} docs/s  {n_tokens/dt/1000:.0f}k tok/s"
            )

    builder.finalize(idx_path)
    dt = time.time() - t0
    logger.info(
        f"DONE  docs={n_docs}  tokens={n_tokens:,}  "
        f"in {dt:.1f}s ({n_docs/dt:.1f} docs/s)"
    )


if __name__ == "__main__":
    main()
