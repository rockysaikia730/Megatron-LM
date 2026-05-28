# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MusicGen-style delay-pattern collator for RVQ audio tokens.

This module turns a *flat* multimodal token document into the per-position
tensors consumed by the multi-codebook GPT model (text head + K audio heads).

Storage format (what lives in the .bin/.idx, one document)
----------------------------------------------------------
A single flat int sequence where audio is delimited by two text-vocab marker
tokens and stored frame-major (frame 0's K codebook ids, then frame 1's, ...):

    [ t0 t1 ... t_{L-1}  <audio_start>  a0_0 a0_1 .. a0_{K-1}  a1_0 ..  a_{T-1}_{K-1}  <audio_end>  t_L ... ]
                          ^marker        \------------- T * K audio codebook ids -------------/   ^marker

  * Text tokens (including the two markers) are ordinary text-vocab ids.
  * Audio codebook ids are raw values in [0, audio_codebook_size); the parser
    knows they are audio purely from their position between the markers.
  * Acoustic/semantic interleaving is already baked into the T frames upstream
    (HCodec emits the interleaved 50Hz stream); the collator is modality-blind.

What the collator produces (one model sequence of length `seq_length`)
----------------------------------------------------------------------
The audio span is *re-laid-out*: T stored frames become ``T + K - 1`` model
positions after applying the delay (layer k shifted right by k). At every model
position exactly one modality is the prediction target (text XOR audio), so the
two loss masks are mutually exclusive per position.

    tokens          [S]      text id at text positions, 0 placeholder at audio
    labels          [S]      next text id (text-target positions), 0 elsewhere
    loss_mask       [S]      1.0 where the text head is trained, else 0.0
    position_ids    [S]      0..S-1
    modality_mask   [S]      True where the INPUT is an audio frame
    audio_tokens    [S, K]   delayed codebook ids at audio-input positions, 0 else
    audio_labels    [S, K]   delayed codebook ids at audio-target positions, pad else
    audio_loss_mask [S, K]   1.0 where an audio head is trained (valid, non-pad)

Design notes
------------
* Pure-Python per-sample logic (operates on lists of ints) so it runs in the
  dataloader workers and is unit-testable without numpy/torch. Batch stacking
  (``collate_delay_pattern_batch``) converts to torch tensors.
* The label at position s is always "the content of position s+1" — standard
  teacher forcing. INPUT modality (modality_mask) and OUTPUT modality (which
  loss mask is active) are decoupled: e.g. the <audio_start> position is a text
  *input* but its *target* is the first audio frame.
"""

from typing import Dict, List, Optional, Tuple

# Item kinds emitted by the parser.
_TEXT = 0
_AUDIO = 1


def apply_delay(
    frames: List[List[int]],
    num_codebooks: int,
    audio_pad_id: int,
) -> Tuple[List[List[int]], List[List[bool]]]:
    """Apply the MusicGen delay pattern to a single audio span.

    Codebook layer ``k`` is shifted right (later in time) by ``k`` positions, so
    that when predicting a frame the model has already seen the coarser layers of
    later-arriving fine layers. Gaps created by the shift are filled with
    ``audio_pad_id``.

    For T input frames this yields ``T + K - 1`` output frames::

        delayed[j][k] = frames[j - k][k]   if 0 <= j - k < T   else audio_pad_id
        valid[j][k]   = (0 <= j - k < T)

    Args:
        frames: ``T`` frames, each a list of exactly ``num_codebooks`` ids.
        num_codebooks: K, the number of RVQ codebook layers.
        audio_pad_id: codebook id used to fill delay gaps (the <audio_pad>).

    Returns:
        ``(delayed, valid)`` each of shape ``[T + K - 1][K]``. ``delayed`` holds
        the shifted ids (pad in the gaps); ``valid`` is True exactly where the id
        is a real (non-pad) token.
    """
    K = num_codebooks
    T = len(frames)
    if K <= 0:
        raise ValueError(f"num_codebooks must be positive, got {K}")
    for f in frames:
        if len(f) != K:
            raise ValueError(
                f"every frame must have exactly {K} codebook ids, got {len(f)}"
            )

    if T == 0:
        return [], []

    out_len = T + K - 1
    delayed: List[List[int]] = [[audio_pad_id] * K for _ in range(out_len)]
    valid: List[List[bool]] = [[False] * K for _ in range(out_len)]
    for j in range(out_len):
        for k in range(K):
            src = j - k
            if 0 <= src < T:
                delayed[j][k] = frames[src][k]
                valid[j][k] = True
    return delayed, valid


def revert_delay(
    delayed: List[List[int]],
    num_codebooks: int,
) -> List[List[int]]:
    """Invert :func:`apply_delay`, recovering the ``T`` original frames.

    Used at inference time to turn the model's delayed predictions back into a
    clean ``[T, K]`` stream for the HCodec decoder.

    Args:
        delayed: ``[T + K - 1][K]`` delayed frames (as produced by apply_delay
            or sampled from the heads).
        num_codebooks: K.

    Returns:
        ``[T][K]`` frames where ``frames[t][k] = delayed[t + k][k]``.
    """
    K = num_codebooks
    out_len = len(delayed)
    if out_len == 0:
        return []
    T = out_len - (K - 1)
    if T <= 0:
        raise ValueError(
            f"delayed length {out_len} too short for num_codebooks {K}"
        )
    frames: List[List[int]] = [[0] * K for _ in range(T)]
    for t in range(T):
        for k in range(K):
            frames[t][k] = delayed[t + k][k]
    return frames


def _parse_items(
    flat_tokens: List[int],
    num_codebooks: int,
    audio_start_id: int,
    audio_end_id: int,
    audio_pad_id: int,
) -> List[Tuple[int, List[int], Optional[List[bool]]]]:
    """Walk the flat document into a list of typed items.

    Each item is ``(kind, ids, valid)``:
      * text item:  ``(_TEXT, [token_id], None)`` — includes the markers.
      * audio item: ``(_AUDIO, [k0..k_{K-1}], [v0..v_{K-1}])`` — one delayed frame.

    Audio spans are expanded in place: the ``<audio_start>`` marker is emitted as
    a text item, then the T stored frames are delay-expanded into ``T + K - 1``
    audio items, then ``<audio_end>`` is emitted as a text item.
    """
    K = num_codebooks
    items: List[Tuple[int, List[int], Optional[List[bool]]]] = []
    i = 0
    n = len(flat_tokens)
    while i < n:
        tok = flat_tokens[i]
        if tok == audio_start_id:
            # Marker itself is a text item.
            items.append((_TEXT, [tok], None))
            i += 1
            # Collect raw audio ids until the end marker.
            span_start = i
            while i < n and flat_tokens[i] != audio_end_id:
                i += 1
            raw = flat_tokens[span_start:i]
            if len(raw) % K != 0:
                raise ValueError(
                    f"audio span length {len(raw)} not divisible by "
                    f"num_codebooks {K}"
                )
            T = len(raw) // K
            frames = [raw[t * K:(t + 1) * K] for t in range(T)]
            delayed, valid = apply_delay(frames, K, audio_pad_id)
            for j in range(len(delayed)):
                items.append((_AUDIO, delayed[j], valid[j]))
            if i < n and flat_tokens[i] == audio_end_id:
                items.append((_TEXT, [audio_end_id], None))
                i += 1
            # If no end marker was found the span ran to EOD; that's fine.
        else:
            items.append((_TEXT, [tok], None))
            i += 1
    return items


def build_delay_pattern_sample(
    flat_tokens: List[int],
    seq_length: int,
    num_codebooks: int,
    audio_start_id: int,
    audio_end_id: int,
    audio_pad_id: int,
    text_pad_id: int = 0,
) -> Dict[str, list]:
    """Build one model sample (length ``seq_length``) from a flat document.

    Implements teacher forcing where ``label(s) == item(s + 1)``. The document
    is parsed into ``N`` typed items (text tokens + delay-expanded audio frames);
    the first ``seq_length`` items become the inputs and items ``1..seq_length``
    become their targets. Shorter documents are right-padded (all masks 0);
    longer documents are truncated from the tail.

    Args:
        flat_tokens: the stored document (text + markers + frame-major audio).
        seq_length: target model sequence length S.
        num_codebooks: K.
        audio_start_id / audio_end_id: text-vocab marker ids delimiting audio.
        audio_pad_id: codebook id for delay gaps (and for masked audio labels).
        text_pad_id: placeholder id at audio-input / padded text positions
            (kept a valid embedding index; its loss is always masked out).

    Returns:
        Dict of plain Python lists with the keys documented at module level.
        ``audio_*`` values are nested lists of shape ``[S][K]``.
    """
    K = num_codebooks
    S = seq_length
    items = _parse_items(
        flat_tokens, K, audio_start_id, audio_end_id, audio_pad_id
    )
    N = len(items)

    tokens: List[int] = [0] * S
    labels: List[int] = [0] * S
    loss_mask: List[float] = [0.0] * S
    position_ids: List[int] = list(range(S))
    modality_mask: List[bool] = [False] * S
    audio_tokens: List[List[int]] = [[0] * K for _ in range(S)]
    audio_labels: List[List[int]] = [[audio_pad_id] * K for _ in range(S)]
    audio_loss_mask: List[List[float]] = [[0.0] * K for _ in range(S)]

    for s in range(S):
        # ---- INPUT side: content of item s. ----
        if s < N:
            kind, ids, _ = items[s]
            if kind == _TEXT:
                tokens[s] = ids[0]
                # modality_mask stays False; audio_tokens stays zeros.
            else:  # _AUDIO
                tokens[s] = text_pad_id
                modality_mask[s] = True
                audio_tokens[s] = list(ids)
        # else: padding position -> tokens stays 0, modality_mask False.

        # ---- TARGET side: content of item s + 1. ----
        if s + 1 < N:
            tkind, tids, tvalid = items[s + 1]
            if tkind == _TEXT:
                labels[s] = tids[0]
                loss_mask[s] = 1.0
                # audio_labels stays pad, audio_loss_mask stays 0.
            else:  # _AUDIO target
                # Text head predicts nothing here; keep labels=0, loss_mask=0.
                audio_labels[s] = list(tids)
                audio_loss_mask[s] = [1.0 if v else 0.0 for v in tvalid]
        # else: last position / past EOD -> both targets masked.

    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "modality_mask": modality_mask,
        "audio_tokens": audio_tokens,
        "audio_labels": audio_labels,
        "audio_loss_mask": audio_loss_mask,
    }


def collate_delay_pattern_batch(samples: List[Dict[str, list]]):
    """Stack per-sample dicts (from :func:`build_delay_pattern_sample`) to tensors.

    This is the dataloader ``collate_fn``. Imported lazily so the pure-Python
    per-sample logic above stays usable (and testable) without torch.

    Args:
        samples: list of B dicts, each as returned by build_delay_pattern_sample.

    Returns:
        Dict[str, torch.Tensor] with batch dimension first:
            tokens/labels/loss_mask/position_ids/modality_mask : [B, S]
            audio_tokens/audio_labels/audio_loss_mask          : [B, S, K]
    """
    import torch  # local import: keeps module importable without torch

    def stack(key, dtype):
        return torch.tensor([s[key] for s in samples], dtype=dtype)

    return {
        "tokens": stack("tokens", torch.long),
        "labels": stack("labels", torch.long),
        "loss_mask": stack("loss_mask", torch.float),
        "position_ids": stack("position_ids", torch.long),
        "modality_mask": stack("modality_mask", torch.bool),
        "audio_tokens": stack("audio_tokens", torch.long),
        "audio_labels": stack("audio_labels", torch.long),
        "audio_loss_mask": stack("audio_loss_mask", torch.float),
    }
