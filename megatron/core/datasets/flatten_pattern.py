# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Flatten-pattern for RVQ audio tokens with 3D (time x depth x stream) positions.

The flatten collator gives every codebook its own sequence position so the model can 
carry a **3D rotary position**:

    axis 0 = physical 25Hz time  tau   (advances by 1 per timestep)
    axis 1 = codebook depth      k     (0 .. K-1)
    axis 2 = stream              s     (0 = acoustic, 1 = semantic)

Audio tokens are mapped into a single vocabulary that extends the text
vocab, so the model uses one embedding table and one output head rather than 
separate audio tables/heads.

Storage format in the .bin/.idx
--------------------------------------------------------------------------
    [ text...  <audio_start>  f0_0..f0_{K-1}  f1_0.. ..  f_{T50-1}_{K-1}  <audio_end>  text... ]

The ``T50`` stored frames are the already-interleaved 50Hz HCodec stream: even
rows are acoustic, odd rows semantic, paired by physical time stored row
``2*tau`` is acoustic at time ``tau`` and row ``2*tau + 1`` is semantic at time
``tau``.

Vocabulary id for an audio token
---------------------------------------
    audio_id = audio_vocab_base + stream*(K*V_a) + k*V_a + codebook_value
with ``codebook_value`` in ``[0, V_a)``. ``audio_vocab_base`` must sit above the
highest text/marker id; the resulting audio block spans
``num_streams * K * V_a`` ids (2*4*1024 = 8192 by default).
"""

from typing import Dict, List, Tuple

_ACOUSTIC = 0
_SEMANTIC = 1


def _stream_order(stream_order: str) -> List[int]:
    """Map the order name to the stream-id emission order within a timestep."""
    if stream_order == "semantic_first":
        return [_SEMANTIC, _ACOUSTIC]
    if stream_order == "acoustic_first":
        return [_ACOUSTIC, _SEMANTIC]
    raise ValueError(
        f"stream_order must be 'semantic_first' or 'acoustic_first', got "
        f"{stream_order!r}"
    )


def _parse_emitted(
    flat_tokens: List[int],
    num_codebooks: int,
    audio_start_id: int,
    audio_end_id: int,
    audio_vocab_base: int,
    audio_codebook_size: int,
    num_streams: int,
    stream_order: str,
) -> List[Tuple[int, Tuple[int, int, int], bool]]:
    """Expand a flat document into the linear list of model positions."""
    K = num_codebooks
    V_a = audio_codebook_size
    order = _stream_order(stream_order)

    emitted: List[Tuple[int, Tuple[int, int, int], bool]] = []
    base = 0
    i = 0
    n = len(flat_tokens)
    while i < n:
        tok = flat_tokens[i]
        if tok == audio_start_id:
            # marker is an ordinary text token at the current physical time
            emitted.append((tok, (base, base, base), False))
            base += 1
            i += 1

            span_start = i
            while i < n and flat_tokens[i] != audio_end_id:
                i += 1
            raw = flat_tokens[span_start:i]
            if len(raw) % K != 0:
                raise ValueError(
                    f"audio span length {len(raw)} not divisible by "
                    f"num_codebooks {K}"
                )
            T50 = len(raw) // K
            if T50 % num_streams != 0:
                raise ValueError(
                    f"stored audio frame count {T50} not divisible by "
                    f"num_streams {num_streams}; cannot pair the streams"
                )
            T_phys = T50 // num_streams
            frames = [raw[t * K:(t + 1) * K] for t in range(T50)]

            for tau in range(T_phys):
                for s in order:
                    stored_idx = num_streams * tau + s  # 2*tau + s
                    frame = frames[stored_idx]
                    for k in range(K):
                        val = frame[k]
                        if not (0 <= val < V_a):
                            raise ValueError(
                                f"audio code {val} out of range [0, {V_a}) "
                                f"at tau={tau} stream={s} k={k}"
                            )
                        uid = audio_vocab_base + s * (K * V_a) + k * V_a + val
                        emitted.append((uid, (base + tau, k, s), True))
            base += T_phys

            if i < n and flat_tokens[i] == audio_end_id:
                emitted.append((audio_end_id, (base, base, base), False))
                base += 1
                i += 1
        else:
            emitted.append((tok, (base, base, base), False))
            base += 1
            i += 1
    return emitted


def build_flatten_pattern_sample(
    flat_tokens: List[int],
    seq_length: int,
    num_codebooks: int,
    audio_start_id: int,
    audio_end_id: int,
    audio_vocab_base: int,
    audio_codebook_size: int = 1024,
    num_streams: int = 2,
    stream_order: str = "semantic_first",
    text_pad_id: int = 0,
) -> Dict[str, list]:
    """Build one model sample (flatten + 3D positions) from a flat document.

    Args:
        flat_tokens: the stored document (text + markers + frame-major audio).
        seq_length: target model sequence length S (sample is padded/truncated).
        num_codebooks: K.
        audio_start_id / audio_end_id: text-vocab marker ids delimiting audio.
        audio_vocab_base: union-vocab offset where audio ids begin; must be above
            every text/marker id.
        audio_codebook_size: V_a (HCodec = 1024).
        num_streams: number of interleaved streams (acoustic+semantic = 2).
        stream_order: 'semantic_first' (default) or 'acoustic_first'.
        text_pad_id: input id used at padding positions (loss always masked there).

    Returns:
        Dict of plain Python lists with the keys documented at module level.
        ``position_ids`` is a ``[3][S]`` nested list (axis-major).
    """
    S = seq_length
    emitted = _parse_emitted(
        flat_tokens,
        num_codebooks,
        audio_start_id,
        audio_end_id,
        audio_vocab_base,
        audio_codebook_size,
        num_streams,
        stream_order,
    )
    N = len(emitted)

    tokens: List[int] = [text_pad_id] * S
    labels: List[int] = [0] * S
    loss_mask: List[float] = [0.0] * S
    pos0: List[int] = [0] * S
    pos1: List[int] = [0] * S
    pos2: List[int] = [0] * S
    modality_mask: List[bool] = [False] * S

    for s in range(S):
        if s < N:
            uid, (a0, a1, a2), is_audio = emitted[s]
            tokens[s] = uid
            pos0[s], pos1[s], pos2[s] = a0, a1, a2
            modality_mask[s] = is_audio
        if s + 1 < N:
            labels[s] = emitted[s + 1][0]
            loss_mask[s] = 1.0

    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": [pos0, pos1, pos2],
        "modality_mask": modality_mask,
    }


def revert_flatten_audio(
    tokens: List[int],
    position_ids: List[List[int]],
    modality_mask: List[bool],
    num_codebooks: int,
    audio_vocab_base: int,
    audio_codebook_size: int = 1024,
    num_streams: int = 2,
) -> List[List[int]]:
    """Reconstruct the stored ``[T50][K]`` frames from a flattened sample.

    Inverse of the audio part of :func:`build_flatten_pattern_sample`; used by the
    round-trip unit test and at inference time to feed the HCodec decoder. Assumes
    a single contiguous audio span (the time axis is normalised by the first audio
    timestep, since axis-0 carries the *global* physical time ``base + tau``).

    Args:
        tokens / position_ids / modality_mask: as produced by the builder
            (``position_ids`` axis-major ``[3][S]``).
        num_codebooks: K.
        audio_vocab_base / audio_codebook_size / num_streams: union-vocab layout.

    Returns:
        ``[T50][K]`` list of codebook ids (the even/odd interleaved 50Hz stream).
    """
    K = num_codebooks
    V_a = audio_codebook_size

    audio_positions = [s for s in range(len(tokens)) if modality_mask[s]]
    if not audio_positions:
        return []
    # axis-0 holds global physical time (base + tau); rebase to span-local tau.
    min_time = min(position_ids[0][s] for s in audio_positions)

    cells: Dict[Tuple[int, int], int] = {}
    max_idx = -1
    for s in audio_positions:
        tau = position_ids[0][s] - min_time
        k = position_ids[1][s]
        stream = position_ids[2][s]
        val = (tokens[s] - audio_vocab_base) % V_a
        stored_idx = num_streams * tau + stream
        cells[(stored_idx, k)] = val
        if stored_idx > max_idx:
            max_idx = stored_idx
    T50 = max_idx + 1
    frames: List[List[int]] = [[0] * K for _ in range(T50)]
    for (idx, k), v in cells.items():
        frames[idx][k] = v
    return frames


def collate_flatten_pattern_batch(samples: List[Dict[str, list]]):
    """Stack per-sample dicts to tensors.

    Args:
        samples: list of B dicts from :func:`build_flatten_pattern_sample`.

    Returns:
        Dict[str, torch.Tensor]:
            tokens/labels/loss_mask/modality_mask : [B, S]
            position_ids                          : [B, 3, S]
    """
    import torch

    def stack(key, dtype):
        return torch.tensor([s[key] for s in samples], dtype=dtype)

    return {
        "tokens": stack("tokens", torch.long),
        "labels": stack("labels", torch.long),
        "loss_mask": stack("loss_mask", torch.float),
        "position_ids": torch.tensor(
            [s["position_ids"] for s in samples], dtype=torch.long
        ),
        "modality_mask": stack("modality_mask", torch.bool),
    }
