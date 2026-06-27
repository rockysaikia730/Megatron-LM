# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the flatten-pattern collator (3D time x depth x stream positions).

Pure-Python: no numpy / torch / pytest fixtures required, so it can be run
directly with ``python3 tests/unit_tests/data/test_flatten_pattern.py`` as well
as under pytest. The torch-only ``collate_flatten_pattern_batch`` is exercised
separately and skipped when torch is unavailable.
"""

try:
    from megatron.core.datasets.flatten_pattern import (
        build_flatten_pattern_sample,
        revert_flatten_audio,
    )
except ModuleNotFoundError:
    # Standalone path: megatron.core.__init__ imports torch, which may be absent
    # in a bare dev env. Load the pure-Python module directly from its file.
    import importlib.util
    import os

    _mod_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "megatron", "core", "datasets", "flatten_pattern.py",
    )
    _spec = importlib.util.spec_from_file_location("flatten_pattern", _mod_path)
    _fp = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_fp)
    build_flatten_pattern_sample = _fp.build_flatten_pattern_sample
    revert_flatten_audio = _fp.revert_flatten_audio

# Shared constants.
AUDIO_START = 9001
AUDIO_END = 9002
K = 4
V_A = 1024
BASE = 100000  # audio_vocab_base (above all text/marker ids)
NUM_STREAMS = 2

# Stored 50Hz interleaved frames: even rows acoustic, odd rows semantic.
# Encode stored value as row*10 + k so positions are readable (all < V_A).
#   row 0 = acoustic tau0, row 1 = semantic tau0,
#   row 2 = acoustic tau1, row 3 = semantic tau1
_FRAMES = [
    [0, 1, 2, 3],        # row 0  acoustic tau0
    [10, 11, 12, 13],    # row 1  semantic tau0
    [20, 21, 22, 23],    # row 2  acoustic tau1
    [30, 31, 32, 33],    # row 3  semantic tau1
]
_RAW = [v for frame in _FRAMES for v in frame]            # 16 ints
_DOC = [5, 6, AUDIO_START] + _RAW + [AUDIO_END, 7]        # text + span + text


def _uid(stream, k, val):
    return BASE + stream * (K * V_A) + k * V_A + val


def _build(stream_order="semantic_first", seq_length=24):
    return build_flatten_pattern_sample(
        flat_tokens=_DOC,
        seq_length=seq_length,
        num_codebooks=K,
        audio_start_id=AUDIO_START,
        audio_end_id=AUDIO_END,
        audio_vocab_base=BASE,
        audio_codebook_size=V_A,
        num_streams=NUM_STREAMS,
        stream_order=stream_order,
    )


def test_lengths_and_count():
    """N = 2 text + 1 start + (2 streams * 2 tau * K) audio + 1 end + 1 text."""
    s = _build(seq_length=24)
    # tokens has fixed length S
    assert len(s["tokens"]) == 24
    assert len(s["position_ids"]) == 3
    assert all(len(axis) == 24 for axis in s["position_ids"])
    # 3 leading text positions, 16 audio, end marker, trailing text -> 21 real;
    # 20 positions have a next-token target (the last real position has none).
    assert sum(1 for v in s["loss_mask"] if v > 0) == 20


def test_semantic_first_order_and_ids():
    """First physical timestep must emit semantic K codebooks, then acoustic."""
    s = _build()
    tok = s["tokens"]
    p0, p1, p2 = s["position_ids"]
    mod = s["modality_mask"]

    # positions 0,1 text (5,6); 2 = audio_start marker
    assert tok[0] == 5 and tok[1] == 6 and tok[2] == AUDIO_START
    assert mod[0] is False and mod[2] is False

    # audio begins at position 3. semantic_first => semantic tau0 k0..3 first.
    # semantic tau0 row is _FRAMES[1] = [10,11,12,13]
    for k in range(K):
        pos = 3 + k
        assert mod[pos] is True
        assert tok[pos] == _uid(1, k, 10 + k)        # stream=1 (semantic)
        assert (p0[pos], p1[pos], p2[pos]) == (3, k, 1)
    # then acoustic tau0 k0..3, row _FRAMES[0] = [0,1,2,3]
    for k in range(K):
        pos = 7 + k
        assert tok[pos] == _uid(0, k, 0 + k)         # stream=0 (acoustic)
        assert (p0[pos], p1[pos], p2[pos]) == (3, k, 0)
    # tau1 advances the time axis to 4 (semantic row3, acoustic row2)
    for k in range(K):
        pos = 11 + k
        assert tok[pos] == _uid(1, k, 30 + k)
        assert (p0[pos], p1[pos], p2[pos]) == (4, k, 1)
    for k in range(K):
        pos = 15 + k
        assert tok[pos] == _uid(0, k, 20 + k)
        assert (p0[pos], p1[pos], p2[pos]) == (4, k, 0)

    # audio_end marker at position 19, then text 7 at position 20
    assert tok[19] == AUDIO_END and mod[19] is False
    assert tok[20] == 7 and mod[20] is False


def test_text_positions_are_degenerate():
    """Every text position has axis0 == axis1 == axis2 (1D-RoPE degenerate)."""
    s = _build()
    p0, p1, p2 = s["position_ids"]
    for i in range(len(s["tokens"])):
        if not s["modality_mask"][i]:
            # only check real (non-padding) text positions
            if s["tokens"][i] != 0 or i <= 2 or i in (19, 20):
                assert p0[i] == p1[i] == p2[i]


def test_time_axis_contiguous_and_physical():
    """Time axis is contiguous: text steps +1, the audio block spans T_phys=2."""
    s = _build()
    p0 = s["position_ids"][0]
    # 5(t=0) 6(t=1) start(t=2) audio(t=3,3,..,4,4) end(t=5) 7(t=6)
    assert p0[0] == 0 and p0[1] == 1 and p0[2] == 2
    assert all(p0[3 + j] == 3 for j in range(8))   # tau0 -> time 3 (8 tokens)
    assert all(p0[11 + j] == 4 for j in range(8))  # tau1 -> time 4 (8 tokens)
    assert p0[19] == 5 and p0[20] == 6             # end marker, trailing text


def test_labels_are_next_token():
    """labels[s] == tokens[s+1] wherever loss_mask is set."""
    s = _build()
    tok, lab, lm = s["tokens"], s["labels"], s["loss_mask"]
    for i in range(len(tok) - 1):
        if lm[i] > 0:
            assert lab[i] == tok[i + 1]
    # last real position (20) has no next-token target
    assert lm[20] == 0.0


def test_acoustic_first_flips_streams():
    """acoustic_first emits the acoustic block before the semantic one."""
    s = _build(stream_order="acoustic_first")
    tok = s["tokens"]
    p2 = s["position_ids"][2]
    # first audio position (3) is now acoustic (stream 0)
    assert p2[3] == 0 and tok[3] == _uid(0, 0, 0)
    # the semantic block follows at 7
    assert p2[7] == 1 and tok[7] == _uid(1, 0, 10)


def test_round_trip_reconstructs_frames():
    """revert_flatten_audio must rebuild the exact stored [T50][K] frames."""
    for order in ("semantic_first", "acoustic_first"):
        s = _build(stream_order=order)
        frames = revert_flatten_audio(
            tokens=s["tokens"],
            position_ids=s["position_ids"],
            modality_mask=s["modality_mask"],
            num_codebooks=K,
            audio_vocab_base=BASE,
            audio_codebook_size=V_A,
            num_streams=NUM_STREAMS,
        )
        assert frames == _FRAMES, f"round-trip failed for {order}: {frames}"


def test_text_only_document():
    """A document with no audio span yields all-degenerate positions, no audio."""
    s = build_flatten_pattern_sample(
        flat_tokens=[5, 6, 7, 8],
        seq_length=8,
        num_codebooks=K,
        audio_start_id=AUDIO_START,
        audio_end_id=AUDIO_END,
        audio_vocab_base=BASE,
        audio_codebook_size=V_A,
    )
    assert not any(s["modality_mask"])
    p0, p1, p2 = s["position_ids"]
    assert p0[:4] == [0, 1, 2, 3]
    assert p1[:4] == [0, 1, 2, 3] and p2[:4] == [0, 1, 2, 3]


def test_padding_positions_masked():
    """Positions beyond the real sequence are zeroed and never trained."""
    s = _build(seq_length=30)
    # real length is 21; positions 21..29 are padding
    for i in range(21, 30):
        assert s["tokens"][i] == 0
        assert s["loss_mask"][i] == 0.0
        assert s["modality_mask"][i] is False


def test_collate_shapes():
    """torch collate stacks position_ids to [B, 3, S]; skipped without torch."""
    try:
        import torch  # noqa: F401
        from megatron.core.datasets.flatten_pattern import (
            collate_flatten_pattern_batch,
        )
    except ModuleNotFoundError:
        return  # torch absent in this env -> skip

    samples = [_build(seq_length=24), _build(seq_length=24)]
    batch = collate_flatten_pattern_batch(samples)
    assert tuple(batch["tokens"].shape) == (2, 24)
    assert tuple(batch["position_ids"].shape) == (2, 3, 24)
    assert batch["modality_mask"].dtype == torch.bool


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"All {len(fns)} flatten-pattern tests passed.")


if __name__ == "__main__":
    _run_all()
