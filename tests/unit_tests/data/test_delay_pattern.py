# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the delay-pattern collator.

Pure-Python: no numpy / torch / pytest fixtures required, so it can be run
directly with ``python3 tests/unit_tests/data/test_delay_pattern.py`` as well as
under pytest. The torch-only ``collate_delay_pattern_batch`` is exercised
separately and skipped when torch is unavailable.
"""

try:
    # Normal path (under pytest / when torch is installed).
    from megatron.core.datasets.delay_pattern import (
        apply_delay,
        build_delay_pattern_sample,
        revert_delay,
    )
except ModuleNotFoundError:
    # Standalone path: megatron.core.__init__ imports torch, which may be absent
    # in a bare dev env. Load the pure-Python module directly from its file so
    # the (torch-free) delay logic stays runnable with `python3 <thisfile>`.
    import importlib.util
    import os

    _mod_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "megatron", "core", "datasets", "delay_pattern.py",
    )
    _spec = importlib.util.spec_from_file_location("delay_pattern", _mod_path)
    _dp = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_dp)
    apply_delay = _dp.apply_delay
    build_delay_pattern_sample = _dp.build_delay_pattern_sample
    revert_delay = _dp.revert_delay

# Marker / pad ids used across the tests.
AUDIO_START = 9001
AUDIO_END = 9002
PAD = 1023  # <audio_pad> id inside the codebook vocab
K = 4


def test_apply_delay_worked_example():
    """T=3, K=4 must match the hand-derived delay layout exactly."""
    # frames[t][k] encoded as t*10 + k so we can read positions back easily.
    frames = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]]
    delayed, valid = apply_delay(frames, K, PAD)

    expected = [
        [0, PAD, PAD, PAD],     # j=0
        [10, 1, PAD, PAD],      # j=1
        [20, 11, 2, PAD],       # j=2
        [PAD, 21, 12, 3],       # j=3
        [PAD, PAD, 22, 13],     # j=4
        [PAD, PAD, PAD, 23],    # j=5
    ]
    assert delayed == expected, delayed
    assert len(delayed) == 3 + K - 1 == 6

    expected_valid = [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
    ]
    assert valid == expected_valid, valid


def test_apply_delay_per_codebook_counts_equal_T():
    """Each codebook layer has exactly T real (non-pad) entries -- the leading
    and trailing triangles are complementary, so counts are EQUAL across k."""
    T = 7
    frames = [[t, t, t, t] for t in range(T)]
    _, valid = apply_delay(frames, K, PAD)
    for k in range(K):
        count = sum(1 for j in range(len(valid)) if valid[j][k])
        assert count == T, (k, count)


def test_revert_delay_roundtrips():
    """revert_delay(apply_delay(x)) == x for several shapes."""
    for T in (1, 2, 5, 11):
        frames = [[t * 10 + k for k in range(K)] for t in range(T)]
        delayed, _ = apply_delay(frames, K, PAD)
        assert revert_delay(delayed, K) == frames, T


def test_apply_delay_empty():
    assert apply_delay([], K, PAD) == ([], [])


def test_text_only_document():
    """No audio span: audio tensors all empty/zero, text trained everywhere a
    next token exists."""
    flat = [5, 6, 7, 8]
    S = 6
    out = build_delay_pattern_sample(flat, S, K, AUDIO_START, AUDIO_END, PAD)

    assert out["tokens"] == [5, 6, 7, 8, 0, 0]
    # labels are next-token; last real token (idx 3) has next item idx4 -> none.
    assert out["labels"] == [6, 7, 8, 0, 0, 0]
    assert out["loss_mask"] == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert all(m is False for m in out["modality_mask"])
    assert all(all(v == 0 for v in row) for row in out["audio_tokens"])
    assert all(all(v == 0.0 for v in row) for row in out["audio_loss_mask"])


def _audio_doc():
    """[t0 t1] <audio_start> (T=2 frames, K=4) <audio_end> [t9].

    Audio raw ids encoded as t*10 + k. T=2 -> delayed length T+K-1 = 5.
    """
    t0, t1, t9 = 100, 101, 109
    raw = [0, 1, 2, 3, 10, 11, 12, 13]  # 2 frames x 4
    flat = [t0, t1, AUDIO_START] + raw + [AUDIO_END, t9]
    return flat, (t0, t1, t9)


def test_audio_document_layout():
    flat, (t0, t1, t9) = _audio_doc()
    # item layout (N): t0 t1 <start> [5 delayed audio frames] <end> t9  -> 10 items
    S = 12
    out = build_delay_pattern_sample(flat, S, K, AUDIO_START, AUDIO_END, PAD)

    # ----- INPUT side -----
    # positions: 0:t0 1:t1 2:<start> 3..7:audio 8:<end> 9:t9 10,11:pad
    assert out["tokens"][0:3] == [t0, t1, AUDIO_START]
    assert out["tokens"][8:10] == [AUDIO_END, t9]
    # audio inputs occupy positions 3..7
    assert out["modality_mask"] == [
        False, False, False,            # t0 t1 <start>
        True, True, True, True, True,   # 5 delayed audio frames
        False, False,                   # <end> t9
        False, False,                   # pad
    ]
    # first delayed input frame == [a0_0, PAD, PAD, PAD]
    assert out["audio_tokens"][3] == [0, PAD, PAD, PAD]
    # last delayed input frame (j=4, T=2) == [PAD, PAD, PAD, a1_3] = [PAD,PAD,PAD,13]
    assert out["audio_tokens"][7] == [PAD, PAD, PAD, 13]
    # non-audio positions have zero audio input
    assert out["audio_tokens"][2] == [0, 0, 0, 0]
    assert out["audio_tokens"][8] == [0, 0, 0, 0]


def test_audio_document_targets_and_boundaries():
    flat, (t0, t1, t9) = _audio_doc()
    S = 12
    out = build_delay_pattern_sample(flat, S, K, AUDIO_START, AUDIO_END, PAD)

    # ----- TEXT targets -----
    # pos0 -> t1, pos1 -> <start>  (text head trained)
    assert out["labels"][0] == t1 and out["loss_mask"][0] == 1.0
    assert out["labels"][1] == AUDIO_START and out["loss_mask"][1] == 1.0
    # pos2 (<start> input) -> target is first audio frame: text head NOT trained
    assert out["loss_mask"][2] == 0.0
    # last audio input (pos7) -> target is <audio_end>: text head trained
    assert out["labels"][7] == AUDIO_END and out["loss_mask"][7] == 1.0
    # <end> (pos8) -> t9: text trained
    assert out["labels"][8] == t9 and out["loss_mask"][8] == 1.0

    # ----- AUDIO targets -----
    # audio targets live at positions 2..6 (predicting delayed frames 0..4).
    # pos2 predicts delayed frame 0 -> valid only at k0.
    assert out["audio_labels"][2] == [0, PAD, PAD, PAD]
    assert out["audio_loss_mask"][2] == [1.0, 0.0, 0.0, 0.0]
    # pos6 predicts delayed frame 4 (T=2) = [PAD,PAD,PAD,13] -> valid only k3.
    assert out["audio_labels"][6] == [PAD, PAD, PAD, 13]
    assert out["audio_loss_mask"][6] == [0.0, 0.0, 0.0, 1.0]
    # pos7's target is <end> (text), so audio loss there is OFF.
    assert out["audio_loss_mask"][7] == [0.0, 0.0, 0.0, 0.0]


def test_text_and_audio_losses_mutually_exclusive():
    """At every position, the text head and the audio heads are never trained
    simultaneously (text XOR audio target)."""
    flat, _ = _audio_doc()
    S = 12
    out = build_delay_pattern_sample(flat, S, K, AUDIO_START, AUDIO_END, PAD)
    for s in range(S):
        text_on = out["loss_mask"][s] > 0
        audio_on = any(v > 0 for v in out["audio_loss_mask"][s])
        assert not (text_on and audio_on), s


def test_padding_short_document():
    flat = [5, 6]
    S = 5
    out = build_delay_pattern_sample(flat, S, K, AUDIO_START, AUDIO_END, PAD)
    assert len(out["tokens"]) == S
    assert len(out["audio_tokens"]) == S
    assert all(len(r) == K for r in out["audio_tokens"])
    # only position 0 trains text (0->6); rest padded.
    assert out["loss_mask"] == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_truncation_long_document():
    flat = list(range(50, 50 + 100))  # 100 text tokens, no audio
    S = 8
    out = build_delay_pattern_sample(flat, S, K, AUDIO_START, AUDIO_END, PAD)
    assert len(out["tokens"]) == S
    assert out["tokens"] == [50, 51, 52, 53, 54, 55, 56, 57]
    # every position has a next item (doc longer than S) -> all text trained.
    assert out["loss_mask"] == [1.0] * S


def test_shapes_are_consistent():
    flat, _ = _audio_doc()
    S = 16
    out = build_delay_pattern_sample(flat, S, K, AUDIO_START, AUDIO_END, PAD)
    for key in ("tokens", "labels", "loss_mask", "position_ids", "modality_mask"):
        assert len(out[key]) == S, key
    for key in ("audio_tokens", "audio_labels", "audio_loss_mask"):
        assert len(out[key]) == S, key
        assert all(len(r) == K for r in out[key]), key
    assert out["position_ids"] == list(range(S))


def test_audio_label_ids_are_valid_indices():
    """Masked audio labels must still be a valid embedding/gather index
    (in [0, audio_codebook_size)) so the loss kernel never indexes OOB."""
    flat, _ = _audio_doc()
    S = 12
    audio_vocab = 1024
    out = build_delay_pattern_sample(flat, S, K, AUDIO_START, AUDIO_END, PAD)
    for row in out["audio_labels"]:
        for v in row:
            assert 0 <= v < audio_vocab, v
    for row in out["audio_tokens"]:
        for v in row:
            assert 0 <= v < audio_vocab, v


def _run_all():
    """Standalone runner (no pytest) so this works without numpy/torch."""
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} delay-pattern tests passed.")


if __name__ == "__main__":
    _run_all()
