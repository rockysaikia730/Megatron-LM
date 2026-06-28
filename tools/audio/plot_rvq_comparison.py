#!/usr/bin/env python3
"""Plot flatten + 3D-RoPE vs delay-pattern RVQ-audio runs from Megatron .out logs.

Parses both the per-iteration *training* lines and the single *validation*
(held-out) line, in BOTH the delay format (``audio loss mean`` / ``audio loss
k0..k3``) and the flatten format (``audio_token_loss`` / ``ac_k*`` / ``sem_k*``),
and produces two PNGs:

  rvq_training_curves.png   audio-NLL training curves (both runs) + random baseline
  rvq_heldout_comparison.png  held-out audio NLL: overall + per-codebook bars

Usage::

  python tools/audio/plot_rvq_comparison.py \\
    --delay-train   logs/slurm/training/rvq-2635583.out \\
    --delay-eval    logs/slurm/training/rvq-2635632.out \\
    --flatten-train logs/slurm/training/rvq-flat-2634194.out \\
    --flatten-eval  logs/slurm/training/rvq-flat-2635621.out \\
    --out-dir plots/

Self-test the parsers (no matplotlib / log files needed)::

  python tools/audio/plot_rvq_comparison.py --selftest
"""

import argparse
import math
import re
from pathlib import Path

# matplotlib is imported lazily inside _plot_* so --selftest runs without it.

_FLOAT = r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
BASELINE = math.log(1024)  # 6.931 — random audio-token NLL

# --- training (per-iteration) line fields -------------------------------------
_TRAIN = {
    "iter":       re.compile(r"iteration\s+(\d+)\s*/"),
    "lm":         re.compile(r"lm loss:\s*" + _FLOAT),
    "text":       re.compile(r"text loss:\s*" + _FLOAT),
    "audio_mean": re.compile(r"audio loss mean:\s*" + _FLOAT),
    "k0":         re.compile(r"audio loss k0:\s*" + _FLOAT),
    "k1":         re.compile(r"audio loss k1:\s*" + _FLOAT),
    "k2":         re.compile(r"audio loss k2:\s*" + _FLOAT),
    "k3":         re.compile(r"audio loss k3:\s*" + _FLOAT),
}

# --- validation (held-out) fields; "<name> value: <float>" --------------------
# Delay names + flatten names are both matched; whichever exist are returned.
_EVAL = {
    "lm":         re.compile(r"lm loss value:\s*" + _FLOAT),
    "text":       re.compile(r"(?:text loss|text_token_loss) value:\s*" + _FLOAT),
    # overall audio NLL: delay 'audio loss mean', flatten 'audio_token_loss'
    "audio":      re.compile(r"(?:audio loss mean|audio_token_loss) value:\s*" + _FLOAT),
    # delay per-codebook
    "k0":         re.compile(r"audio loss k0 value:\s*" + _FLOAT),
    "k1":         re.compile(r"audio loss k1 value:\s*" + _FLOAT),
    "k2":         re.compile(r"audio loss k2 value:\s*" + _FLOAT),
    "k3":         re.compile(r"audio loss k3 value:\s*" + _FLOAT),
    # flatten per-(stream, codebook)
    "ac_k0":      re.compile(r"ac_k0_token_loss value:\s*" + _FLOAT),
    "ac_k1":      re.compile(r"ac_k1_token_loss value:\s*" + _FLOAT),
    "ac_k2":      re.compile(r"ac_k2_token_loss value:\s*" + _FLOAT),
    "ac_k3":      re.compile(r"ac_k3_token_loss value:\s*" + _FLOAT),
    "sem_k0":     re.compile(r"sem_k0_token_loss value:\s*" + _FLOAT),
    "sem_k1":     re.compile(r"sem_k1_token_loss value:\s*" + _FLOAT),
    "sem_k2":     re.compile(r"sem_k2_token_loss value:\s*" + _FLOAT),
    "sem_k3":     re.compile(r"sem_k3_token_loss value:\s*" + _FLOAT),
}


def parse_training(path):
    """Return {field: [values...]} for every per-iteration line (aligned by iter).

    Only fields present in the log are populated (delay has the audio fields;
    flatten pre-reporting has just ``lm``). srun rank prefixes (``0: ``) are fine.
    """
    series = {k: [] for k in _TRAIN}
    for line in Path(path).read_text(errors="ignore").splitlines():
        if "iteration" not in line or "lm loss:" not in line:
            continue
        m_iter = _TRAIN["iter"].search(line)
        if not m_iter:
            continue
        series["iter"].append(int(m_iter.group(1)))
        for key, pat in _TRAIN.items():
            if key == "iter":
                continue
            m = pat.search(line)
            if m:
                series[key].append(float(m.group(1)))
    return {k: v for k, v in series.items() if v}


def parse_eval(text_or_path):
    """Return {field: float} from the validation line. Accepts a path or a string.

    Reads the whole content and searches globally, so it is robust to the very
    long single validation line (and srun rank prefixes).
    """
    text = text_or_path if "\n" in str(text_or_path) or "value:" in str(text_or_path) \
        else Path(text_or_path).read_text(errors="ignore")
    out = {}
    for key, pat in _EVAL.items():
        m = pat.search(text)
        if m:
            out[key] = float(m.group(1))
    return out


# ----------------------------------------------------------------------------- #
# Plotting
# ----------------------------------------------------------------------------- #

def _plot_training(delay_tr, flat_tr, out_path, baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax_cb) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if flat_tr and "lm" in flat_tr:
        ax.plot(flat_tr["iter"], flat_tr["lm"], lw=2, color="tab:blue",
                label="flatten 3D-RoPE — lm loss (≈ audio NLL)")
    if delay_tr and "audio_mean" in delay_tr:
        ax.plot(delay_tr["iter"], delay_tr["audio_mean"], lw=2, color="tab:red",
                label="delay — audio loss mean")
    ax.set_ylabel("audio NLL (nats/token)")
    ax.set_title("Training audio NLL — flatten + 3D-RoPE vs delay (FLEURS en_us)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # per-codebook training curves (delay only; flatten train has no audio fields)
    if delay_tr and "k0" in delay_tr:
        for k, c in zip(("k0", "k1", "k2", "k3"),
                        ("#7bccc4", "#43a2ca", "#0868ac", "#084081")):
            if k in delay_tr:
                ax_cb.plot(delay_tr["iter"], delay_tr[k], lw=1.5, color=c,
                           label=f"delay {k}")
        ax_cb.legend(loc="upper right", ncol=2, title="delay per-codebook")
    else:
        ax_cb.text(0.5, 0.5, "no per-codebook training fields", ha="center",
                   va="center", transform=ax_cb.transAxes, color="gray")
    ax_cb.set_xlabel("iteration")
    ax_cb.set_ylabel("per-codebook loss")
    ax_cb.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


def _plot_heldout(delay_ev, flat_ev, out_path, baseline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_all, ax_cb) = plt.subplots(1, 2, figsize=(13, 5),
                                        gridspec_kw={"width_ratios": [1, 2.2]})

    # Panel A: overall held-out audio NLL
    names, vals, colors = [], [], []
    if flat_ev.get("audio") is not None:
        names.append("flatten\n3D-RoPE"); vals.append(flat_ev["audio"]); colors.append("tab:blue")
    if delay_ev.get("audio") is not None:
        names.append("delay"); vals.append(delay_ev["audio"]); colors.append("tab:red")
    bars = ax_all.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax_all.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)
    ax_all.set_ylabel("held-out audio NLL (nats/token)")
    ax_all.set_title("Overall")
    ax_all.grid(True, axis="y", alpha=0.3)

    # Panel B: per-codebook held-out (flatten 8 stream/codebook + delay 4)
    labels, vals2, colors2 = [], [], []
    for k in ("k0", "k1", "k2", "k3"):
        sem = flat_ev.get(f"sem_{k}")
        if sem is not None:
            labels.append(f"sem {k}"); vals2.append(sem); colors2.append("#2c7fb8")
    for k in ("k0", "k1", "k2", "k3"):
        ac = flat_ev.get(f"ac_{k}")
        if ac is not None:
            labels.append(f"ac {k}"); vals2.append(ac); colors2.append("#41b6c4")
    for k in ("k0", "k1", "k2", "k3"):
        d = delay_ev.get(k)
        if d is not None:
            labels.append(f"delay {k}"); vals2.append(d); colors2.append("tab:red")
    bars2 = ax_cb.bar(range(len(vals2)), vals2, color=colors2)
    for b, v in zip(bars2, vals2):
        ax_cb.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}",
                   ha="center", va="bottom", fontsize=8)
    ax_cb.set_xticks(range(len(labels)))
    ax_cb.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax_cb.set_ylabel("held-out NLL (nats/token)")
    ax_cb.set_title("Per stream / codebook (flatten)  vs  per codebook (delay)")
    ax_cb.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Held-out audio NLL — flatten + 3D-RoPE vs delay (FLEURS en_us, 500 iters)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"saved {out_path}")


# ----------------------------------------------------------------------------- #
# Self-test (parsers, against the exact pasted log lines)
# ----------------------------------------------------------------------------- #

_FLAT_EVAL = ("validation loss at iteration 500 on validation set | lm loss value: "
    "3.669814E+00 | lm loss PPL: 3.92E+01 | text_token_loss value: 5.217469E+00 | "
    "audio_token_loss value: 3.646357E+00 | audio_token_loss PPL: 3.83E+01 | "
    "ac_k0_token_loss value: 3.477592E+00 | ac_k1_token_loss value: 3.984113E+00 | "
    "ac_k2_token_loss value: 4.810050E+00 | ac_k3_token_loss value: 5.149229E+00 | "
    "sem_k0_token_loss value: 3.387852E+00 | sem_k1_token_loss value: 2.992944E+00 | "
    "sem_k2_token_loss value: 2.581524E+00 | sem_k3_token_loss value: 2.787553E+00 |")

_DELAY_EVAL = ("validation loss at iteration 500 on validation set | lm loss value: "
    "1.238816E+01 | text loss value: 4.002913E+00 | audio loss mean value: 8.509657E+00 | "
    "audio loss k0 value: 8.784400E+00 | audio loss k1 value: 1.018868E+01 | "
    "audio loss k2 value: 8.124148E+00 | audio loss k3 value: 6.874165E+00 |")

_DELAY_TRAIN = ("0: iteration      500/     500 | consumed samples: 2000 | lm loss: "
    "7.619929E+00 | text loss: 3.559623E+00 | audio loss mean: 4.073831E+00 | "
    "audio loss k0: 3.721728E+00 | audio loss k1: 4.101208E+00 | audio loss k2: "
    "4.353648E+00 | audio loss k3: 4.345811E+00 | grad norm: 7.639 |")

_FLAT_TRAIN = ("0: iteration      500/     500 | consumed tokens: 0.016B | lm loss: "
    "3.665127E+00 | loss scale: 1.0 | grad norm: 1.191 |")


def _selftest():
    fe = parse_eval(_FLAT_EVAL)
    assert abs(fe["audio"] - 3.646357) < 1e-5, fe
    assert abs(fe["ac_k3"] - 5.149229) < 1e-5 and abs(fe["sem_k2"] - 2.581524) < 1e-5, fe
    assert abs(fe["text"] - 5.217469) < 1e-5, fe
    de = parse_eval(_DELAY_EVAL)
    assert abs(de["audio"] - 8.509657) < 1e-5, de
    assert abs(de["k1"] - 10.18868) < 1e-4 and abs(de["k3"] - 6.874165) < 1e-5, de
    # training lines (single-line files)
    import tempfile, os
    for txt, want in ((_DELAY_TRAIN, ("audio_mean", 4.073831)), (_FLAT_TRAIN, ("lm", 3.665127))):
        p = Path(tempfile.mkdtemp()) / "log.out"
        p.write_text(txt)
        tr = parse_training(p)
        assert tr["iter"] == [500], tr
        assert abs(tr[want[0]][0] - want[1]) < 1e-5, tr
        os.remove(p)
    print("selftest OK: all parsers match the pasted delay + flatten lines.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delay-train")
    ap.add_argument("--delay-eval")
    ap.add_argument("--flatten-train")
    ap.add_argument("--flatten-eval")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--baseline", type=float, default=BASELINE)
    ap.add_argument("--selftest", action="store_true", help="Run parser self-test and exit.")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    delay_tr = parse_training(args.delay_train) if args.delay_train else {}
    flat_tr = parse_training(args.flatten_train) if args.flatten_train else {}
    delay_ev = parse_eval(args.delay_eval) if args.delay_eval else {}
    flat_ev = parse_eval(args.flatten_eval) if args.flatten_eval else {}

    print(f"delay train: {len(delay_tr.get('iter', []))} iters | "
          f"flatten train: {len(flat_tr.get('iter', []))} iters")
    print(f"delay held-out audio = {delay_ev.get('audio')} | "
          f"flatten held-out audio = {flat_ev.get('audio')}")

    if delay_tr or flat_tr:
        _plot_training(delay_tr, flat_tr, out / "rvq_training_curves.png", args.baseline)
    if delay_ev or flat_ev:
        _plot_heldout(delay_ev, flat_ev, out / "rvq_heldout_comparison.png", args.baseline)


if __name__ == "__main__":
    main()
