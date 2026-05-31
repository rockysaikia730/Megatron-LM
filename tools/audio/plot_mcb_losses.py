#!/usr/bin/env python3
"""Parse Megatron iteration log lines and plot the 6 multi-codebook losses.

The training loop prints one line per iteration like::

  iteration  100/  100 | ... | lm loss: 1.32E+01 | text loss: 8.42E+00 |
    audio loss mean: 5.00E+00 | audio loss k0: 4.79E+00 | audio loss k1: ... |
    audio loss k2: ... | audio loss k3: ... | loss scale: 1.0 | grad norm: 5.7 | ...

This script greps those lines, pulls out the 7 numbers, and emits a single PNG.

Usage::

  python tools/audio/plot_mcb_losses.py \\
    --log /iopsstor/.../logs/interactive/llama3-8b-fleurs-20260531-100000.log \\
    --out loss_curves.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


# One regex per metric. We anchor each on " <name>: <number> |" so we don't
# get tripped up by reordering or by other " | " separators in the line.
_FIELDS = {
    "iter":        r"iteration\s+(\d+)/",
    "lm":          r"lm loss:\s+([\d.E+-]+)",
    "text":        r"text loss:\s+([\d.E+-]+)",
    "audio_mean":  r"audio loss mean:\s+([\d.E+-]+)",
    "k0":          r"audio loss k0:\s+([\d.E+-]+)",
    "k1":          r"audio loss k1:\s+([\d.E+-]+)",
    "k2":          r"audio loss k2:\s+([\d.E+-]+)",
    "k3":          r"audio loss k3:\s+([\d.E+-]+)",
    "grad_norm":   r"grad norm:\s+([\d.E+-]+)",
}
_PAT = {k: re.compile(v) for k, v in _FIELDS.items()}


def parse_log(path: Path) -> dict:
    """Return {field: [values...]} for every iteration line in `path`."""
    series: dict = {k: [] for k in _FIELDS}
    with path.open() as f:
        for line in f:
            if "iteration" not in line or "lm loss:" not in line:
                continue
            row: dict = {}
            ok = True
            for key, pat in _PAT.items():
                m = pat.search(line)
                if m is None:
                    ok = False
                    break
                row[key] = float(m.group(1))
            if ok:
                for k, v in row.items():
                    series[k].append(v)
    return series


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="Path to the tee'd training log.")
    ap.add_argument("--out", default="loss_curves.png", help="Output PNG path.")
    ap.add_argument("--title", default="Multi-codebook training: Llama3-8B + RVQ on FLEURS en_us")
    args = ap.parse_args()

    series = parse_log(Path(args.log))
    n = len(series["iter"])
    if n == 0:
        raise SystemExit(f"no iteration lines parsed from {args.log}")
    print(f"parsed {n} iterations from {args.log}")

    iters = series["iter"]

    # Two-panel layout: losses on top, grad norm + audio random-baseline on bottom.
    fig, (ax_loss, ax_aud) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel 1: text + lm + audio_mean
    ax_loss.plot(iters, series["lm"],         label="lm loss (combined)",       linewidth=2)
    ax_loss.plot(iters, series["text"],       label="text loss",                linewidth=2)
    ax_loss.plot(iters, series["audio_mean"], label="audio loss (mean over K)", linewidth=2)
    ax_loss.axhline(6.93, ls="--", color="grey", alpha=0.6,
                    label="audio random baseline = ln(1024)")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title(args.title)
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    # Panel 2: per-codebook losses
    for k in ("k0", "k1", "k2", "k3"):
        ax_aud.plot(iters, series[k], label=f"audio loss {k}", linewidth=1.5)
    ax_aud.axhline(6.93, ls="--", color="grey", alpha=0.6)
    ax_aud.set_xlabel("iteration")
    ax_aud.set_ylabel("per-codebook loss")
    ax_aud.legend(loc="upper right", ncol=2)
    ax_aud.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
