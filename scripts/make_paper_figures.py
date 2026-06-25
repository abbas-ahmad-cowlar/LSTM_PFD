"""Generate the manuscript figures (F1, F2, F3) for paper/figures/.

Every data-bearing number is read from the committed result artifact, not
hardcoded:
  F1, F2 <- results/p7_strengthen/p7_strengthen_record_level.json  (the n=12 grid)
  F3      <- the docs/PHYSICS.md S4 class->signature map (mirrored below, cited)

Run (laptop, CPU):
  $env:PYTHONIOENCODING='utf-8'; .\\venv\\Scripts\\python.exe scripts\\make_paper_figures.py

Outputs vector PDFs (portable, offline LaTeX build):
  paper/figures/F1_degradation_spread.pdf
  paper/figures/F2_n3_to_n12_dissolution.pdf
  paper/figures/F3_dataset_overview.pdf

GUARDRAILS honored in every label/caption-facing string: complete negative on the
TESTED methods; record-level (528) + seed-level estimands only; the within-seed
McNemar is never presented as evidence (F2's whole point is that it dissolves).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID_JSON = ROOT / "results" / "p7_strengthen" / "p7_strengthen_record_level.json"
OUTDIR = ROOT / "paper" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Consistent, restrained styling.
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

TAU = 1.0  # pre-registered robustness threshold (pts), PROTOCOL S8.8

# Arm display order + labels + colors (CE-only is the reference baseline).
ARMS = [
    ("w0_CEonly", "CE-only\n(baseline)", "#444444"),
    ("w1.0_correct", "correct\nphysics", "#1f77b4"),
    ("w1.0_scramble", "scramble\ncontrol", "#ff7f0e"),
    ("w1.0_random", "random-band\ncontrol", "#2ca02c"),
]


def load_grid() -> dict:
    with open(GRID_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


def fig1_spread(grid: dict) -> None:
    """F1: per-seed clean->5dB degradation spread for the four arms (n=12)."""
    arms = grid["arms"]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    rng = np.random.default_rng(0)  # jitter only; not data

    for i, (key, label, color) in enumerate(ARMS):
        degr = np.asarray(arms[key]["degradation_per_seed"], dtype=float)
        x = i + (rng.uniform(-0.12, 0.12, size=degr.size))
        ax.scatter(x, degr, s=26, color=color, alpha=0.75, zorder=3,
                   edgecolors="white", linewidths=0.4)
        mean = float(arms[key]["degradation_seedmean"])
        ax.hlines(mean, i - 0.25, i + 0.25, color=color, lw=2.2, zorder=4)
        robust = int(arms[key][f"robust_seed_count_tau{TAU}"])
        ax.text(i, -1.8, f"{robust}/12\nrobust", ha="center", va="top",
                fontsize=7.5, color=color)

    ax.axhline(TAU, ls="--", lw=1.0, color="grey", zorder=1)
    ax.text(3.45, TAU + 0.15, r"$\tau=1.0$ pt", ha="right", va="bottom",
            fontsize=7.5, color="grey")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([a[1] for a in ARMS])
    ax.set_ylabel(r"clean$\rightarrow$5 dB degradation (pts)")
    ax.set_xlim(-0.5, len(ARMS) - 0.5)
    ax.set_ylim(-3.0, 18)
    ax.set_title("Per-seed degradation, record level (528), $n=12$",
                 fontsize=9)
    fig.tight_layout()
    out = OUTDIR / "F1_degradation_spread.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def fig2_dissolution(grid: dict) -> None:
    """F2 (HEADLINE): the apparent n=3 benefit dissolves at the n=12 seed level."""
    arms = grid["arms"]
    ce = np.asarray(arms["w0_CEonly"]["degradation_per_seed"], dtype=float)
    co = np.asarray(arms["w1.0_correct"]["degradation_per_seed"], dtype=float)
    p_seed = float(grid["wilcoxon_vs_CEonly"]["w1.0_correct"]["p_two_sided"])

    # Bar heights / printed means: the n=12 panel uses the canonical artifact
    # seed-means (3.54 / 3.47) so every number matches the JSON exactly; the
    # n=3 panel re-averages the first three seeds (4.29 / 0.06, == FINDINGS S0),
    # since no n=3 mean is stored. Per-seed scatter always uses the raw seeds.
    ce_mean12 = float(arms["w0_CEonly"]["degradation_seedmean"])
    co_mean12 = float(arms["w1.0_correct"]["degradation_seedmean"])
    ce3, co3 = ce[:3], co[:3]
    panels = [
        ("$n=3$  (seeds 0--2)", ce3, co3, None, None),
        ("$n=12$  (pre-registered)", ce, co, p_seed, (ce_mean12, co_mean12)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    colors = {"CE-only": "#444444", "correct": "#1f77b4"}

    for ax, (title, ce_d, co_d, p, mean_override) in zip(axes, panels):
        means = list(mean_override) if mean_override is not None \
            else [ce_d.mean(), co_d.mean()]
        ax.bar([0, 1], means, width=0.55,
               color=[colors["CE-only"], colors["correct"]], alpha=0.35,
               zorder=2)
        rng = np.random.default_rng(1)
        for j, d in enumerate((ce_d, co_d)):
            xj = j + rng.uniform(-0.1, 0.1, size=d.size)
            ax.scatter(xj, d, s=22,
                       color=[colors["CE-only"], colors["correct"]][j],
                       zorder=3, edgecolors="white", linewidths=0.4)
        for j, m in enumerate(means):
            ax.text(j, m + 0.3, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["CE-only", "correct\nphysics"])
        gap = means[0] - means[1]
        if p is None:
            ax.set_title(f"{title}\napparent gap {gap:+.2f} pt", fontsize=8.5)
        else:
            ax.set_title(f"{title}\n$\\Delta$={gap:+.2f} pt, Wilcoxon $p$={p:.2f} (n.s.)",
                         fontsize=8.5)
        ax.set_xlim(-0.6, 1.6)

    axes[0].set_ylabel(r"clean$\rightarrow$5 dB degradation (pts)")
    axes[0].set_ylim(-1.0, 12.5)
    fig.suptitle("The apparent noise-robustness benefit is a seed artifact",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    out = OUTDIR / "F2_n3_to_n12_dissolution.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def fig3_overview() -> None:
    """F3: JBFD-11 signature map (classes vs characteristic frequencies at
    Omega_0=60 Hz). Mirrors docs/PHYSICS.md S4; see that file for the normative
    equations. Frequencies are nominal centers for visualization only."""
    # (class label, [tonal/point Hz...], (broadband span Hz or None), family note)
    OM = 60.0
    classes = [
        ("sain", [], None, "noise floor only"),
        ("desalignement", [2 * OM, 3 * OM], None, "2X, 3X harmonics"),
        ("desequilibre", [1 * OM], None, "1X (speed$^2$)"),
        ("jeu", [0.45 * OM, OM, 2 * OM], None, "sub-sync + 1X/2X"),
        ("lubrification", [3.5], None, "2--5 Hz stick-slip"),
        ("cavitation", [2000.0], (1400, 2600), "1.4--2.6 kHz bursts"),
        ("usure", [OM, 2 * OM], (100, 8000), "broadband + asperity"),
        ("oilwhirl", [0.45 * OM], None, "0.42--0.48X whirl"),
        ("mixed_misalign_imbalance", [OM, 2 * OM, 3 * OM], None, "1X+2X+3X"),
        ("mixed_wear_lube", [3.5, OM, 2 * OM], (100, 8000), "broadband+lube"),
        ("mixed_cavit_jeu", [0.45 * OM, OM, 2000.0], (1400, 2600), "HF+sub-sync"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    n = len(classes)
    for row, (name, tones, span, note) in enumerate(classes):
        y = n - 1 - row
        if span is not None:
            ax.fill_betweenx([y - 0.28, y + 0.28], span[0], span[1],
                             color="#2ca02c", alpha=0.18, zorder=1)
        for f in tones:
            ax.scatter(f, y, s=34, color="#1f77b4", zorder=3,
                       edgecolors="white", linewidths=0.4)
        ax.text(1.05, y, note, va="center", ha="left", fontsize=7,
                color="#666666", transform=ax.get_yaxis_transform())

    ax.set_yticks(range(n))
    ax.set_yticklabels([c[0] for c in classes][::-1], fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlim(1, 1.2e4)
    ax.set_xlabel(r"characteristic frequency (Hz) at $\Omega_0=60$ Hz "
                  r"(log scale)")
    ax.set_title("JBFD-11 signature map (11 classes)", fontsize=9)
    ax.grid(axis="x", which="both", ls=":", lw=0.5, alpha=0.5)
    # legend proxies
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markersize=6, label="tonal / shaft-order"),
        Patch(facecolor="#2ca02c", alpha=0.25, label="broadband / HF band"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    fig.tight_layout()
    out = OUTDIR / "F3_dataset_overview.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    grid = load_grid()
    fig1_spread(grid)
    fig2_dissolution(grid)
    fig3_overview()
    print("done.")


if __name__ == "__main__":
    main()
