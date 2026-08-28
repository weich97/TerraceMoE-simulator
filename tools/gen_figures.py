# -*- coding: utf-8 -*-
"""Generate all result figures under docs/assets/ (SVG; text is rendered client-side, GitHub renders directly).

**The numbers embedded in this script are the published numbers** — the figures,
the tables in docs/06, and this script must agree in all three places; all numbers
come from internal measurement records, with the aggregation conventions noted
per section in docs/06-troute-results.md.
Run: python tools/gen_figures.py   (requires matplotlib)
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

plt.rcParams.update({
    "svg.fonttype": "none",           # keep text as selectable <text> elements; the browser renders it with client-side fonts
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "figure.dpi": 100,
    # Byte-for-byte reproducible SVGs: a fixed salt pins the element ids that
    # matplotlib would otherwise randomise, and every savefig below strips the
    # timestamp. Without both, rerunning this script rewrites six files with no
    # change of content and a diff that hides real ones.
    "svg.hashsalt": "terracemoe-route-ablation",
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets")
os.makedirs(OUT, exist_ok=True)

# Set TERRACE_FIG_PDF=1 to also emit PDFs into paper/figures for LaTeX. GitHub
# renders the SVGs and Overleaf needs vector PDF; both come from this one script so
# a figure in the paper can never drift from the same figure in the repository.
PDF_DIR = os.path.join(ROOT, "paper", "figures") if os.environ.get("TERRACE_FIG_PDF") else None
if PDF_DIR:
    os.makedirs(PDF_DIR, exist_ok=True)
_savefig = plt.Figure.savefig


def _savefig_both(self, fname, *a, **kw):
    _savefig(self, fname, *a, **kw)
    if PDF_DIR and str(fname).endswith(".svg"):
        kw.pop("metadata", None)
        _savefig(self, os.path.join(PDF_DIR,
                 os.path.basename(str(fname))[:-4] + ".pdf"), *a, **kw)


plt.Figure.savefig = _savefig_both

C = {"gl": "#5B8DB8", "qo": "#C4823B", "full": "#3A7D5C", "base": "#8A8F98"}
MODES3 = ["group_limited\n(group limit only)", "quota_only\n(equal quota only = MoGE)", "full\n(T-Route)"]
CK = ["gl", "qo", "full"]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- F1 headline
# 13.14B/A1.33B, 4 modes x 4 seeds, holdout val loss, paired deltas vs unconstrained, 90% CI
D20 = {"gl": (0.00276, 0.00223, 0.00329), "qo": (0.00895, 0.00726, 0.01064),
       "full": (0.00339, 0.00224, 0.00455)}
D30 = {"gl": (0.00192, 0.00114, 0.00270), "qo": (0.00895, 0.00751, 0.01039),
       "full": (0.00372, 0.00300, 0.00444)}

fig, ax = plt.subplots(figsize=(7.6, 4.2))
x = range(3)
for i, (dat, off, lab) in enumerate([(D20, -0.17, "readout @20k steps"),
                                     (D30, +0.17, "readout @30k steps")]):
    xs = [j + off for j in x]
    ys = [dat[k][0] for k in CK]
    lo = [dat[k][0] - dat[k][1] for k in CK]
    hi = [dat[k][2] - dat[k][0] for k in CK]
    ax.bar(xs, ys, width=0.3, color=[C[k] for k in CK],
           alpha=(0.95 if i else 0.55), label=lab, edgecolor="white")
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="#333", capsize=4, lw=1.2)
ax.axhline(0.02, color="#B33", ls="--", lw=1.2)
ax.text(2.46, 0.0205, "0.02 nats, tighter than the prespecified margin", color="#B33", ha="right", fontsize=9)
ax.axhspan(0.0056, 0.0058, color="#888", alpha=0.35)
ax.text(2.46, 0.0060, "seed noise from the training-log era 0.0056–0.0058", color="#555",
        ha="right", fontsize=8.5)
ax.set_xticks(list(x), MODES3)
ax.set_ylabel("Δ val loss vs unconstrained top-k (nats, lower is better)")
ax.set_title("Quality cost: T-Route +0.0034 nats, 3.4% of the prespecified 0.1 nats, which sits off-chart")
ax.legend(frameon=False, loc="upper left")
_style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f1-loss-axis.svg"), bbox_inches="tight",
            metadata={"Date": None})
plt.close(fig)

# ---------------------------------------------------------------- F2 per-seed
# All 24 paired deltas (3 constraint tiers x 4 seeds x 2 readouts) are positive — the effect is real and small
SEED = {
    "20k": {"gl": [0.002659, 0.002195, 0.002931, 0.003251],
            "qo": [0.009686, 0.008026, 0.007498, 0.010593],
            "full": [0.003032, 0.003252, 0.002501, 0.004791]},
    "30k": {"gl": [0.001513, 0.002640, 0.001219, 0.002305],
            "qo": [0.010748, 0.008523, 0.007984, 0.008553],
            "full": [0.003432, 0.003636, 0.003208, 0.004601]},
}
fig, ax = plt.subplots(figsize=(7.6, 3.6))
for j, k in enumerate(CK):
    for rp, mk in (("20k", "o"), ("30k", "s")):
        ax.scatter([j + (-0.08 if rp == "20k" else 0.08)] * 4, SEED[rp][k],
                   marker=mk, s=42, color=C[k], alpha=0.85,
                   edgecolor="white", linewidth=0.8,
                   label=("readout @%s" % rp) if j == 0 else None)
ax.axhline(0, color="#333", lw=1)
ax.set_xticks(range(3), MODES3)
ax.set_ylabel("per-seed paired delta (nats)")
ax.set_title("24/24 paired deltas share a positive sign: the difference is real, and each one is far below the tolerance")
ax.legend(frameon=False, loc="upper left")
_style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f2-per-seed.svg"), bbox_inches="tight",
            metadata={"Date": None})
plt.close(fig)

# ---------------------------------------------------------------- F3 downstream TOST
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8),
                              gridspec_kw={"width_ratios": [1, 1.25]})
pts = [("HellaSwag\nacc_norm", 0.158, -0.095, 0.411),
       ("LAMBADA\naccuracy", -0.084, -0.406, 0.238)]
ax.axhspan(-1.0, 1.0, color=C["full"], alpha=0.10)
ax.axhline(1.0, color=C["full"], ls="--", lw=1)
ax.axhline(-1.0, color=C["full"], ls="--", lw=1)
ax.text(1.45, 1.06, "prespecified equivalence bounds ±1.0 pp", color=C["full"], ha="right", fontsize=9)
for i, (n, m, lo, hi) in enumerate(pts):
    ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=C["full"],
                capsize=5, ms=7, lw=1.5)
ax.axhline(0, color="#333", lw=0.8)
ax.set_xticks(range(2), [p[0] for p in pts])
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-1.4, 1.4)
ax.set_ylabel("T-Route Δ vs unconstrained (pp, 90% CI)")
ax.set_title("Downstream: both axes equivalent (TOST)")
_style(ax)

# LAMBADA swing across three readouts: a single readout can fool you
swing = {"gl": [0.369, -0.097, -0.645], "full": [0.752, -0.427, -0.577],
         "qo": [0.398, 0.058, -1.586]}
xs = [10, 20, 30]
for k in ("gl", "qo", "full"):
    ax2.plot(xs, swing[k], "o-", color=C[k], lw=1.6, ms=5,
             label={"gl": "group_limited", "qo": "quota_only", "full": "full (T-Route)"}[k])
ax2.axhspan(-1.0, 1.0, color=C["full"], alpha=0.08)
ax2.axhline(0, color="#333", lw=0.8)
ax2.set_xticks(xs, ["@10k", "@20k", "@30k"])
ax2.set_ylabel("LAMBADA Δ (pp)")
ax2.set_title("A single readout can fool you: across three readouts it swings 1–2 pp and flips sign = noise")
ax2.legend(frameon=False, fontsize=8.5)
_style(ax2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f3-downstream.svg"), bbox_inches="tight",
            metadata={"Date": None})
plt.close(fig)

# ---------------------------------------------------------------- F4 load axis
ENT = {"gl": (0.00018, 0.00033), "qo": (0.00035, 0.00026), "full": (0.00026, 0.00040)}
CV = {"gl": (-0.0061, 0.0125), "qo": (-0.0137, 0.0094), "full": (-0.0094, 0.0147)}
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
for a, dat, ttl, better in ((ax, ENT, "expert-level load entropy Δ (higher = more balanced)", "≥0 means no hidden capacity loss"),
                            (ax2, CV, "expert-level load CV Δ (lower = more balanced)", "≤0 means more balanced")):
    ys = [dat[k][0] for k in CK]
    es = [dat[k][1] for k in CK]
    a.bar(range(3), ys, width=0.5, color=[C[k] for k in CK], edgecolor="white")
    a.errorbar(range(3), ys, yerr=es, fmt="none", ecolor="#333", capsize=4, lw=1.1)
    a.axhline(0, color="#333", lw=0.9)
    a.set_xticks(range(3), MODES3, fontsize=8.5)
    a.set_title(ttl + "\n" + better, fontsize=10)
    _style(a)
fig.suptitle("Load axis: none of the three constraint tiers is worse than unconstrained (paired, n=4)", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f4-load-axis.svg"), bbox_inches="tight",
            metadata={"Date": None})
plt.close(fig)

# ---------------------------------------------------------------- F5 step-time neutrality
fig, ax = plt.subplots(figsize=(6.2, 3.2))
runs = [0.9882, 1.0070]
ax.axhspan(0.97, 1.03, color="#888", alpha=0.15)
ax.text(1.44, 1.024, "run-to-run spread band of same-type jobs ±3%", color="#555", ha="right", fontsize=8.5)
ax.scatter([0, 1], runs, s=70, color=C["full"], zorder=3, edgecolor="white")
ax.axhline(1.0, color="#333", lw=1)
ax.axhline(sum(runs) / 2, color=C["full"], ls="--", lw=1.2)
ax.text(1.44, sum(runs) / 2 - 0.004, "mean 0.9976", color=C["full"], ha="right", fontsize=9)
ax.set_xticks([0, 1], ["run 1", "run 2"])
ax.set_xlim(-0.45, 1.5)
ax.set_ylim(0.955, 1.045)
ax.set_ylabel("G = t(unconstrained) / t(T-Route)")
ax.set_title("Step time: both arms use the baseline a2a and differ only in routing — T-Route charges no step-time tax")
_style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f5-step-neutral.svg"), bbox_inches="tight",
            metadata={"Date": None})
plt.close(fig)

# ---------------------------------------------------------------- F6 forerunner 2x2
M1 = [("global_topk", 6.0302, 0.0003, 0.924), ("quota_only", 6.0313, 0.0002, 0.907),
      ("group_limited", 6.0305, 0.0011, 0.930), ("full(T-Route)", 6.0306, 0.0002, 0.946)]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
cols = [C["gl"], C["qo"], C["gl"], C["full"]]
ax.bar(range(4), [m[1] for m in M1], width=0.55, color=cols, edgecolor="white")
ax.errorbar(range(4), [m[1] for m in M1], yerr=[m[2] for m in M1],
            fmt="none", ecolor="#333", capsize=4, lw=1.1)
ax.set_ylim(6.024, 6.036)
ax.set_xticks(range(4), [m[0] for m in M1], fontsize=8, rotation=12)
ax.set_ylabel("holdout val loss")
ax.set_title("Forerunner (small-scale synthetic, 2×2 factorization): loss spread across the four tiers ≤0.0011 < seed range", fontsize=10)
_style(ax)
ax2.bar(range(4), [m[3] for m in M1], width=0.55, color=cols, edgecolor="white")
ax2.set_ylim(0.88, 0.96)
ax2.set_xticks(range(4), [m[0] for m in M1], fontsize=8, rotation=12)
ax2.set_ylabel("load entropy (higher = balanced)")
ax2.set_title("Same testbed: full has the highest load entropy", fontsize=10)
_style(ax2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f6-forerunner.svg"), bbox_inches="tight",
            metadata={"Date": None})
plt.close(fig)

print("6 figures ->", OUT)
