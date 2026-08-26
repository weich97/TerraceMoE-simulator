# -*- coding: utf-8 -*-
"""Figures explaining T-Route: what the four routing modes do, and what they cost.

F14 is a schematic. It is the one figure in this repository that shows a mechanism
rather than a measurement, and it exists because every result about T-Route is
unreadable without knowing precisely what each constrained mode does to expert
selection. F15 puts the measured outcome of all four modes side by side.

Numbers in F15 are the ablation measurements, embedded here and kept in sync with
docs/06. Run: python tools/gen_route_figures.py   (requires matplotlib)
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

plt.rcParams.update({
    "svg.fonttype": "none",
    "svg.hashsalt": "terracemoe-route",
    "font.sans-serif": ["Helvetica Neue", "Arial", "Liberation Sans",
                        "DejaVu Sans", "sans-serif"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "figure.dpi": 100,
})

OUT = os.path.join(ROOT, "docs", "assets")
os.makedirs(OUT, exist_ok=True)
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

SEL = "#C4823B"       # a selected expert
OFF = "#DDE1E6"       # an unselected expert
GRP = "#3A7D5C"       # a group that receives traffic
COLD = "#B0B6BC"      # a group that does not

# ------------------------------------------------------------------ F14 schematic
# Toy geometry chosen so every mode is expressible and visibly different:
# E = 16 experts, N_g = 4 groups of 4, k = 4 experts per token, M = 2 groups.
NG, PER, K, M = 4, 4, 4, 2

MODES = [
    ("(a) unconstrained top-k",
     [(0, 0), (0, 2), (1, 1), (3, 3)],
     "fan-out up to min(k, N_g) = 4 groups\nmessage sizes 1..k rows, data dependent"),
    ("(b) group-limited only",
     [(1, 0), (1, 1), (1, 3), (2, 2)],
     "fan-out bounded: at most M = 2 groups\nsizes still vary, here 3 rows and 1 row"),
    ("(c) equal quota only (MoGE)",
     [(0, 1), (1, 2), (2, 0), (3, 3)],
     "every group gets exactly k/N_g = 1 row\nconstant size, but fan-out is all N_g groups"),
    ("(d) T-Route: both constraints",
     [(1, 0), (1, 2), (3, 1), (3, 3)],
     "fan-out = M = 2, and exactly k/M = 2 rows each\nbounded and constant: a compile-time envelope"),
]

fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.9))
for ax, (title, picks, note) in zip(axes, MODES):
    chosen_groups = sorted({g for g, _ in picks})
    for g in range(NG):
        hot = g in chosen_groups
        ax.add_patch(Rectangle((-0.42, g - 0.45), PER - 0.16, 0.9,
                               facecolor=(GRP if hot else COLD), alpha=0.13,
                               edgecolor=(GRP if hot else COLD), linewidth=1.2))
        ax.text(-0.72, g, "g%d" % g, ha="right", va="center", fontsize=9,
                color=(GRP if hot else "#98A0A8"))
        for e in range(PER):
            on = (g, e) in picks
            ax.add_patch(Rectangle((e - 0.30, g - 0.30), 0.60, 0.60,
                                   facecolor=(SEL if on else OFF),
                                   edgecolor="white", linewidth=1.4))
    n_rows = {}
    for g, _ in picks:
        n_rows[g] = n_rows.get(g, 0) + 1
    for g, c in sorted(n_rows.items()):
        ax.annotate("%d row%s" % (c, "" if c == 1 else "s"),
                    xy=(PER - 0.45, g), xytext=(PER + 0.15, g),
                    fontsize=8.5, color=SEL, va="center",
                    arrowprops=dict(arrowstyle="->", color=SEL, lw=1.1))
    ax.set_xlim(-1.15, PER + 1.25)
    ax.set_ylim(-0.75, NG - 0.25)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.text(0.5, -0.16, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.6, color="#3A4149")

fig.suptitle("What each routing mode constrains. One token, 16 experts in 4 groups of 4, "
             "k = 4 experts selected, M = 2 groups.\nOrange is a selected expert; a shaded "
             "group is one that receives cross-group traffic. The arrows are what crosses the slow link.",
             fontsize=11, y=1.06)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f14-routing-modes.svg"), bbox_inches="tight",
            metadata={"Date": None})
plt.close(fig)

# ------------------------------------------------------------- F15 measured costs
# Ablation measurements: 13.14B total / 1.33B active, E=128 in 8 groups of 16,
# k=8, M=4, four seeds per mode, 62.9B tokens per arm. Deltas are paired by seed
# against unconstrained top-k, holdout validation loss.
MODE_KEYS = ["group_limited", "quota_only", "full"]
LABEL = {"group_limited": "group-limited\nonly",
         "quota_only": "equal quota\nonly (MoGE)",
         "full": "T-Route\n(both)"}
COLOR = {"group_limited": "#5B8DB8", "quota_only": "#C4823B", "full": "#3A7D5C"}
QUALITY = {"group_limited": (0.00276, 0.00223, 0.00329),
           "quota_only": (0.00895, 0.00726, 0.01064),
           "full": (0.00339, 0.00224, 0.00455)}
ENTROPY = {"group_limited": (0.00018, 0.00033), "quota_only": (0.00035, 0.00026),
           "full": (0.00026, 0.00040)}
# Forerunner testbed, small synthetic 2x2: absolute load entropy per mode
FORE_ENTROPY = {"global_topk": 0.924, "group_limited": 0.930,
                "quota_only": 0.907, "full": 0.946}

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))

ax = axes[0]
xs = range(3)
ys = [QUALITY[k][0] for k in MODE_KEYS]
lo = [QUALITY[k][0] - QUALITY[k][1] for k in MODE_KEYS]
hi = [QUALITY[k][2] - QUALITY[k][0] for k in MODE_KEYS]
ax.bar(xs, ys, width=0.55, color=[COLOR[k] for k in MODE_KEYS], edgecolor="white")
ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="#333", capsize=4, lw=1.2)
ax.axhline(0.1, color="#B33", ls="--", lw=1.2)
ax.text(2.45, 0.093, "preregistered no-loss threshold 0.1", color="#B33",
        ha="right", va="top", fontsize=8.5)
ax.set_yscale("log")
ax.set_ylim(0.0015, 0.16)
ax.set_xticks(list(xs), [LABEL[k] for k in MODE_KEYS], fontsize=8.5)
ax.set_ylabel("delta val loss vs top-k (nats, log)")
ax.set_title("Quality cost: T-Route pays 38% of what\nthe quota alone costs", fontsize=10)

ax = axes[1]
ys = [ENTROPY[k][0] for k in MODE_KEYS]
es = [ENTROPY[k][1] for k in MODE_KEYS]
ax.bar(xs, ys, width=0.55, color=[COLOR[k] for k in MODE_KEYS], edgecolor="white")
ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor="#333", capsize=4, lw=1.1)
ax.axhline(0, color="#333", lw=0.9)
ax.set_xticks(list(xs), [LABEL[k] for k in MODE_KEYS], fontsize=8.5)
ax.set_ylabel("delta load entropy vs top-k")
ax.set_title("Load: every constrained mode is at or above\nthe control, so no hidden capacity loss",
             fontsize=10)

ax = axes[2]
fk = ["global_topk", "group_limited", "quota_only", "full"]
fl = ["top-k", "group-\nlimited", "quota\nonly", "T-Route"]
fc = ["#8A8F98", COLOR["group_limited"], COLOR["quota_only"], COLOR["full"]]
ax.bar(range(4), [FORE_ENTROPY[k] for k in fk], width=0.55, color=fc,
       edgecolor="white")
for i, k in enumerate(fk):
    ax.text(i, FORE_ENTROPY[k] + 0.002, "%.3f" % FORE_ENTROPY[k], ha="center",
            fontsize=8.5)
ax.set_ylim(0.88, 0.96)
ax.set_xticks(range(4), fl, fontsize=8.5)
ax.set_ylabel("load entropy (higher is more even)")
ax.set_title("Forerunner testbed: both constraints\nbalance better than either alone",
             fontsize=10)

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", alpha=0.25, linewidth=0.6)
    a.set_axisbelow(True)

fig.suptitle("What the constraints cost, measured. 13.14B total / 1.33B active, "
             "E=128 in 8 groups, k=8, M=4, four seeds per mode, 62.9B tokens per arm.",
             fontsize=11, y=1.05)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f15-routing-costs.svg"), bbox_inches="tight",
            metadata={"Date": None})
plt.close(fig)

print("2 figures -> %s" % OUT)
