# -*- coding: utf-8 -*-
"""Generate simulator result figures F7-F10 (SVG; text uses client-side fonts, GitHub renders them directly).

Unlike tools/gen_figures.py (T-Route ablation, numbers embedded), every number in
this script is **computed live by calling the sim/ modules** (single exception:
F12 embeds internal measurement records, noted at the top of that section) --
figure = f(code + calibration constants), and a rerun reproduces the SVG
byte-for-byte (fixed Monte Carlo seed + fixed svg.hashsalt + timestamp stripped
from metadata). The numbers in the tables of docs/05, docs/07 and docs/08 defer
to the output of this script and the sim modules.
Run: python tools/gen_sim_figures.py   (requires matplotlib)
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.colors import TwoSlopeNorm   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sim.calibrate import flat_supernode, synthetic          # noqa: E402
from sim.core import MoEGeometry, one_hop_call, two_hop_call  # noqa: E402
from sim.overlap import evaluate                              # noqa: E402
from sim.sweep import CHAIN_SCENARIOS                         # noqa: E402
from sim.uncertainty import breakeven_ratio, heatmap, mc_band  # noqa: E402

plt.rcParams.update({
    "svg.fonttype": "none",
    "svg.hashsalt": "terracemoe-sim",   # fix the random ids; together with metadata Date=None
                                        # this makes a rerun of the same code reproduce the SVG byte-for-byte
    "font.sans-serif": ["Helvetica Neue", "Arial", "Liberation Sans",
                        "DejaVu Sans", "sans-serif"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "figure.dpi": 100,
})

OUT = os.path.join(ROOT, "docs", "assets")
os.makedirs(OUT, exist_ok=True)

C = {"m0": "#8A8F98", "m1": "#B48EAD", "m2": "#5B8DB8", "m3": "#C4823B",
     "m4": "#3A7D5C", "m5": "#B35A5A", "band": "#5B8DB8"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------- F7 overlap model families: all fail
res = evaluate(flat_supernode(), verbose=False)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0),
                              gridspec_kw={"width_ratios": [1.15, 1]})
order = ["M0", "M1", "M2", "M3", "M4", "M5"]
mk = {"M0": "x", "M1": "v", "M2": "o", "M3": "^", "M4": "s", "M5": "D"}
lo = hi = None
for k in order:
    r = res[k]
    xs = [row[1] for row in r["rows"]]     # measured G
    ys = [row[2] for row in r["rows"]]     # predicted G
    ax.scatter(xs, ys, marker=mk[k], s=44, color=C[k.lower()], alpha=0.85,
               label="%s (MAE %.3f)" % (k, r["mae"]))
    pts = xs + ys
    lo = min(pts) if lo is None else min(lo, min(pts))
    hi = max(pts) if hi is None else max(hi, max(pts))
pad = 0.04
ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#333", lw=1)
ax.fill_between([lo - pad, hi + pad], [lo - pad - 0.035, hi + pad - 0.035],
                [lo - pad + 0.035, hi + pad + 0.035], color="#3A7D5C", alpha=0.12)
ax.text(hi + pad - 0.01, hi + pad - 0.032, "±0.035 tolerance band", color="#3A7D5C",
        ha="right", fontsize=8.5)
ax.set_xlabel("Measured G (6 holdout geometries)")
ax.set_ylabel("Predicted G")
ax.set_title("Six single-parameter overlap model families: none lands in the band", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper left")
_style(ax)

maes = [res[k]["mae"] for k in order]
ax2.bar(range(6), maes, width=0.55, color=[C[k.lower()] for k in order],
        edgecolor="white")
ax2.axhline(0.025, color="#B33", ls="--", lw=1.2)
ax2.text(5.3, 0.028, "gate MAE≤0.025", color="#B33", ha="right", fontsize=9)
for i, k in enumerate(order):
    ax2.text(i, maes[i] + 0.003, "%.3f" % maes[i], ha="center", fontsize=8)
ax2.set_xticks(range(6), order)
ax2.set_ylabel("Holdout MAE")
ax2.set_title("Even the best family (M4) misses the gate by 2x -- negative result stands", fontsize=11)
_style(ax2)
fig.suptitle("Tier-2 push: single-parameter global overlap models are a dead end (fit uses flag only; holdout points never enter the fit)",
             y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f7-overlap-families.svg"), bbox_inches="tight", metadata={"Date": None})
plt.close(fig)

# ------------------------------------------------------- F8 breakeven heatmap
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
mats = [heatmap(chain) for _, chain in CHAIN_SCENARIOS[:2]]
_all = [v for mat, _, _ in mats for row in mat for v in row]
norm = TwoSlopeNorm(vmin=min(_all), vcenter=1.0, vmax=max(_all))   # shared by both panels, so the colorbar holds for both
for ax, (name, chain), (mat, ratios, toks) in zip(axes, CHAIN_SCENARIOS[:2], mats):
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", norm=norm,
                   origin="lower")
    for i in range(len(toks)):
        for j in range(len(ratios)):
            ax.text(j, i, "%.2f" % mat[i][j], ha="center", va="center",
                    fontsize=7.2,
                    color="#222" if 0.6 < mat[i][j] < 1.8 else "#eee")
    ax.set_xticks(range(len(ratios)), ["%g" % r for r in ratios], fontsize=8)
    ax.set_yticks(range(len(toks)), ["%d" % t for t in toks], fontsize=8)
    ax.set_xlabel("Hierarchy ratio (fast-side / slow-side bandwidth)")
    ax.set_ylabel("token / rank")
    be = breakeven_ratio(chain)
    ax.set_title("%s\nbreakeven hierarchy ratio = %.2f (q=3)" % (name, be), fontsize=10)
fig.colorbar(im, ax=axes, shrink=0.85, label="one-hop time / two-hop time (>1 = two-hop faster)")
fig.suptitle("Where two-hop is worth it: green = two-hop wins (communication-level simulation, 16 groups × 8)", y=1.04)
fig.savefig(os.path.join(OUT, "f8-breakeven-map.svg"), bbox_inches="tight", metadata={"Date": None})
plt.close(fig)

# ------------------------------------------------------- F9 Monte Carlo uncertainty band
fig, ax = plt.subplots(figsize=(8.6, 4.6))
band_ratios = [1.03, 1.5, 2, 2.5, 3.2, 4, 4.5, 5.5, 6.5, 8, 11, 15.7]
tier_colors = ["#B35A5A", "#5B8DB8", "#3A7D5C"]
for (name, chain), col in zip(CHAIN_SCENARIOS, tier_colors):
    p5s, meds, p95s = [], [], []
    for r in band_ratios:
        p5, med, p95 = mc_band(r, chain)
        p5s.append(p5); meds.append(med); p95s.append(p95)
    ax.plot(band_ratios, meds, "o-", color=col, lw=1.6, ms=4, label=name)
    ax.fill_between(band_ratios, p5s, p95s, color=col, alpha=0.18)
    be = breakeven_ratio(chain)
    if 1.0 < be < 32.0:
        ax.axvline(be, color=col, ls=":", lw=1)
        ax.text(be, 0.32, "%.1f" % be, color=col, ha="center", fontsize=8.5)
ax.axhline(1.0, color="#333", lw=1)
ax.set_xscale("log")
ax.set_xticks(band_ratios)
ax.set_xticklabels(["%g" % r for r in band_ratios], fontsize=8)
ax.minorticks_off()
ax.set_xlabel("Hierarchy ratio (log axis); dotted lines = breakeven per implementation tier")
ax.set_ylabel("one-hop time / two-hop time (>1 = two-hop faster)")
_, _, flat_p95 = mc_band(1.03, CHAIN_SCENARIOS[2][1])
hier_p5, _, _ = mc_band(8.0, CHAIN_SCENARIOS[0][1])
ax.set_title("Calibration uncertainty propagation (400 Monte Carlo draws, band = [p5, p95]):\n"
             "flat column, most favorable case p95=%.2f %s\n"
             "8x column, least favorable case p5=%.2f %s"
             % (flat_p95, "≤1 (negative verdict robust)" if flat_p95 <= 1.0 else "!! >1",
                hier_p5, ">1 (direction robust)" if hier_p5 > 1.0 else "!! <1"),
             fontsize=10)
ax.legend(frameon=False, fontsize=9, loc="upper left")
_style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f9-uncertainty.svg"), metadata={"Date": None}, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------- F10 scale effect and where α comes from
fig, ax = plt.subplots(figsize=(8.2, 4.2))
worlds = [32, 64, 128, 256, 512]
for like_meas, col, lab in ((True, "#3A7D5C", "α = shape measured on this machine (>128 are borrowed points)"),
                            (False, "#8A8F98", "α = flat 0.05 ms (counterfactual machine)")):
    ys = []
    for w in worlds:
        ng = w // 8
        c = synthetic(3.2, alpha_like_measured=like_meas,
                      chain_us_per_row=CHAIN_SCENARIOS[1][1])
        g = MoEGeometry(name="scale", n_groups=ng, R=8, k=6, M=2,
                        seq=4096, mbs=1, gbs=w * 4096)
        ys.append(one_hop_call(c, g) / two_hop_call(c, g))
    ax.plot(worlds, ys, "o-", color=col, lw=1.8, ms=5, label=lab)
ax.axvspan(128, 512, color="#C4823B", alpha=0.10)
ax.text(256, ax.get_ylim()[0] + 0.06, "α(256)/α(512) is a borrowed curve shape\n"
        "-- conclusions there swing with the α source;\nrecalibrate on any new machine",
        color="#9A6B2F", fontsize=8.5, ha="center")
ax.axhline(1.0, color="#333", lw=1)
ax.set_xscale("log", base=2)
ax.set_xticks(worlds)
ax.set_xticklabels([str(w) for w in worlds])
ax.set_xlabel("Total dies (groups × 8; hierarchy ratio fixed at 3.2, fused-kernel tier)")
ax.set_ylabel("one-hop time / two-hop time")
ax.set_title("Scale effect: on large clusters two-hop's extra edge\ncomes mostly from the growth of α(world)\n"
             "-- and the α shape is a machine property, not a topology property", fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
_style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f10-scale-alpha.svg"), metadata={"Date": None}, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------- F11 verdict testbed panorama
# G values come straight from sim.validate.HOLDOUTS (step-level measured ground truth, one single source);
# per-point run counts / spreads are measurement records kept in the HOLDOUTS comments, embedded here
# and maintained in sync.
from sim.validate import HOLDOUTS  # noqa: E402

GM = {h.geom.name: h.g_measured for h in HOLDOUTS}
NINFO = {"flag": "n=7\nt=3.27, p≈0.017", "tok2x": "n=3\nspread 0.8%",
         "tok4x": "n=3\nspread 0.6%", "k8m2": "n=1", "k8m4": "n=1",
         "n8": "n=1", "n4": "n=1"}

fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
panels = [
    ("Scale axis: node count (on a bandwidth-flat\nmachine, fewer nodes = less for two-hop to save)",
     [("4", "n4"), ("8", "n8"), ("16", "flag")], "node count"),
    ("Token axis: micro-batch (more tokens =\nmore bandwidth-bound = worse for two-hop)",
     [("×1", "flag"), ("×2", "tok2x"), ("×4", "tok4x")], "token multiplier"),
    ("Load axis: k and M (wider cross-group\nfan-out = worse for two-hop)",
     [("k6M2", "flag"), ("k8M2", "k8m2"), ("k8M4", "k8m4")], "routing load"),
]
for ax, (title, pts, xlab) in zip(axes, panels):
    xs = list(range(len(pts)))
    ys = [GM[key] for _, key in pts]
    cols = ["#3A7D5C" if y > 1 else "#B35A5A" for y in ys]
    ax.bar(xs, [y - 1.0 for y in ys], width=0.5, bottom=1.0, color=cols,
           edgecolor="white")
    for x, y, (_, key) in zip(xs, ys, pts):
        ax.text(x, y + (0.006 if y >= 1 else -0.006), "%.4f" % y,
                ha="center", va="bottom" if y >= 1 else "top", fontsize=8.5)
        ax.text(x, 0.796, NINFO[key], ha="center", fontsize=7, color="#666")
    ax.axhline(1.0, color="#333", lw=1)
    ax.set_xticks(xs, [lab for lab, _ in pts])
    ax.set_xlabel(xlab)
    ax.set_ylim(0.78, 1.08)
    ax.set_title(title, fontsize=9.5)
    _style(ax)
axes[0].set_ylabel("G = t(one-hop) / t(two-hop)\n(>1 = two-hop faster)")
fig.suptitle("Verdict testbed panorama: end-to-end G for 7 geometries (bandwidth-flat supernode, 16 groups × 8, same config, only the a2a swapped)"
             " -- the only positive value lands inside the microbenchmark's two-hop-favorable window (4096-token tier); all three axes trend as the mechanism predicts",
             y=1.04, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f11-verdict-bed.svg"), bbox_inches="tight", metadata={"Date": None})
plt.close(fig)

# ------------------------------------------------------- F12 one-sided vs collective a2a
# Note: unlike every other figure in this script, the F12 numbers are **embedded internal
# measurement records** (bandwidth-flat supernode, intra-node 8 dies, row-width-aligned
# convention, medians; instrument and criterion in tools/onesided/ and docs/08),
# not sim output -- to remeasure, run tools/onesided/bench_onesided.py.
MB = [2.10, 4.19, 6.29, 8.39, 12.58, 16.78, 25.17]
A2A = [78.9, 92.2, 98.4, 100.8, 102.8, 102.9, 104.0]
OFF1 = [17.6, 23.3, 41.0, 46.5, 52.4, 53.6, 58.1]        # official form, block_dim=1
BFY1 = [13.6, 14.5, 14.7, 14.9, 14.8, 13.5, 13.0]        # butterfly, block_dim=1
BFY8 = [43.3, 50.1, 59.4, 63.3, 65.2, 64.2, 70.5]        # butterfly, block_dim=8

# Voided tiers per arm (time under 3x that arm's instrument floor; reading voided, drawn hollow)
# -- indices match the archived JSON:
# butterfly x8: over_floor 1.3/2.3/2.9 -> {0,1,2}; official x1: 2.1/-/2.7 -> {0,2}; butterfly x1: 2.7 -> {0}
VOIDS = {"BFY8": {0, 1, 2}, "OFF1": {0, 2}, "BFY1": {0}}


def _arm(ax, ys, void_idx, marker, color, label):
    ax.plot(MB, ys, marker + "-", color=color, lw=1.6, ms=5, label=label)
    vx = [MB[i] for i in sorted(void_idx)]
    vy = [ys[i] for i in sorted(void_idx)]
    ax.plot(vx, vy, marker, color=color, ms=7, markerfacecolor="white", zorder=3)


fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.plot(MB, A2A, "o-", color="#333333", lw=2.0, ms=5, label="collective a2a (HCCL, control arm)")
_arm(ax, BFY8, VOIDS["BFY8"], "s", "#3A7D5C", "one-sided · butterfly 8-core (hollow = under 3× floor, voided)")
_arm(ax, OFF1, VOIDS["OFF1"], "^", "#5B8DB8", "one-sided · official alltoall form (one stream per peer)")
_arm(ax, BFY1, VOIDS["BFY1"], "v", "#B35A5A", "one-sided · butterfly single-core (as-shipped block_dim=1)")
best = max(b / a for i, (b, a) in enumerate(zip(BFY8, A2A)) if i not in VOIDS["BFY8"])
ax.annotate("one-sided best = %.2f× a2a\n(criterion ≥1.15× on ≥2 tiers, preregistered)" % best,
            xy=(MB[-1], BFY8[-1]), xytext=(11, 44), fontsize=9, color="#3A7D5C",
            arrowprops=dict(arrowstyle="->", color="#3A7D5C", lw=1))
ax.set_xlabel("Payload per peer (MB, row-width-aligned convention)")
ax.set_ylabel("Effective bandwidth (GB/s, egress bytes of this die / wall clock)")
ax.set_ylim(0, 115)
ax.set_title("One-sided (aclshmem put) vs collective a2a, intra-node 8 dies:\n"
             "every one-sided configuration fails the criterion\n"
             "single-core MTE ceiling ~14 → 8 cores give a 5× lift → still only 0.68× of a2a", fontsize=10.5)
ax.legend(frameon=False, fontsize=8.5, loc="center right")
_style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f12-onesided.svg"), metadata={"Date": None}, bbox_inches="tight")
plt.close(fig)

print("6 figures ->", OUT)
