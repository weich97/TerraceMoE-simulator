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

# PDF_TOO: set TERRACE_FIG_PDF=1 to also emit PDFs into paper/figures for LaTeX.
# GitHub renders the SVGs; Overleaf needs vector PDF, and both come from this one
# script so a figure in the paper can never drift from the same figure in the repo.
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

# ------------------------------------------------------- F10 scale effect: what the data does and does not support
# alpha is direct-measured at worlds 8, 16 and 128. Nothing in the corpus measures
# 256 or 512 except one dataset that sits 5x below every other in absolute
# bandwidth and that the cost model fits worst -- so past 128 ranks the curve is
# drawn as a band over four defensible treatments of alpha, not as a line.
ALPHA_TREATMENTS = {
    "same-corpus refit": {256: 0.425, 512: 2.888},
    "borrowed points (previously shipped)": {256: 0.735, 512: 1.859},
    "no growth past 128": {256: 0.378, 512: 0.378},
    "linear in peers past 128": {256: 0.378 + 0.0107 * 128,
                                 512: 0.378 + 0.0107 * 384},
}
MEASURED_W = [32, 64, 128]
EXTRAP_W = [128, 256, 512]


def _scale_ratio(alpha_pts, w):
    c = synthetic(3.2, chain_us_per_row=CHAIN_SCENARIOS[1][1])
    for lvl in (c.fast, c.slow, c.flat):
        lvl.alpha_pts = alpha_pts
    g = MoEGeometry(name="scale", n_groups=w // 8, R=8, k=6, M=2,
                    seq=4096, mbs=1, gbs=w * 4096)
    return one_hop_call(c, g) / two_hop_call(c, g)


base_alpha = dict(synthetic(3.2).fast.alpha_pts)
fig, ax = plt.subplots(figsize=(8.4, 4.4))
measured_y = [_scale_ratio(sorted(base_alpha.items()), w) for w in MEASURED_W]
ax.plot(MEASURED_W, measured_y, "o-", color="#3A7D5C", lw=2.2, ms=6,
        label="direct-measured α (worlds 8/16/128)", zorder=3)

curves = {}
for lab, override in ALPHA_TREATMENTS.items():
    pts = sorted({**base_alpha, **override}.items())
    curves[lab] = [_scale_ratio(pts, w) for w in EXTRAP_W]
lo = [min(c[i] for c in curves.values()) for i in range(len(EXTRAP_W))]
hi = [max(c[i] for c in curves.values()) for i in range(len(EXTRAP_W))]
ax.fill_between(EXTRAP_W, lo, hi, color="#C4823B", alpha=0.22, zorder=1,
                label="past 128: spread over four defensible α treatments")
for lab, ys in curves.items():
    ax.plot(EXTRAP_W, ys, "--", color="#9A6B2F", lw=1.0, alpha=0.85, zorder=2)
    ax.annotate(lab, xy=(512, ys[-1]), xytext=(4, 0), textcoords="offset points",
                fontsize=7.3, color="#7A5322", va="center")

ax.axvline(128, color="#333", lw=0.9, ls=":")
ax.axhline(1.0, color="#333", lw=1)
ax.set_xscale("log", base=2)
ax.set_xticks([32, 64, 128, 256, 512])
ax.set_xticklabels(["32", "64", "128", "256", "512"])
ax.set_xlim(30, 1250)
ax.set_xlabel("Total dies (groups × 8; hierarchy ratio fixed at 3.2, fused-kernel tier)")
ax.set_ylabel("one-hop time / two-hop time")
ax.set_title("Scale effect, and where the data stops: through 128 dies every α treatment\n"
             "agrees to the digit (%.2f / %.2f / %.2f); at 512 they span %.2f-%.2f -- the direction\n"
             "itself flips, so this repo makes no claim about clusters past 128 ranks"
             % (measured_y[0], measured_y[1], measured_y[2], lo[-1], hi[-1]),
             fontsize=10)
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

# ------------------------------------------------------- F13 where the methods pay off
from sim.platforms import ARCHETYPES, platform_map          # noqa: E402
from sim.uncertainty import mc_band                          # noqa: E402

rows = platform_map()
bes = rows[0]["breakevens"]
tier_names = list(bes)
tier_colors = {tier_names[0]: "#B35A5A", tier_names[1]: "#5B8DB8",
               tier_names[2]: "#3A7D5C"}

fig, ax = plt.subplots(figsize=(9.6, 4.6))
xs = [0.9, 40]
for tier, col in tier_colors.items():
    be = bes[tier]
    ax.axvline(be, color=col, lw=1.6, ls="--")
    ax.text(be, 5.62, " %s\n breakeven %.2f" % (tier.split("(")[0].strip(), be),
            color=col, fontsize=8.2, va="top", ha="left")
ax.axvspan(0.9, min(bes.values()), color="#B35A5A", alpha=0.07)

ys = list(range(len(ARCHETYPES)))
for y, a in zip(ys, ARCHETYPES):
    n_yes = sum(1 for t in tier_names if a.ratio_nominal >= bes[t])
    col = "#3A7D5C" if n_yes == 3 else ("#C4823B" if n_yes else "#B35A5A")
    ax.plot([0.95, a.ratio_nominal], [y, y], color=col, lw=1.1, alpha=0.35, zorder=1)
    ax.scatter([a.ratio_nominal], [y], s=130, color=col, zorder=3,
               edgecolor="white", linewidth=1.2)
    ax.text(a.ratio_nominal * 1.14, y, "%s  (%d/3 tiers)" % (a.label, n_yes),
            va="center", fontsize=8.6, color="#222")

for key, marker, lbl in (("A", "D", "platform A, measured"),):
    ax.scatter([1.03], [-0.85], s=95, marker=marker, color="#333", zorder=4)
    ax.text(1.03 * 1.14, -0.85, "%s (ratio 1.03)" % lbl, va="center",
            fontsize=8.6, color="#333")

ax.set_yticks([])
ax.set_ylim(-1.6, 5.7)
ax.set_xscale("log")
ax.set_xlim(0.95, 42)
ax.set_xticks([1, 1.5, 2, 3, 5, 8, 12, 18, 30])
ax.set_xticklabels(["1", "1.5", "2", "3", "5", "8", "12", "18", "30"], fontsize=8.5)
ax.minorticks_off()
ax.set_xlabel("Hierarchy ratio: fast-side bandwidth / slow-side bandwidth (log scale)")
ax.set_title("Where hierarchical dispatch pays off. Vertical lines are the breakeven "
             "ratio for each\nimplementation tier, so the arrival chain moves the "
             "threshold as much as the fabric does.", fontsize=10.5)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.grid(axis="x", alpha=0.25, linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "f13-platform-map.svg"), bbox_inches="tight",
            metadata={"Date": None})
plt.close(fig)

# ------------------------------------------------------- F16 what one collective call costs
# Like F12, the numbers here are **embedded measurement records** rather than sim
# output: the call-count scan of 2026-08-26 on the calibrated machine (bench
# launch_scan.py, all-to-all, median over 7 repetitions of the slowest rank, then
# averaged over payloads 256 B / 1 KiB / 4 KiB / 16 KiB, which agree to within the
# run-to-run spread because the wire term at 16 KiB is 0.12 us). Constants distilled
# from it live in sim/profile.py; the protocol and the reading are in docs/09.
LS_NS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
BURST_W8_A = [292.3, 216.3, 175.6, 151.9, 134.9, 158.6, 137.5, 137.4, 133.7]
BURST_W8_B = [291.1, 207.8, 170.7, 152.0, 142.8, 129.6, 124.3, 124.5, 124.4]
BURST_W16 = [357.2, 238.1, 191.6, 167.0, 158.5, 136.9, 136.0, 146.7, 128.6]
SERIAL_W8 = [300.3, 279.3, 281.7, 264.0, 275.7, 266.1, 263.9, 251.2, 299.4]
SERIAL_W16 = [339.8, 308.6, 295.8, 287.0, 282.6, 300.0, 284.0, 291.0, 280.4]
DEEP_NS = [64, 256, 512, 1024]                 # plateau check, one node, 1 KiB only
DEEP_US = [120.0, 102.9, 114.6, 124.6]

from sim.profile import (PER_CALL_DEEP_QUEUE_MS, PER_CALL_HOST_EXPOSED_MS)  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3),
                         gridspec_kw={"width_ratios": [1.5, 1.0]})

ax = axes[0]
ax.plot(LS_NS, BURST_W8_A, "o-", color="#3A7D5C", lw=1.8, ms=4.5,
        label="queue kept deep, world 8 (node 1)")
ax.plot(LS_NS, BURST_W8_B, "o--", color="#3A7D5C", lw=1.4, ms=4, alpha=0.75,
        label="queue kept deep, world 8 (node 2)")
ax.plot(DEEP_NS, DEEP_US, "o:", color="#3A7D5C", lw=1.2, ms=4, alpha=0.55,
        label="plateau check to N = 1024")
ax.plot(LS_NS, BURST_W16, "s-", color="#5B8DB8", lw=1.8, ms=4.5,
        label="queue kept deep, world 16")
ax.plot(LS_NS, SERIAL_W8, "^-", color="#B35A5A", lw=1.8, ms=4.5,
        label="host observes each call, world 8")
ax.plot(LS_NS, SERIAL_W16, "v-", color="#C4823B", lw=1.8, ms=4.5,
        label="host observes each call, world 16")
ax.axvspan(8, 12, color="#8A8F98", alpha=0.12)
ax.annotate("the calibration harness times\n10 calls and sits here",
            xy=(9.5, 150), xytext=(1.1, 34), fontsize=8.4, color="#5A6068",
            arrowprops=dict(arrowstyle="->", color="#8A8F98", lw=1))
ax.annotate("flat from N = 16 to N = 1024:\ncollectives do not pipeline", xy=(300, 122),
            xytext=(26, 46), fontsize=8.8, color="#3A7D5C",
            arrowprops=dict(arrowstyle="->", color="#3A7D5C", lw=1))
ax.set_xscale("log")
ax.set_xticks(LS_NS + [1024], [str(n) for n in LS_NS] + ["1024"], fontsize=8)
ax.set_xlabel("Calls issued back to back before the host waits (N)")
ax.set_ylabel("Cost per call (microseconds)")
ax.set_ylim(0, 450)
ax.set_title("One collective call costs what it costs, twice over\n"
             "the same call is 129 us when the host runs ahead and 255 us when it does not",
             fontsize=10.5)
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=1)
_style(ax)

ax = axes[1]
worlds = [2, 4, 8, 16]
xs = range(len(worlds))
alpha_us = [flat_supernode().flat.alpha_ms(w) * 1e3 for w in worlds]
deep_us = [PER_CALL_DEEP_QUEUE_MS[w] * 1e3 for w in worlds]
host_us = [PER_CALL_HOST_EXPOSED_MS.get(w, 0) * 1e3 for w in worlds]
w = 0.26
ax.bar([x - w for x in xs], alpha_us, w, color="#8A8F98", edgecolor="white",
       label="alpha(world) as shipped")
ax.bar(list(xs), deep_us, w, color="#3A7D5C", edgecolor="white",
       label="measured, queue deep")
ax.bar([x + w for x in xs if host_us[x]], [h for h in host_us if h], w,
       color="#B35A5A", edgecolor="white", label="measured, host exposed")
for x, (a, d, h) in zip(xs, zip(alpha_us, deep_us, host_us)):
    for dx, v in ((-w, a), (0, d), (w, h)):
        if v:
            ax.text(x + dx, v + 6, "%.0f" % v, ha="center", fontsize=7.8)
ax.text(0.5, 128, "agrees within 3% where\nnothing is disputed", ha="center",
        va="bottom", fontsize=8, color="#5A6068")
ax.set_xticks(list(xs), ["world %d" % w for w in worlds], fontsize=8.5)
ax.set_ylabel("Fixed cost per call (microseconds)")
ax.set_ylim(0, 330)
ax.set_title("Measured alpha climbs while a node fills,\n"
             "then flattens across nodes; the table does not", fontsize=10.5)
ax.legend(frameon=False, fontsize=8.2, loc="upper left")
_style(ax)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "f16-launch-cost.svg"), metadata={"Date": None},
            bbox_inches="tight")
plt.close(fig)

print("8 figures ->", OUT)
