# -*- coding: utf-8 -*-
"""Platform registry: every measured machine is one sample, none is the story.

The simulator's product is a **method** -- measure primitives, fit the cost model,
pass the gates, then extrapolate -- and a machine is what you apply it to. Each
entry below is one calibration sample: its constants, where they came from, how
well the model fits it, and where it sits on the hierarchy-ratio axis that decides
whether two-hop communication is worth anything at all.

Adding a platform is deliberately small: measure the primitives with the
conventions in docs/05, fit with `sim/fit.py`, and append a `Platform` here. The
registry is what makes the coverage visible -- including its gaps.

## Coverage today

Only platform A has a measured hierarchy ratio, 1.03.  Platform B contributes a
same-corpus fit/consistency check after machine-specific refitting, but its corpus
does not separate the fast and slow levels, so its hierarchy ratio is unresolved.
Nothing here has been calibrated against a machine where the fast side is several
times the slow side, which is precisely the regime the sensitivity study is about.

Wanted, in rough order of what each would buy:

    ratio ~8      one node's NVLink/HCCS domain vs its cross-node fabric --
                  the regime the whole extrapolation targets, currently unmeasured
    ratio ~2-4    anything in between, to test whether the curve bends where the
                  model says it does
    another ~1    cheap to add, buys little: the axis is already sampled here
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .calibrate import (ALPHA_PTS, BETA_FAST, BETA_FLAT, CHAIN_US_PER_ROW,
                        CROSS_NODE_RATIO, SECOND_ALPHA_PTS, SECOND_BETA_FLAT,
                        SPLITS_SYNC_MS, X_HALF_FLAT, saturating_beta)
from .core import ClusterSpec, Level


@dataclass
class Platform:
    """One measured machine, with the provenance that makes its numbers readable."""
    key: str
    label: str
    hierarchy_ratio: float | None   # fast/slow beta only when both were measured
    alpha_pts: list
    beta_inf: float
    x_half: float
    beta_fast: float
    cross_node_ratio: float
    R: int = 8
    splits_sync_ms: float = SPLITS_SYNC_MS
    chain_us_per_row: float = CHAIN_US_PER_ROW
    provenance: str = ""
    fit_median_err: float = float("nan")
    n_targets: int = 0
    notes: str = ""

    def spec(self) -> ClusterSpec:
        flat = saturating_beta(self.beta_inf, self.x_half)
        fast = ([(1e3, self.beta_fast), (1e9, self.beta_fast)]
                if self.beta_fast else flat)
        slow = [(x, b * self.cross_node_ratio) for x, b in flat]
        return ClusterSpec(
            name=self.label, R=self.R,
            fast=Level("node-internal", self.alpha_pts, fast),
            slow=Level("cross-node", self.alpha_pts, slow),
            flat=Level("full-fabric", self.alpha_pts, flat),
            splits_sync_ms=self.splits_sync_ms,
            chain_us_per_row=self.chain_us_per_row)


PLATFORMS = {
    "A": Platform(
        key="A", label="platform A (bandwidth-flat supernode)",
        hierarchy_ratio=1.0 / CROSS_NODE_RATIO,
        alpha_pts=ALPHA_PTS, beta_inf=BETA_FLAT, x_half=X_HALF_FLAT,
        beta_fast=BETA_FAST, cross_node_ratio=CROSS_NODE_RATIO,
        provenance="alpha direct-measured at worlds 8/16/128; beta fitted on this "
                   "machine's own sweeps with alpha pinned; x_half a borrowed shape",
        fit_median_err=0.019, n_targets=6,
        notes="Cross-node and intra-node bandwidth differ by 2.6%, so two-hop has "
              "nothing to buy here. Useful as a calibration sample and as the low "
              "end of the ratio axis; it settles nothing about hierarchical machines."),
    "B": Platform(
        key="B", label="platform B (second machine, re-fitted)",
        hierarchy_ratio=None,
        alpha_pts=SECOND_ALPHA_PTS, beta_inf=SECOND_BETA_FLAT,
        x_half=X_HALF_FLAT, beta_fast=0.0, cross_node_ratio=1.0,
        provenance="alpha, flat beta, and x_half fitted on this machine's C3 corpus; "
                   "the level split and arrival-chain assumptions are retained "
                   "rather than independently measured",
        fit_median_err=0.093, n_targets=44,
        notes="A fit/consistency check on the same C3 corpus used to fit the "
              "machine-specific constants: 44 targets at 9.3% median error. It "
              "shows that the same functional form remains usable after refitting; "
              "it is not an out-of-sample transfer test and supplies no hierarchy "
              "ratio. C4 on platform A is the post-freeze holdout."),
}


# ---------------------------------------------------------------------------
# Ratio scenarios. Only 1.03 is measured (platform A); higher values are synthetic.
# ---------------------------------------------------------------------------


@dataclass
class Archetype:
    """A ratio-only sensitivity point, not a target-platform prediction."""
    key: str
    label: str
    fast_side: str
    slow_side: str
    ratio_nominal: float
    note: str = ""


ARCHETYPES = [
    Archetype("measured-a", "Platform A measured ratio",
              "measured fast side", "measured slow side", 1.03,
              "The only row whose hierarchy ratio is measured."),
    Archetype("ratio-2", "Synthetic ratio scenario",
              "synthetic", "synthetic", 2.0,
              "Sensitivity point; not assigned to a named platform."),
    Archetype("ratio-4", "Synthetic ratio scenario",
              "synthetic", "synthetic", 4.0,
              "Sensitivity point; not assigned to a named platform."),
    Archetype("ratio-9", "Synthetic ratio scenario",
              "synthetic", "synthetic", 9.0,
              "Sensitivity point; not assigned to a named platform."),
    Archetype("ratio-18", "Synthetic ratio scenario",
              "synthetic", "synthetic", 18.0,
              "Sensitivity point; not assigned to a named platform."),
]


def verdict(ratio: float, breakevens: dict) -> dict:
    """Which implementation tiers pay off at a given hierarchy ratio."""
    return {tier: ratio >= be for tier, be in breakevens.items()}


def platform_map(q: int = 3, tok: int = 4096) -> list:
    """For each ratio-only scenario, report which tiers cross their breakeven.

    Only the 1.03 ratio is measured. The remaining rows are synthetic sensitivity
    points and must not be presented as target-platform verdicts.
    """
    from .sweep import CHAIN_SCENARIOS
    from .uncertainty import breakeven_ratio
    bes = {name: breakeven_ratio(chain, q=q, tok=tok)
           for name, chain in CHAIN_SCENARIOS}
    rows = []
    for a in ARCHETYPES:
        rows.append({"archetype": a, "breakevens": bes,
                     "verdict": verdict(a.ratio_nominal, bes)})
    return rows


def get(key: str) -> Platform:
    if key not in PLATFORMS:
        raise KeyError("unknown platform %r; have %s"
                       % (key, sorted(PLATFORMS)))
    return PLATFORMS[key]


def coverage() -> dict:
    """Where the samples sit on the axis that decides two-hop's value."""
    ratios = sorted(p.hierarchy_ratio for p in PLATFORMS.values()
                    if p.hierarchy_ratio is not None)
    return {"n_platforms": len(PLATFORMS),
            "n_ratio_measured": len(ratios),
            "ratio_min": ratios[0], "ratio_max": ratios[-1],
            "spans_hierarchical": ratios[-1] >= 1.5,
            "targets_total": sum(p.n_targets for p in PLATFORMS.values())}


def main() -> None:
    print("=== measured platforms ===")
    print("%-4s %-42s %7s %9s %8s" %
          ("key", "platform", "ratio", "fit err", "targets"))
    for k in sorted(PLATFORMS):
        p = PLATFORMS[k]
        ratio = "%.2f" % p.hierarchy_ratio if p.hierarchy_ratio is not None else "unresolved"
        print("%-4s %-42s %10s %8.1f%% %8d"
              % (k, p.label[:42], ratio, p.fit_median_err * 100,
                 p.n_targets))
    c = coverage()
    print("\n%d platforms, %d validation targets; ratio measured on %d platform(s): %.2f-%.2f"
          % (c["n_platforms"], c["targets_total"], c["n_ratio_measured"],
             c["ratio_min"], c["ratio_max"]))
    if not c["spans_hierarchical"]:
        print("**No measured ratio above 1.5.** Every statement beyond 1.03 is a")
        print("synthetic sensitivity; that gap closes only by calibrating a target")
        print("in the hierarchical regime.")

    rows = platform_map()
    bes = rows[0]["breakevens"]
    print()
    print("=== ratio-only sensitivity scenarios ===")
    tiers = list(bes)
    print("%-44s %6s  %s" % ("scenario", "ratio",
                             "  ".join("%-22s" % t[:22] for t in tiers)))
    print("%-44s %6s  %s" % ("", "",
                             "  ".join("%-22s" % ("breakeven %.2f" % bes[t])
                                       for t in tiers)))
    for r in rows:
        a = r["archetype"]
        cells = "  ".join("%-22s" % ("yes" if r["verdict"][t] else "no")
                          for t in tiers)
        print("%-44s %6.2f  %s" % (a.label[:44], a.ratio_nominal, cells))
    print()
    print("Read the columns, not just the rows: these are ratio-only sensitivity")
    print("scenarios, not deployment verdicts.")
    print()
    print("T-Route is evaluated separately. It is what makes the two-hop")
    print("shape expressible at all -- it bounds each token's cross-group fan-out at")
    print("compile time and fixes the per-token quota for each selected group. Aggregate")
    print("per-peer counts remain data-dependent, so a counts exchange or padding may")
    print("still be required. The measured quality effect is small but nonzero")
    print("(+0.0034 nats); downstream equivalence is reported with incomplete estimator")
    print("provenance, and load/step-time evidence is thinner. See docs/06.")


if __name__ == "__main__":
    main()
