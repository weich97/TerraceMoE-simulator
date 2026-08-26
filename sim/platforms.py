# -*- coding: utf-8 -*-
"""Platform registry: every calibrated machine is one sample, none is the story.

The simulator's product is a **method** -- measure primitives, fit the cost model,
pass the gates, then extrapolate -- and a machine is what you apply it to. Each
entry below is one calibration sample: its constants, where they came from, how
well the model fits it, and where it sits on the hierarchy-ratio axis that decides
whether two-hop communication is worth anything at all.

Adding a platform is deliberately small: measure the primitives with the
conventions in docs/05, fit with `sim/fit.py`, and append a `Platform` here. The
registry is what makes the coverage visible -- including its gaps.

## Coverage today

The two samples on file sit at essentially the same point of the axis that matters
(hierarchy ratio ~1.0). That is a real limitation of the sample set, not of the
method: nothing here has been calibrated against a machine where the fast side is
several times the slow side, which is precisely the regime the extrapolation is
about. Reading the table as "the method works" is fair; reading it as "hierarchical
clusters are covered" is not.

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
    """One calibrated machine, with the provenance that makes its numbers readable."""
    key: str
    label: str
    hierarchy_ratio: float          # fast-side beta / slow-side beta, as measured
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
        hierarchy_ratio=1.0,
        alpha_pts=SECOND_ALPHA_PTS, beta_inf=SECOND_BETA_FLAT,
        x_half=X_HALF_FLAT, beta_fast=0.0, cross_node_ratio=1.0,
        provenance="every constant re-fitted from this machine's own corpora with "
                   "the x_half shape pinned; nothing inherited from platform A",
        fit_median_err=0.093, n_targets=44,
        notes="The transfer test: same model form, all constants re-fitted, 44 "
              "independent targets at 9.3% median with no bias. This is what "
              "licenses re-calibration on hardware we have never seen."),
}


# ---------------------------------------------------------------------------
# Archetypes: platforms we have NOT measured, specified from public/nominal
# numbers so the method can be pointed at the machines people actually run on.
# ---------------------------------------------------------------------------


@dataclass
class Archetype:
    """A platform class described by its hierarchy ratio, not measured by us.

    Ratios here are **nominal**: vendor link rates, not collective-communication
    measurements. The repo's own rule is that effective ratios differ from nominal
    by more than 2x often enough to flip a verdict, so these entries say where a
    class of machine plausibly lands, and every one of them carries the same
    instruction: measure your own ratio before believing the row.
    """
    key: str
    label: str
    fast_side: str
    slow_side: str
    ratio_nominal: float
    note: str = ""


ARCHETYPES = [
    Archetype("flat-supernode", "Unified-fabric supernode",
              "cross-node switching", "same fabric", 1.0,
              "One switching domain, no fast side to exploit. Platform A is one of "
              "these, measured at 1.03."),
    Archetype("pcie-ib", "PCIe-attached accelerators + InfiniBand",
              "PCIe Gen5 x16 (~60 GB/s)", "IB NDR (~25 GB/s)", 2.4,
              "The narrowest hierarchy in common use; the fast side is barely fast."),
    Archetype("hccs-roce", "Intra-server HCCS + cross-server RoCE",
              "HCCS (~200 GB/s)", "RoCE (~25 GB/s)", 8.0,
              "The configuration TeleChat3-MoE reports +15% throughput on at EP=16."),
    Archetype("nvlink-ib", "8-GPU NVLink node + InfiniBand",
              "NVLink (~450 GB/s/GPU)", "IB NDR (~50 GB/s)", 9.0,
              "The mainstream training node. DeepSeek-V3's node-limited routing plus "
              "IB-to-NVLink forwarding is this family."),
    Archetype("rack-domain", "Rack-scale NVLink/UB domain + cross-rack fabric",
              "in-rack domain (~900 GB/s)", "cross-rack (~50 GB/s)", 18.0,
              "GB200 NVL72 and CloudMatrix-class racks. Largest ratio in production, "
              "and the domain is big enough that EP may fit inside it -- in which "
              "case the cross-domain hop disappears instead of being optimised."),
]


def verdict(ratio: float, breakevens: dict) -> dict:
    """Which implementation tiers pay off at a given hierarchy ratio."""
    return {tier: ratio >= be for tier, be in breakevens.items()}


def platform_map(q: int = 3, tok: int = 4096) -> list:
    """The deliverable: for each platform class, which tiers make two-hop worth it.

    Returns one row per archetype with the verdict per implementation tier. The
    tiers are what the arrival chain costs, so the table's real message is that
    implementation quality decides as much as topology does over most of the axis.
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
    ratios = sorted(p.hierarchy_ratio for p in PLATFORMS.values())
    return {"n_platforms": len(PLATFORMS),
            "ratio_min": ratios[0], "ratio_max": ratios[-1],
            "spans_hierarchical": ratios[-1] >= 1.5,
            "targets_total": sum(p.n_targets for p in PLATFORMS.values())}


def main() -> None:
    print("=== measured platforms ===")
    print("%-4s %-42s %7s %9s %8s" %
          ("key", "platform", "ratio", "fit err", "targets"))
    for k in sorted(PLATFORMS):
        p = PLATFORMS[k]
        print("%-4s %-42s %7.2f %8.1f%% %8d"
              % (k, p.label[:42], p.hierarchy_ratio, p.fit_median_err * 100,
                 p.n_targets))
    c = coverage()
    print("\n%d platforms, %d validation targets, hierarchy ratio spanned %.2f-%.2f"
          % (c["n_platforms"], c["targets_total"], c["ratio_min"], c["ratio_max"]))
    if not c["spans_hierarchical"]:
        print("**No sample above ratio 1.5.** Every statement about hierarchical")
        print("clusters is extrapolation from machines that are not hierarchical;")
        print("that gap closes only by calibrating one, not by more of these.")

    rows = platform_map()
    bes = rows[0]["breakevens"]
    print()
    print("=== where the methods pay off (nominal ratios; measure your own) ===")
    tiers = list(bes)
    print("%-44s %6s  %s" % ("platform class", "ratio",
                             "  ".join("%-22s" % t[:22] for t in tiers)))
    print("%-44s %6s  %s" % ("", "",
                             "  ".join("%-22s" % ("breakeven %.2f" % bes[t])
                                       for t in tiers)))
    for r in rows:
        a = r["archetype"]
        cells = "  ".join("%-22s" % ("yes" if r["verdict"][t] else "no")
                          for t in tiers)
        print("%-44s %6.1f  %s" % (a.label[:44], a.ratio_nominal, cells))
    print()
    print("Read the columns, not just the rows: over most of the axis the")
    print("implementation tier decides as much as the topology does.")
    print()
    print("T-Route's status is different and stronger. It is what makes the two-hop")
    print("shape expressible at all -- it bounds each token's cross-group fan-out at")
    print("compile time and makes every cross-group message constant-size -- and it")
    print("was **measured to be free**: quality-neutral (+0.0034 nats, 3.4% of the")
    print("preregistered threshold, 24/24 paired deltas same sign), downstream-")
    print("equivalent under TOST on two benchmarks, load-neutral, and step-time-")
    print("neutral (G = 0.9976 over two runs). See docs/06. A constraint that buys a")
    print("compile-time traffic envelope at no measurable cost is worth adopting on")
    print("any machine in the 'yes' region, and costs nothing on the others.")


if __name__ == "__main__":
    main()
