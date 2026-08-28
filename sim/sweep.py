# -*- coding: utf-8 -*-
"""Extrapolation: swap cluster parameters, compare one-hop vs two-hop. **The gate sits at the entrance, not in a comment.**

- Communication-level extrapolation: requires Tier-1 pass (sim/validate_micro.py; currently: PASS).
- Step-level (end-to-end G) extrapolation: requires Tier-2 pass (sim/validate.py;
  currently: **FAIL**, blocked by the phase-ledger/step-ledger contradiction in the
  internal measurement records -- phase delta x call count disagrees with the step-level
  delta by ~5x; the two arms overlap differently on dual streams, so event-timed phase
  spans do not add up to step time. Unlocking Tier-2 takes an overlap-aware composition
  model + its own holdout points, not parameter tuning).

Run:
    python -m sim.sweep
"""
from __future__ import annotations

from .calibrate import CHAIN_US_PER_ROW, aug_flat, synthetic
from .core import MoEGeometry, one_hop_call, two_hop_call
from .validate import validate
from .validate_micro import validate_micro

CHAIN_SCENARIOS = [
    ("PyTorch chain (measured)", CHAIN_US_PER_ROW),   # single provenance: the calibration layer
    # A design target used for sensitivity analysis.  No fused end-to-end kernel
    # has been measured at this cost, so the label must not read like a result.
    ("hypothetical fused target", 0.012),
    ("zero implementation overhead (upper bound)", 0.0),
]


def comm_speedup(cluster, q: int, tok: int, k_base: int = 6) -> float:
    """Communication-time ratio one-hop/two-hop (>1 = two-hop faster). q is realized via M=k/q."""
    k = k_base if k_base % q == 0 and k_base >= q else q
    m = max(k // q, 1)
    g = MoEGeometry(name="sweep", n_groups=16, R=cluster.R, k=k, M=m,
                    seq=tok, mbs=1, gbs=16 * cluster.R * tok)
    return one_hop_call(cluster, g) / two_hop_call(cluster, g)


def main() -> None:
    aug = aug_flat()
    t1_ok, _ = validate_micro(aug, verbose=False)
    print("Tier-1 (communication level): %s" % ("PASS" if t1_ok else "FAIL"))
    t2_ok, _ = validate(aug, verbose=False)
    print("Tier-2 (step level): %s" % ("PASS" if t2_ok else
                                "FAIL -- step-level extrapolation locked (phase/step ledger contradiction in the internal measurement records unresolved)"))
    if not t1_ok:
        raise SystemExit("Tier-1 fails; all extrapolation forbidden.")

    print()
    print("=" * 74)
    print("Communication-level extrapolation (**simulation**; labeling discipline: these are not measurements)")
    print("x-axis = fast/slow bandwidth ratio; cell = one-hop time / two-hop time (>1 = two-hop faster)")
    print("Geometry: 16 groups x R=8, k=6, T=4096 tok/rank (control-testbed operating point)")
    print("=" * 74)
    ratios = [1.03, 2.0, 3.2, 4.5, 8.0, 15.7]
    labels = ["ratio 1.03", "ratio 2", "ratio 3.2", "ratio 4.5",
              "ratio 8", "ratio 15.7"]
    for chain_name, chain in CHAIN_SCENARIOS:
        print("\n-- implementation tier: %s (%.4f us/row) --" % (chain_name, chain))
        print("%-16s" % "q \\ ratio", "".join("%12s" % l[:10] for l in labels))
        for q in (2, 3, 6):
            cells = []
            for rt in ratios:
                c = synthetic(rt, chain_us_per_row=chain)
                cells.append("%12.2f" % comm_speedup(c, q, 4096))
            print("%-16s" % ("q=%d" % q), "".join(cells))
    print()
    print("How to read:")
    print("  · The flat column (1.03) stays <=1 at every implementation tier -- matches our measured negative verdict (internal consistency).")
    print("  · Columns with hierarchy ratio >=3.2 are all >1 inside the calibrated")
    print("    sensitivity model; the ratios are synthetic rather than target measurements.")
    print("  · The implementation tier changes the magnitude materially.  Because the")
    print("    fused tier is hypothetical and the high-ratio columns are sensitivity")
    print("    scenarios, none of these cells is a target-platform deployment verdict.")


if __name__ == "__main__":
    main()
