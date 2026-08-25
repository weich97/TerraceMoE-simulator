# -*- coding: utf-8 -*-
"""Calibration: distill the measurements into a ClusterSpec. **The public version embeds constants; no raw sweep data ships.**

Every constant is annotated with its convention and the nature of its provenance; the raw
measurements (dozens of sweep JSONs, pairwise probes) are internal data and do not ship
with the repo -- but the **validation-gate targets** (validate_micro) come from equally
independent measurements, so external users can rerun the same pipeline on their own
machine: measure the primitives -> fill a ClusterSpec -> pass the validation gates ->
only then extrapolate.

Produces two kinds of clusters:
  flat_supernode()  the bandwidth-flat supernode we actually measured (16 nodes x 8 cards)
  synthetic(...)    synthetic hierarchical clusters -- usable for extrapolation only after
                    the validation gates pass
"""
from __future__ import annotations

from .core import ClusterSpec, Level

# ---------------------------------------------------------------------------
# Distilled constants (all measured, conventions noted; machine run-to-run
# drift ~20%, medians govern)
# ---------------------------------------------------------------------------

# alpha(world) [ms]: fixed overhead per collective call.
#   The 8 / 16 / 128 points are direct measurements on this machine (1 token/rank tier,
#   median of two runs).
#
#   **The 256 and 512 entries are low confidence and no conclusion may rest on them.**
#   A later audit of every size sweep we own (331 usable a2a points over four datasets
#   and two machines) found that the only corpus covering worlds above 128 is also the
#   corpus whose absolute bandwidth sits ~5x below every other one and that the cost
#   model fits worst -- 40% median relative error, against 5-11% everywhere else.
#   Re-fitting that same corpus yields alpha(256)=0.425 and alpha(512)=2.888, nowhere
#   near the entries below. Propagating four defensible treatments of these two points
#   through the scale extrapolation leaves the ratio at 512 dies anywhere in 1.37-2.94,
#   and the sign of the trend flips between treatments (figure F10 plots that band).
#   Conclusions in this repository are therefore stated for worlds <= 128 only, where
#   every treatment agrees to the digit. These two entries stay in the table so that
#   synthetic large clusters remain runnable, not because they are trustworthy.
#
#   The direct measurements also give one important
#   fact: **on this machine alpha(16)+alpha(8) ≈ alpha(128)** -- two-hop saves almost
#   nothing on alpha. Whether alpha is savable is a machine property; do not assume it
#   across machines.
ALPHA_PTS = [(2, 0.09), (8, 0.111), (16, 0.157), (128, 0.378),
             (256, 0.735), (512, 1.859)]

# beta [GB/s] (aligned convention: per-peer bytes are integer multiples of the real row
# width; unaligned sizes fall into implementation behavior that steps by powers of 2,
# and you end up measuring the alignment effect, not the link -- we stepped on this).
BETA_FLAT = 113.4    # 128-card full-fabric a2a: asymptotic bandwidth.
                     #   Refitting every size sweep we own (see sim/fit.py) put this at
                     #   117.8 / 111.0 / 113.4 / 130.3 GB/s across four independent
                     #   datasets -- a spread narrower than the machine's own drift.
                     #   The value used is the one from the corpus built to resolve size
                     #   dependence (19 distinct sizes over 4.5 decades), the same corpus
                     #   that fixes X_HALF_FLAT below; taking both from one well-conditioned
                     #   dataset keeps the pair self-consistent.

# Half-performance message size for the full-fabric level: the per-peer size at which
# the collective reaches half of BETA_FLAT. A single flat bandwidth over-credits small
# messages, and the bias was visible -- the flat-only model ran 8-27% fast against every
# sweep corpus, always the same sign.
#   Estimated at 54 KiB, bootstrap 90% interval [30, 87] KiB, from the 19-size sweep;
#   the coarse 5-6 size corpora cannot resolve this parameter and are not used for it.
#   **Estimated before looking at the gate**, then checked against it: Tier-1 median
#   error improves from 8.1% to 4.5% and the gate still passes. Note the crossover
#   position moves to the upper edge of the preregistered window, so an x_half much
#   above this interval would fail the gate -- see tests/test_sim.py.
#   Fitting caveat (sim/fit.py): alpha and x_half trade off, so this number is only
#   meaningful with alpha pinned to its direct measurements, which is how it was fitted.
X_HALF_FLAT = 54 * 1024
X_HALF_CI = (30 * 1024, 87 * 1024)
BETA_FAST = 122.4    # intra-node 8-card a2a: **physics-endorsed** --
                     #   measured 88.08 MB / 0.719 ms = 122.6,
                     #   physical aggregate egress (6x intra-node links 112.1 + 1x in-package direct 185)/7 = 122.4
                     #   the two differ by 0.2%; this tier's run-to-run spread is <0.3%,
                     #   the steadiest in the whole dataset
CROSS_NODE_RATIO = 0.974   # cross-node / intra-node (pairwise probes, 360 pairs, CV<0.4%) -- flat

SPLITS_SYNC_MS = 0.044     # host-side retrieval of splits for variable-length a2a, per call (measured 0.042-0.046)
CHAIN_US_PER_ROW = 2.15 * 1000.0 / 24576.0   # arrival chain, PyTorch op chain, per row
                                             # (measured 2.15 ms/call @ 24576 rows)


def saturating_beta(beta_inf: float, x_half: float,
                    lo: float = 1e3, hi: float = 1e9, n: int = 48) -> list:
    """Sample beta_inf * x/(x + x_half) into the (bytes, GB/s) table ClusterSpec reads.

    Logarithmic sampling keeps the interpolation error far below measurement noise
    over the whole range. x_half <= 0 gives back a flat table.
    """
    xs = [lo * (hi / lo) ** (i / (n - 1)) for i in range(n)]
    if x_half <= 0:
        return [(x, beta_inf) for x in xs]
    return [(x, beta_inf * x / (x + x_half)) for x in xs]


def flat_supernode() -> ClusterSpec:
    """The machine we actually measured: a bandwidth-flat supernode (cross-node / intra-node = 0.974)."""
    # The full-fabric and cross-node levels saturate with message size; the intra-node
    # level keeps a flat bandwidth because its value is physics-endorsed (link
    # aggregation, 0.2% from measurement) rather than fitted, and the sweep corpora
    # never isolate that level.
    beta_flat = saturating_beta(BETA_FLAT, X_HALF_FLAT)
    beta_fast = [(1e3, BETA_FAST), (1e9, BETA_FAST)]
    beta_slow = [(x, b * CROSS_NODE_RATIO) for x, b in beta_flat]
    return ClusterSpec(
        name="flat_supernode (calibrated from measurements)", R=8,
        fast=Level("node-internal", ALPHA_PTS, beta_fast),
        slow=Level("cross-node", ALPHA_PTS, beta_slow),
        flat=Level("full-fabric", ALPHA_PTS, beta_flat),
        splits_sync_ms=SPLITS_SYNC_MS,
        chain_us_per_row=CHAIN_US_PER_ROW,
    )


# Backward-compatible alias (internal code/tests use this name)
aug_flat = flat_supernode


def synthetic(ratio: float, name: str = "", R: int = 8,
              base_beta_gbps: float = 100.0, alpha_like_measured: bool = True,
              chain_us_per_row: float = CHAIN_US_PER_ROW,
              x_half: float = X_HALF_FLAT,
              **_compat) -> ClusterSpec:
    """Synthetic hierarchical cluster: fast-side beta = base, slow-side beta = base/ratio, alpha borrowed from the measured table.

    **Only usable after the validation gates pass** (guarded by sim/validate_micro.py /
    sim/validate.py); outputs are always labeled "simulated extrapolation". The borrowed
    alpha table carries the warning above: alpha's shape is a machine property.
    """
    ap = ALPHA_PTS if alpha_like_measured else [(2, 0.05), (512, 0.05)]
    # Same saturation shape as the calibrated machine: a synthetic cluster of the
    # same family has no reason to reach line rate on small messages either. Pass
    # x_half=0 for the older flat behaviour, and re-estimate it on any real machine.
    flat_b = saturating_beta(base_beta_gbps / ratio, x_half)
    fast_b = saturating_beta(base_beta_gbps, x_half)
    return ClusterSpec(
        name=name or ("synthetic(ratio=%.2f)" % ratio), R=R,
        fast=Level("fast", ap, fast_b),
        slow=Level("slow", ap, flat_b),
        flat=Level("flat", ap, flat_b),   # one-hop effective bandwidth is set by the slow edge
        splits_sync_ms=SPLITS_SYNC_MS, chain_us_per_row=chain_us_per_row,
    )
