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
#
#   **Independently re-measured 2026-08-26** by a call-count scan (sim/profile.py,
#   PER_CALL_DEEP_QUEUE_MS), months later and with a different benchmark: 129 us at
#   world 8 against the 111 here, and 134 at world 16 against 157. Both sit inside the
#   20% run-to-run drift this file documents, so the two entries are corroborated at
#   the level. The *step* between them is not: alpha rises 41% from world 8 to 16
#   where the scan rises 4%, and the same scan reproduces the world 2 and 4 entries
#   to within 3%, so the shape disagreement is the scan's clearest statement.
#   Neither number moves. What the scan adds is the convention: **alpha as tabulated
#   belongs to the deep-queue regime**, where the host stays far enough ahead that
#   submission hides under execution. Where the host has to observe each call the
#   same collective costs about twice as much, and that regime is not priced here.
ALPHA_PTS = [(2, 0.09), (8, 0.111), (16, 0.157), (128, 0.378),
             (256, 0.735), (512, 1.859)]

# beta [GB/s] (aligned convention: per-peer bytes are integer multiples of the real row
# width; unaligned sizes fall into implementation behavior that steps by powers of 2,
# and you end up measuring the alignment effect, not the link -- we stepped on this).
BETA_FLAT = 117.8    # 128-card full-fabric a2a: asymptotic bandwidth, fitted on **this
                     #   machine's own** sweeps with alpha pinned (sim/fit.py). Refits of
                     #   every corpus we own land at 117.8 here and 111.0-130.3 on a second
                     #   machine of the same family -- a spread narrower than either
                     #   machine's run-to-run drift, which is why beta is treated as
                     #   transferable in a way alpha is not.

# Half-performance message size for the full-fabric level: the per-peer size at which
# the collective reaches half of BETA_FLAT. A single flat bandwidth over-credits small
# messages, and the bias was visible -- the flat-only model ran 8-27% fast against every
# sweep corpus, always the same sign.
#   Estimated at 54 KiB, bootstrap 90% interval [30, 87] KiB. **This is a borrowed
#   shape**: it comes from a 19-size sweep taken on the *second* machine, because this
#   machine's own sweep corpus has only 6 sizes and cannot resolve the parameter. Same
#   discipline as the alpha curve -- borrow the shape, never the level; the level
#   (BETA_FLAT) is measured here. Sensitivity is reassuring rather than alarming: the
#   whole bootstrap interval passes Tier-1 (30 KiB -> 2.5% median, 87 KiB -> 5.0%).
#   **Estimated before looking at the gate**, then checked against it: Tier-1 median
#   error improves from 8.1% to 2.8% and the gate still passes. Note the crossover
#   position moves to the upper edge of the preregistered window, so an x_half much
#   above this interval would fail the gate -- see tests/test_sim.py.
#   Fitting caveat (sim/fit.py): alpha and x_half trade off, so this number is only
#   meaningful with alpha pinned to its direct measurements, which is how it was fitted.
X_HALF_FLAT = 54 * 1024
X_HALF_CI = (30 * 1024, 87 * 1024)
# How much room the *gates* leave x_half, which is a different quantity from the
# bootstrap interval above: sweep x_half, rerun Tier-1 and all three Tier-1b corpora,
# and see where every one of them still passes.
#
# The answer is one-sided. Machine B rules out anything above 77 KiB, tightening the
# bootstrap upper end slightly. Nothing rules out the low end -- 1 KiB passes every
# gate, because at a small x_half beta_eff is simply beta_inf across the payload
# range the corpora cover, and no corpus has the small-message resolution to object.
# So the gates corroborate the upper half of the bootstrap interval and say nothing
# about the lower half. x_half stays a weakly determined, borrowed constant, and
# sim/fit.py's identifiability caveat is the reason.
#
# Recorded because an earlier revision of this file claimed [46, 76] KiB, deriving a
# lower bound from a single-run version of the world-16 corpus. Repeating that sweep
# six times and taking medians dissolved the bound. **An admissibility interval
# computed from a noisy corpus comes out too tight**, and too tight is the flattering
# direction, so it reads as a stronger result than the data supports.
X_HALF_GATE_UPPER = 77 * 1024
BETA_FAST = 122.4    # intra-node 8-card a2a: **physics-endorsed** --
                     #   measured 88.08 MB / 0.719 ms = 122.6,
                     #   physical aggregate egress (6x intra-node links 112.1 + 1x in-package direct 185)/7 = 122.4
                     #   the two differ by 0.2%; this tier's run-to-run spread is <0.3%,
                     #   the steadiest in the whole dataset
CROSS_NODE_RATIO = 0.974   # cross-node / intra-node (pairwise probes, 360 pairs, CV<0.4%) -- flat

SPLITS_SYNC_MS = 0.044     # host-side retrieval of splits for variable-length a2a, per call (measured 0.042-0.046)
CHAIN_US_PER_ROW = 2.15 * 1000.0 / 24576.0   # arrival chain, PyTorch op chain, per row
                                             # (measured 2.15 ms/call @ 24576 rows)
# This is the constant everything hinges on -- it alone moves the breakeven ratio
# from 1.07 to 3.87 -- and until 2026-08-26 it rested on that single point. It was
# then swept properly: nine row counts from 1024 to 65536, on each of two nodes,
# thirty iterations each, op-level and single-card (bench/machine/dispatch_oplevel).
#
# **The linear form holds where it is used.** Above 8192 rows the per-row cost is
# 0.0865 to 0.1018 us, a 15% spread with no trend, and the two nodes agree to 2%.
# The reference geometry sits at 24576 rows, inside that range.
#
# **The level re-measures 13% higher**: 0.0987 us/row today against the 0.0875 here,
# and 2.50 ms at the original 24576-row point against 2.15. Inside the documented
# 20% drift, so the constant does not move -- but note which way it would move if it
# did. Adopting 0.0987 raises the breakeven from 3.87 to 4.23, against the method
# this repository is proposing. Recorded because the new evidence is the stronger of
# the two, eighteen measurements against one, and a future recalibration should
# probably take it.
CHAIN_US_PER_ROW_REMEASURED = 0.0987      # 2026-08-26, 9 row counts x 2 nodes
# **And a boundary the linear form does not have.** Below about 8192 rows the chain
# hits a fixed floor: 0.248 ms at 1024 rows where the linear form predicts 0.090.
# The model is therefore optimistic about the arrival chain on small geometries, and
# optimistic about the chain means optimistic about two-hop, since only two-hop pays
# it. At 3072 rows the gap is 23%, at 6144 rows 7%, and above 8192 it closes. Not
# applied in core.py, for the same reason the skew model is not: it would model an
# effect the Tier-1 targets do not contain. It is a stated limit on where the model
# may be used, and tests/test_sim.py pins it.
CHAIN_FLOOR_MS = 0.248                    # whole-chain wall clock at 1024 rows
CHAIN_LINEAR_MIN_ROWS = 8192              # below this the linear form under-prices


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


def flat_supernode(x_half: float = None) -> ClusterSpec:
    """The machine we actually measured: a bandwidth-flat supernode (cross-node / intra-node = 0.974).

    x_half is overridable so the gates can be swept over it (see X_HALF_ADMISSIBLE
    and tests/test_sim.py); leaving it None uses the shipped estimate.
    """
    # The full-fabric and cross-node levels saturate with message size; the intra-node
    # level keeps a flat bandwidth because its value is physics-endorsed (link
    # aggregation, 0.2% from measurement) rather than fitted, and the sweep corpora
    # never isolate that level.
    beta_flat = saturating_beta(BETA_FLAT,
                                X_HALF_FLAT if x_half is None else x_half)
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


# Constants for a second machine of the same family, fitted the same way from its own
# corpora with the shape pinned (sim/fit.py). They exist so Tier-1b can ask the question
# that matters for anyone else adopting this: does the model *form* survive a change of
# machine once its constants are re-fitted? Note how close the two machines land --
# beta within 5%, alpha(128) 0.394 against 0.378 -- which is a fact about these two
# machines, not a licence to skip re-calibration.
SECOND_ALPHA_PTS = [(8, 0.130), (16, 0.141), (32, 0.252), (64, 0.231), (128, 0.394)]
SECOND_BETA_FLAT = 111.9


def second_machine(x_half: float = None) -> ClusterSpec:
    """A second machine of the same family; every constant re-fitted, form unchanged.

    x_half is overridable for the same reason as in flat_supernode: it is the one
    shape both machines share, so a sweep over it has to move both or it is not
    sweeping the shared parameter at all.
    """
    beta = saturating_beta(SECOND_BETA_FLAT,
                           X_HALF_FLAT if x_half is None else x_half)
    return ClusterSpec(
        name="second_machine (re-fitted from its own corpora)", R=8,
        fast=Level("node-internal", SECOND_ALPHA_PTS, beta),
        slow=Level("cross-node", SECOND_ALPHA_PTS, beta),
        flat=Level("full-fabric", SECOND_ALPHA_PTS, beta),
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
