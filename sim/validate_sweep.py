# -*- coding: utf-8 -*-
"""Tier-1b: does the model hold on measurement corpora it was never tuned against?

Tier-1 (validate_micro) checks the model against the crossover benchmark, which is
also what the calibration is anchored to -- internally consistent, but
self-referential. Tier-1b closes that loop with an independent family: plain a2a
size sweeps, on two machines, taken weeks apart by a different benchmark.

There are now three corpora, and they are not equally informative. A and B were both
available when the gate was written, so they test consistency. C was collected on
2026-08-26, months after the constants were frozen, at a world machine A had never
been scored at, and nothing was refitted for it -- so it tests prediction. It passes
with the thinnest margin of the three, 11.3% against a 12% gate, and that is the
honest reading: the model predicts an unseen world on a machine it knows, at roughly
the accuracy the gate was set to demand and no better.

## What is being tested, precisely

Machine A is the calibrated machine. Machine B is a second machine of the same
family, and **every constant is re-fitted for it; only the model form is shared**.
Passing on B is the real claim -- not that our constants transfer (they do not, and
alpha least of all), but that the shape of the cost model does, so someone else can
re-calibrate on their own hardware and expect it to work.

## Targets

Median wall clock per a2a call, keyed by (world, total send bytes per rank).
Medians rather than minima, for the reason validate_micro gives: a single run sits
below the machine's own drift. Machine A's corpus is a drift study whose per-point
spread reaches 3.06x, so its six targets carry much less information than their run
count suggests -- which is why the gate scores the two machines separately instead
of pooling 50 points into one average.

## Preregistered gate

  per machine   median relative error <= 12%
  per machine   at most 1 in 5 targets may exceed 35%
  per machine   median *signed* error within [-8%, +8%]  (no systematic bias)

Thresholds come from the corpora's own resolution rather than the current fit:
machine A's points scatter by up to 3.06x run to run, and the bias band is what a
model carrying no contention term can honestly promise against corpora that contain
contended runs. **This gate was written after the calibration, so it is a regression
guard, not evidence for it** -- Tier-1 stays the gate that had to be passed blind.

Excluded, and why: one corpus reaching worlds 256 and 512 sits ~5x below every other
in absolute bandwidth and fits worst by a factor of four, so it is quarantined (see
calibrate.py on the alpha entries above world 128). Non-power-of-2 worlds are also
out -- they collapse ~7x from a library artifact, documented in docs/03.
"""
from __future__ import annotations

# (world, total send bytes per rank, median ms, runs behind that median)
TARGETS_A = [
    (128, 8388608, 0.6114, 33),
    (128, 16777216, 0.5758, 33),
    (128, 33554432, 0.7072, 33),
    (128, 67108864, 1.0162, 33),
    (128, 100663296, 1.3705, 33),
    (128, 201326592, 2.0985, 33),
]

TARGETS_B = [
    (8, 1048576, 0.1882, 1),
    (8, 4194304, 0.1613, 1),
    (8, 16777216, 0.2186, 1),
    (8, 67108864, 0.6039, 1),
    (8, 268435456, 2.3391, 1),
    (8, 1073741824, 9.4678, 1),
    (16, 1048576, 0.3089, 1),
    (16, 4194304, 0.1637, 1),
    (16, 16777216, 0.2394, 1),
    (16, 67108864, 0.6912, 1),
    (16, 268435456, 2.6408, 1),
    (16, 1073741824, 10.4827, 1),
    (32, 1048576, 0.3893, 1),
    (32, 4194304, 0.2269, 1),
    (32, 16777216, 0.5865, 1),
    (32, 67108864, 1.1790, 1),
    (32, 268435456, 2.6190, 1),
    (32, 1073741824, 10.1870, 1),
    (64, 1048576, 0.5376, 1),
    (64, 4194304, 0.2599, 1),
    (64, 16777216, 0.3601, 1),
    (64, 67108864, 0.8765, 1),
    (64, 268435456, 2.5861, 1),
    (64, 1073741824, 10.0869, 1),
    (128, 31457, 0.5658, 1),
    (128, 62914, 0.4691, 1),
    (128, 131072, 0.3933, 1),
    (128, 262144, 0.4054, 1),
    (128, 524288, 0.5148, 1),
    (128, 1048576, 0.4402, 2),
    (128, 2097152, 0.5585, 1),
    (128, 4194304, 0.7636, 2),
    (128, 8388608, 0.5637, 4),
    (128, 16777216, 0.6208, 5),
    (128, 25165824, 0.5971, 1),
    (128, 26214400, 0.6305, 3),
    (128, 33554432, 0.6958, 4),
    (128, 50331648, 0.8315, 1),
    (128, 67108864, 1.0672, 5),
    (128, 100663296, 1.3586, 1),
    (128, 134217728, 1.7219, 1),
    (128, 268435456, 2.7575, 2),
    (128, 536870912, 5.0484, 1),
    (128, 1073741824, 9.9033, 2),
]

# Machine A again, world 16, collected 2026-08-26 -- long after every constant in
# calibrate.py was frozen, on a world machine A had never been validated at (its
# other corpus is world 128 only), with the same benchmark and the same aligned
# convention. This is the closest thing here to a blind test: the gate thresholds
# below were preregistered, the constants were fixed months earlier, and this corpus
# did not exist when either was written. It is scored on its own rather than pooled
# into TARGETS_A, so the original six points keep working as the regression guard
# they were written to be.
TARGETS_C = [
    (16, 32768, 0.1990, 1),
    (16, 131072, 0.1942, 1),
    (16, 524288, 0.1897, 1),
    (16, 1048576, 0.1928, 1),
    (16, 2097152, 0.1824, 1),
    (16, 4194304, 0.1866, 1),
    (16, 8388608, 0.1867, 1),
    (16, 16777216, 0.2412, 1),
    (16, 33554432, 0.3652, 1),
    (16, 67108864, 0.6551, 1),
    (16, 134217728, 1.2656, 1),
    (16, 268435456, 2.4990, 1),
]

GATE_MEDIAN = 0.12
GATE_OUTLIER_FRACTION = 0.20
GATE_OUTLIER_THRESHOLD = 0.35
GATE_BIAS = 0.08


def _predict(cluster, world: int, total_bytes: float) -> float:
    """One a2a call on the full-fabric level -- the same account as sim/core.py."""
    wire = total_bytes * (world - 1) / world
    beta = cluster.flat.beta_gbps(total_bytes / world)
    return cluster.flat.alpha_ms(world) + wire / (beta * 1e6)


def validate_sweep(cluster, targets, label: str = "", verbose: bool = True):
    rows, signed = [], []
    for world, nbytes, ms, runs in targets:
        pred = _predict(cluster, world, nbytes)
        rel = (pred - ms) / ms
        signed.append(rel)
        rows.append((world, nbytes, ms, pred, rel, runs))
    absolute = sorted(abs(r) for r in signed)
    median = absolute[len(absolute) // 2]
    outliers = [r for r in signed if abs(r) > GATE_OUTLIER_THRESHOLD]
    frac = len(outliers) / len(signed)
    ordered = sorted(signed)
    bias = ordered[len(ordered) // 2]
    ok = (median <= GATE_MEDIAN and frac <= GATE_OUTLIER_FRACTION
          and abs(bias) <= GATE_BIAS)
    if verbose:
        print("Tier-1b %s: %d targets" % (label, len(targets)))
        print("  median |error| %5.1f%% (gate %.0f%%)   over %.0f%%: %d/%d = %.0f%% "
              "(gate %.0f%%)   bias %+5.1f%% (gate +-%.0f%%)"
              % (median * 100, GATE_MEDIAN * 100, GATE_OUTLIER_THRESHOLD * 100,
                 len(outliers), len(signed), frac * 100,
                 GATE_OUTLIER_FRACTION * 100, bias * 100, GATE_BIAS * 100))
        print("  **%s**" % ("PASS" if ok else "FAIL"))
    return ok, {"median": median, "outlier_fraction": frac, "bias": bias,
                "rows": rows}


def main() -> None:
    from .calibrate import flat_supernode, second_machine
    ok_a, _ = validate_sweep(flat_supernode(), TARGETS_A, "machine A (calibrated)")
    ok_b, _ = validate_sweep(second_machine(), TARGETS_B, "machine B (re-fitted)")
    ok_c, _ = validate_sweep(flat_supernode(), TARGETS_C,
                             "machine A, world 16, collected after the freeze")
    print()
    print("Tier-1b overall: %s" % ("PASS" if (ok_a and ok_b and ok_c) else "FAIL"))


if __name__ == "__main__":
    main()
