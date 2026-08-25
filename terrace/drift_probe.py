"""**On-device** probe for the residual T-A2A seam drift (active only with TERRACE_DRIFT_PROBE=1).

Why this file exists (evening of 2026-08-20, blocker on the verdict-testbed eq gate):
the on arm of the 6-arg overlap seam drifts step by step against the vendor overlap on
the alignment testbed (early window clean at 2e-5, |delta|=1.31e-4 by iter80), while the
legacy seam's on arm — same testbed, same method — stays <=2e-5 throughout. Already ruled
out: gate-plane dtype (one internal commit), topk tie ordering (1ec68687),
measurement-scale mismatch. On CPU, the four gradient paths across the three revisions
HEAD / pre-C1 / f73c041e are **bit-for-bit equal** => if an eliminable implementation
difference still exists, it arises only on the device. On the device there is no
ready-made control: every bit-level contract (the two seams produce identical values,
quota branch == generic branch, plan fast-path sel == generic-branch sel) has only ever
been proven in CPU unit tests. This module fills that gap: it **executes** those
contracts on the NPU, and gives both arms a fingerprint at shared probe points that can
be diffed offline.

Three probe families, all inside `torch.no_grad()`, none of them mutates any tensor:

  1. `note` / `note_int` — point fingerprints. Floats get (numel, sum|x|, sum x^2, max|x|),
     which is **row-permutation invariant**: at the same point the two arms order rows
     differently by construction (T-A2A dedups and reorders by node, the vendor reorders
     by expert), so element-wise comparison is off the table, while permutation
     invariants are sensitive to "did the set of values change" — sum|x| has no
     cancellation, an fp32 tree reduction's own error is ~1e-10 relative, and a
     bf16-ulp-level value difference on one element in a thousand already pushes past
     1e-6 relative. Ints get an exact int64 checksum + min/max.
  2. `check_equal` — are two tensors **bit-for-bit** equal on the device (device-side
     execution of a CPU contract).
  3. `check_reduction` — reduction-point recheck: (1) does re-running the same index_add
     reproduce the result bit-for-bit (whether NPU index_add has a stable atomic/blocking
     order is documented nowhere); (2) deviation from a **deterministic** fp32 reference
     (sort by index, then a fixed-shape reshape-sum) — this measures "the accumulator
     width and ordering of the device reduction", which is exactly the one structural
     difference between the on and off arms.

Discipline:
  - switch unset = **zero behavior**: each point costs one extra read of a cached bool;
    no tensor is built, not one line is printed.
  - the probe itself never changes any numerics: everything is detach + no_grad, read-only.
  - a probe error is never silent: print one WARN line and **self-disable** (a dead
    diagnostic must be visible in the log, but must not take the training run down with
    it); when a contract probe (check_equal / plan.sel) hits a mismatch,
    TERRACE_DRIFT_PROBE_STRICT=1 raises immediately.

Offline diff: `python -m terrace.drift_probe compare on.log off.log` pairs points by
pt/step/layer and reports the **first** out-of-tolerance point in log order — "the first
layer / first step where the drift is injected" is read off directly from that.
"""
from __future__ import annotations

import os
import re
import sys

import torch

ENV_SWITCH = "TERRACE_DRIFT_PROBE"
ENV_EVERY = "TERRACE_DRIFT_PROBE_EVERY"      # emit one round every N dispatches (default 1)
ENV_MAXCALL = "TERRACE_DRIFT_PROBE_MAXCALL"  # self-stop after N dispatches (default 8000)
ENV_RANKS = "TERRACE_DRIFT_PROBE_RANKS"      # comma-separated rank whitelist (default "0")
ENV_STRICT = "TERRACE_DRIFT_PROBE_STRICT"    # raise when a contract probe hits a mismatch
ENV_CHUNK = "TERRACE_DRIFT_PROBE_CHUNK"      # fingerprint chunk size, elements (default 4M, caps peak device memory)

TAG = "[terrace-drift]"

# Process-level state. `on` is tri-state: None = undecided, True/False = decided
# (the hot path reads it exactly once).
_S: dict = {"on": None, "every": 1, "maxcall": 8000, "chunk": 1 << 22,
            "strict": False, "rank": 0, "ranks_ok": True,
            "step": 0, "layer": -1, "call": 0, "dispatch": 0, "dead": False}


def reset():
    """Re-read the environment and reset counters. For unit tests; a training
    process never flips the switch mid-run."""
    _S.update({"on": None, "step": 0, "layer": -1, "call": 0, "dispatch": 0,
               "dead": False})


def _decide():
    _S["on"] = os.environ.get(ENV_SWITCH) == "1"
    if not _S["on"]:
        return
    _S["every"] = max(1, int(os.environ.get(ENV_EVERY, "1") or "1"))
    _S["maxcall"] = int(os.environ.get(ENV_MAXCALL, "8000") or "8000")
    _S["chunk"] = max(1, int(os.environ.get(ENV_CHUNK, str(1 << 22)) or (1 << 22)))
    _S["strict"] = os.environ.get(ENV_STRICT) == "1"
    _S["rank"] = int(os.environ.get("RANK", "0") or "0")
    allow = os.environ.get(ENV_RANKS, "0") or "0"
    _S["ranks_ok"] = (allow.strip() == "*"
                      or _S["rank"] in {int(x) for x in allow.split(",") if x.strip()})


def enabled() -> bool:
    """The hot path's only switch read. Decide once if undecided; after that it
    is a single dict lookup."""
    if _S["on"] is None:
        _decide()
    return bool(_S["on"]) and not _S["dead"]


def _live() -> bool:
    """Should this call actually emit (switch + rank whitelist + period + cap)."""
    if not enabled() or not _S["ranks_ok"]:
        return False
    if 0 <= _S["maxcall"] <= _S["dispatch"]:
        return False
    # After tick, dispatch is 1-based; the -1 makes the **first** call always emit
    # (however coarse the period, the first frame is never dropped — and the first frame
    # is exactly where "the first step of drift injection" is most likely to land).
    # With no wrapper attached, dispatch=0, which also emits.
    return (max(_S["dispatch"] - 1, 0) % _S["every"]) == 0


def set_where(step=None, layer=None, call=None):
    """Point coordinates (the pairing key for the offline diff). The wrapper
    sets them once before each call."""
    if step is not None:
        _S["step"] = int(step)
    if layer is not None:
        _S["layer"] = int(layer)
    if call is not None:
        _S["call"] = int(call)


def tick_dispatch():
    """Dispatch call counter (the unit the period/cap are measured in). The
    wrapper calls this at the dispatch-half entry."""
    _S["dispatch"] += 1
    return _S["dispatch"]


def where() -> str:
    return "step=%d layer=%d call=%d rank=%d" % (
        _S["step"], _S["layer"], _S["call"], _S["rank"])


def _emit(line: str):
    print("%s %s" % (TAG, line), flush=True)


def _die(exc: Exception, what: str):
    """The probe itself failed: print one WARN line and self-disable. The
    diagnostic may die; it must not take the training run down with it."""
    _S["dead"] = True
    _emit("WARN probe failed at %s (%r) — self-disabled, no further points; "
          "points missing after this line mean the probe died, not that the "
          "data matched" % (what, exc))


class DriftProbeMismatch(RuntimeError):
    """A contract probe hit a mismatch (TERRACE_DRIFT_PROBE_STRICT=1).

    Its own class, so that the `except Exception` below (the probe's
    self-protection) can **let it through** — swallowing STRICT's hard stop as
    "the probe itself failed" is exactly the failure mode the probe must never
    have.
    """


def _fail(point: str, msg: str):
    _emit("MISMATCH pt=%s %s %s" % (point, where(), msg))
    if _S["strict"]:
        raise DriftProbeMismatch("%s contract probe mismatch pt=%s %s" % (TAG, point, msg))


# ---------------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------------

def digest(x: torch.Tensor, chunk: int | None = None):
    """Permutation-invariant three-stat fingerprint. Returns (numel, sum|x|, sum x^2, max|x|); float input.

    Chunked accumulation: peak temporary device memory is capped at `chunk` elements
    (upcasting a [pairs, 2048] block to fp32 in one shot is ~0.5 GB on the verdict
    testbed; chunked it is constant), and the two arms share the same shape => same
    chunk boundaries => same reduction tree, so the fingerprints differ only in the
    **values** being summed, never in **how** they are summed.
    """
    xd = x.detach().reshape(-1)
    n = int(xd.numel())
    if n == 0:
        return (0, 0.0, 0.0, 0.0)
    step = int(chunk or _S["chunk"])
    dev = xd.device
    s_abs = torch.zeros((), dtype=torch.float32, device=dev)
    s_sq = torch.zeros((), dtype=torch.float32, device=dev)
    amax = torch.zeros((), dtype=torch.float32, device=dev)
    for i in range(0, n, step):
        c = xd[i:i + step].to(torch.float32)
        a = c.abs()
        s_abs = s_abs + a.sum()
        s_sq = s_sq + (c * c).sum()
        amax = torch.maximum(amax, a.max())
    vals = torch.stack([s_abs, s_sq, amax]).tolist()      # the single sync in this function
    return (n, vals[0], vals[1], vals[2])


def digest_int(x: torch.Tensor):
    """Exact fingerprint for integer/bool planes. Returns (numel, sum x, min, max), exact in int64."""
    xd = x.detach().reshape(-1)
    n = int(xd.numel())
    if n == 0:
        return (0, 0, 0, 0)
    xi = xd.to(torch.int64)
    vals = torch.stack([xi.sum(), xi.min(), xi.max()]).tolist()
    return (n, int(vals[0]), int(vals[1]), int(vals[2]))


def _line(point: str, dt, fields: str) -> str:
    return "pt=%s %s dt=%s %s" % (point, where(), str(dt).replace("torch.", ""),
                                  fields)


def note(point: str, x, **_ignored):
    """Float point fingerprint. Switch off = zero behavior."""
    if not _live() or x is None:
        return
    try:
        with torch.no_grad():
            if not torch.is_tensor(x):
                return
            if not x.is_floating_point():
                return note_int(point, x)
            n, sa, sq, am = digest(x)
            _emit(_line(point, x.dtype,
                        "n=%d sabs=%.17g ssq=%.17g amax=%.17g" % (n, sa, sq, am)))
    except DriftProbeMismatch:
        raise
    except Exception as e:                                    # noqa: BLE001
        _die(e, "note(%s)" % point)


def note_layout(point: str, x, **_ignored):
    """**Layout** fingerprint: shape / dtype / device / contiguity / stride. Switch off = zero behavior.

    Added 2026-08-23. `note` records only the dtype and a numeric digest
    (numel / abs-sum / square-sum / max), nothing about layout — yet the open item in
    the internal measurement records is exactly layout:

      `seam_gap` (between the two seams, = the expert GEMM forward) held steady at
      −0.478 / −0.483 ms/call across two rounds, i.e. −73 ms/step, a sizable share of
      the base-tier win, and **has no explanation**. The host-sync hypothesis is ruled
      out (that `.cpu()` line belongs to a dispatcher we do not use). The remaining
      candidate: both arms feed the grouped GEMM the same set of (token, expert) pairs,
      but the **permutation order and memory contiguity may differ**.

    All of this is metadata — **no device-memory reads, no syncs, no kernel launches** —
    so it is even cheaper than `note` and can stay on long-term on both arms.
    """
    if not _live() or x is None:
        return
    try:
        if not torch.is_tensor(x):
            return
        _emit(_line(point + "#layout", x.dtype,
                    "shape=%s dev=%s contig=%d stride=%s" % (
                        "x".join(str(d) for d in x.shape),
                        x.device.type, int(x.is_contiguous()),
                        ",".join(str(v) for v in x.stride()))))
    except Exception as e:                                    # noqa: BLE001
        _die(e, "note_layout(%s)" % point)


def note_int(point: str, x, **_ignored):
    """Integer/bool point fingerprint (exact). Switch off = zero behavior."""
    if not _live() or x is None:
        return
    try:
        with torch.no_grad():
            if not torch.is_tensor(x):
                return
            n, s, lo, hi = digest_int(x)
            _emit(_line(point, x.dtype,
                        "n=%d isum=%d imin=%d imax=%d" % (n, s, lo, hi)))
    except DriftProbeMismatch:
        raise
    except Exception as e:                                    # noqa: BLE001
        _die(e, "note_int(%s)" % point)


# ---------------------------------------------------------------------------------
# Contract probes
# ---------------------------------------------------------------------------------

def check_equal(point: str, a, b, note_a: str = ""):
    """Are two tensors **bit-for-bit** equal on the device. A contract proven in CPU
    unit tests gets actually executed here, once."""
    if not _live() or a is None or b is None:
        return
    try:
        with torch.no_grad():
            ad, bd = a.detach(), b.detach()
            if ad.shape != bd.shape or ad.dtype != bd.dtype:
                _fail(point, "shape/dtype differ %s%s vs %s%s %s"
                      % (tuple(ad.shape), ad.dtype, tuple(bd.shape), bd.dtype, note_a))
                return
            if torch.equal(ad, bd):
                _emit(_line(point + ".eq", ad.dtype, "bitequal=1 n=%d" % ad.numel()))
                return
            d = (ad.to(torch.float32) - bd.to(torch.float32)).abs()
            _fail(point, "bitequal=0 n=%d ndiff=%d maxabs=%.17g %s"
                  % (ad.numel(), int((d > 0).sum()), float(d.max()), note_a))
    except DriftProbeMismatch:
        raise
    except Exception as e:                                    # noqa: BLE001
        _die(e, "check_equal(%s)" % point)


def check_reduction(point: str, out, src, index, n_out: int, per=None):
    """Reduction-point recheck: reproducibility + deviation from a deterministic fp32 reference.

    `out` must be the result of `zeros(n_out, W).index_add_(0, index, src)`.

      det=1/0      whether re-running the same index_add reproduces the result
                   **bit-for-bit**. Nothing documents the accumulation order of NPU
                   index_add on duplicate indices (on CUDA it is atomic, not
                   reproducible); det=0 means the on arm's own reduction order varies
                   from step to step — that is not an on/off difference, it is
                   uncontrolled variation inside the on arm, and the eq gate's noise
                   floor must include it.
      maxabs/maxrel deviation from the deterministic reference (group by index, then a
                   fixed-shape reshape-sum, fp32 accumulation). Measures the accumulator
                   width of the device reduction: a bf16 accumulator gives ~2^-9
                   relative, an fp32 accumulator ~2^-24 relative — 5 orders of magnitude
                   apart, tellable at a glance. Only computable when every output row
                   has exactly `per` contributions (holds by construction on the quota
                   fast path).
    """
    if not _live() or out is None:
        return
    try:
        with torch.no_grad():
            o, s, idx = out.detach(), src.detach(), index.detach()
            again = torch.zeros_like(o).index_add_(0, idx, s)
            det = 1 if torch.equal(again, o) else 0
            extra = ""
            if per and idx.numel() == n_out * int(per):
                p = torch.argsort(idx, stable=True)
                ref = s[p].to(torch.float32).reshape(
                    n_out, int(per), -1).sum(1)
                d = (o.to(torch.float32) - ref).abs()
                scale = ref.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
                extra = " per=%d maxabs=%.17g maxrel=%.17g" % (
                    int(per), float(d.max()), float(d.max() / scale))
            _emit(_line(point + ".chk", o.dtype,
                        "det=%d n_out=%d npair=%d%s"
                        % (det, n_out, int(idx.numel()), extra)))
            if det == 0:
                _fail(point + ".chk", "index_add is **not reproducible** on this "
                                      "device (same input, two runs, bit-for-bit "
                                      "different results)")
    except DriftProbeMismatch:
        raise
    except Exception as e:                                    # noqa: BLE001
        _die(e, "check_reduction(%s)" % point)


# ---------------------------------------------------------------------------------
# Offline diff
# ---------------------------------------------------------------------------------

_LINE_RE = re.compile(
    r"\[terrace-drift\] pt=(?P<pt>\S+) step=(?P<step>\d+) layer=(?P<layer>-?\d+) "
    r"call=(?P<call>\d+) rank=(?P<rank>\d+) dt=(?P<dt>\S+) (?P<rest>.*)$")
_KV_RE = re.compile(r"(\w+)=(-?[0-9.eE+-]+)")

# Diff criterion: upper bound on relative deviation. An fp32 tree reduction by itself
# stays below 1e-9; allow three orders of magnitude of headroom.
CMP_RTOL = 1e-6


def parse_line(line: str):
    """One line -> (key, dict) or None. key = (pt, step, layer, call)."""
    m = _LINE_RE.search(line)
    if m is None:
        return None
    d = {k: float(v) for k, v in _KV_RE.findall(m.group("rest"))}
    d["dt"] = m.group("dt")
    return ((m.group("pt"), int(m.group("step")), int(m.group("layer")),
             int(m.group("call"))), d)


def parse_log(path):
    """Log -> (dict[key] = fields, [key...] in order of appearance)."""
    out, order = {}, []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            got = parse_line(line)
            if got is None:
                continue
            key, fields = got
            if key not in out:
                order.append(key)
            out[key] = fields
    return out, order


def _reldiff(a, b):
    scale = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / scale


def compare(path_a, path_b, rtol=CMP_RTOL, limit=20):
    """Diff the two arms' logs point by point; report the first out-of-tolerance
    point in A's order of appearance. Returns an exit code."""
    A, order = parse_log(path_a)
    B, _ = parse_log(path_b)
    common = [k for k in order if k in B]
    if not common:
        print("NOCOMMON: the two logs share no points (pt/step/layer/call never coincide)")
        return 2
    bad = []
    for k in common:
        fa, fb = A[k], B[k]
        if fa.get("dt") != fb.get("dt"):
            bad.append((k, "dt", fa.get("dt"), fb.get("dt"), float("inf")))
            continue
        for f in ("n", "sabs", "ssq", "amax", "isum", "imin", "imax", "det",
                  "bitequal", "maxabs"):
            if f in fa and f in fb:
                r = _reldiff(fa[f], fb[f])
                if r > rtol:
                    bad.append((k, f, fa[f], fb[f], r))
    print("%d common points (%d only in A, %d only in B)"
          % (len(common), len(order) - len(common), len(B) - len(common)))
    if not bad:
        print("CLEAN: all common points agree within rtol=%g — the drift is not at any instrumented point" % rtol)
        return 0
    k, f, va, vb, r = bad[0]
    print("FIRST-DIVERGENCE: pt=%s step=%d layer=%d call=%d field %s "
          "A=%.17g B=%.17g rel=%.3e" % (k[0], k[1], k[2], k[3], f,
                                        va if isinstance(va, float) else float("nan"),
                                        vb if isinstance(vb, float) else float("nan"),
                                        r))
    print("%d points out of tolerance, first %d:" % (len(bad), min(limit, len(bad))))
    for k, f, va, vb, r in bad[:limit]:
        print("  pt=%-22s step=%-4d layer=%-3d call=%-4d %-8s rel=%.3e"
              % (k[0], k[1], k[2], k[3], f, r))
    return 1


def main(argv):
    if len(argv) >= 3 and argv[0] == "compare":
        rtol = float(argv[3]) if len(argv) > 3 else CMP_RTOL
        return compare(argv[1], argv[2], rtol)
    print("usage: python -m terrace.drift_probe compare <A.log> <B.log> [rtol]")
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
