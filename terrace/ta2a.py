"""T-A2A (OM-1): hierarchy-aligned all-to-all for the expert dispatch.

Drop-in alternative to `ep_dist.ep_moe_forward` that must produce the SAME routed
output (see tests/test_ta2a.py) while moving fewer, larger messages over the
contended HCCS fabric.

Why the levers are free -- and why a kernel is required (internal measurement records):
The baseline already builds its send buffer with an indexed gather sorted by
destination rank (`ep_dist.py`: `payload = x_local[src[perm]]`). T-A2A changes only
WHAT that permutation is, so both levers cost nothing extra:

  1. dedup      A token whose top-k experts include several experts on the SAME
                destination node is sent to that node ONCE, not once per expert; the
                node fans it out locally. Measured on real reference-operating-point
                traces (internal measurement records) T-Route's fan-out is exactly
                M groups, so this is a hard k/M byte reduction -- 2.00x at the
                reference operating point (k=8, M=4), 3x at the flagship (k=6, M=2).
  2. aggregation Rows are ordered by destination NODE, and each rank sends a node's
                whole payload to ONE designated peer on that node instead of to all
                `rpn` of its ranks. Same bytes cross the fabric, in `rpn`x larger
                messages -- which is worth a lot, because the measured beta(size) curve
                is steep at the dispatch operating point (world=128: 41.4 GB/s at 25 MB vs
                ~102 GB/s at 200 MB, lever 2.46x -- but world=128 was NOT directly
                measured: beta(25) is the 2026-07-22 error-bar run, beta(200) is
                extrapolated along the saturated curve; see the internal measurement
                records for the measured testbeds: w64 57.2 -> 102.0,
                w32 65.0-68.4 -> 101.7-102.9).

A composition of stock collectives cannot do this: it has to physically move the whole
payload two extra times to re-lay-out the send buffers, which measured 0.58x at the
operating point (internal measurement records). Here the only extra motion is
one intra-node forward of the RECEIVED slice (1/n_nodes of the traffic) over the fast
tier.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .layer import grouped_mm
from .ep_dist import _a2a
from . import drift_probe as _dp


# (T, k, n_nodes, M) combinations whose exact fan-out has already been checked against
# the device. Keyed by shape rather than global, so a change of micro-batch or of M
# re-verifies instead of inheriting an earlier shape's clean bill of health.
_VERIFIED: set = set()

# Per-key dispatch counter for the periodic re-verification below.
_VERIFY_COUNT: dict = {}

# Index tensors that depend only on the shape, not on the routing. Training calls the plan
# once per layer per micro-batch with the same shape every time, so rebuilding these is a
# per-call allocation and kernel launch for a value that never changes. Keyed by
# (length, device, dtype); the plan's shapes are fixed for a run, so this cannot grow.
_ARANGE_CACHE: dict = {}


def _arange(n: int, device, dtype=torch.int64) -> torch.Tensor:
    key = (n, str(device), dtype)
    t = _ARANGE_CACHE.get(key)
    if t is None:
        t = torch.arange(n, device=device, dtype=dtype)
        _ARANGE_CACHE[key] = t
    return t


def plan_ta2a(expert_idx: torch.Tensor, world: int, n_experts: int, rpn: int = 8,
              groups_m: int | None = None):
    """Build the dedup'd, node-ordered dispatch plan. Pure indexing -- no collectives.

    Pass `groups_m` = M only when the routing satisfies BOTH halves of the T-Route
    invariant: every token touches exactly M nodes AND each touched node holds exactly
    k/M of that token's experts (equal quota -- t_route mode 'full' or 'quota_only').
    Exact fan-out alone is NOT enough: group-limit without quota (mode 'group_limited')
    can produce a 3/1 split whose fan-out is still exactly M, and the structured fast
    path then corrupts the dispatch mask silently. Both halves are verified on the first
    call and every 256th thereafter. The row count is then T*M, known before the data is
    looked at, and the plan needs no host synchronisation at all. Leave it None for
    routings whose fan-out or quota is data-dependent; the count is then read back from
    the device, which costs a sync.

    The sync is worth 0.12-0.18 ms at the reference operating point -- real, but not the reason
    the plan was ever slow. That was `torch.argsort(stable=True)`, since replaced; see the
    compaction comment below and the internal measurement records. Under `groups_m`
    the compaction is now sort-free entirely: the known row count lets one searchsorted
    over the cumsum replace the argsort, bit-for-bit.

    Returns
      u_src   [S] token index of each row to send (S = sum over tokens of |dest nodes|)
      u_node  [S] destination node of each row, sorted ascending
      counts  [n_nodes] rows per destination node
      inverse [T*k] for each (token, expert) slot, the row in the send buffer that
              carries it -- used to expand back after arrival and to scatter results.
    """
    epr = n_experts // world
    n_nodes = world // rpn
    T, k = expert_idx.shape
    # float32 exactness bound (2026-08-02 review): the plan's cumsum and sort keys hold
    # values up to n_nodes*T in float32, exact only below 2^24. Every configuration to date
    # sits near 2^21, which is why nothing has corrupted -- but nothing guarded the edge,
    # while _stable_argsort_small asserts the same class of bound for its buckets. Refuse
    # loudly instead of silently mis-sorting.
    if n_nodes * T >= (1 << 24):
        raise ValueError(
            f"plan_ta2a float32 key range exceeded: n_nodes*T = {n_nodes * T} >= 2^24; "
            f"shard tokens below {(1 << 24) // max(n_nodes, 1)} per rank or extend the "
            f"plan to int64 keys first")
    dev = expert_idx.device
    # One divide, not two: e // epr // rpn == e // (epr*rpn) for non-negative e, and every
    # op here costs ~0.05 ms of fixed dispatch overhead regardless of how few elements it
    # touches, which is what the plan is actually made of (measured: the per-stage times
    # barely move between T=512 and T=8192).
    dest_node = torch.div(expert_idx, epr * rpn, rounding_mode="floor")   # [T, k]

    # Dedup without torch.unique. Profiling the previous version on NPU
    # (internal benchmark script, not shipped with the repo) showed unique(+inverse) at 0.815 ms and the
    # arange+repeat_interleave that fed it at 0.601 ms -- together 55% of the 2.59 ms
    # plan, i.e. 3x what the whole fabric hop costs. Both are avoidable: a token's
    # (token, node) pairs are just the distinct entries of a [T, n_nodes] occupancy
    # table, which is a fixed-shape scatter, and the source index is a pure function of
    # (T, k) so it can be cached. Sorting by node then falls out of flattening the table
    # in node-major order -- no argsort of the rows either.
    occ = torch.zeros(T, n_nodes, dtype=torch.bool, device=dev)
    occ.scatter_(1, dest_node, True)                                   # [T, n_nodes]

    # Node-major flatten => rows are already node-contiguous, no sort needed.
    node_first = occ.t().reshape(-1)                                   # [n_nodes * T]
    # float32 cumsum, not integer. Measured on this device at n=65536
    # (internal benchmark script, not shipped with the repo): cumsum int64 0.230 ms, int32 0.281 ms,
    # float32 0.101 ms -- integer scans have no fast path here, and an earlier "obvious"
    # int64 -> int32 change actually made this slower. float32 represents every integer up
    # to 2^24 exactly and the running count cannot exceed n_nodes*T (65536 at our largest
    # shape), so the values are exact, not approximate. The inclusive count is kept as its
    # own tensor because the groups_m fast path below reads the compaction straight off it.
    row_count = torch.cumsum(node_first.to(torch.float32), 0)          # [n_nodes*T], inclusive
    row_pos = (row_count - 1).to(torch.int64)

    # The number of rows IS known when the caller says so: under the group constraint every
    # token touches exactly M nodes, so n_rows = T*M. This used to be a module-level
    # `_EXACT_ROWS` that no caller ever assigned, so the sync-free path it documented never
    # once ran and every dispatch paid the `.item()` -- the same shape of silent failure as
    # the vendor arg drop and the CRLF deployment. A parameter cannot be left unset by
    # accident: it is visible at every call site, and the ones that know M now pass it.
    if groups_m is None:
        n_rows = int(occ.sum().item())      # host sync; unavoidable when fan-out varies
    else:
        n_rows = T * groups_m
        # A groups_m that is too SMALL silently drops real rows; one that is too LARGE is
        # NOT harmless either -- under the old sliced-argsort compaction it pulled UNSET
        # positions in as ghost rows (measured: gm=6 admitted 16 ghost rows, gm=8 admitted
        # 32; the searchsorted compaction below turns the same mistake into out-of-range
        # row indices instead). Either way the plan is wrong -- and the ghost-row variant
        # was numerically plausible, the worst failure mode this codebase has. Both
        # directions are caught by the fan-out check below.
        # Verify against the device once per (shape, M), then never again: the check costs
        # exactly the sync we are removing, so paying it every step would defeat the point,
        # while paying it once catches a misconfigured caller on its first dispatch.
        key = (T, k, n_nodes, groups_m)
        # Re-verify every 256th call as well as the first (2026-08-02 review): the original
        # once-per-shape check proves the FIRST batch's fan-out and then inherits trust
        # forever -- a routing that drifts later behind the same shapes would silently
        # truncate rows. Every-256 keeps the amortised sync cost ~0.4% while bounding how
        # long a drift can stay hidden.
        cnt = _VERIFY_COUNT.get(key, 0)
        _VERIFY_COUNT[key] = cnt + 1
        if key not in _VERIFIED or cnt % 256 == 0:
            actual = int(occ.sum().item())
            if actual != n_rows:
                raise ValueError(
                    f"groups_m={groups_m} implies {n_rows} dispatch rows for T={T}, but the "
                    f"routing actually produces {actual} (fan-out "
                    f"{actual / T:.4f} != {groups_m}). Pass groups_m only for routings with "
                    f"an exact fan-out; leave it None otherwise.")
            # Fan-out == M is NOT the precondition the structured fast path needs; it needs
            # the strictly stronger "each touched node holds exactly k/M of the token's
            # experts". The gap between the two is a silent-corruption zone: with k=4, M=2
            # and a 3/1 split the fan-out is exactly M (this check passes), yet
            # build_expansion's [T, M, k/M] reshape mixes experts across the node boundary
            # -- the second node's row comes out ZERO and the first node's mask carries a
            # bit belonging to the OTHER node's expert (reproduced 2026-08-11: reference 16
            # mask bits, fast path 8, per-row 0b1101 vs 0b1010). Reachable with the repo's
            # own router: t_route(mode="group_limited") has exact fan-out but no quota.
            # Equal-quota (mode="full") is what guarantees this invariant -- so verify the
            # invariant itself, on the same first + every-256th cadence.
            if k % groups_m != 0:
                raise ValueError(
                    f"groups_m={groups_m} does not divide k={k}: the structured path's "
                    f"per-node quota k/M is not an integer, and the arrival-side demand "
                    f"extraction would floor it and drop experts. Leave groups_m None.")
            per = torch.zeros(T, n_nodes, dtype=torch.float32, device=dev)
            per.scatter_add_(1, dest_node, torch.ones_like(dest_node, dtype=torch.float32))
            bad = int((per[per > 0] != float(k // groups_m)).sum().item())
            if bad:
                raise ValueError(
                    f"groups_m={groups_m} requires every touched node to hold exactly "
                    f"k/M={k // groups_m} of a token's experts, but {bad} (token, node) "
                    f"pairs violate the quota (e.g. a 3/1 split with fan-out still == M). "
                    f"This routing is group-limited but not equal-quota; pass groups_m "
                    f"only for equal-quota routings (t_route mode='full'/'quota_only'), "
                    f"leave it None otherwise.")
            _VERIFIED.add(key)
    # Compact the set bits: both branches below compute the SAME tensor -- `sel`, the
    # ascending positions of the set bits in `node_first` -- and must stay bit-for-bit
    # equal (tests/test_ta2a.py::test_fastpath_sortfree_construction_is_bitwise_equal).
    # Row order fixes the downstream reduction order, so any divergence here changes
    # numerics, not just speed.
    if groups_m is None:
        # General branch: sort on FLOAT keys. The stable argsort this construction
        # replaced was 2.483 ms of a 3.034 ms plan at T=4096 on 64 die -- 82% -- for
        # 32768 uint8 elements, i.e. ~76 ns per element (internal benchmark script,
        # not shipped with the repo; internal measurement records):
        # torch.argsort(stable=True) evidently
        # has no fast path here. The primitive bench that condemned the integer sort
        # (5.32 ms at int64, 4.44 at int32) puts float32 sort at 0.107 ms -- 50x -- and
        # that also beat an interim scatter-based compaction (scatter_ int64, 0.487 ms).
        # The original mistake was never "sorting is slow", it was sorting an integer
        # dtype.
        #
        # No stable sort needed either: the key is `position` for set bits and
        # `position + N` for unset ones, which is already a total order putting the set
        # bits first in index order. Stability was the other half of what made the old
        # argsort expensive.
        pos = _arange(n_nodes * T, dev, torch.float32)
        key = torch.where(node_first, pos, pos + float(n_nodes * T))
        sel = torch.argsort(key)[:n_rows]
    else:
        # Fast path: no sort at all. `row_count[p]` is the number of set bits at or
        # before position p, so the j-th compacted row sits at the FIRST p with
        # row_count[p] > j -- readable straight off the cumsum with one searchsorted
        # (binary search; deterministic, a pure function of row_count, no scatter whose
        # write order could vary). Legal only here because n_rows = T*M is known by
        # construction and verified above: the general branch cannot size the query
        # vector without the host sync.
        #
        # Launch/work accounting for the NPU tally (vs the general branch, N = n_nodes*T,
        # S = T*M): removes the where [N] and the argsort [N] (0.107 ms at N=65536 on the
        # prim bench), adds one searchsorted with S queries over N keys -- S*log2(N)
        # ~ 8192*16 comparisons at the reference operating point against the sort's N*log2(N) ~ 65536*16.
        # Net one fewer kernel launch and O(N log N) -> O(S log N) work; everything else
        # in the plan is unchanged. float32 stays exact: all values are integers bounded
        # by n_nodes*T < 2^24, guarded at the top of this function.
        #
        # Failure containment: if the routing drifts off the equal-quota invariant
        # between the periodic re-verifications above (<= 255 calls exposed), queries
        # past the true set-bit count return N -- an out-of-range row index that makes
        # the downstream gather fail loudly, where the argsort construction would have
        # fabricated in-range ghost rows instead.
        sel = torch.searchsorted(row_count, _arange(n_rows, dev, torch.float32),
                                 right=True)
        if _dp.enabled():
            # Drift probe: execute the contract above ("sel is bit-for-bit equal
            # across the two branches") ON THE DEVICE. So far it has only been proven
            # in a CPU unit test (test_fastpath_sortfree_construction_is_
            # bitwise_equal), the NPU's searchsorted quality is an explicitly listed
            # un-reconciled item (an internal commit), and it runs **only on the fast
            # path** -- fast path on/off is precisely one of the divides between
            # "runs that drift" and "runs that do not". One bit of sel moved = row
            # order moved = every downstream reduction order moved.
            _pos = _arange(n_nodes * T, dev, torch.float32)
            _dp.check_equal("plan.sel", sel, torch.argsort(
                torch.where(node_first, _pos, _pos + float(n_nodes * T)))[:n_rows])
    u_node = torch.div(sel, T, rounding_mode="floor")
    u_src = sel % T
    # float32 reduction: sum int64 is 0.475 ms at n=65536 against 0.013 ms for float32,
    # a 37x gap for a value bounded by T (8192), which float32 holds exactly.
    counts = occ.to(torch.float32).sum(0).to(torch.int64)              # rows per node

    # inverse: for each (token, expert) slot, the send row carrying it. The row for
    # (token t, node n) sits at row_pos[n * T + t], so this is a gather, not a remap.
    flat_pos = dest_node * T + _arange(T, dev, dest_node.dtype).unsqueeze(1)   # [T, k]
    inverse = row_pos[flat_pos.reshape(-1)].to(torch.int64)
    return u_src, u_node, counts, inverse


def dispatch_stats(expert_idx, world: int, n_experts: int, rpn: int = 8):
    """Byte/message accounting for one dispatch, baseline vs T-A2A. No collectives.

    Used by the kernel benchmark and by tests to assert the two levers actually fire:
      rows_baseline  T*k  (one row per (token, expert))
      rows_ta2a      one row per (token, destination node)  -> the dedup lever
      msgs_baseline  destinations per rank = world          -> small messages
      msgs_ta2a      destinations per rank = n_nodes        -> rpn x larger messages
    """
    T, k = expert_idx.shape
    n_nodes = world // rpn
    _, _, counts, _ = plan_ta2a(expert_idx, world, n_experts, rpn)
    rows_ta2a = int(counts.sum())
    return {
        "rows_baseline": T * k,
        "rows_ta2a": rows_ta2a,
        "dedup_x": (T * k) / max(rows_ta2a, 1),
        "msgs_baseline": world,
        "msgs_ta2a": n_nodes,
        "msg_growth_x": rpn,
    }
