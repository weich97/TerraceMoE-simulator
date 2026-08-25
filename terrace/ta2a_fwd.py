"""T-A2A forward: the expansion plane and the routed-expert forward (OM-1).

`ta2a.plan_ta2a` builds the dedup'd, node-ordered send buffer. This module adds the
piece that makes dedup legal end-to-end: after a token is sent to a node ONCE, the
node still has to hand it to every local expert that wanted it. That demand travels
alongside each row in one of two wire formats, either a rounding error against the
4-16 KB [hidden]-wide bf16 payload row:
  - general branch: a compact int64 bitmask (8 B/row) plus a [n_rows, slots] sparse
    gate plane, unpacked at arrival by bit extraction + nonzero;
  - quota fast path (groups_m, C1 2026-08-20): [n_rows, quota] ascending slot ids +
    [n_rows, quota] dense gate values -- the slot table IS the arrival pair list, so
    the pack loses the pow2/scatter stage and the arrival loses bit-extract + topk,
    bit-identical outputs (see _pack_quota_wire).

The result must equal `ep_dist.ep_moe_forward` exactly (tests/test_ta2a.py); T-A2A is
a communication schedule, not a different model.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .layer import grouped_mm
from .ep_dist import _a2a, _a2a_raw
from .ta2a import plan_ta2a
# K1 kernel gate (terrace.ops only imports THIS module lazily inside functions, so the
# module-level import here cannot cycle; custom_ops_enabled() is a cached-bool read).
from . import ops as _tops

# The intra-node subgroup is created once (new_group is collective and expensive); the
# forward is called per layer per step, so creating it inline would deadlock or crawl.
_INTRA_GROUP = [None]
_INTER_GROUP = [None]   # inter-node subgroup: same local index across all nodes (see init_ta2a_groups)
_POW2_CACHE: dict = {}
_TIEBREAK_CACHE: dict = {}

# Iota prefix buffers for DATA-dependent lengths (arrival row/pair counts vary per
# micro-batch). ta2a._ARANGE_CACHE keys by exact length, which is right for the plan's
# fixed shapes but would grow one entry per distinct length here; a single growing buffer
# per (device, dtype), sliced to size, keeps the cached-kernel saving with bounded memory.
_ARANGE_BUF: dict = {}

# The Hop A count-exchange scatter index (arange(n_nodes) * rpn + my_local) is a pure
# function of the geometry -- three kernels per call for a value that never changes.
_SEND_IDX_CACHE: dict = {}


def _arange_prefix(n: int, device, dtype) -> torch.Tensor:
    """Read-only arange prefix from a per-(device, dtype) growing buffer.

    Returned tensor is a VIEW of the shared buffer -- callers must never write into it
    (every use below feeds it to an out-of-place op).
    """
    key = (str(device), dtype)
    buf = _ARANGE_BUF.get(key)
    if buf is None or buf.numel() < n:
        buf = torch.arange(max(n, 4096), device=device, dtype=dtype)
        _ARANGE_BUF[key] = buf
    return buf[:n]


def _pow2_table(slots: int, dev, dtype) -> torch.Tensor:
    """Cached [slots] power-of-two table (see the in-cache-miss comment for why int pow).

    Building the table is one place this can go wrong: `2.0 ** arange(..., float32)` is a
    FLOATING-POINT pow, which the NPU evaluates as exp2/log and returns 32767.99985 for
    2**15. Truncating that to int64 loses one bit and every affected mask comes out
    exactly 1 too small -- which is what it did, silently, until a device-side
    equivalence check caught it (internal benchmark script, not shipped with the repo;
    the CPU unit tests all passed). Integer pow is exact, and int -> float32 is exact for powers of two well
    past 2**24; it is only their SUM that is not (see build_expansion's mdtype boundary).
    """
    key = (dev, slots, dtype)
    tbl = _POW2_CACHE.get(key)
    if tbl is None:
        tbl = (2 ** torch.arange(slots, device=dev, dtype=torch.int64)).to(dtype)
        _POW2_CACHE[key] = tbl
    return tbl


def _tiebreak_table(slots: int, dev, dtype) -> torch.Tensor:
    """Cached [slots] strictly DECREASING column key `slots-1, slots-2, ..., 1, 0`.

    Added to the {0,1} demand bits before `_expand_arrival`'s quota `topk` so that no two
    entries of a row can tie -- see the comment at the use site for why a tie there is a
    silent numerical hazard. Values are integers below 2*slots <= 126, exact in float32.
    """
    key = (dev, slots, dtype)
    tbl = _TIEBREAK_CACHE.get(key)
    if tbl is None:
        tbl = (slots - 1) - torch.arange(slots, device=dev, dtype=torch.int64)
        tbl = tbl.to(dtype)
        _TIEBREAK_CACHE[key] = tbl
    return tbl


def _send_index(n_nodes: int, rpn: int, my_local: int, dev) -> torch.Tensor:
    """Cached scatter index for the Hop A count exchange: node n's designated peer."""
    key = (n_nodes, rpn, my_local, str(dev))
    t = _SEND_IDX_CACHE.get(key)
    if t is None:
        t = torch.arange(n_nodes, device=dev, dtype=torch.int64) * rpn + my_local
        _SEND_IDX_CACHE[key] = t
    return t


def _splits_to_lists(a: torch.Tensor, b: torch.Tensor):
    """Materialise two split-count tensors with ONE device->host sync instead of two.

    `torch.cat([a, b]).tolist()` reads both in a single transfer; the values are the
    same integers `.tolist()` on each would produce. Every sync here drains the device
    dispatch queue, so halving them is worth the one small cat kernel.
    """
    n = a.numel()
    both = torch.cat([a, b]).tolist()
    return both[:n], both[n:]


def _expand_arrival(rmask: torch.Tensor, slots: int, quota):
    """Arrival-side (row, slot) demand pairs from the shipped bitmasks, pre-reorder.

    Bit extraction: NPU's __rshift__ has no int64 kernel (aclnnBitwiseRightShift fails),
    so bits come from divide + modulo. Below 25 slots the mask VALUE is < 2**24, so the
    whole unpack runs exactly in float32: int64 -> float32 is exact there, dividing by a
    power of two only moves the exponent, and floor / % 2 of integer-valued floats are
    exact -- the same 0/1 bits as the int64 unpack for a fraction of its cost, and the
    quota branch's topk no longer needs a separate cast. Wider nodes keep the exact
    int64 unpack, the same 24-bit boundary build_expansion's accumulator uses.

    quota is not None: every arrived row is one (token, node) pair and exactly `quota`
    of the node's experts want it, so the pair count R*quota is known before the data is
    read. `nonzero` would produce the same pairs but with a data-dependent output shape,
    which forces a host synchronisation in the middle of the return path; topk over a
    statically known count does not. quota None: the general nonzero extraction.

    NEVER hand that topk a TIED key. The pair ORDER this function returns is not free
    metadata: it fixes `ordo`, the Hop B pair sequence and -- the part that shows up in
    the loss -- the order of the per-row `index_add_` that sums a row's `quota`
    contributions back together in the combine. Over raw {0,1} bits all `quota` winners
    tie at 1.0, and PyTorch does not specify the order of tied maxima: CPU happens to
    emit them ascending, another backend is free not to, and nothing downstream would
    ever complain because the (row, slot) pairing stays self-consistent either way --
    only the reduction order moves. That is a divergence NO CPU test can see. Measured on
    the CPU bit-parity bed with the tie order reversed (a legal topk output on a fully
    tied key): forward and every gradient move by one ulp at once (fp32 y 7.2e-7, bf16
    x.grad 3.1e-2) against the bit-identical run -- i.e. exactly the "extra float
    reordering" signature an eq bed reports, from a line that looks value-preserving.
    So make the key DISTINCT before selecting: `slots-1-col` ranks the set bits above
    every unset one and, among the set bits, in ASCENDING column order -- which is the
    order `nonzero` gives the general branch and the order `_pack_quota_wire` ships on
    the C1 wire. Same composite-key trick, and the same reason, as
    `_stable_argsort_small`: buy a total order instead of trusting a tie-break.
    """
    dev = rmask.device
    R = rmask.shape[0]
    if slots <= 24:
        tbl = _pow2_table(slots, dev, torch.float32)
        bits = (rmask.to(torch.float32).unsqueeze(1) // tbl) % 2         # [R, slots]
    else:
        tbl = _pow2_table(slots, dev, torch.int64)
        bits = (rmask.unsqueeze(1) // tbl) % 2
    if quota is not None:
        fbits = bits if bits.dtype == torch.float32 else bits.to(torch.float32)
        # add(col_desc, fbits, alpha=slots) == fbits*slots + (slots-1-col) bit-for-bit:
        # every operand and every intermediate is an integer below 2*slots <= 126, so no
        # grouping of the multiply-add can round. One elementwise kernel, no sort.
        key = torch.add(_tiebreak_table(slots, dev, torch.float32), fbits,
                        alpha=float(slots))
        slot_idx = torch.topk(key, quota, dim=1).indices                 # [R, quota]
        r_idx = _arange_prefix(R, dev, torch.int64).unsqueeze(1).expand(
            R, quota).reshape(-1)
        return r_idx, slot_idx.reshape(-1)
    return torch.nonzero(bits, as_tuple=True)


def _pack_quota_wire(expert_idx, gates, inverse, payload, n_rows: int, slots: int,
                     quota: int, n_experts: int, *, sorted_rows: bool = False):
    """Quota fast-path Hop A id/gate planes (C1, 2026-08-20): [n_rows, quota] ASCENDING
    slot ids (int64) + [n_rows, quota] gate values, replacing the int64 bitmask +
    [n_rows, slots] sparse gate plane.

    Wire contract, and why nothing downstream moves by a bit:
      - Row r of the slot table lists, ascending, the destination-node expert slots
        that want payload row r; the gate table is co-indexed (gate[r, i] belongs to
        slot[r, i]). Row-major + slot-ascending is THE arrival pair-order contract,
        shared by all three formats: `nonzero` gives it to the general branch by
        definition, `_expand_arrival`'s quota branch now forces it with a distinct
        composite key, and this packer ships it. So the flattened (r_idx, slot_idx)
        pair sequence, every ordering derived from it (ordo, Hop B splits, expert
        order) and every reduction order are bit-identical across formats. C1 changes
        bytes and kernels on the wire, not one output bit.
        The contract used to rest instead on "topk returns tied maxima in index
        order", checked exhaustively over C(slots, quota) bit patterns -- on CPU. That
        is not a property PyTorch guarantees, so the check could not transfer to the
        device, and a backend that broke the tie the other way would have made this
        format and the old one disagree numerically while every CPU test stayed green
        (see _expand_arrival). The premise is now enforced, not assumed.
      - The gate values are the SAME scalars the sparse plane carried at
        (row, slot), packed dense: the arrival gather `rgate[r_idx, slot_idx]`
        degenerates to `rgate.reshape(-1)[ordo]` -- the flat position r*quota + i IS
        the pair's position in the pre-ordo enumeration.

    Why the packing is cheap: under equal quota a token's k experts split into
    exactly M = k/quota runs of `quota` slots, one run per touched node, and every
    member of a run maps via `inverse` to the same send row -- so both planes are
    one row-scatter of a [T*M, quota] view. The pow2 table, the per-run sum, the
    mask cast and the [n_rows, slots] zeros + per-slot scatter are simply gone; the
    arrival side loses the whole bit-extract + topk stage.

    `sorted_rows=True` promises ascending rows (the seam extraction's construction,
    routing_map_to_topk). Ascending expert ids are ascending (node, slot), so the
    runs are already contiguous, node-ascending and slot-ascending within each run
    -- no sort at all. Otherwise one per-row argsort makes it so: expert ids are
    distinct within a row, so the ascending permutation is unique and the float32
    keys (exact below 2**24, same boundary and fallback as routing_map_to_topk)
    cannot tie.

    The gate plane is allocated from `payload` ON PURPOSE, not from `gates`: the
    caller owns the rounding point (the overlap seam casts to payload dtype BEFORE
    packing -- the same point the sparse plane rounded at), and a gates/payload
    dtype mismatch must keep failing loudly in the index_put, exactly as the
    [n_rows, slots] plane failed. Deriving the plane from `gates` would silently
    ship a wider gate plane instead -- the entry point of the drift bug fixed in an
    internal commit.
    """
    T, k = expert_idx.shape
    if sorted_rows:
        slot_vals = (expert_idx.reshape(-1) % slots).view(-1, quota)
        gate_vals = gates.reshape(-1, quota)
        row_of_run = inverse.view(T, k)[:, ::quota]                    # [T, M]
    else:
        if n_experts < (1 << 24):
            order = torch.argsort(expert_idx.to(torch.float32), dim=1)
        else:                                    # unreachable guard, same as the seam
            order = torch.argsort(expert_idx, dim=1)
        slot_vals = (torch.gather(expert_idx, 1, order) % slots).reshape(-1, quota)
        gate_vals = torch.gather(gates, 1, order).reshape(-1, quota)
        row_of_run = torch.gather(inverse.view(T, k), 1, order[:, ::quota])
    rof = row_of_run.reshape(-1)
    # zeros, not empty: if the routing drifts off the equal-quota invariant between
    # plan_ta2a's periodic re-verifications, a row can drop out of `rof`; zeros makes
    # that row ship slot 0 / gate 0 -- deterministic and harmless-shaped, the same
    # containment the zeroed mask plane had (missing row == no demand bits).
    slot_tab = torch.zeros(n_rows, quota, dtype=torch.int64, device=expert_idx.device)
    slot_tab[rof] = slot_vals
    gate_tab = payload.new_zeros(n_rows, quota)
    gate_tab[rof] = gate_vals
    return slot_tab, gate_tab


def _expand_arrival_quota(rslot: torch.Tensor):
    """Arrival-side (row, slot) pairs straight off the C1 quota wire table.

    The slot table IS the pair list: a row-major flatten yields the same
    (r_idx, slot_idx) sequence `_expand_arrival`'s quota branch produces from the
    bitmask -- both are row-major with slots ascending within a row, the arrival
    pair-order contract (_pack_quota_wire; `_expand_arrival` enforces its half with a
    distinct topk key rather than trusting a tie-break). So ordo, the Hop B splits and
    every downstream ordering are bit-identical -- while the bit extraction and the
    topk no longer exist.
    """
    R, quota = rslot.shape
    r_idx = _arange_prefix(R, rslot.device, torch.int64).unsqueeze(1).expand(
        R, quota).reshape(-1)
    return r_idx, rslot.reshape(-1)


def init_ta2a_groups(world: int, rpn: int = 8):
    """Create the intra-node and **inter-node** sub-communicators. EVERY rank must call this, in the same order.

    Inter-node subgroup (added 2026-08-24): {l, l+rpn, l+2*rpn, ...}, i.e. the same
    local index across all nodes. This is Hop A's communication domain in the standard
    2D decomposition.

    **Why it must exist**: Hop A used to run over the whole EP group
    (`inter_group=ep`, 128 ranks), with 16 non-zeros padded into a length-128 splits
    vector. Consequences:
      - pays the full α(127)=0.704 -- exactly the cost two-hop was supposed to remove
        (the subgroup only needs α(15)=0.485)
      - the payload is spread over 128 peers, 0.11 MB per peer, at the very bottom of
        the β(size) curve
    Measured ledger gap: **0.219 ms per call**, 12% of the 1.799 dispatch gap -- and it
    is **pure waste, not a trade-off**.

    **Both groups are built here, once**: `new_group` is a world-wide collective;
    building it lazily mid-training-step deadlocks (stepped on in 2026-08, see the
    usercustomize note).
    """
    groups = [dist.new_group(list(range(n * rpn, (n + 1) * rpn)))
              for n in range(world // rpn)]
    _INTRA_GROUP[0] = groups[dist.get_rank() // rpn]
    # Order identical on every rank: build for local index 0..rpn-1 first, then each rank takes its own
    inter = [dist.new_group(list(range(l, world, rpn))) for l in range(rpn)]
    _INTER_GROUP[0] = inter[dist.get_rank() % rpn]
    return _INTRA_GROUP[0]


def fixed_hist(idx: torch.Tensor, n_buckets: int) -> torch.Tensor:
    """Fixed-length histogram -- a **bit-for-bit** replacement for `torch.bincount(idx, minlength=n_buckets)`.

    Why not bincount (operator-level table, 2026-08-23): bincount's output length is
    `max(minlength, idx.max()+1)`, so it takes a max first **even when minlength is
    given** -- a device->host sync that drains the whole pipeline. Measured on the
    verdict testbed: **0.784 ms per call**, 66.8% of the arrival chain -- an 8-bucket
    histogram costing 4.8x more than the big [24576, 2048] gather. Per step that is
    94 ms, larger than the entire remaining gap at the time (71 ms/step).

    Here the output length is **fixed at build time** to n_buckets (idx at every call
    site is in [0, n_buckets) by construction); no data needs to be inspected, hence
    no sync.

    Bit-for-bit identical: both are exact integer counts (tests/test_fixed_hist.py
    covers n=0/1/100/24576 and random distributions). **And safer**: an out-of-range
    index silently returns a longer array under bincount, and fails on the spot under
    scatter_add_.
    """
    out = torch.zeros(n_buckets, dtype=torch.int64, device=idx.device)
    if idx.numel() == 0:
        return out
    return out.scatter_add_(0, idx, torch.ones_like(idx))


def _stable_argsort_small(key: torch.Tensor, n_buckets: int) -> torch.Tensor:
    """Stable ascending argsort of a small-range integer key, without an integer sort.

    `torch.argsort(..., stable=True)` on an integer dtype is the most expensive primitive
    on this device: 5.32 ms for 65536 int64 values against 0.107 ms for float32
    (internal benchmark script, not shipped with the repo). Both of the return path's sorts have keys with a
    tiny range -- destination local rank (rpn = 8 values) and local expert index
    (epr = 2) -- so a composite key `bucket * n + position` is already a total order, which
    makes the sort stable without asking for stability, and it fits in float32 exactly as
    long as n_buckets * n stays under 2**24.

    Falls back to the integer stable sort when it does not fit, because a silently
    approximate ordering here would corrupt the routing rather than merely slow it.
    """
    n = key.numel()
    if n_buckets * n >= (1 << 24):
        return torch.argsort(key, stable=True)
    pos = _arange_prefix(n, key.device, torch.float32)
    # add(pos, key, alpha=n) computes pos + n*key == key*n + pos bit-for-bit: every
    # operand and every intermediate is an integer below 2**24 (guarded above), so no
    # grouping of the multiply-add can round -- one fewer kernel, zero bits moved.
    return torch.argsort(torch.add(pos, key.to(torch.float32), alpha=float(n)))


def build_expansion(expert_idx, inverse, n_rows: int, world: int, n_experts: int,
                    rpn: int = 8, groups_m: int | None = None, *,
                    sorted_rows: bool = False, slot_flat=None):
    """Per-send-row bitmask of the destination node's expert slots that want the token.

    Bit j of row r is set iff local-expert-slot j (0 <= j < epr*rpn, numbered within the
    destination node) is one of the token's top-k experts. Requires epr*rpn <= 63.

    `groups_m` takes a path that exploits T-Route's structure instead of scattering; see
    below. It changes cost only, and the two paths are asserted equal in tests/test_ta2a.py.

    `sorted_rows=True` promises each ROW of expert_idx is ascending. Only the seam
    extraction (routing_map_to_topk) guarantees that by construction; it lets the
    structured path skip its per-row argsort (see comment there). Cost only -- the mask
    bits are identical either way (tests/test_ta2a_dispatch_slim.py).

    `slot_flat` is an optional precomputed `expert_idx.reshape(-1) % slots`: both seam
    halves need that exact tensor for the gate plane too, so computing it twice per
    dispatch was pure waste. Values are asserted-by-construction identical.
    """
    epr = n_experts // world
    slots = epr * rpn
    if slots > 63:
        raise ValueError(f"{slots} expert slots per node exceeds the int64 mask")
    dev = expert_idx.device
    T, k = expert_idx.shape
    # 2**slot via a small lookup table (NPU has no int64 shift kernel). The table depends
    # only on `slots`, so building it per call was pure waste -- cache it per (device,
    # slots, dtype).
    # A row's mask is a SUM of up to k distinct powers of two, so float32 holds it exactly
    # only while the highest set bit stays below 2**24 -- that is, while slots <= 24. The
    # guard above tests 63, the width of the int64 the mask is finally cast to, NOT the
    # width of the float32 it is ACCUMULATED in. Every geometry with 25..63 slots per node
    # therefore produced a silently truncated mask: bits below 2**(slots-24) were rounded
    # away, rows were delivered to the wrong experts, and the dispatch stayed fast and
    # entirely plausible. It went unseen because every configuration measured between
    # 2026-07-27 and 2026-07-30 had slots = 16 (128 experts over 8 nodes), which is exact;
    # 128 experts over 4 nodes -> slots 32 fails equivalence at max_rel_diff 0.46-0.64, and
    # so does 256 experts over 8 nodes, which is how we know the trigger is slots and not
    # the node count. Above 24 slots, pay for the exact int64 accumulation instead.
    mdtype = torch.float32 if slots <= 24 else torch.int64
    # Table construction is the OTHER place this can go wrong -- see _pow2_table for the
    # floating-pow truncation story (2**15 -> 32767.99985) and why integer pow is exact.
    tbl = _pow2_table(slots, dev, mdtype)

    if groups_m is not None and k % groups_m == 0:
        # Structure-exploiting path. Under T-Route every token touches exactly M nodes with
        # exactly k/M experts on each, so sorting a token's k slots by destination node
        # partitions them into M contiguous runs of equal length quota = k/M. The mask for
        # a send row is then a reduction over the last axis of a [T, M, quota] view -- no
        # scatter at all.
        #
        # Why it matters: `scatter_add_` on int64 is the single most expensive primitive in
        # the whole dispatch, 0.920 ms at n=65536 (internal benchmark script, not
        # shipped with the repo), while
        # a float32 sort of the same data is 0.107 ms and the reduction is 0.013 ms. The
        # routing property the paper is about is exactly what buys the cheaper formulation.
        #
        # Sorting by destination node is what makes the runs contiguous; we do not assume
        # the router emits the k experts already grouped -- UNLESS the caller promises it
        # (sorted_rows). Ascending rows imply ascending dest per row, so the runs are
        # already contiguous and in ascending node order: the argsort, the dest divide and
        # the gather all vanish. Bit-safety of the skip: per_run sums the SAME set of
        # distinct powers of two (exact in float32 below 24 slots, exact in int64 above,
        # under ANY summation order), and every member of a run maps via `inverse` to the
        # same send row -- so run_first picking a different member could not change the
        # scatter target either. Identical mask, fewer kernels.
        quota = k // groups_m
        if sorted_rows:
            if slot_flat is None:
                slot_flat = expert_idx.reshape(-1) % slots
            vals = tbl[slot_flat].reshape(T, groups_m, quota)
            row_of_run = inverse.reshape(T, k)[:, ::quota]         # [T, M] run entry rows
        else:
            dest = torch.div(expert_idx, slots, rounding_mode="floor").to(torch.float32)
            order = torch.argsort(dest, dim=1)
            slot_sorted = torch.gather(expert_idx, 1, order) % slots
            vals = tbl[slot_sorted].reshape(T, groups_m, quota)
            # Rows are laid out node-major (node 0's tokens, then node 1's ...), and within
            # a token the runs are already in ascending node order, so a row's mask is
            # found by the same `inverse` mapping used by the general path -- take it from
            # the first slot of each run, which is where that (token, node) pair enters
            # `inverse`.
            run_first = order[:, ::quota]                          # [T, M] column of run
            row_of_run = torch.gather(inverse.reshape(T, k), 1, run_first)
        per_run = vals.sum(-1)                                     # [T, M]
        mask = torch.zeros(n_rows, dtype=mdtype, device=dev)
        mask[row_of_run.reshape(-1)] = per_run.reshape(-1)
        return mask.to(torch.int64)

    if slot_flat is None:
        slot_flat = expert_idx.reshape(-1) % slots   # expert's position inside its node
    mask = torch.zeros(n_rows, dtype=mdtype, device=dev)
    mask.scatter_add_(0, inverse, tbl[slot_flat])  # rows unique per (token,node): no dup bits
    return mask.to(torch.int64)


def ta2a_moe_forward(x_local, expert_idx, gates, w13_shard, w2_shard,
                     world: int, n_experts: int, rpn: int = 8,
                     groups_m: int | None = None) -> torch.Tensor:
    """Routed-expert output under T-A2A dispatch. Same contract as ep_moe_forward.

    `groups_m` is the exact per-token destination-node count under T-Route. Passing it
    lets the plan skip its host synchronisation (see plan_ta2a). It is only
    result-preserving when the routing is EQUAL-QUOTA (each touched node holds exactly
    k/M of the token's experts); plan_ta2a verifies that invariant on the first call and
    every 256th thereafter and refuses otherwise. Between verification windows a routing
    that drifts off the invariant is NOT caught -- do not pass groups_m for routings that
    merely have exact fan-out (group-limit without quota).
    """
    epr = n_experts // world
    n_nodes = world // rpn
    if n_nodes < 2:                       # one node: no fabric hop to reshape
        from .ep_dist import ep_moe_forward
        return ep_moe_forward(x_local, expert_idx, gates, w13_shard, w2_shard,
                              world, n_experts)

    T, k = expert_idx.shape
    dev = x_local.device
    rank = dist.get_rank()
    my_local = rank % rpn

    u_src, u_node, node_counts, inverse = plan_ta2a(expert_idx, world, n_experts, rpn,
                                                groups_m=groups_m)
    n_rows = u_src.numel()
    payload = x_local[u_src]                              # dedup'd, node-contiguous
    slots = epr * rpn
    if groups_m:
        # Floor division would silently drop k % M experts per token from the wire
        # packing / demand extraction even where the general branch stays correct
        # (measured pre-C1: k=3, M=2 loses 4 of 12 pairs). Unreachable via
        # TRouteConfig, which rejects k % M != 0, but this function is also called
        # directly by benches.
        assert k % groups_m == 0, f"k={k} not divisible by groups_m={groups_m}"
    quota = (k // groups_m) if groups_m else None
    if quota is not None:
        # C1 quota wire format: [n_rows, quota] ascending slot ids + [n_rows, quota]
        # gate values (see _pack_quota_wire for the bit-parity argument). No
        # sorted_rows promise at this entry point -- arbitrary callers (benches,
        # tests) -- so the packer pays one per-row argsort to establish it.
        mask, gate_rows = _pack_quota_wire(expert_idx, gates, inverse, payload,
                                           n_rows, slots, quota, n_experts)
    else:
        # One modulo, shared by the mask and gate planes below -- it used to be
        # computed twice per dispatch (here and inside build_expansion).
        slot_flat = expert_idx.reshape(-1) % slots
        # No sorted_rows here: this entry point takes expert_idx from arbitrary
        # callers with no ascending-row guarantee; only the seam extraction has one.
        mask = build_expansion(expert_idx, inverse, n_rows, world, n_experts, rpn,
                               groups_m=groups_m, slot_flat=slot_flat)
        gate_rows = payload.new_zeros(n_rows, slots)
        gate_rows[inverse, slot_flat] = gates.reshape(-1)
    # Three places the gate COULD be applied; two are correct and one of the correct
    # ones is free (Step 2, internal design record (not shipped with the repo), landed 2026-08-20):
    #   - at the ORIGIN: WRONG. A returned row is the sum over every expert of that node
    #     that wanted the token -- granularity has already collapsed to per-(token, node)
    #     by then, so weighting once per (token, expert) slot at the origin would both
    #     double-count the row and use the wrong weights.
    #   - at the EXPERT rank: correct, but the per-pair gate has to ride an extra
    #     intra-node exchange (`my_gate`) to get there -- one more differentiable
    #     collective in the forward AND one more in the backward.
    #   - at the ARRIVAL rank, on the return leg just before `red.index_add_`: correct
    #     and FREE. Granularity there is still per-(row, slot) pair, and the gate is
    #     already local (`rgate` lands on this rank). That is where it is applied below.
    # The gate must still CROSS THE FABRIC via `rgate` (the differentiable exchange
    # further down) -- the router learns through it, and the origin cannot apply it
    # (first bullet). Deleting `rgate` is NOT a further simplification, it is the bug
    # the first bullet describes. Ship it in the id plane -- slots_per_node floats
    # (quota floats on the C1 fast path) against a [hidden] bf16 payload row, still
    # a rounding error of the payload.

    # Each rank sends a node's whole payload to ONE designated peer (same local index),
    # so the fabric sees n_nodes large messages instead of `world` small ones. The
    # scatter index is a pure function of the geometry -- cached, not rebuilt per call.
    send = torch.zeros(world, dtype=torch.long, device=dev)
    send[_send_index(n_nodes, rpn, my_local, dev)] = node_counts
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
    # Materialise the split counts ONCE, with ONE sync (_splits_to_lists). `_a2a_raw`
    # accepts lists (see ep_dist.py) and each of the four exchanges below would otherwise
    # call `.tolist()` on both tensors -- eight device->host syncs for two values the
    # fabric hop already had to know. The order of the pair is load-bearing: (send, recv)
    # going out, (recv, send) coming back at the return exchange. Swapping those silently
    # misroutes rows while still producing a plausible forward, so the reversed use below
    # is spelled out rather than derived.
    send_l, recv_l = _splits_to_lists(send, recv)

    # DO NOT FUSE THE GATE INTO THE PAYLOAD. Tried and measured on 2026-08-01, and it is
    # much worse -- kept as a comment so it is not tried again.
    #
    # The idea was sound on paper: `rx` and `rgate` share splits and group, as did
    # `node_rx` and the since-removed `my_gate` exchange (Step 2, 2026-08-20, keeps the
    # gate local instead -- subtraction, not fusion). The backward is where T-A2A
    # loses (6 differentiable exchanges to the baseline's 2, 5 after Step 2; 36% slower at equal volume,
    # internal measurement records), and each exchange costs ~0.1 ms of device-side
    # fixed overhead regardless of size (M0 Sec.6, A-9), so removing two round trips looked
    # free -- the gate is a few columns next to a 2048-wide hidden.
    #
    # Measured, 64 die / 16384 tok, 3 reps, still numerically and gradient equivalent:
    #     unfused   fwd 2.13x   bwd +7.03 ms    fwd+bwd 1.494x
    #     fused     fwd 1.09x   bwd +55.05 ms   fwd+bwd 0.730x
    # The concatenation copies the whole ~268 MB payload, and the slices that undo it are
    # non-contiguous, so every downstream gather and the second cat pay for strided access.
    # Two HCCL round trips are worth ~0.2 ms; the copies cost tens of ms. The fixed-cost
    # argument was right about the collectives and silent about the memory traffic.
    rx = _a2a(payload, send_l, recv_l)                    # the one fabric-crossing hop
    # `mask` is the id plane in either wire format: [n_rows] int64 bitmask (general)
    # or [n_rows, quota] ascending slot table (C1 fast path). Same splits, same call
    # count -- C1 changes each call's payload layout, never the number of collectives
    # (pinned by test_ta2a_gate_at_arrival's counts).
    rmask = _a2a_raw(mask, send_l, recv_l)
    # The gate goes through the DIFFERENTIABLE exchange, unlike the mask and the slot ids.
    # Masks and indices carry no gradient and `_a2a_raw` is right for them; the gate is the
    # router's output and the router learns through it. Shipping it with `_a2a_raw` left
    # `gates.grad` as None while every other gradient matched the baseline to 1e-7 and the
    # forward was bit-identical -- a router that silently stops learning, invisible to any
    # output comparison. Caught by tests/test_ta2a_grad.py.
    rgate = _a2a(gate_rows, send_l, recv_l)

    # After the hop a node's rows are spread over its `rpn` ranks (partitioned by the
    # SOURCE local index), but the experts that want a given row may live on any rank of
    # the node. The first version all_gathered the node's whole arrival set onto every
    # rank: correct, but each rank then held 8x the rows it actually serves, and that
    # redundant intra-node traffic was ONE cause of the kernel being slower than baseline --
    # but a minor one: removing it moved world=16 from 0.40x only to 0.465x (git 13f5954,
    # 2026-07-26; world=16 is 2 nodes, an admittedly poor operating point). The dominant
    # cost was the PLAN: internal measurement records had the fabric hop
    # already at 1.67x with plan = 2.593 ms, and the real culprit turned out to be
    # argsort(stable) at 2.483 ms (internal measurement records).
    # Instead route each row only to the ranks that want it -- one intra-node all_to_all
    # sized by the ACTUAL demand, so intra-node traffic is ~1x instead of rpn x.
    if _INTRA_GROUP[0] is None:
        raise RuntimeError("call init_ta2a_groups(world, rpn) once before the forward")
    intra = _INTRA_GROUP[0]
    R = rx.shape[0]

    # Which local rank each arrived row owes work to. Quota fast path: the received
    # slot table IS the (row, slot) pair list, row-major (_expand_arrival_quota) --
    # no bit extraction, no topk. General branch: bit j of the mask is expert slot j,
    # owned by rank j // epr; extraction and the nonzero pair enumeration live in
    # _expand_arrival (shared with both seam halves; the float32 unpack there is
    # bitwise-identical to the int64 one below 25 slots).
    # K1 (AscendC kernel, 2026-08-20): on the quota fast path this whole arrival-side
    # block -- pair expansion, owner bucketing, stable reorder, i_send histogram, the
    # [pairs, H] send gather and the co-indexed gate gather -- is ONE fused kernel.
    # Its two-pass counting sort is bit-identical to _stable_argsort_small's stable
    # order (proof: terrace/ops/ascendc/op_kernel/terrace_k1_arrival.cpp header; CPU
    # executable spec: terrace.ops.k1_arrival_ref). The composite chain in the else
    # branch is the pre-K1 text, UNCHANGED, and remains the only path when the kernel
    # is absent (TERRACE_CUSTOM_OPS=0 / no .so): zero behaviour change. `exp_rx` /
    # `gate_pairs` are hoisted out of the collectives below so both branches meet the
    # Hop B exchange with the same five tensors -- pure data-flow reshuffle, the ops
    # and their operands are unchanged.
    if quota is not None and _tops.custom_ops_enabled():
        exp_rx, gate_pairs, r_idx, slot_idx, i_send = _tops.k1_arrival(
            rx, rmask, rgate, quota, epr, rpn, my_local)
    else:
        if quota is not None:
            r_idx, slot_idx = _expand_arrival_quota(rmask)
        else:
            r_idx, slot_idx = _expand_arrival(rmask, slots, quota)
        owner = slot_idx // epr
        ordo = _stable_argsort_small(owner, rpn)
        r_idx, slot_idx = r_idx[ordo], slot_idx[ordo]
        # bincount is permutation-blind: counting the unpermuted owners equals
        # counting owner[ordo], so the third [pairs]-sized gather bought nothing.
        i_send = fixed_hist(owner, rpn)                                # rows per peer (fixed length, no sync)
        exp_rx = rx[r_idx]
        # The per-pair gate is NOT exchanged (Step 2): it stays on this rank and is
        # applied to `ret` when the expert results land back here. Indexing AFTER the
        # `ordo` reorder above is load-bearing: `gate_pairs[j]` must describe the same
        # (row, slot) pair that `ret[j]` answers, and `ret` comes back in the
        # reordered pair order. Indexing before the reorder silently misaligns gate
        # against ret while staying shape-correct. Quota fast path: the dense gate
        # table is co-indexed with the pair enumeration, so the 2-D gather degenerates
        # to a flat gather -- `rgate.reshape(-1)[ordo][j]` and
        # `rgate[r_idx[j], slot_idx[j]]` name the SAME element (flat pos =
        # row*quota + i is the pair's pre-ordo position), bit-identical by
        # construction.
        gate_pairs = (rgate.reshape(-1)[ordo] if quota is not None
                      else rgate[r_idx, slot_idx])        # kept local; no collective
    i_recv = torch.empty_like(i_send)
    dist.all_to_all_single(i_recv, i_send, group=intra)
    # Same once-only, one-sync materialisation as the fabric hop above; (is_l, ir_l)
    # outbound, (ir_l, is_l) on the way back.
    is_l, ir_l = _splits_to_lists(i_send, i_recv)

    # Ship only the demanded rows, plus the slot each copy is for. (Not fused into one
    # exchange -- see the measured result at the fabric hop above.)
    node_rx = _a2a(exp_rx, is_l, ir_l, group=intra)
    my_slot = _a2a_raw(slot_idx, is_l, ir_l, group=intra)

    # Every received row is already destined for one of THIS rank's experts.
    exp_j = my_slot - my_local * epr
    order = _stable_argsort_small(exp_j, epr)
    row_i = order                                    # rows are 1:1 with (row, expert) pairs
    counts = fixed_hist(exp_j, epr)                  # permutation-blind: no exp_j[order]

    a, b = grouped_mm(node_rx[row_i], w13_shard, counts).chunk(2, dim=-1)
    ye = grouped_mm(F.silu(a) * b, w2_shard, counts)

    # Send the expert results straight back along the reverse of the demand exchange,
    # UNWEIGHTED, and weight each contribution by ITS OWN gate as it lands (see the
    # three-position comment above `gate_rows`). `ye * my_gate` at the expert rank and
    # `ret * gate_pairs` here are the same elementwise product of the same operands --
    # the exchange only moves rows -- so the result is bit-identical, while one
    # intra-node collective disappears from the forward (shipping my_gate out) and one
    # from the backward (its `_A2A.backward` shipping the gate gradient back; the gate
    # gradient is now computed where the gate lives). No reduce_scatter over a padded
    # rpn-sized buffer: the return traffic is again exactly the demand.
    back_pairs = ye.new_empty((ye.shape[0], ye.shape[1]))
    back_pairs[order] = ye
    ret = _a2a(back_pairs, ir_l, is_l, group=intra)       # REVERSED on purpose: ir_l, is_l
    red = rx.new_zeros(R, ret.shape[1])
    red.index_add_(0, r_idx, ret * gate_pairs.unsqueeze(1).to(ret.dtype))
    back = _a2a(red, recv_l, send_l)     # rows return to their origin -- reversed: recv, send

    # Each returned row already carries the gate-weighted sum over the destination
    # node's experts, so the origin adds it EXACTLY ONCE per (token, node) -- u_src names
    # that token. (Adding once per (token, expert) slot was the earlier bug: it repeated
    # the row M times and reapplied gates that had already been consumed.)
    y = x_local.new_zeros(T, x_local.shape[1])
    return y.index_add(0, u_src, back)
