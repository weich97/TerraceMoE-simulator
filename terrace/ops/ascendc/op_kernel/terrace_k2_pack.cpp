/**
 * terrace_k2_pack -- AscendC kernel: T-A2A send-side fused pack chain (K2, C1 quota wire format).
 *
 * ===================== Functional spec (bitwise replication) =====================
 *
 * Replicates, under the quota fast path (equal-quota, groups_m=M), the entire existing
 * composed chain before the Hop A count exchange (terrace/ta2a.py::plan_ta2a fast path +
 * the send stage of terrace/ta2a_fwd.py; ta2a_moe_forward and
 * ta2a_dispatch.ta2a_permute[_overlap] share this same stage, isomorphically):
 *
 *     u_src, u_node, node_counts, inverse = plan_ta2a(expert_idx, world,
 *                                                     n_experts, rpn, groups_m=M)
 *     payload = hidden[u_src]                       # deduplicated gather, node-major order
 *     mask, gate_rows = _pack_quota_wire(expert_idx, gates, inverse, payload,
 *                                        n_rows, slots, quota, n_experts)
 *
 * Inputs: hidden [T, H] (fp16/bf16/fp32), expert_idx [T, k] (int64), gates [T, k]
 * (same dtype as hidden -- the C1 rounding-point contract: the gate plane derives from
 * payload, and a mismatch dies loudly on the torch side); attributes world / n_experts /
 * rpn / groups_m.
 * Outputs: payload [T*M, H], mask [T*M, quota] (ascending slot table, int64), gate_rows
 * [T*M, quota], u_src [T*M] int64, node_counts [n_nodes] int64; quota = k/M.
 * u_node is not output: none of the three call sites uses it (ta2a_permute discards it,
 * the overlap half binds it to `_`, the fused forward no longer reads it); inverse is not
 * output: on the quota branch it is only consumed by packing, which the kernel absorbs.
 *
 * No within-row ascending order is assumed (the kernel sorts each row itself, see below),
 * so one and the same kernel covers, bit for bit, both _pack_quota_wire entry points:
 * sorted_rows=True (the seam; the row sort is the identity when rows are already
 * ascending) and =False (fused forward; one in-row sort).
 *
 * ========== Why the two-pass method matches the stable order bit for bit ==========
 *
 * The live chain's row order is fixed by plan_ta2a: the occupancy table occ
 * [T, n_nodes] is flattened in **node-major order** (node_first = occ.t().reshape(-1));
 * the ascending positions of the set bits are the send rows -- rows are laid out in
 * (ascending node, ascending token) order; u_src = sel % T, counts[n] = number of tokens
 * touching n. The quota branch's searchsorted compaction and the generic branch's argsort
 * compaction are bitwise identical
 * (tests/test_ta2a.py::test_fastpath_sortfree_construction_is_bitwise_equal).
 *
 * The kernel computes the same mathematical object, enumerating in units of runs (under
 * equal quota, a token's k experts, sorted ascending, split into exactly M segments of
 * quota each, every segment landing on a single node -- the invariant plan_ta2a verifies
 * on its first call and every 256 calls):
 *   run enumeration order p = t*M + j (ascending t, ascending in-row segment id j); a
 *   run's destination node d(t,j) = e_sorted[t, j*quota] / slots, strictly increasing in
 *   j (ascending row => non-decreasing node, equal quota => exactly one segment per node
 *   => strictly increasing).
 *
 *   Pass 1 (count): scan the **slot plane** in full (f = 0..T*k-1), slotHist[e/slots]++,
 *     snapshotting at f == t0*k the partial histogram mineSlots[] of everything before
 *     this core's range. Under equal quota, node d's slot count = quota * (number of
 *     touched tokens), so the divisions hist[d] = slotHist[d]/quota and
 *     mine[d] = mineSlots[d]/quota are **exact** (they divide evenly), and no row sorting
 *     is needed -- pass 1 has zero sorting and zero inter-core sync (same as K1: the full
 *     scan lets each core independently compute the global bucket bases base[d] and its
 *     own cursor starts cur[d] = base[d] + mine[d]; no SyncAll / no workspace cursor
 *     table). The cost is one extra read of the expert_idx plane per core (T*k*8B; tens
 *     of thousands = a few hundred KB), a rounding error next to the [T*M, H] payload
 *     copy (4-16KB per row).
 *   Pass 2 (pack and write rows): each core scans its own [t0, t1) in token order,
 *     insertion-sorts each row (k <= 64, scalar; expert ids are distinct within a row, so
 *     the permutation is unique -- the same permutation as _pack_quota_wire's float32-key
 *     argsort, see the <2^24 exactness argument in ta2a_fwd), and for each run:
 *     dst = cur[d]++, writing u_src[dst] = t, mask[dst, i] = e_sorted[jq+i] % slots
 *     (ascending, the C1 wire contract), gate_rows[dst, i] = element jq+i of the gates
 *     row under the same permutation (pure bit move), payload[dst, :] = hidden[t, :].
 *
 * Bitwise-identity proof: for any node bucket d, the runs written into it are exactly the
 * runs with d(t,j)==d; cores partition contiguous ranges in ascending token order and
 * scan in ascending (t, j) within a core, and a core's cursor start in bucket d skips
 * exactly the same-bucket runs with smaller t (mine[d] counts precisely the token < t0
 * part); under equal quota each token has at most one run per node, so the in-bucket
 * landing order == ascending token == the plan's in-bucket order; buckets follow
 * ascending base[] == ascending node. Hence the dst permutation equals the plan's sel
 * enumeration element for element => payload/u_src/node_counts are bitwise identical.
 * mask/gate_rows: _pack_quota_wire's row-scatter target
 * rof[t*M+j] = inverse[t, order[j*quota]] is exactly the send row of (t, d(t,j)) ==
 * dst(t,j), and the scattered values = the slot/gate segments after the row's ascending
 * permutation == what the kernel writes; gates see no floating-point arithmetic anywhere
 * (bit moves), bitwise identical to torch's gather.
 *
 * ============================ 910C hardware boundary ============================
 *
 *   - No int64 vector arithmetic/shifts/cheap sorting (internal engineering records;
 *     roadmap "hardware negatives"). This kernel emits no int64 **vector** instructions:
 *     the int64 planes (expert_idx/mask/u_src/node_counts) are all accessed as int32
 *     pairs (little-endian lo/hi) via scalar reads/writes -- expert ids < n_experts <
 *     2^31 (host-enforced; upstream in the live chain the float32 key further caps
 *     them < 2^24), token ids < T < 2^31, counts <= T*M < 2^31, the high 32 bits are
 *     always 0, so reading the low word and writing (lo, 0) is a bit-complete
 *     non-negative int64.
 *   - The only sort is the in-row k-element insertion sort (scalar unit, k <= 64); the
 *     bucket sort is avoided entirely (cursor method).
 *   - Payload rows move via DataCopy (GM->UB->GM, double-buffered queues, same pattern as
 *     K1/passthrough); host tiling guarantees H*sizeof(dtype) is a multiple of 32B
 *     (always true for the real hidden sizes 2048/7168), so no DataCopyPad is needed.
 *   - Pass 1's expert_idx scan goes through UB staging (DataCopy the int32 view into UB,
 *     then scalar reads); the tail of <4 int64s short of 32B alignment degrades to scalar
 *     GM reads -- no out-of-bounds reads. Pass 2 reads only this core's range
 *     (1/usedCores of the plane) via direct scalar GM reads, never contending with
 *     pass 1's staging buffer.
 *
 * Containment of corrupted inputs (no bitwise promise, only a no-overrun promise):
 * out-of-range expert ids (<0 or >= n_experts) are skipped under the same criterion in
 * both passes; if the equal-quota invariant drifts (the unverified window between
 * plan_ta2a's every-256-calls checks), cursors may cross bucket boundaries, and any write
 * with dst >= nRows skips the whole row -- on the torch side mask/gate_rows/u_src/
 * node_counts are allocated as zeros (_pack_quota_wire's original zeros containment: a
 * missing row == slot 0/gate 0, shape-harmless), while the live composed chain in the
 * same window dies loudly at the downstream gather on a searchsorted out-of-range row id
 * -- neither containment silently fabricates data; gate passage is judged on bitwise
 * equivalence for well-formed inputs.
 *
 * Cluster-compile verification points ([V1]-[V4] identical to K1; conservative form
 * everywhere we were unsure):
 *   [V1] PipeBarrier<PIPE_ALL>: sledgehammer sync between staging and scalar reads. If
 *        this overload is unavailable in this CANN drop, switch to the
 *        SetFlag/WaitFlag<HardEvent::MTE2_S> and S_MTE2 pair.
 *   [V2] GlobalTensor<int32_t>::GetValue/SetValue scalar GM access; if a particular drop
 *        requires an explicit dcache flush for host visibility, add
 *        DataCacheCleanAndInvalid(ENTIRE_DATA_CACHE) at the end of Process().
 *   [V3] Scalar GM writes and MTE3 DataCopy writes target disjoint address ranges, so no
 *        aliasing conflict; if bitwise verification shows u_src/mask/node_counts
 *        occasionally holding stale values, check [V2] first.
 *   [V4] When DTYPE_GATES is bf16, move it as uint16 bits (sizeof branch), never touching
 *        bf16 scalar arithmetic semantics -- a pure move, bitwise identical to torch's
 *        gather.
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;      // double buffer for payload row copy
constexpr uint32_t MAX_NODES = 256;    // node-bucket cap (host tiling enforces the same value)
constexpr uint32_t MAX_K = 64;         // in-row sort array cap (host tiling enforces the same value)

class KernelTerraceK2Pack {
public:
    __aicore__ inline KernelTerraceK2Pack() {}

    __aicore__ inline void Init(GM_ADDR hidden, GM_ADDR expertIdx, GM_ADDR gates,
                                GM_ADDR payload, GM_ADDR mask, GM_ADDR gateRows,
                                GM_ADDR uSrc, GM_ADDR nodeCounts,
                                uint32_t tokens, uint32_t hiddenW, uint32_t topk,
                                uint32_t quota, uint32_t groupsM, uint32_t slots,
                                uint32_t nNodes, uint32_t nExperts, uint32_t nRows,
                                uint32_t tokensPerCoreBase, uint32_t tokensRem,
                                uint32_t idxChunk, uint32_t rowTile)
    {
        this->tokens = tokens;
        this->hiddenW = hiddenW;
        this->topk = topk;
        this->quota = quota;
        this->groupsM = groupsM;
        this->slots = slots;
        this->nNodes = nNodes;
        this->nExperts = nExperts;
        this->nRows = nRows;
        this->idxChunk = idxChunk;
        this->rowTile = rowTile;

        // This core's contiguous token range [t0, t1): the first tokensRem cores get 1 extra each (matches host).
        uint32_t idx = static_cast<uint32_t>(GetBlockIdx());
        this->t0 = idx * tokensPerCoreBase + (idx < tokensRem ? idx : tokensRem);
        this->t1 = this->t0 + tokensPerCoreBase + (idx < tokensRem ? 1u : 0u);

        hiddenGm.SetGlobalBuffer((__gm__ DTYPE_HIDDEN *)hidden,
                                 static_cast<uint64_t>(tokens) * hiddenW);
        payloadGm.SetGlobalBuffer((__gm__ DTYPE_PAYLOAD *)payload,
                                  static_cast<uint64_t>(nRows) * hiddenW);
        // int32 views of the int64 planes (little-endian lo/hi pairs; see "hardware boundary" in the file header).
        idxGm32.SetGlobalBuffer((__gm__ int32_t *)expertIdx,
                                static_cast<uint64_t>(tokens) * topk * 2);
        maskGm32.SetGlobalBuffer((__gm__ int32_t *)mask,
                                 static_cast<uint64_t>(nRows) * quota * 2);
        usrcGm32.SetGlobalBuffer((__gm__ int32_t *)uSrc,
                                 static_cast<uint64_t>(nRows) * 2);
        countsGm32.SetGlobalBuffer((__gm__ int32_t *)nodeCounts,
                                   static_cast<uint64_t>(nNodes) * 2);
        // The gate plane moves as integer bit patterns by width ([V4]): build both views, pick by sizeof at run time.
        uint64_t flat = static_cast<uint64_t>(tokens) * topk;
        gatesGm16.SetGlobalBuffer((__gm__ uint16_t *)gates, flat);
        gateOutGm16.SetGlobalBuffer((__gm__ uint16_t *)gateRows, flat);
        gatesGm32.SetGlobalBuffer((__gm__ uint32_t *)gates, flat);
        gateOutGm32.SetGlobalBuffer((__gm__ uint32_t *)gateRows, flat);

        pipe.InitBuffer(rowQueue, BUFFER_NUM,
                        this->rowTile * sizeof(DTYPE_HIDDEN));
        // **The VECOUT queue is mandatory.** Measured on passthrough on 2026-08-24: a
        // `TQue`'s position decides which two pipelines it synchronizes -- VECIN pairs
        // MTE2->V; only VECOUT pairs V->MTE3. CopyRow is GM->UB->GM; with VECIN alone
        // nothing fences that MTE3, and dirty data gets copied out.
        // This bug propagated along the comment chain passthrough -> K1 -> K2, three
        // copies, all of them **silent**: it compiled, loaded, and ran; only bitwise
        // comparison exposed it (8x256, 2047/2048 wrong).
        pipe.InitBuffer(rowOutQueue, BUFFER_NUM,
                        this->rowTile * sizeof(DTYPE_HIDDEN));
        pipe.InitBuffer(idxStage, this->idxChunk * 2 * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        uint32_t slotHist[MAX_NODES];  // global slot histogram (every core independently computes the same one)
        uint32_t mineSlots[MAX_NODES]; // partial slot histogram for tokens < t0
        uint32_t cur[MAX_NODES];       // this core's write cursors (run granularity)
        for (uint32_t b = 0; b < nNodes; b++) {
            slotHist[b] = 0;
            mineSlots[b] = 0;
        }

        // ---- Pass 1: full count over the slot plane + snapshot at f == t0*topk (zero
        // inter-core sync, zero sorting -- under equal quota, run count = slot count /
        // quota, exact division; see file header) ----
        uint32_t flatTotal = tokens * topk;
        uint32_t f0 = t0 * topk;
        bool snapped = (f0 == 0);
        for (uint32_t chunkStart = 0; chunkStart < flatTotal;
             chunkStart += idxChunk) {
            uint32_t n = flatTotal - chunkStart;
            if (n > idxChunk) {
                n = idxChunk;
            }
            StageIdx(chunkStart, n);
            for (uint32_t i = 0; i < n; i++) {
                uint32_t f = chunkStart + i;
                if (!snapped && f == f0) {
                    for (uint32_t b = 0; b < nNodes; b++) {
                        mineSlots[b] = slotHist[b];
                    }
                    snapped = true;
                }
                int32_t e = IdxAt(chunkStart, i);
                if (e < 0 || static_cast<uint32_t>(e) >= nExperts) {
                    continue;   // out-of-range expert id: the live chain dies loudly
                }               // at plan's gather/scatter; the kernel skips to avoid
                                // overruns (corrupted input, no bitwise promise)
                slotHist[static_cast<uint32_t>(e) / slots]++;
            }
        }
        if (!snapped) {          // t0 == tokens: this core has no tokens; mine is only formally assigned
            for (uint32_t b = 0; b < nNodes; b++) {
                mineSlots[b] = slotHist[b];
            }
        }

        // Global bucket bases + this core's cursor starts (the key to stable order:
        // base[b] + mine[b]; proof in the file header). node_counts = the run histogram,
        // single-core write ([V3], same as K1's i_send).
        uint32_t base = 0;
        for (uint32_t b = 0; b < nNodes; b++) {
            uint32_t h = slotHist[b] / quota;          // divides evenly under equal quota, exact
            cur[b] = base + mineSlots[b] / quota;
            if (GetBlockIdx() == 0) {
                countsGm32.SetValue(2 * b, static_cast<int32_t>(h));
                countsGm32.SetValue(2 * b + 1, 0);
            }
            base += h;
        }

        // ---- Pass 2: over this core's range [t0, t1) in token order, sort each row + pack and write per run ----
        uint32_t eVal[MAX_K];    // in-row expert ids (int64 low words; high words always 0, see file header)
        uint32_t gBits[MAX_K];   // in-row raw gate bits (16/32-bit, pure move [V4])
        bool eOk[MAX_K];         // per-element in-row out-of-range flags (same criterion in both passes)
        for (uint32_t t = t0; t < t1; t++) {
            uint64_t rowBase = static_cast<uint64_t>(t) * topk;
            for (uint32_t j = 0; j < topk; j++) {
                int32_t lo = idxGm32.GetValue((rowBase + j) * 2);
                eOk[j] = !(lo < 0 || static_cast<uint32_t>(lo) >= nExperts);
                eVal[j] = static_cast<uint32_t>(lo);
                gBits[j] = (sizeof(DTYPE_GATES) == 2)
                               ? static_cast<uint32_t>(gatesGm16.GetValue(rowBase + j))
                               : gatesGm32.GetValue(rowBase + j);
            }
            // In-row insertion sort, ascending by expert id (distinct within a row =>
            // unique permutation, == the live chain's argsort(float32) permutation;
            // insertion sort is stable, so even corrupted inputs with duplicate ids stay
            // deterministic).
            for (uint32_t a = 1; a < topk; a++) {
                uint32_t ev = eVal[a];
                uint32_t gv = gBits[a];
                bool ok = eOk[a];
                int32_t b = static_cast<int32_t>(a) - 1;
                while (b >= 0 && eVal[b] > ev) {
                    eVal[b + 1] = eVal[b];
                    gBits[b + 1] = gBits[b];
                    eOk[b + 1] = eOk[b];
                    b--;
                }
                eVal[b + 1] = ev;
                gBits[b + 1] = gv;
                eOk[b + 1] = ok;
            }
            for (uint32_t j = 0; j < groupsM; j++) {
                uint32_t lead = j * quota;
                if (!eOk[lead]) {
                    continue;    // run leader out of range: same criterion as pass 1 (that slot never entered the histogram)
                }
                uint32_t d = eVal[lead] / slots;       // < nNodes (lead < nExperts)
                uint32_t dst = cur[d]++;
                if (dst >= nRows) {
                    continue;    // containment for quota-invariant drift: skip the write, no overrun (file header)
                }
                usrcGm32.SetValue(2 * static_cast<uint64_t>(dst),
                                  static_cast<int32_t>(t));
                usrcGm32.SetValue(2 * static_cast<uint64_t>(dst) + 1, 0);
                for (uint32_t i = 0; i < quota; i++) {
                    uint32_t p = lead + i;
                    uint64_t o = static_cast<uint64_t>(dst) * quota + i;
                    // Out-of-range members write slot 0/gate 0 (zeros containment, verbatim from _pack_quota_wire).
                    int32_t s = eOk[p] ? static_cast<int32_t>(eVal[p] % slots) : 0;
                    maskGm32.SetValue(2 * o, s);
                    maskGm32.SetValue(2 * o + 1, 0);
                    if (sizeof(DTYPE_GATES) == 2) {    // [V4] pure bit move
                        gateOutGm16.SetValue(o, eOk[p]
                            ? static_cast<uint16_t>(gBits[p]) : uint16_t(0));
                    } else {
                        gateOutGm32.SetValue(o, eOk[p] ? gBits[p] : 0u);
                    }
                }
                CopyRow(t, dst);
            }
        }
        // [V2] If bitwise verification shows the scalar-write planes occasionally holding stale values, add a dcache flush here.
    }

private:
    // Stage the flattened expert_idx chunk [chunkStart, chunkStart+n) into UB: DataCopy
    // the 32B-aligned part; the tail of <4 int64s is left to IdxAt's scalar GM reads. In
    // the int32 view, 4 int64 = 8 int32 = 32B. Used by pass 1 only (the full scan);
    // pass 2 reads only this core's 1/usedCores range via scalar GM reads.
    __aicore__ inline void StageIdx(uint32_t chunkStart, uint32_t n)
    {
        stagedAligned = n & ~3u;
        stagedBase = chunkStart;
        if (stagedAligned > 0) {
            LocalTensor<int32_t> sl = idxStage.Get<int32_t>();
            // [V1] Reused buffer: the previous chunk's scalar reads must complete before this MTE2 overwrite.
            PipeBarrier<PIPE_ALL>();
            DataCopy(sl, idxGm32[static_cast<uint64_t>(chunkStart) * 2],
                     stagedAligned * 2);
            // [V1] Scalar reads must wait for MTE2 to land.
            PipeBarrier<PIPE_ALL>();
        }
    }

    // Expert id at flattened slot chunkStart+i (int64 low word; high word always 0, see file header).
    __aicore__ inline int32_t IdxAt(uint32_t chunkStart, uint32_t i)
    {
        if (i < stagedAligned) {
            LocalTensor<int32_t> sl = idxStage.Get<int32_t>();
            return sl.GetValue(2 * i);
        }
        return idxGm32.GetValue((static_cast<uint64_t>(chunkStart) + i) * 2);
    }

    // Row srcTok of hidden -> row dst of payload: GM->UB->GM, split by rowTile, double
    // **Two queues**: VECIN handles MTE2->V, VECOUT handles V->MTE3. The
    // "K1/passthrough template" this originally copied had only one VECIN queue -- that
    // template itself was wrong.
    __aicore__ inline void CopyRow(uint32_t srcTok, uint32_t dst)
    {
        uint64_t srcBase = static_cast<uint64_t>(srcTok) * hiddenW;
        uint64_t dstBase = static_cast<uint64_t>(dst) * hiddenW;
        for (uint32_t t = 0; t < hiddenW; t += rowTile) {
            uint32_t len = hiddenW - t;
            if (len > rowTile) {
                len = rowTile;
            }
            LocalTensor<DTYPE_HIDDEN> inBuf = rowQueue.AllocTensor<DTYPE_HIDDEN>();
            DataCopy(inBuf, hiddenGm[srcBase + t], len);
            rowQueue.EnQue(inBuf);                  // MTE2 -> V

            inBuf = rowQueue.DeQue<DTYPE_HIDDEN>();
            LocalTensor<DTYPE_HIDDEN> outBuf = rowOutQueue.AllocTensor<DTYPE_HIDDEN>();
            DataCopy(outBuf, inBuf, len);           // UB -> UB
            rowOutQueue.EnQue(outBuf);              // V -> MTE3
            rowQueue.FreeTensor(inBuf);

            outBuf = rowOutQueue.DeQue<DTYPE_HIDDEN>();
            DataCopy(payloadGm[dstBase + t], outBuf, len);
            rowOutQueue.FreeTensor(outBuf);
        }
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> rowQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rowOutQueue;
    TBuf<TPosition::VECCALC> idxStage;

    GlobalTensor<DTYPE_HIDDEN> hiddenGm;
    GlobalTensor<DTYPE_PAYLOAD> payloadGm;
    GlobalTensor<int32_t> idxGm32, maskGm32, usrcGm32, countsGm32;
    GlobalTensor<uint16_t> gatesGm16, gateOutGm16;
    GlobalTensor<uint32_t> gatesGm32, gateOutGm32;

    uint32_t tokens = 0, hiddenW = 0, topk = 1, quota = 1, groupsM = 1;
    uint32_t slots = 1, nNodes = 1, nExperts = 1, nRows = 0;
    uint32_t idxChunk = 0, rowTile = 0;
    uint32_t t0 = 0, t1 = 0;
    uint32_t stagedAligned = 0, stagedBase = 0;
};

// Entry symbol: snake_case of the op type TerraceK2Pack (msopgen skeleton convention).
// The DTYPE_HIDDEN / DTYPE_GATES / DTYPE_PAYLOAD macros are injected by the build system
// when instantiating per dtype combination.
extern "C" __global__ __aicore__ void terrace_k2_pack(
    GM_ADDR hidden, GM_ADDR expert_idx, GM_ADDR gates, GM_ADDR payload,
    GM_ADDR mask, GM_ADDR gate_rows, GM_ADDR u_src, GM_ADDR node_counts,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.nRows == 0) {
        return;                       // T == 0: all five outputs empty/zero; host has already allocated by shape
    }
    KernelTerraceK2Pack op;
    op.Init(hidden, expert_idx, gates, payload, mask, gate_rows, u_src, node_counts,
            tilingData.tokens, tilingData.hidden, tilingData.topk, tilingData.quota,
            tilingData.groupsM, tilingData.slots, tilingData.nNodes,
            tilingData.nExperts, tilingData.nRows, tilingData.tokensPerCoreBase,
            tilingData.tokensRem, tilingData.idxChunk, tilingData.rowTile);
    op.Process();
}
