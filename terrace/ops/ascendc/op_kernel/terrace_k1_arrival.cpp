/**
 * terrace_k1_arrival -- AscendC kernel: T-A2A arrival-side fused chain (K1, C1 quota wire format).
 *
 * ===================== Functional spec (bitwise replication) =====================
 *
 * Replicates the live composed chain in terrace/ta2a_fwd.py::ta2a_moe_forward and
 * ta2a_dispatch.py::ta2a_permute between C1 and Hop B (quota fast-path branch):
 *
 *     r_idx, slot_idx = _expand_arrival_quota(rslot)      # pair expansion (the slot table IS the pair table)
 *     owner = slot_idx // epr                             # destination local rank (bucket id)
 *     ordo  = _stable_argsort_small(owner, rpn)           # stable ascending (owner bucket sort)
 *     r_idx, slot_idx = r_idx[ordo], slot_idx[ordo]       # pair reordering
 *     i_send = torch.bincount(owner, minlength=rpn)       # rows per peer (permutation-blind)
 *     send_buf   = rx[r_idx]                              # [P, H] expanded send buffer
 *     gate_pairs = rgate.reshape(-1)[ordo]                # gates in pair order (dense table flattened)
 *
 * Inputs: rx [R, H] (fp16/bf16/fp32), rslot [R, quota] (int64, C1 wire format: ascending
 * slot ids per row), rgate [R, quota] (same dtype as rx); attributes quota/epr/rpn/my_local.
 * Outputs: send_buf [P, H], gate_pairs [P], r_idx [P] int64, slot_idx [P] int64,
 * i_send [rpn] int64, P = R*quota. my_local is unused by the math in this stage -- the
 * interface reserves it per the K1 spec for the expert-order regrouping half-stage after
 * Hop B of the arrival chain (the isomorphic bucket sort on exp_j = my_slot - my_local*epr,
 * enabled later if that half gets fused as a second form of this same kernel); the tiling
 * already carries it.
 *
 * ========== Why the two-pass method matches the stable order bit for bit ==========
 *
 * The live chain's sorting primitive _stable_argsort_small(owner, rpn) is a **stable
 * ascending sort**: output order = buckets in ascending owner order, within a bucket
 * ascending by the pair's flattened position p (= r*quota + i). It is implemented via the
 * composite key pos + n*key (float32 is exact for integers < 2^24, otherwise it falls back
 * to an integer stable sort); both paths yield **the same permutation**, so the kernel only
 * has to replicate the mathematical object "stable bucket sort" itself.
 *
 * Two-pass method (counting sort):
 *   Pass 1 (count): scan rslot in pair-flattened order p = 0..P-1, owner(p) = slot(p)/epr,
 *     producing the histogram hist[rpn] (== i_send; bincount is permutation-blind). With
 *     multiple cores each core owns one contiguous pair range [p0, p1), but **every core
 *     scans the full range [0, P)**, snapshotting at p == p0 the partial histogram mine[]
 *     of "everything before this core's range". What the full scan buys: each core
 *     independently computes the global bucket bases base[b] = sum_{b'<b} hist[b'] and its
 *     own cursor starts cur[b] = base[b] + mine[b], with **zero inter-core synchronization**
 *     (no SyncAll / no workspace cursor table -- smallest version-divergence surface).
 *     The cost is one extra read of the rslot plane per core (P*8B; tens of thousands of
 *     pairs = a few hundred KB), a rounding error next to the [P, H] payload copy
 *     (4-16KB per row).
 *   Pass 2 (expand and write rows): each core scans its own [p0, p1) in flattened order,
 *     dst = cur[owner(p)]++, writing r_idx[dst] = p / quota, slot_idx[dst] = slot(p),
 *     gate_pairs[dst] = element p of flattened rgate, send_buf[dst, :] = rx[p / quota, :].
 *
 * Bitwise-identity proof: for any bucket b, the pairs written into it are exactly the pairs
 * with owner==b; cores partition contiguous ranges in ascending p0 order and scan in
 * ascending p within a core, and a core's cursor start in bucket b skips exactly the
 * same-bucket pairs with smaller p (mine[b] counts precisely the owner==b pairs in
 * [0, p0)), so the in-bucket landing order == ascending flattened position p == the
 * in-bucket order of the live chain's stable sort; buckets are laid out in ascending
 * base[] order == ascending owner. Hence the dst permutation equals ordo element for
 * element, and all five outputs are bitwise identical. r_idx comes from p / quota rather
 * than a table lookup: _expand_arrival_quota's r_idx is just arange(R) expanded by quota,
 * so the row id of pair p is always p / quota, which after the ordo reorder is
 * ordo[j] / quota.
 *
 * ============================ 910C hardware boundary ============================
 *
 *   - No int64 vector arithmetic/shifts/cheap sorting (internal engineering records;
 *     roadmap "hardware negatives"). This kernel emits no int64 **vector** instructions:
 *     the int64 planes (rslot/r_idx/slot_idx/i_send) are all accessed as int32 pairs
 *     (little-endian lo/hi) via scalar reads/writes -- slot ids < epr*rpn <= 63, row ids
 *     < R < 2^31, counts <= P < 2^31, the high 32 bits are always 0, so reading the low
 *     word and writing (lo, 0) is a bit-complete non-negative int64. The scalar unit
 *     itself is a general-purpose core; uint32/uint64 scalar arithmetic is not constrained
 *     by the vector ISA.
 *   - Sorting is avoided entirely (cursor method); owner is just a scalar division.
 *   - Payload rows move via DataCopy (GM->UB->GM, double-buffered queues, same pattern as
 *     the passthrough template); host tiling guarantees H*sizeof(dtype) is a multiple of
 *     32B (always true for the real hidden sizes 2048/7168), rowTile alignment is
 *     guaranteed by the host, so no DataCopyPad is needed.
 *   - The rslot scan goes through UB staging (DataCopy the int32 view into UB, then scalar
 *     reads); the tail of <4 pairs short of 32B alignment degrades to scalar GM reads --
 *     no out-of-bounds reads.
 *
 * Cluster-compile verification points (everywhere we were unsure we chose the conservative
 * form, tagged below):
 *   [V1] PipeBarrier<PIPE_ALL>: sledgehammer sync between staging and scalar reads. If this
 *        overload is unavailable in this CANN drop, switch to the
 *        SetFlag/WaitFlag<HardEvent::MTE2_S> and S_MTE2 pair.
 *   [V2] GlobalTensor<int32_t>::GetValue/SetValue scalar GM access; if a particular drop
 *        requires an explicit dcache flush for host visibility, add
 *        DataCacheCleanAndInvalid(ENTIRE_DATA_CACHE) at the end of Process().
 *   [V3] Scalar GM writes and MTE3 DataCopy writes target disjoint address ranges, so no
 *        aliasing conflict; if bitwise verification shows i_send/r_idx occasionally holding
 *        stale values, check [V2] first.
 *   [V4] When DTYPE_RGATE is bf16, move it as uint16 bits (sizeof branch), never touching
 *        bf16 scalar arithmetic semantics -- a pure move, bitwise identical to torch's
 *        gather.
 */
#include "kernel_operator.h"
// The tiling struct is a plain C struct shared by host and kernel (CANN 9.0.0 ASC system).
// Without this line, TerraceK1ArrivalTilingData is undefined when GET_TILING_DATA expands;
// the kernel compile fails inside the binary sub-build and the main log only shows the
// downstream "Target path not found".
#include "terrace_k1_arrival_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;      // double buffer for payload row copy
constexpr uint32_t MAX_RPN = 64;       // bucket-count cap: slots = epr*rpn <= 63 (hard constraint of the live chain)

class KernelTerraceK1Arrival {
public:
    __aicore__ inline KernelTerraceK1Arrival() {}

    __aicore__ inline void Init(GM_ADDR rx, GM_ADDR rslot, GM_ADDR rgate,
                                GM_ADDR sendBuf, GM_ADDR gatePairs, GM_ADDR rIdx,
                                GM_ADDR slotIdx, GM_ADDR iSend,
                                uint32_t rows, uint32_t hidden, uint32_t quota,
                                uint32_t epr, uint32_t rpn, uint32_t pairCount,
                                uint32_t pairsPerCoreBase, uint32_t pairsRem,
                                uint32_t slotChunk, uint32_t rowTile)
    {
        this->hidden = hidden;
        this->quota = quota;
        this->epr = epr;
        this->rpn = rpn;
        this->pairCount = pairCount;
        this->slotChunk = slotChunk;
        this->rowTile = rowTile;

        // This core's contiguous pair range [p0, p1): the first pairsRem cores get 1 extra each (matches host).
        uint32_t idx = static_cast<uint32_t>(GetBlockIdx());
        this->p0 = idx * pairsPerCoreBase + (idx < pairsRem ? idx : pairsRem);
        this->p1 = this->p0 + pairsPerCoreBase + (idx < pairsRem ? 1u : 0u);

        rxGm.SetGlobalBuffer((__gm__ DTYPE_RX *)rx,
                             static_cast<uint64_t>(rows) * hidden);
        sendGm.SetGlobalBuffer((__gm__ DTYPE_SEND_BUF *)sendBuf,
                               static_cast<uint64_t>(pairCount) * hidden);
        // int32 views of the int64 planes (little-endian lo/hi pairs; see "hardware boundary" in the file header).
        rslotGm32.SetGlobalBuffer((__gm__ int32_t *)rslot,
                                  static_cast<uint64_t>(pairCount) * 2);
        ridxGm32.SetGlobalBuffer((__gm__ int32_t *)rIdx,
                                 static_cast<uint64_t>(pairCount) * 2);
        slotIdxGm32.SetGlobalBuffer((__gm__ int32_t *)slotIdx,
                                    static_cast<uint64_t>(pairCount) * 2);
        isendGm32.SetGlobalBuffer((__gm__ int32_t *)iSend,
                                  static_cast<uint64_t>(rpn) * 2);
        // The gate plane moves as integer bit patterns by width ([V4]): build both views, pick by sizeof at run time.
        rgateGm16.SetGlobalBuffer((__gm__ uint16_t *)rgate, pairCount);
        gateOutGm16.SetGlobalBuffer((__gm__ uint16_t *)gatePairs, pairCount);
        rgateGm32.SetGlobalBuffer((__gm__ uint32_t *)rgate, pairCount);
        gateOutGm32.SetGlobalBuffer((__gm__ uint32_t *)gatePairs, pairCount);

        pipe.InitBuffer(rowQueue, BUFFER_NUM,
                        this->rowTile * sizeof(DTYPE_RX));
        // **The VECOUT queue is mandatory.** Measured on passthrough on 2026-08-24:
        // a `TQue`'s position decides which two pipelines it synchronizes -- VECIN's
        // EnQue/DeQue pairs MTE2->V; only VECOUT pairs V->MTE3. CopyRow is GM->UB->GM;
        // with VECIN alone nothing fences that MTE3, and what gets copied out is data not
        // yet fully landed or already recycled by the next round's AllocTensor.
        // The symptom on passthrough: **mostly-zero output** (8x256, 2047/2048 wrong).
        pipe.InitBuffer(rowOutQueue, BUFFER_NUM,
                        this->rowTile * sizeof(DTYPE_RX));
        pipe.InitBuffer(slotStage, this->slotChunk * 2 * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        uint32_t hist[MAX_RPN];    // global histogram (every core independently computes the same one)
        uint32_t mine[MAX_RPN];    // partial histogram of [0, p0) (this core's per-bucket cursor-start offset)
        uint32_t cur[MAX_RPN];     // this core's write cursors
        for (uint32_t b = 0; b < rpn; b++) {
            hist[b] = 0;
            mine[b] = 0;
        }

        // ---- Pass 1: full count + snapshot the partial histogram at p == p0 (zero inter-core sync) ----
        bool snapped = (p0 == 0);
        for (uint32_t chunkStart = 0; chunkStart < pairCount;
             chunkStart += slotChunk) {
            uint32_t n = pairCount - chunkStart;
            if (n > slotChunk) {
                n = slotChunk;
            }
            StageSlots(chunkStart, n);
            for (uint32_t i = 0; i < n; i++) {
                uint32_t p = chunkStart + i;
                if (!snapped && p == p0) {
                    for (uint32_t b = 0; b < rpn; b++) {
                        mine[b] = hist[b];
                    }
                    snapped = true;
                }
                int32_t s = SlotAt(chunkStart, i);
                if (s < 0 || static_cast<uint32_t>(s) >= epr * rpn) {
                    continue;   // out-of-range slot id: the live chain dies loudly in
                }               // bincount/gather; the kernel skips it to avoid writing
                                // out of bounds (corrupted input; no bitwise promise)
                hist[static_cast<uint32_t>(s) / epr]++;
            }
        }
        if (!snapped) {          // p0 == pairCount: this core has no pairs; mine is only formally assigned
            for (uint32_t b = 0; b < rpn; b++) {
                mine[b] = hist[b];
            }
        }

        // Global bucket bases + this core's cursor starts (the key to stable order: base[b] + mine[b]; proof in the file header).
        uint32_t base = 0;
        for (uint32_t b = 0; b < rpn; b++) {
            cur[b] = base + mine[b];
            base += hist[b];
        }

        // i_send = the histogram (bincount is permutation-blind). Single-core write, to
        // avoid multi-core scalar writes to the same address ([V3]).
        if (GetBlockIdx() == 0) {
            for (uint32_t b = 0; b < rpn; b++) {
                isendGm32.SetValue(2 * b, static_cast<int32_t>(hist[b]));
                isendGm32.SetValue(2 * b + 1, 0);
            }
        }

        // ---- Pass 2: expand and write rows over this core's range [p0, p1) in flattened order ----
        // Chunks follow the global alignment grid (integer multiples of slotChunk; slotChunk
        // is a multiple of 4): p0 itself is not guaranteed 4-aligned, and staging straight
        // from p0 would put the DataCopy GM source address on an 8B rather than a 32B
        // boundary. On the grid, each chunk's GM start = chunkStart*8B with
        // chunkStart % 4 == 0 always true, so 32B alignment always holds; this core only
        // consumes the intersection with [p0, p1) inside each chunk.
        for (uint32_t chunkStart = (p0 / slotChunk) * slotChunk; chunkStart < p1;
             chunkStart += slotChunk) {
            uint32_t n = pairCount - chunkStart;
            if (n > slotChunk) {
                n = slotChunk;
            }
            StageSlots(chunkStart, n);
            uint32_t iLo = (p0 > chunkStart) ? (p0 - chunkStart) : 0u;
            uint32_t iHi = (p1 - chunkStart < n) ? (p1 - chunkStart) : n;
            for (uint32_t i = iLo; i < iHi; i++) {
                uint32_t p = chunkStart + i;
                int32_t s = SlotAt(chunkStart, i);
                if (s < 0 || static_cast<uint32_t>(s) >= epr * rpn) {
                    continue;   // same skip criterion as pass 1; the two passes agree, so cursors neither miss nor double-count
                }
                uint32_t b = static_cast<uint32_t>(s) / epr;
                uint32_t dst = cur[b]++;
                uint32_t srcRow = p / quota;
                // int64 outputs = (lo, hi=0) int32 pairs ([V2]).
                ridxGm32.SetValue(2 * dst, static_cast<int32_t>(srcRow));
                ridxGm32.SetValue(2 * dst + 1, 0);
                slotIdxGm32.SetValue(2 * dst, s);
                slotIdxGm32.SetValue(2 * dst + 1, 0);
                if (sizeof(DTYPE_RGATE) == 2) {          // [V4] pure bit move
                    gateOutGm16.SetValue(dst, rgateGm16.GetValue(p));
                } else {
                    gateOutGm32.SetValue(dst, rgateGm32.GetValue(p));
                }
                CopyRow(srcRow, dst);
            }
        }
        // [V2] If bitwise verification shows the scalar-write planes occasionally holding stale values, add a dcache flush here.
    }

private:
    // Stage the rslot pair chunk [chunkStart, chunkStart+n) into UB: DataCopy the
    // 32B-aligned part; the tail of <4 pairs is left to SlotAt's scalar GM reads.
    // In the int32 view, 4 pairs = 8 int32 = 32B.
    __aicore__ inline void StageSlots(uint32_t chunkStart, uint32_t n)
    {
        stagedAligned = n & ~3u;
        if (stagedAligned > 0) {
            LocalTensor<int32_t> sl = slotStage.Get<int32_t>();
            // [V1] Reused buffer: the previous chunk's scalar reads must complete before this MTE2 overwrite.
            PipeBarrier<PIPE_ALL>();
            DataCopy(sl, rslotGm32[static_cast<uint64_t>(chunkStart) * 2],
                     stagedAligned * 2);
            // [V1] Scalar reads must wait for MTE2 to land.
            PipeBarrier<PIPE_ALL>();
        }
    }

    // Slot id of pair chunkStart+i (int64 low word; high word always 0, see file header).
    __aicore__ inline int32_t SlotAt(uint32_t chunkStart, uint32_t i)
    {
        if (i < stagedAligned) {
            LocalTensor<int32_t> sl = slotStage.Get<int32_t>();
            return sl.GetValue(2 * i);
        }
        return rslotGm32.GetValue((static_cast<uint64_t>(chunkStart) + i) * 2);
    }

    // Row srcRow of rx -> row dst of send_buf: GM->UB->GM, split by rowTile, double buffered.
    // **Two queues, not one**: VECIN handles MTE2->V, VECOUT handles V->MTE3. With only a
    // VECIN queue, nobody inserts that MTE3 barrier (see the comment at InitBuffer).
    // The middle UB->UB copy is not waste -- it is the link that pairs the two queues;
    // and with double buffering, tile i+1's MTE2 can overlap tile i's MTE3, which is
    // faster than serializing with PipeBarrier<PIPE_ALL> (StageSlots uses PipeBarrier
    // because it is a TBuf and moves only a small chunk, off the bandwidth path).
    __aicore__ inline void CopyRow(uint32_t srcRow, uint32_t dst)
    {
        uint64_t srcBase = static_cast<uint64_t>(srcRow) * hidden;
        uint64_t dstBase = static_cast<uint64_t>(dst) * hidden;
        for (uint32_t t = 0; t < hidden; t += rowTile) {
            uint32_t len = hidden - t;
            if (len > rowTile) {
                len = rowTile;
            }
            LocalTensor<DTYPE_RX> inBuf = rowQueue.AllocTensor<DTYPE_RX>();
            DataCopy(inBuf, rxGm[srcBase + t], len);
            rowQueue.EnQue(inBuf);                  // MTE2 -> V

            inBuf = rowQueue.DeQue<DTYPE_RX>();
            LocalTensor<DTYPE_RX> outBuf = rowOutQueue.AllocTensor<DTYPE_RX>();
            DataCopy(outBuf, inBuf, len);           // UB -> UB
            rowOutQueue.EnQue(outBuf);              // V -> MTE3
            rowQueue.FreeTensor(inBuf);

            outBuf = rowOutQueue.DeQue<DTYPE_RX>();
            DataCopy(sendGm[dstBase + t], outBuf, len);
            rowOutQueue.FreeTensor(outBuf);
        }
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> rowQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rowOutQueue;
    TBuf<TPosition::VECCALC> slotStage;

    GlobalTensor<DTYPE_RX> rxGm;
    GlobalTensor<DTYPE_SEND_BUF> sendGm;
    GlobalTensor<int32_t> rslotGm32, ridxGm32, slotIdxGm32, isendGm32;
    GlobalTensor<uint16_t> rgateGm16, gateOutGm16;
    GlobalTensor<uint32_t> rgateGm32, gateOutGm32;

    uint32_t hidden = 0, quota = 1, epr = 1, rpn = 1;
    uint32_t pairCount = 0, slotChunk = 0, rowTile = 0;
    uint32_t p0 = 0, p1 = 0;
    uint32_t stagedAligned = 0;
};

// Entry symbol: snake_case of the op type TerraceK1Arrival (msopgen skeleton convention).
// The DTYPE_RX / DTYPE_RGATE / DTYPE_SEND_BUF macros are injected by the build system when
// instantiating per dtype combination.
extern "C" __global__ __aicore__ void terrace_k1_arrival(
    GM_ADDR rx, GM_ADDR rslot, GM_ADDR rgate, GM_ADDR send_buf, GM_ADDR gate_pairs,
    GM_ADDR r_idx, GM_ADDR slot_idx, GM_ADDR i_send, GM_ADDR workspace,
    GM_ADDR tiling)
{
    // The ASC system requires the kernel to declare this op's tiling struct explicitly, so
    // the GET_TILING_DATA macro expansion (generated per SOC at build time) knows which
    // type to deserialize into.
    REGISTER_TILING_DEFAULT(TerraceK1ArrivalTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.pairCount == 0) {
        return;                       // R == 0: all five outputs empty/zero; host has already allocated by shape
    }
    KernelTerraceK1Arrival op;
    op.Init(rx, rslot, rgate, send_buf, gate_pairs, r_idx, slot_idx, i_send,
            tilingData.rows, tilingData.hidden, tilingData.quota, tilingData.epr,
            tilingData.rpn, tilingData.pairCount, tilingData.pairsPerCoreBase,
            tilingData.pairsRem, tilingData.slotChunk, tilingData.rowTile);
    op.Process();
}
