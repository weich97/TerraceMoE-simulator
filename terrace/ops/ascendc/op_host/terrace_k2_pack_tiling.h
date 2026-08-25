/**
 * terrace_k2_pack -- host-side tiling data definition (K2: the send-side pack chain).
 *
 * All uint32 (avoid int64 planes on 910C; same discipline as K1). Bound
 * accounting:
 *   flat = tokens * topk and nRows * quota (= flat) are host-limited to < 2^31;
 *   the payload element total nRows * hidden can exceed 2^32, so every GM
 *   offset in the kernel accumulates products in uint64, and tiling stores no
 *   product fields (the 2x subscript of the int64 plane likewise uses uint64).
 */
#ifndef TERRACE_K2_PACK_TILING_H
#define TERRACE_K2_PACK_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TerraceK2PackTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, tokens);            // T: expert_idx dim 0
    TILING_DATA_FIELD_DEF(uint32_t, hidden);            // H: hidden row width (elements)
    TILING_DATA_FIELD_DEF(uint32_t, topk);              // k: expert_idx dim 1
    TILING_DATA_FIELD_DEF(uint32_t, quota);             // k / groups_m (equal quota)
    TILING_DATA_FIELD_DEF(uint32_t, groupsM);           // M: nodes touched per token
    TILING_DATA_FIELD_DEF(uint32_t, epr);               // experts per rank
    TILING_DATA_FIELD_DEF(uint32_t, rpn);               // ranks per node
    TILING_DATA_FIELD_DEF(uint32_t, slots);             // epr * rpn (<= 63, hard bound of the current chain)
    TILING_DATA_FIELD_DEF(uint32_t, nNodes);            // world / rpn (bucket count)
    TILING_DATA_FIELD_DEF(uint32_t, nExperts);          // total expert count (slot-id out-of-range criterion)
    TILING_DATA_FIELD_DEF(uint32_t, nRows);             // T * M: send-buffer row count (static)
    TILING_DATA_FIELD_DEF(uint32_t, tokensPerCoreBase); // T / usedCores
    TILING_DATA_FIELD_DEF(uint32_t, tokensRem);         // T % usedCores (first rem cores get one extra)
    TILING_DATA_FIELD_DEF(uint32_t, idxChunk);          // int64 count staged per pass-1 step (multiple of 4)
    TILING_DATA_FIELD_DEF(uint32_t, rowTile);           // UB tile element count for payload-row copies
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TerraceK2Pack, TerraceK2PackTilingData)

}  // namespace optiling

#endif  // TERRACE_K2_PACK_TILING_H
