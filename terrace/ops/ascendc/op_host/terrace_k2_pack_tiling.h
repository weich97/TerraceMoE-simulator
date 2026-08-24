/**
 * terrace_k2_pack -- host 侧 tiling 数据定义(K2:发送侧打包链)。
 *
 * 全部 uint32(910C 上避免 int64 平面;与 K1 同纪律)。上限核算:
 *   flat = tokens * topk 与 nRows * quota(= flat)由 host 限 < 2^31;
 *   payload 元素总数 nRows * hidden 可能超 2^32,kernel 内所有 GM 偏移一律
 *   uint64 累乘,tiling 里不存任何乘积字段(int64 平面的 2x 下标同理 uint64)。
 */
#ifndef TERRACE_K2_PACK_TILING_H
#define TERRACE_K2_PACK_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TerraceK2PackTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, tokens);            // T:expert_idx 第 0 维
    TILING_DATA_FIELD_DEF(uint32_t, hidden);            // H:hidden 行宽(元素数)
    TILING_DATA_FIELD_DEF(uint32_t, topk);              // k:expert_idx 第 1 维
    TILING_DATA_FIELD_DEF(uint32_t, quota);             // k / groups_m(等额配额)
    TILING_DATA_FIELD_DEF(uint32_t, groupsM);           // M:每 token 触达节点数
    TILING_DATA_FIELD_DEF(uint32_t, epr);               // 每 rank 专家数
    TILING_DATA_FIELD_DEF(uint32_t, rpn);               // 节点内 rank 数
    TILING_DATA_FIELD_DEF(uint32_t, slots);             // epr * rpn(<= 63,现链硬约束)
    TILING_DATA_FIELD_DEF(uint32_t, nNodes);            // world / rpn(桶数)
    TILING_DATA_FIELD_DEF(uint32_t, nExperts);          // 专家总数(槽号越界判据)
    TILING_DATA_FIELD_DEF(uint32_t, nRows);             // T * M:发送缓冲行数(静态)
    TILING_DATA_FIELD_DEF(uint32_t, tokensPerCoreBase); // T / usedCores
    TILING_DATA_FIELD_DEF(uint32_t, tokensRem);         // T % usedCores(前 rem 核多 1)
    TILING_DATA_FIELD_DEF(uint32_t, idxChunk);          // 第 1 遍每次 stage 的 int64 数(4 的倍数)
    TILING_DATA_FIELD_DEF(uint32_t, rowTile);           // 载荷行搬运的 UB tile 元素数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TerraceK2Pack, TerraceK2PackTilingData)

}  // namespace optiling

#endif  // TERRACE_K2_PACK_TILING_H
