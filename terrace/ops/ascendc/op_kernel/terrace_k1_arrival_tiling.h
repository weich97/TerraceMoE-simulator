/**
 * terrace_k1_arrival -- tiling 数据定义(K1:到达侧融合链)。
 *
 * **放在 op_kernel/ 而不是 op_host/**:CANN 9.0.0 的 ASC 构建体系里 tiling 结构体
 * 是 host/kernel 共享的普通 C 结构体(理由与迁移原因见
 * op_kernel/terrace_passthrough_tiling.h 文件头 + 本文件头的踩坑记录 坑 4)。
 *
 * 全部 uint32(910C 上避免 int64 平面;内部工程记录)。上限核算:
 *   pairCount = R * quota。R 是本 rank 收到的去重行数(实测 64 die/16384 tok 档
 *   万级),quota = k/M <= 8,pairCount 远小于 2^32;send_buf 的元素总数
 *   pairCount * hidden 可能超 2^32,kernel 内所有 GM 偏移一律 uint64 累乘,
 *   本结构体里不存任何乘积字段。
 */
#ifndef TERRACE_K1_ARRIVAL_TILING_H
#define TERRACE_K1_ARRIVAL_TILING_H

#include <cstdint>

struct TerraceK1ArrivalTilingData {
    uint32_t rows;              // R:到达行数(rslot 第 0 维)
    uint32_t hidden;            // H:rx 行宽(元素数)
    uint32_t quota;             // 每行配对数(rslot 第 1 维)
    uint32_t epr;               // 每 rank 专家数
    uint32_t rpn;               // 节点内 rank 数(桶数)
    uint32_t myLocal;           // 本 rank 节点内序(见 kernel 文件头:预留)
    uint32_t pairCount;         // P = R * quota
    uint32_t pairsPerCoreBase;  // P / usedCores
    uint32_t pairsRem;          // P % usedCores(前 rem 核多 1)
    uint32_t slotChunk;         // 每次 stage 进 UB 的配对数(4 的倍数)
    uint32_t rowTile;           // 载荷行搬运的 UB tile 元素数
};

#endif  // TERRACE_K1_ARRIVAL_TILING_H
