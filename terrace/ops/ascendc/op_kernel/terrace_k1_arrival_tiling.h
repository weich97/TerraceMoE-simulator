/**
 * terrace_k1_arrival -- tiling data definition (K1: arrival-side fused chain).
 *
 * **Lives in op_kernel/, not op_host/**: in the CANN 9.0.0 ASC build system the tiling
 * struct is a plain C struct shared by host and kernel (rationale and migration history
 * in the header of op_kernel/terrace_passthrough_tiling.h + pitfall 4 of the pitfall
 * notes in this file's header).
 *
 * All uint32 (avoids int64 planes on 910C; internal engineering records). Bound check:
 *   pairCount = R * quota. R is the deduplicated row count this rank receives (measured
 *   in the tens of thousands at the 64 die/16384 tok tier), quota = k/M <= 8, so
 *   pairCount is far below 2^32; send_buf's total element count pairCount * hidden can
 *   exceed 2^32, so every GM offset inside the kernel is accumulated as a uint64 product,
 *   and this struct stores no product fields.
 */
#ifndef TERRACE_K1_ARRIVAL_TILING_H
#define TERRACE_K1_ARRIVAL_TILING_H

#include <cstdint>

struct TerraceK1ArrivalTilingData {
    uint32_t rows;              // R: arrival row count (dim 0 of rslot)
    uint32_t hidden;            // H: rx row width (element count)
    uint32_t quota;             // pairs per row (dim 1 of rslot)
    uint32_t epr;               // experts per rank
    uint32_t rpn;               // ranks per node (bucket count)
    uint32_t myLocal;           // this rank's index within the node (see kernel file header: reserved)
    uint32_t pairCount;         // P = R * quota
    uint32_t pairsPerCoreBase;  // P / usedCores
    uint32_t pairsRem;          // P % usedCores (first rem cores get 1 extra)
    uint32_t slotChunk;         // pairs staged into UB per chunk (multiple of 4)
    uint32_t rowTile;           // UB tile element count for payload row copy
};

#endif  // TERRACE_K1_ARRIVAL_TILING_H
