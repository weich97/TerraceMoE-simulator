/**
 * terrace_k1_arrival -- host side: tiling + InferShape/InferDataType + op prototype.
 *
 * The three slots for K1 (the arrival-side fused chain); build.sh overwrites the
 * same-named stubs in the msopgen skeleton with this file. Functional spec and
 * the two-pass argument live in the op_kernel/terrace_k1_arrival.cpp file header.
 *
 * Shape contract (every shape is static on the equal-quota fast path -- this is
 * what buys K1 its freedom from host synchronization):
 *   rx [R, H], rslot [R, quota], rgate [R, quota]
 *   -> send_buf [R*quota, H], gate_pairs [R*quota],
 *      r_idx [R*quota], slot_idx [R*quota], i_send [rpn]
 */
// The tiling struct lives in op_kernel/ (shared by host and kernel, CANN 9.0.0
// ASC layout).
#include "../op_kernel/terrace_k1_arrival_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

// Pairs per chunk for rslot staging: multiple of 4 (4 pairs = 32B aligned under
// the int32 view), 2048 pairs = 16KB UB. Payload-row tile capped at 8192
// elements (bf16 16KB x double buffer).
// UB budget (CopyRow uses two queues since 2026-08-24): rowQueue 8192x2Bx2 = 32KB
// + rowOutQueue another 32KB + slotStage 16KB = **80KB**; UB is 192KB, plenty of
// headroom. The second queue is not optional: without it there is no
// MTE2->MTE3 barrier and the copy-out ships dirty data.
constexpr uint32_t SLOT_CHUNK = 2048;
constexpr uint32_t ROW_TILE_MAX = 8192;
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t MAX_RPN = 64;      // kernel bucket-array cap (hard bound of the current chain: slots <= 63)

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // GetTilingData<T>() returns a T* into the tiling buffer and calls
    // SetDataSize(sizeof(T)) as a side effect.
    TerraceK1ArrivalTilingData *tiling =
        context->GetTilingData<TerraceK1ArrivalTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = platform.GetCoreNumAiv();   // trust the 910C measured value (header comment of the build script ascendc/build.sh)
    if (aivNum == 0) {
        return ge::GRAPH_FAILED;
    }

    // ---- input shapes ----
    const gert::StorageShape *rxShape = context->GetInputShape(0);
    const gert::StorageShape *rslotShape = context->GetInputShape(1);
    const gert::StorageShape *rgateShape = context->GetInputShape(2);
    if (rxShape == nullptr || rslotShape == nullptr || rgateShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &rxS = rxShape->GetStorageShape();
    const gert::Shape &rsS = rslotShape->GetStorageShape();
    const gert::Shape &rgS = rgateShape->GetStorageShape();
    if (rxS.GetDimNum() != 2 || rsS.GetDimNum() != 2 || rgS.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    int64_t R = rsS.GetDim(0);
    int64_t Q = rsS.GetDim(1);
    int64_t H = rxS.GetDim(1);
    if (rxS.GetDim(0) != R || rgS.GetDim(0) != R || rgS.GetDim(1) != Q ||
        R < 0 || Q <= 0 || H <= 0) {
        return ge::GRAPH_FAILED;
    }

    // ---- attributes ----
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *quota = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *epr = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *rpn = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *myLocal = attrs->GetAttrPointer<int64_t>(3);
    if (quota == nullptr || epr == nullptr || rpn == nullptr ||
        myLocal == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // fail loud: attributes must be consistent with the tensor geometry (quota
    // is rslot dim 1; bucket-array cap).
    if (*quota != Q || *epr <= 0 || *rpn <= 0 || *rpn > MAX_RPN ||
        (*epr) * (*rpn) > 63 || *myLocal < 0 || *myLocal >= *rpn) {
        return ge::GRAPH_FAILED;
    }

    // ---- payload-row alignment: H*esize must be divisible by 32B (always true
    // for the real hidden sizes 2048/7168) ----
    int64_t esizeRaw = ge::GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
    if (esizeRaw <= 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t esize = static_cast<uint32_t>(esizeRaw);
    if ((static_cast<uint64_t>(H) * esize) % ALIGN_BYTES != 0) {
        return ge::GRAPH_FAILED;   // fail loud; the torch-side csrc intercepts first with a human-readable error
    }

    // ---- core split: pairs divided contiguously and evenly (the inter-core
    // prerequisite for stable ordering, see the kernel file-header argument) ----
    uint64_t P64 = static_cast<uint64_t>(R) * static_cast<uint64_t>(Q);
    if (P64 > 0xFFFFFFFFull) {
        return ge::GRAPH_FAILED;
    }
    uint32_t P = static_cast<uint32_t>(P64);
    uint32_t usedCores = aivNum;
    if (P < usedCores) {
        usedCores = (P > 0) ? P : 1;
    }
    context->SetBlockDim(usedCores);

    // rowTile: <= ROW_TILE_MAX while keeping 32B alignment (H is already
    // aligned, so min suffices; when H > ROW_TILE_MAX, the 8192 elements of
    // ROW_TILE_MAX stay aligned for 2/4-byte dtypes).
    uint32_t rowTile = (H < static_cast<int64_t>(ROW_TILE_MAX))
                           ? static_cast<uint32_t>(H) : ROW_TILE_MAX;

    tiling->rows = static_cast<uint32_t>(R);
    tiling->hidden = static_cast<uint32_t>(H);
    tiling->quota = static_cast<uint32_t>(Q);
    tiling->epr = static_cast<uint32_t>(*epr);
    tiling->rpn = static_cast<uint32_t>(*rpn);
    tiling->myLocal = static_cast<uint32_t>(*myLocal);
    tiling->pairCount = P;
    tiling->pairsPerCoreBase = P / usedCores;
    tiling->pairsRem = P % usedCores;
    tiling->slotChunk = SLOT_CHUNK;
    tiling->rowTile = rowTile;

    // Cursor method, zero inter-core synchronization, no custom workspace; the
    // system workspace stays as in the template.
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *rxShape = context->GetInputShape(0);
    const gert::Shape *rslotShape = context->GetInputShape(1);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (rxShape == nullptr || rslotShape == nullptr || attrs == nullptr) {
        return GRAPH_FAILED;
    }
    const int64_t *rpn = attrs->GetAttrPointer<int64_t>(2);
    if (rpn == nullptr) {
        return GRAPH_FAILED;
    }
    int64_t P = rslotShape->GetDim(0) * rslotShape->GetDim(1);
    int64_t H = rxShape->GetDim(1);

    gert::Shape *sendShape = context->GetOutputShape(0);
    sendShape->SetDimNum(2);
    sendShape->SetDim(0, P);
    sendShape->SetDim(1, H);
    gert::Shape *gateShape = context->GetOutputShape(1);
    gateShape->SetDimNum(1);
    gateShape->SetDim(0, P);
    gert::Shape *ridxShape = context->GetOutputShape(2);
    ridxShape->SetDimNum(1);
    ridxShape->SetDim(0, P);
    gert::Shape *slotShape = context->GetOutputShape(3);
    slotShape->SetDimNum(1);
    slotShape->SetDim(0, P);
    gert::Shape *isendShape = context->GetOutputShape(4);
    isendShape->SetDimNum(1);
    isendShape->SetDim(0, *rpn);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));   // send_buf ~ rx
    context->SetOutputDataType(1, context->GetInputDataType(2));   // gate_pairs ~ rgate
    context->SetOutputDataType(2, ge::DT_INT64);                   // r_idx
    context->SetOutputDataType(3, ge::DT_INT64);                   // slot_idx
    context->SetOutputDataType(4, ge::DT_INT64);                   // i_send
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class TerraceK1Arrival : public OpDef {
public:
    explicit TerraceK1Arrival(const char *name) : OpDef(name)
    {
        // dtype combos correspond by position: bf16 is the main line, fp16/fp32
        // ride along (same policy as passthrough). rslot is an int64 **storage**
        // plane -- the kernel accesses it as int32 scalars and issues no int64
        // vector instructions (910C boundary, see the kernel file header).
        this->Input("rx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("rslot")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("rgate")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("send_buf")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("gate_pairs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("r_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("slot_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("i_send")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("quota").AttrType(REQUIRED).Int();
        this->Attr("epr").AttrType(REQUIRED).Int();
        this->Attr("rpn").AttrType(REQUIRED).Int();
        this->Attr("my_local").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        // The soc string placeholder follows the passthrough policy: build.sh
        // grabs the authoritative value from the skeleton stub and substitutes
        // it, then reads it back to verify (a leftover placeholder stops the
        // build). Measured value on 910C/CANN 9.0.0: ascend910_93.
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("__TERRACE_SOC__");
    }
};

OP_ADD(TerraceK1Arrival);

}  // namespace ops
