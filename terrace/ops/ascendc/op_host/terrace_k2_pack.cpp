/**
 * terrace_k2_pack -- host side: tiling + InferShape/InferDataType + op prototype.
 *
 * The three slots for K2 (the send-side pack chain); build.sh overwrites the
 * same-named stubs in the msopgen skeleton with this file (the graft diff lives
 * in the terrace/ops/ internal grafting notes (not published with this repo);
 * this op is not yet in build.sh's OP_JSONS -- it gets added at graft time, and
 * existing files stay untouched while K1 is under active modification).
 * Functional spec and the two-pass argument live in the
 * op_kernel/terrace_k2_pack.cpp file header.
 *
 * Shape contract (every shape is static on the equal-quota fast path -- this is
 * what buys K2 its freedom from host synchronization; the original
 * n_rows = T*M argument for plan_ta2a is in terrace/ta2a.py):
 *   hidden [T, H], expert_idx [T, k], gates [T, k] (gates shares hidden's dtype,
 *   the C1 rounding-point contract; see the gate-plane-from-payload argument in
 *   _pack_quota_wire)
 *   -> payload [T*M, H], mask [T*M, quota] (ascending slot-id table),
 *      gate_rows [T*M, quota], u_src [T*M], node_counts [world/rpn];
 *      quota = k / groups_m, M = groups_m.
 */
#include "terrace_k2_pack_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

// int64 count per chunk for pass-1 expert_idx staging: multiple of 4 (4 int64 =
// 32B aligned under the int32 view), 2048 = 16KB UB. Payload-row tile capped at
// 8192 elements (bf16 16KB x double buffer).
constexpr uint32_t IDX_CHUNK = 2048;
constexpr uint32_t ROW_TILE_MAX = 8192;
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t MAX_NODES = 256;   // kernel histogram-array cap (reference operating point 512die/rpn8 = 64)
constexpr uint32_t MAX_K = 64;        // kernel in-row sort-array cap (current tiers have k<=8)

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TerraceK2PackTilingData tiling;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = platform.GetCoreNumAiv();   // trust the 910C measured value (header comment of the build script ascendc/build.sh)
    if (aivNum == 0) {
        return ge::GRAPH_FAILED;
    }

    // ---- input shapes ----
    const gert::StorageShape *hidShape = context->GetInputShape(0);
    const gert::StorageShape *idxShape = context->GetInputShape(1);
    const gert::StorageShape *gateShape = context->GetInputShape(2);
    if (hidShape == nullptr || idxShape == nullptr || gateShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &hS = hidShape->GetStorageShape();
    const gert::Shape &iS = idxShape->GetStorageShape();
    const gert::Shape &gS = gateShape->GetStorageShape();
    if (hS.GetDimNum() != 2 || iS.GetDimNum() != 2 || gS.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    int64_t T = iS.GetDim(0);
    int64_t K = iS.GetDim(1);
    int64_t H = hS.GetDim(1);
    if (hS.GetDim(0) != T || gS.GetDim(0) != T || gS.GetDim(1) != K ||
        T < 0 || K <= 0 || H <= 0) {
        return ge::GRAPH_FAILED;
    }

    // ---- attributes ----
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *world = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *nExperts = attrs->GetAttrPointer<int64_t>(1);
    const int64_t *rpn = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *groupsM = attrs->GetAttrPointer<int64_t>(3);
    if (world == nullptr || nExperts == nullptr || rpn == nullptr ||
        groupsM == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // fail loud: attributes must be consistent with the tensor geometry. Each
    // clause maps to a hard precondition of the current chain:
    //   world%rpn / nE%world divisibility (geometry definitions); k%M
    //   divisibility (equal quota, same assert as ta2a_fwd); slots<=63 (hard
    //   bound from the int64-mask era; the C1 slot-id table keeps its numeric
    //   domain); nNodes/k array caps (kernel scalar arrays); nE/T*k < 2^31
    //   (int32 lo-word contract).
    if (*world <= 0 || *rpn <= 0 || *groupsM <= 0 || *nExperts <= 0 ||
        (*world) % (*rpn) != 0 || (*nExperts) % (*world) != 0 ||
        K % (*groupsM) != 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t epr = *nExperts / *world;
    int64_t slots = epr * (*rpn);
    int64_t nNodes = *world / *rpn;
    int64_t quota = K / *groupsM;
    if (slots <= 0 || slots > 63 || nNodes > MAX_NODES || K > MAX_K ||
        *nExperts != nNodes * slots || *nExperts >= (1LL << 31)) {
        return ge::GRAPH_FAILED;
    }
    int64_t flat = T * K;                         // = nRows * quota
    int64_t nRows = T * (*groupsM);
    if (flat >= (1LL << 31) || T >= (1LL << 31)) {
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

    // ---- core split: tokens divided contiguously and evenly (the inter-core
    // prerequisite for stable ordering, see the kernel file-header argument;
    // split by token rather than by flattened slot so a run never crosses cores
    // and pass 2 never has to stitch rows across cores) ----
    uint32_t usedCores = aivNum;
    if (T < static_cast<int64_t>(usedCores)) {
        usedCores = (T > 0) ? static_cast<uint32_t>(T) : 1;
    }
    context->SetBlockDim(usedCores);

    uint32_t rowTile = (H < static_cast<int64_t>(ROW_TILE_MAX))
                           ? static_cast<uint32_t>(H) : ROW_TILE_MAX;

    tiling.set_tokens(static_cast<uint32_t>(T));
    tiling.set_hidden(static_cast<uint32_t>(H));
    tiling.set_topk(static_cast<uint32_t>(K));
    tiling.set_quota(static_cast<uint32_t>(quota));
    tiling.set_groupsM(static_cast<uint32_t>(*groupsM));
    tiling.set_epr(static_cast<uint32_t>(epr));
    tiling.set_rpn(static_cast<uint32_t>(*rpn));
    tiling.set_slots(static_cast<uint32_t>(slots));
    tiling.set_nNodes(static_cast<uint32_t>(nNodes));
    tiling.set_nExperts(static_cast<uint32_t>(*nExperts));
    tiling.set_nRows(static_cast<uint32_t>(nRows));
    tiling.set_tokensPerCoreBase(static_cast<uint32_t>(T) / usedCores);
    tiling.set_tokensRem(static_cast<uint32_t>(T) % usedCores);
    tiling.set_idxChunk(IDX_CHUNK);
    tiling.set_rowTile(rowTile);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

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
    const gert::Shape *hidShape = context->GetInputShape(0);
    const gert::Shape *idxShape = context->GetInputShape(1);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (hidShape == nullptr || idxShape == nullptr || attrs == nullptr) {
        return GRAPH_FAILED;
    }
    const int64_t *world = attrs->GetAttrPointer<int64_t>(0);
    const int64_t *rpn = attrs->GetAttrPointer<int64_t>(2);
    const int64_t *groupsM = attrs->GetAttrPointer<int64_t>(3);
    if (world == nullptr || rpn == nullptr || groupsM == nullptr ||
        *rpn == 0 || *groupsM == 0) {
        return GRAPH_FAILED;
    }
    int64_t T = idxShape->GetDim(0);
    int64_t K = idxShape->GetDim(1);
    int64_t H = hidShape->GetDim(1);
    int64_t nRows = T * (*groupsM);
    int64_t quota = K / (*groupsM);

    gert::Shape *payloadShape = context->GetOutputShape(0);
    payloadShape->SetDimNum(2);
    payloadShape->SetDim(0, nRows);
    payloadShape->SetDim(1, H);
    gert::Shape *maskShape = context->GetOutputShape(1);
    maskShape->SetDimNum(2);
    maskShape->SetDim(0, nRows);
    maskShape->SetDim(1, quota);
    gert::Shape *gateShape = context->GetOutputShape(2);
    gateShape->SetDimNum(2);
    gateShape->SetDim(0, nRows);
    gateShape->SetDim(1, quota);
    gert::Shape *usrcShape = context->GetOutputShape(3);
    usrcShape->SetDimNum(1);
    usrcShape->SetDim(0, nRows);
    gert::Shape *countsShape = context->GetOutputShape(4);
    countsShape->SetDimNum(1);
    countsShape->SetDim(0, *world / *rpn);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));   // payload ~ hidden
    context->SetOutputDataType(1, ge::DT_INT64);                   // mask (slot-id table)
    context->SetOutputDataType(2, context->GetInputDataType(2));   // gate_rows ~ gates
    context->SetOutputDataType(3, ge::DT_INT64);                   // u_src
    context->SetOutputDataType(4, ge::DT_INT64);                   // node_counts
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class TerraceK2Pack : public OpDef {
public:
    explicit TerraceK2Pack(const char *name) : OpDef(name)
    {
        // dtype combos correspond by position: bf16 is the main line, fp16/fp32
        // ride along (same policy as K1). gates paired position-wise with hidden
        // == same-dtype enforcement: the C1 gate plane is derived from the
        // payload (_pack_quota_wire); a mismatch must die loudly, never silently
        // ship a wider gate plane. expert_idx is an int64 **storage** plane --
        // the kernel accesses it as int32 scalars and issues no int64 vector
        // instructions (910C boundary, see the kernel file header).
        this->Input("hidden")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("expert_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("gates")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("payload")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("gate_rows")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("u_src")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("node_counts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("world").AttrType(REQUIRED).Int();
        this->Attr("n_experts").AttrType(REQUIRED).Int();
        this->Attr("rpn").AttrType(REQUIRED).Int();
        this->Attr("groups_m").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        // The soc string placeholder follows the passthrough/K1 policy: build.sh
        // grabs the authoritative value from the skeleton stub and substitutes it.
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(TerraceK2Pack);

}  // namespace ops
