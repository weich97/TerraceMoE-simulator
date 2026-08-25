/**
 * terrace_passthrough -- host side: tiling function + InferShape/InferDataType + op prototype.
 *
 * Engineering-pipeline template (see the header comment of the build script
 * ascendc/build.sh). The three sections fill the three slots of the same-named
 * stub under op_host/ in the msopgen-generated skeleton; build.sh overwrites
 * the stub with this file wholesale.
 *
 * Change here when implementing K1 for real:
 *   - TilingFunc: split cores by rmask row count R and pair count (R*quota,
 *     static under equal quota), and reserve room in the workspace for the
 *     inter-core prefix cursor table (appended beyond GetLibApiWorkSpaceSize());
 *   - InferShape: outputs r_idx/slot_idx shaped [R*quota], i_send shaped [rpn]
 *     (every shape is static on the equal-quota fast path -- this is exactly
 *     what buys K2/K1 their freedom from host synchronization);
 *   - prototype: input rmask (int64, ND), three output planes + the
 *     slots/quota/epr/rpn attributes.
 */
// The tiling struct lives in op_kernel/ (a plain C struct shared by host and
// kernel, CANN 9.0.0 ASC layout); the host includes it from over there -- that
// is exactly how the msopgen skeleton stub is written.
#include "../op_kernel/terrace_passthrough_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

// Tiles per core. The template takes the same constant as the official sample;
// change here when implementing K1 for real: derive it from UB capacity
// (platform.GetCoreMemSize(UB)) and the row width instead of hard-coding a
// constant.
constexpr uint32_t TILE_NUM = 8;
// The kernel-side DataCopy 32B alignment requirement: the element count per
// core per tile (after double buffering) must land the byte count on a 32B
// boundary. The template does no tail block; the host checks divisibility and
// errors out when it fails (the torch-side wrapper intercepts first and takes
// the composite chain, see csrc/terrace_ops.cpp).
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t ALIGN_BYTES = 32;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // GetTilingData<T>() returns a T* into the tiling buffer and calls
    // SetDataSize(sizeof(T)) as a side effect (see
    // exe_graph/runtime/tiling_context.h); it returns nullptr when the capacity
    // is short or the context is missing.
    TerracePassthroughTilingData *tiling =
        context->GetTilingData<TerracePassthroughTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // Vector core count: on 910C trust the measured value (header comment of
    // the build script ascendc/build.sh); the template trusts the platform query.
    uint32_t aivNum = platform.GetCoreNumAiv();
    if (aivNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    int64_t esizeRaw = ge::GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
    if (esizeRaw <= 0 || (ALIGN_BYTES % static_cast<uint32_t>(esizeRaw)) != 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t esize = static_cast<uint32_t>(esizeRaw);
    uint32_t elemsPerChunk = TILE_NUM * BUFFER_NUM * (ALIGN_BYTES / esize);
    // Use the full core count while keeping divisibility; shed cores when it
    // does not divide, falling back to a single-core divisibility check at the
    // end. The template keeps it simple -- real shapes (row planes with
    // hidden=2048/7168) are almost always 32B aligned.
    uint32_t usedCores = aivNum;
    while (usedCores > 1 && (totalLength % (usedCores * elemsPerChunk)) != 0) {
        usedCores--;
    }
    if (totalLength % (usedCores * elemsPerChunk) != 0) {
        // The template does no tail block (change here when implementing K1 for
        // real: DataCopyPad / a tail-block branch).
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(usedCores);
    tiling->totalLength = totalLength;
    tiling->tileNum = TILE_NUM;
    // System workspace: aclnn single-op on the 910B/910C family requires the
    // library-internal workspace reservation; the template's custom workspace
    // is 0 (K1's cursor table gets appended here).
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;                      // passthrough: output shape == input shape
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class TerracePassthrough : public OpDef {
public:
    explicit TerracePassthrough(const char *name) : OpDef(name)
    {
        // dtype roster: bf16 is the main line (internal engineering record: FP8
        // unavailable, BF16 main line), fp16/fp32 ride along for benches; int64
        // is not listed (910C has no int64 shift cores; K1's int64 bit planes go
        // through in-kernel specialization and never pass through here).
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        // soc config string: the legal value is whatever the msopgen-generated
        // stub on this machine says (measured on 910C/CANN 9.0.0:
        // "ascend910_93"). build.sh grabs the authoritative string from the
        // stub, substitutes it for the placeholder below, and reads it back to
        // verify -- if the placeholder is still there it stops outright, so a
        // package with the wrong SOC never gets built.
        // The placeholder is deliberately an illegal soc string: if the
        // substitution logic ever breaks, the build blows up at compile time
        // instead of going silent.
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("__TERRACE_SOC__");
    }
};

OP_ADD(TerracePassthrough);

}  // namespace ops
