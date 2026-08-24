/**
 * terrace_k2_pack -- host 侧:tiling + InferShape/InferDataType + 算子原型。
 *
 * K2(发送侧打包链)的三个坑位,build.sh 会用本文件覆盖 msopgen 骨架里的同名
 * stub(挂接 diff 见 terrace/ops/内部嫁接记录(未随仓发布);本算子尚未进 build.sh 的
 * OP_JSONS —— 嫁接时加,K1 热改期间不动既有文件)。功能规格与两遍法论证见
 * op_kernel/terrace_k2_pack.cpp 文件头。
 *
 * 形状契约(等配额快路径全形状静态 —— K2 免主机同步的本钱,plan_ta2a 的
 * n_rows = T*M 论证原文见 terrace/ta2a.py):
 *   hidden [T, H], expert_idx [T, k], gates [T, k](gates 与 hidden 同 dtype,
 *   C1 圆整点契约,见 _pack_quota_wire 的 gate-plane-from-payload 论证)
 *   -> payload [T*M, H], mask [T*M, quota](升序槽号表), gate_rows [T*M, quota],
 *      u_src [T*M], node_counts [world/rpn];quota = k / groups_m,M = groups_m。
 */
#include "terrace_k2_pack_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

// 第 1 遍 expert_idx staging 的每块 int64 数:4 的倍数(int32 视图下 4 个 int64 =
// 32B 对齐),2048 = 16KB UB。载荷行 tile 上限 8192 元素(bf16 16KB x double buffer)。
constexpr uint32_t IDX_CHUNK = 2048;
constexpr uint32_t ROW_TILE_MAX = 8192;
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t MAX_NODES = 256;   // kernel 直方图数组上限(参考工作点 512die/rpn8 = 64)
constexpr uint32_t MAX_K = 64;        // kernel 行内排序数组上限(现档 k<=8)

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TerraceK2PackTilingData tiling;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = platform.GetCoreNumAiv();   // 910C 实测为准(README §6)
    if (aivNum == 0) {
        return ge::GRAPH_FAILED;
    }

    // ---- 输入形状 ----
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

    // ---- 属性 ----
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
    // fail loud:属性与张量几何必须自洽。逐条对应现链的硬前提:
    //   world%rpn / nE%world 整除(几何定义);k%M 整除(等额配额,ta2a_fwd 的
    //   assert 同款);slots<=63(int64 掩码时代的硬约束,C1 槽号表沿用其数值域);
    //   nNodes/k 的数组上限(kernel 标量数组);nE/T*k < 2^31(int32 lo 词契约)。
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

    // ---- 载荷行对齐:H*esize 必须 32B 整除(真实 hidden 2048/7168 恒真)----
    int64_t esizeRaw = ge::GetSizeByDataType(context->GetInputDesc(0)->GetDataType());
    if (esizeRaw <= 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t esize = static_cast<uint32_t>(esizeRaw);
    if ((static_cast<uint64_t>(H) * esize) % ALIGN_BYTES != 0) {
        return ge::GRAPH_FAILED;   // fail loud;torch 侧 csrc 先行拦截并给人话报错
    }

    // ---- 切核:token 连续均分(稳定序的核间前提,见 kernel 文件头论证;按 token
    // 而非按平铺槽位切,run 不跨核,第 2 遍不需要跨核拼行)----
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

    // 游标法零核间同步,无自定义 workspace;系统 workspace 照样板保留。
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
    context->SetOutputDataType(1, ge::DT_INT64);                   // mask(槽号表)
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
        // dtype 组合按位置对应:bf16 主线,fp16/fp32 陪跑(与 K1 同策)。gates 与
        // hidden 同位组合 == 同 dtype 强制:C1 的 gate 平面从 payload 派生
        // (_pack_quota_wire),失配必须大声死,不许静默送更宽的 gate 平面上线。
        // expert_idx 是 int64 **存储**平面 —— kernel 内以 int32 对标量访问,
        // 不发 int64 向量指令(910C 边界,见 kernel 文件头)。
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

        // soc 串占位与 passthrough/K1 同策:build.sh 从骨架 stub 抓权威值替换。
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(TerraceK2Pack);

}  // namespace ops
