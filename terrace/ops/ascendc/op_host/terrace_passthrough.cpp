/**
 * terrace_passthrough -- host 侧:tiling 函数 + InferShape/InferDataType + 算子原型。
 *
 * 工程链路样板(构建脚本 ascendc/build.sh 的头注)。三块内容对应 msopgen 生成骨架里 op_host/ 下同名 stub
 * 的三个坑位,build.sh 会用本文件整体覆盖 stub。
 *
 * K1 正式实现时改这里:
 *   - TilingFunc:按 rmask 行数 R 与 pair 数(R*quota,等配额下静态)切核,
 *     workspace 里给核间前缀游标表留空间(GetLibApiWorkSpaceSize() 之外追加);
 *   - InferShape:输出 r_idx/slot_idx 形状 [R*quota],i_send 形状 [rpn]
 *     (等配额快路径全形状静态 —— 这正是 K2/K1 免主机同步的本钱);
 *   - 原型定义:输入 rmask(int64, ND),输出三平面 + slots/quota/epr/rpn 属性。
 */
// tiling 结构体住在 op_kernel/(host/kernel 共享的普通 C 结构体,CANN 9.0.0 ASC
// 体系),host 侧反向 include —— msopgen 骨架的 stub 就是这么写的。
#include "../op_kernel/terrace_passthrough_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

// 每核 tile 数。样板取官方样例同款常数;K1 正式实现时改这里:按 UB 容量
// (platform.GetCoreMemSize(UB))与行宽反推,而不是拍常数。
constexpr uint32_t TILE_NUM = 8;
// kernel 侧 DataCopy 的 32B 对齐要求:每核每 tile(double buffer 后)的元素数
// 必须让字节数落在 32B 边界上。样板不做尾块,由 host 校验整除性,不满足即报错
// (torch 侧封装先行拦截并走组合链,见 csrc/terrace_ops.cpp)。
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t ALIGN_BYTES = 32;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // GetTilingData<T>() 返回指向 tiling buffer 的 T*,并顺带 SetDataSize(sizeof(T))
    // (见 exe_graph/runtime/tiling_context.h);容量不足或上下文缺失时返回 nullptr。
    TerracePassthroughTilingData *tiling =
        context->GetTilingData<TerracePassthroughTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // 向量核数:910C 上须以实测为准(构建脚本 ascendc/build.sh 的头注);样板信平台查询值。
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
    // 用满核数且满足整除;不齐则减核,最后退到单核整除校验。样板从简 —— 真实
    // 形状(hidden=2048/7168 的行平面)按 32B 对齐几乎恒真。
    uint32_t usedCores = aivNum;
    while (usedCores > 1 && (totalLength % (usedCores * elemsPerChunk)) != 0) {
        usedCores--;
    }
    if (totalLength % (usedCores * elemsPerChunk) != 0) {
        // 尾块样板不做(K1 正式实现时改这里:DataCopyPad / 尾块分支)。
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(usedCores);
    tiling->totalLength = totalLength;
    tiling->tileNum = TILE_NUM;
    // 系统 workspace:aclnn 单算子在 910B/910C 系上要求预留库内工作区;
    // 自定义 workspace 样板为 0(K1 的游标表在这里追加)。
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
    *yShape = *xShape;                      // passthrough:输出形状 == 输入形状
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
        // dtype 名单:bf16 为主线(内部工程记录:FP8 不可用,BF16 主线),
        // fp16/fp32 陪跑供 bench;不列 int64(910C 无 int64 移位核,K1 的
        // int64 位平面走 kernel 内部特化,不经这里)。
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

        // soc 配置串:合法值以本机 msopgen 生成的 stub 为准(910C/CANN 9.0.0 上
        // 实测是 "ascend910_93")。build.sh 从 stub 抓权威串替换下面的占位符,并在
        // 替换后回读校验 —— 占位符还在就直接停,绝不会编出错 SOC 的包。
        // 占位符故意写成一个非法 soc 串:万一替换逻辑失灵,编译期就炸,不静默。
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("__TERRACE_SOC__");
    }
};

OP_ADD(TerracePassthrough);

}  // namespace ops
