/**
 * terrace_ops -- torch.library 绑定:torch.ops.terrace.* -> aclnn 两段式调用。
 *
 * 只在集群编译(python csrc/build_ext.py,需 torch_npu + 已安装的 opp vendor 包)。
 * 结构照 CANN 官方 PytorchInvocation 路数:schema 注册(TORCH_LIBRARY)+
 * PrivateUse1(NPU)实现 + Meta 实现(fake tensor 形状推导用)。
 *
 * aclnn 符号(aclnnTerracePassthrough{GetWorkspaceSize,}) 来自 opp 包装出的
 * libcust_opapi.so,build_ext.py 直接链它 -- 编译期缺符号即失败,fail loud,
 * 不做 dlopen 延迟绑定。
 *
 * K1(terrace::k1_arrival,2026-08-20 落地):
 *   - schema:k1_arrival(Tensor rx, Tensor rslot, Tensor rgate, int quota,
 *     int epr, int rpn, int my_local) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
 *     = (send_buf, gate_pairs, r_idx, slot_idx, i_send);
 *   - 功能规格 = 到达侧现组合链逐位复刻,见 op_kernel/terrace_k1_arrival.cpp
 *     文件头(两遍法 + 稳定序逐位一致论证);
 *   - send_buf/gate_pairs 可微(反向 = 现组合链 scatter-add/gather,Python 侧
 *     autograd.Function 承担);r_idx/slot_idx/i_send 是索引/计数平面,Python 侧
 *     mark_non_differentiable;
 *   - int64 rslot 直接进 kernel(int32 lo/hi 标量访问,不发 int64 向量指令)。
 */
#include <map>
#include <vector>

#include <ATen/ATen.h>
#include <torch/library.h>

#include "acl/acl.h"
// aclTensor 构造/销毁。个别 CANN 8.x 发行版头文件名不同:若编译报找不到,
// 在 ${ASCEND_HOME_PATH}/include 下 grep -rl aclCreateTensor 换成实名(构建脚本 ascendc/build.sh 的头注)。
#include "aclnn/acl_meta.h"
// opp vendor 包安装后生成的算子专属头(vendors/<vendor>/op_api/include)。
#include "aclnn_terrace_passthrough.h"
#include "aclnn_terrace_k1_arrival.h"

// torch_npu:取当前 NPU 流。头/库路径由 build_ext.py 注入。
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {

// at::Tensor -> aclTensor 视图(不拷数据)。contiguous 由调用方保证。
aclTensor *MakeAclTensor(const at::Tensor &t)
{
    static const std::map<at::ScalarType, aclDataType> kDtype = {
        {at::kHalf, ACL_FLOAT16},
        {at::kBFloat16, ACL_BF16},
        {at::kFloat, ACL_FLOAT},
        {at::kLong, ACL_INT64},        // K1 的 rslot / r_idx / slot_idx / i_send
    };
    auto it = kDtype.find(t.scalar_type());
    TORCH_CHECK(it != kDtype.end(), "terrace ops: unsupported dtype ",
                t.scalar_type(), " (fp16/bf16/fp32/int64 only, see op prototypes)");
    std::vector<int64_t> shape(t.sizes().begin(), t.sizes().end());
    std::vector<int64_t> strides(t.strides().begin(), t.strides().end());
    return aclCreateTensor(shape.data(), shape.size(), it->second, strides.data(),
                           0 /*offset*/, ACL_FORMAT_ND, shape.data(), shape.size(),
                           const_cast<void *>(t.storage().data()));
}

at::Tensor passthrough_npu(const at::Tensor &x)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1,
                "terrace::passthrough expects an NPU tensor");
    at::Tensor xc = x.contiguous();
    at::Tensor y = at::empty_like(xc);

    aclTensor *xAcl = MakeAclTensor(xc);
    aclTensor *yAcl = MakeAclTensor(y);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    auto ret = aclnnTerracePassthroughGetWorkspaceSize(xAcl, yAcl, &workspaceSize,
                                                       &executor);
    TORCH_CHECK(ret == ACL_SUCCESS,
                "aclnnTerracePassthroughGetWorkspaceSize failed: ", ret,
                " (host tiling 拒绝了这个形状? 样板要求 32B 整除,构建脚本 ascendc/build.sh 的头注)");

    // workspace 用 torch 的 NPU 缓存分配器拿,免手工 aclrtMalloc/生命周期管理。
    at::Tensor workspace;
    void *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        workspace = at::empty({static_cast<int64_t>(workspaceSize)},
                              xc.options().dtype(at::kByte));
        workspacePtr = workspace.data_ptr();
    }

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    ret = aclnnTerracePassthrough(workspacePtr, workspaceSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnTerracePassthrough failed: ", ret);

    aclDestroyTensor(xAcl);
    aclDestroyTensor(yAcl);
    return y;
}

at::Tensor passthrough_meta(const at::Tensor &x)
{
    return at::empty_like(x);   // 形状推导:输出形状/dtype == 输入
}

// ======================================================================================
// K1:到达侧融合链(功能规格与逐位论证见 op_kernel/terrace_k1_arrival.cpp 文件头)。
// ======================================================================================

using K1Out = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

K1Out k1_arrival_npu(const at::Tensor &rx, const at::Tensor &rslot,
                     const at::Tensor &rgate, int64_t quota, int64_t epr,
                     int64_t rpn, int64_t my_local)
{
    TORCH_CHECK(rx.device().type() == c10::DeviceType::PrivateUse1,
                "terrace::k1_arrival expects NPU tensors");
    TORCH_CHECK(rx.dim() == 2 && rslot.dim() == 2 && rgate.dim() == 2,
                "terrace::k1_arrival: rx/rslot/rgate must be 2-D");
    TORCH_CHECK(rslot.scalar_type() == at::kLong,
                "terrace::k1_arrival: rslot must be int64 (C1 wire format)");
    TORCH_CHECK(rgate.scalar_type() == rx.scalar_type(),
                "terrace::k1_arrival: rgate dtype must equal rx dtype (the C1 gate "
                "plane is allocated from the payload -- see _pack_quota_wire)");
    TORCH_CHECK(rslot.size(1) == quota && rgate.size(1) == quota &&
                rslot.size(0) == rx.size(0) && rgate.size(0) == rx.size(0),
                "terrace::k1_arrival: geometry mismatch (rx ", rx.sizes(),
                ", rslot ", rslot.sizes(), ", rgate ", rgate.sizes(),
                ", quota ", quota, ")");
    TORCH_CHECK(quota > 0 && epr > 0 && rpn > 0 && epr * rpn <= 63 &&
                my_local >= 0 && my_local < rpn,
                "terrace::k1_arrival: bad geometry scalars");
    // host tiling 拒绝非 32B 整除的行宽(样板同款 fail loud);这里先给人话报错。
    TORCH_CHECK((rx.size(1) * rx.element_size()) % 32 == 0,
                "terrace::k1_arrival: hidden*esize must be 32B-aligned, got H=",
                rx.size(1), " esize=", rx.element_size());

    at::Tensor rxc = rx.contiguous();
    at::Tensor rslotc = rslot.contiguous();
    at::Tensor rgatec = rgate.contiguous();
    const int64_t R = rxc.size(0), H = rxc.size(1), P = R * quota;

    at::Tensor send_buf = at::empty({P, H}, rxc.options());
    at::Tensor gate_pairs = at::empty({P}, rgatec.options());
    at::Tensor r_idx = at::empty({P}, rslotc.options());
    at::Tensor slot_idx = at::empty({P}, rslotc.options());
    // i_send 用 zeros:P == 0 时 kernel 不发射(下方短路),零直方图即正确答案。
    at::Tensor i_send = at::zeros({rpn}, rslotc.options());
    if (P == 0) {
        return {send_buf, gate_pairs, r_idx, slot_idx, i_send};
    }

    aclTensor *rxAcl = MakeAclTensor(rxc);
    aclTensor *rslotAcl = MakeAclTensor(rslotc);
    aclTensor *rgateAcl = MakeAclTensor(rgatec);
    aclTensor *sendAcl = MakeAclTensor(send_buf);
    aclTensor *gateAcl = MakeAclTensor(gate_pairs);
    aclTensor *ridxAcl = MakeAclTensor(r_idx);
    aclTensor *slotAcl = MakeAclTensor(slot_idx);
    aclTensor *isendAcl = MakeAclTensor(i_send);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    // aclnn 生成签名顺序:输入张量 -> int 属性 -> 输出张量(msopgen aclnn 约定;
    // 若本 drop 生成的头顺序不同,以 aclnn_terrace_k1_arrival.h 为准 —— 集群编译
    // 验证点,编译期签名不符即报错,fail loud)。
    auto ret = aclnnTerraceK1ArrivalGetWorkspaceSize(
        rxAcl, rslotAcl, rgateAcl, quota, epr, rpn, my_local,
        sendAcl, gateAcl, ridxAcl, slotAcl, isendAcl, &workspaceSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS,
                "aclnnTerraceK1ArrivalGetWorkspaceSize failed: ", ret,
                " (host tiling 拒绝了这个几何? 见 op_host/terrace_k1_arrival.cpp)");

    at::Tensor workspace;
    void *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        workspace = at::empty({static_cast<int64_t>(workspaceSize)},
                              rxc.options().dtype(at::kByte));
        workspacePtr = workspace.data_ptr();
    }

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    ret = aclnnTerraceK1Arrival(workspacePtr, workspaceSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnTerraceK1Arrival failed: ", ret);

    aclDestroyTensor(rxAcl);
    aclDestroyTensor(rslotAcl);
    aclDestroyTensor(rgateAcl);
    aclDestroyTensor(sendAcl);
    aclDestroyTensor(gateAcl);
    aclDestroyTensor(ridxAcl);
    aclDestroyTensor(slotAcl);
    aclDestroyTensor(isendAcl);
    return {send_buf, gate_pairs, r_idx, slot_idx, i_send};
}

K1Out k1_arrival_meta(const at::Tensor &rx, const at::Tensor &rslot,
                      const at::Tensor &rgate, int64_t quota, int64_t epr,
                      int64_t rpn, int64_t my_local)
{
    const int64_t P = rslot.size(0) * quota, H = rx.size(1);
    return {at::empty({P, H}, rx.options()),
            at::empty({P}, rgate.options()),
            at::empty({P}, rslot.options()),
            at::empty({P}, rslot.options()),
            at::empty({rpn}, rslot.options())};
}

}  // namespace

// schema 与实现分开注册:autograd 由 Python 侧 autograd.Function 承担
// (terrace/ops/__init__.py),这里不注册 Autograd key -- K1/K2 反向走组合链,
// kernel 只出现在 forward(内部设计记录(未随仓发布)拍板)。
TORCH_LIBRARY(terrace, m)
{
    m.def("passthrough(Tensor x) -> Tensor");
    m.def("k1_arrival(Tensor rx, Tensor rslot, Tensor rgate, int quota, int epr, "
          "int rpn, int my_local) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(terrace, PrivateUse1, m)
{
    m.impl("passthrough", &passthrough_npu);
    m.impl("k1_arrival", &k1_arrival_npu);
}

TORCH_LIBRARY_IMPL(terrace, Meta, m)
{
    m.impl("passthrough", &passthrough_meta);
    m.impl("k1_arrival", &k1_arrival_meta);
}
