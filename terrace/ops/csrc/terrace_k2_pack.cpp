/**
 * terrace_k2_pack -- torch.library 绑定:torch.ops.terrace.k2_pack -> aclnn 两段式。
 *
 * 独立编译单元,**不动 terrace_ops.cpp**(K1 热改期间零冲突,「备好即嫁接」):
 *   - TORCH_LIBRARY(terrace) 的唯一定义在 terrace_ops.cpp;本文件用
 *     TORCH_LIBRARY_FRAGMENT(terrace) 往同一命名空间追加 schema —— torch 官方
 *     多编译单元扩展同一 namespace 的标准写法,两个 .cpp 链进同一个 .so 即可。
 *   - 挂进构建:build_ext.py 的 sources 列表加本文件(一行 diff,见
 *     terrace/ops/内部嫁接记录(未随仓发布);不嫁接则本文件不参与编译,零影响)。
 *
 * schema:k2_pack(Tensor hidden, Tensor expert_idx, Tensor gates, int world,
 *   int n_experts, int rpn, int groups_m) -> (Tensor, Tensor, Tensor, Tensor,
 *   Tensor) = (payload, mask, gate_rows, u_src, node_counts)。
 * 功能规格 = quota 快路径发送段现组合链(plan_ta2a 快路径 + 去重 gather +
 * _pack_quota_wire)逐位复刻,论证见 op_kernel/terrace_k2_pack.cpp 文件头。
 * payload/gate_rows 可微(反向 = 现组合链 index_add/gather,Python 侧
 * autograd.Function 承担,注册块见 内部嫁接记录(未随仓发布));mask/u_src/node_counts 是
 * 索引/计数平面,Python 侧 mark_non_differentiable。
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
#include "aclnn_terrace_k2_pack.h"

// torch_npu:取当前 NPU 流。头/库路径由 build_ext.py 注入。
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {

// at::Tensor -> aclTensor 视图(不拷数据)。contiguous 由调用方保证。
// (与 terrace_ops.cpp 的同名函数同文,匿名命名空间隔离,无 ODR 冲突。)
aclTensor *MakeAclTensor(const at::Tensor &t)
{
    static const std::map<at::ScalarType, aclDataType> kDtype = {
        {at::kHalf, ACL_FLOAT16},
        {at::kBFloat16, ACL_BF16},
        {at::kFloat, ACL_FLOAT},
        {at::kLong, ACL_INT64},        // expert_idx / mask / u_src / node_counts
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

using K2Out = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

K2Out k2_pack_npu(const at::Tensor &hidden, const at::Tensor &expert_idx,
                  const at::Tensor &gates, int64_t world, int64_t n_experts,
                  int64_t rpn, int64_t groups_m)
{
    TORCH_CHECK(hidden.device().type() == c10::DeviceType::PrivateUse1,
                "terrace::k2_pack expects NPU tensors");
    TORCH_CHECK(hidden.dim() == 2 && expert_idx.dim() == 2 && gates.dim() == 2,
                "terrace::k2_pack: hidden/expert_idx/gates must be 2-D");
    TORCH_CHECK(expert_idx.scalar_type() == at::kLong,
                "terrace::k2_pack: expert_idx must be int64");
    TORCH_CHECK(gates.scalar_type() == hidden.scalar_type(),
                "terrace::k2_pack: gates dtype must equal hidden dtype (the C1 gate "
                "plane is allocated from the payload; a mismatch must keep failing "
                "loudly exactly as _pack_quota_wire's index_put did -- cast gates at "
                "the caller, the caller owns the rounding point)");
    TORCH_CHECK(expert_idx.size(0) == hidden.size(0) &&
                gates.sizes() == expert_idx.sizes(),
                "terrace::k2_pack: geometry mismatch (hidden ", hidden.sizes(),
                ", expert_idx ", expert_idx.sizes(), ", gates ", gates.sizes(), ")");
    TORCH_CHECK(world > 0 && rpn > 0 && n_experts > 0 && groups_m > 0 &&
                world % rpn == 0 && n_experts % world == 0,
                "terrace::k2_pack: bad geometry scalars (world ", world, ", n_experts ",
                n_experts, ", rpn ", rpn, ", groups_m ", groups_m, ")");
    const int64_t T = expert_idx.size(0), K = expert_idx.size(1);
    const int64_t H = hidden.size(1);
    const int64_t epr = n_experts / world, slots = epr * rpn;
    const int64_t n_nodes = world / rpn, quota = K / groups_m;
    TORCH_CHECK(K % groups_m == 0,
                "terrace::k2_pack: k=", K, " not divisible by groups_m=", groups_m,
                " (equal-quota fast path only; same assert as ta2a_moe_forward)");
    TORCH_CHECK(slots >= 1 && slots <= 63,
                "terrace::k2_pack: ", slots, " expert slots per node exceeds the "
                "63-slot chain bound");
    TORCH_CHECK(n_nodes <= 256 && K <= 64 && n_experts < (1LL << 31) &&
                T * K < (1LL << 31),
                "terrace::k2_pack: geometry exceeds kernel bounds (n_nodes<=256, "
                "k<=64, n_experts<2^31, T*k<2^31)");
    // host tiling 拒绝非 32B 整除的行宽(K1 同款 fail loud);这里先给人话报错。
    TORCH_CHECK((H * hidden.element_size()) % 32 == 0,
                "terrace::k2_pack: hidden*esize must be 32B-aligned, got H=", H,
                " esize=", hidden.element_size());

    at::Tensor hc = hidden.contiguous();
    at::Tensor ic = expert_idx.contiguous();
    at::Tensor gc = gates.contiguous();
    const int64_t n_rows = T * groups_m;

    // payload 正常路径下每行恰写一次(计数排序是 [0, T*M) 上的双射)-> empty;
    // mask/gate_rows/u_src/node_counts 用 zeros:_pack_quota_wire 的 zeros 收容
    // 原文(配额不变量漂移窗口内被跳过的行 == 槽 0/gate 0,确定且形状无害),
    // 兼 T == 0 时 kernel 不发射(下方短路)、零平面即正确答案(K1 i_send 同款)。
    at::Tensor payload = at::empty({n_rows, H}, hc.options());
    at::Tensor mask = at::zeros({n_rows, quota}, ic.options());
    at::Tensor gate_rows = at::zeros({n_rows, quota}, gc.options());
    at::Tensor u_src = at::zeros({n_rows}, ic.options());
    at::Tensor node_counts = at::zeros({n_nodes}, ic.options());
    if (n_rows == 0) {
        return {payload, mask, gate_rows, u_src, node_counts};
    }

    aclTensor *hAcl = MakeAclTensor(hc);
    aclTensor *iAcl = MakeAclTensor(ic);
    aclTensor *gAcl = MakeAclTensor(gc);
    aclTensor *pAcl = MakeAclTensor(payload);
    aclTensor *mAcl = MakeAclTensor(mask);
    aclTensor *grAcl = MakeAclTensor(gate_rows);
    aclTensor *uAcl = MakeAclTensor(u_src);
    aclTensor *cAcl = MakeAclTensor(node_counts);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    // aclnn 生成签名顺序:输入张量 -> int 属性 -> 输出张量(msopgen aclnn 约定;
    // 若本 drop 生成的头顺序不同,以 aclnn_terrace_k2_pack.h 为准 —— 集群编译
    // 验证点,编译期签名不符即报错,fail loud)。
    auto ret = aclnnTerraceK2PackGetWorkspaceSize(
        hAcl, iAcl, gAcl, world, n_experts, rpn, groups_m,
        pAcl, mAcl, grAcl, uAcl, cAcl, &workspaceSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS,
                "aclnnTerraceK2PackGetWorkspaceSize failed: ", ret,
                " (host tiling 拒绝了这个几何? 见 op_host/terrace_k2_pack.cpp)");

    at::Tensor workspace;
    void *workspacePtr = nullptr;
    if (workspaceSize > 0) {
        workspace = at::empty({static_cast<int64_t>(workspaceSize)},
                              hc.options().dtype(at::kByte));
        workspacePtr = workspace.data_ptr();
    }

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
    ret = aclnnTerraceK2Pack(workspacePtr, workspaceSize, executor, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnTerraceK2Pack failed: ", ret);

    aclDestroyTensor(hAcl);
    aclDestroyTensor(iAcl);
    aclDestroyTensor(gAcl);
    aclDestroyTensor(pAcl);
    aclDestroyTensor(mAcl);
    aclDestroyTensor(grAcl);
    aclDestroyTensor(uAcl);
    aclDestroyTensor(cAcl);
    return {payload, mask, gate_rows, u_src, node_counts};
}

K2Out k2_pack_meta(const at::Tensor &hidden, const at::Tensor &expert_idx,
                   const at::Tensor &gates, int64_t world, int64_t n_experts,
                   int64_t rpn, int64_t groups_m)
{
    const int64_t n_rows = expert_idx.size(0) * groups_m;
    const int64_t quota = expert_idx.size(1) / groups_m;
    const int64_t H = hidden.size(1);
    return {at::empty({n_rows, H}, hidden.options()),
            at::empty({n_rows, quota}, expert_idx.options()),
            at::empty({n_rows, quota}, gates.options()),
            at::empty({n_rows}, expert_idx.options()),
            at::empty({world / rpn}, expert_idx.options())};
}

}  // namespace

// FRAGMENT,不是 TORCH_LIBRARY:terrace 命名空间的唯一定义权在 terrace_ops.cpp,
// 这里只追加。autograd 由 Python 侧 autograd.Function 承担(内部嫁接记录(未随仓发布)),
// 不注册 Autograd key —— K2 反向走组合链,kernel 只出现在 forward(路线图拍板)。
TORCH_LIBRARY_FRAGMENT(terrace, m)
{
    m.def("k2_pack(Tensor hidden, Tensor expert_idx, Tensor gates, int world, "
          "int n_experts, int rpn, int groups_m) -> "
          "(Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(terrace, PrivateUse1, m)
{
    m.impl("k2_pack", &k2_pack_npu);
}

TORCH_LIBRARY_IMPL(terrace, Meta, m)
{
    m.impl("k2_pack", &k2_pack_meta);
}
