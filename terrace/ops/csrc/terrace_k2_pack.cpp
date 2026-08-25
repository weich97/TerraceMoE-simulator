/**
 * terrace_k2_pack -- torch.library binding: torch.ops.terrace.k2_pack -> aclnn two-phase call.
 *
 * A standalone compilation unit that **does not touch terrace_ops.cpp** (zero
 * conflicts while K1 is under active modification; "prepare now, graft later"):
 *   - The sole TORCH_LIBRARY(terrace) definition lives in terrace_ops.cpp; this
 *     file appends a schema to the same namespace via
 *     TORCH_LIBRARY_FRAGMENT(terrace) -- torch's standard idiom for extending
 *     one namespace from multiple compilation units; just link both .cpp into
 *     the same .so.
 *   - Hooking into the build: add this file to build_ext.py's sources list (a
 *     one-line diff, see the terrace/ops/ internal grafting notes (not
 *     published with this repo); ungrafted, this file is not compiled and has
 *     zero impact).
 *
 * schema: k2_pack(Tensor hidden, Tensor expert_idx, Tensor gates, int world,
 *   int n_experts, int rpn, int groups_m) -> (Tensor, Tensor, Tensor, Tensor,
 *   Tensor) = (payload, mask, gate_rows, u_src, node_counts).
 * Functional spec = a bit-for-bit replica of the current composite chain of the
 * quota fast-path send stage (plan_ta2a fast path + dedup gather +
 * _pack_quota_wire); the argument lives in the op_kernel/terrace_k2_pack.cpp
 * file header.
 * payload/gate_rows are differentiable (backward = the composite chain's
 * index_add/gather, carried by a Python-side autograd.Function; the
 * registration block is in the internal grafting notes (not published with
 * this repo)); mask/u_src/node_counts are index/count planes,
 * mark_non_differentiable on the Python side.
 */
#include <map>
#include <vector>

#include <ATen/ATen.h>
#include <torch/library.h>

#include "acl/acl.h"
// aclTensor create/destroy. Some CANN 8.x releases name this header
// differently: if the compile cannot find it, grep -rl aclCreateTensor under
// ${ASCEND_HOME_PATH}/include and use the real name (header comment of the
// build script ascendc/build.sh).
#include "aclnn/acl_meta.h"
// Op-specific header generated once the opp vendor package is installed
// (vendors/<vendor>/op_api/include).
#include "aclnn_terrace_k2_pack.h"

// torch_npu: grab the current NPU stream. Header/library paths injected by
// build_ext.py.
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {

// at::Tensor -> aclTensor view (no data copy). The caller guarantees contiguous.
// (Verbatim copy of the same-named function in terrace_ops.cpp; the anonymous
// namespace isolates it, no ODR conflict.)
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
    // Host tiling rejects row widths not divisible into 32B (same fail-loud as
    // K1); give the human-readable error here first.
    TORCH_CHECK((H * hidden.element_size()) % 32 == 0,
                "terrace::k2_pack: hidden*esize must be 32B-aligned, got H=", H,
                " esize=", hidden.element_size());

    at::Tensor hc = hidden.contiguous();
    at::Tensor ic = expert_idx.contiguous();
    at::Tensor gc = gates.contiguous();
    const int64_t n_rows = T * groups_m;

    // On the normal path payload writes each row exactly once (the counting
    // sort is a bijection on [0, T*M)) -> empty; mask/gate_rows/u_src/
    // node_counts use zeros: the original zeros-containment rationale of
    // _pack_quota_wire (rows skipped inside a quota-invariant drift window ==
    // slot 0 / gate 0, deterministic and shape-harmless), and doubly so because
    // when T == 0 the kernel never launches (short-circuit below) and the zero
    // planes are the correct answer (same as K1's i_send).
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
    // aclnn generated signature order: input tensors -> int attributes ->
    // output tensors (the msopgen aclnn convention; if this drop's generated
    // header orders things differently, aclnn_terrace_k2_pack.h is
    // authoritative -- a cluster-compile verification point; a signature
    // mismatch errors at compile time, fail loud).
    auto ret = aclnnTerraceK2PackGetWorkspaceSize(
        hAcl, iAcl, gAcl, world, n_experts, rpn, groups_m,
        pAcl, mAcl, grAcl, uAcl, cAcl, &workspaceSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS,
                "aclnnTerraceK2PackGetWorkspaceSize failed: ", ret,
                " (did host tiling reject this geometry? see "
                "op_host/terrace_k2_pack.cpp)");

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

// FRAGMENT, not TORCH_LIBRARY: sole definition rights over the terrace
// namespace belong to terrace_ops.cpp; this file only appends. autograd is
// carried by the Python-side autograd.Function (internal grafting notes, not
// published with this repo); no Autograd key is registered -- the K2 backward
// takes the composite chain, the kernel appears in forward only (roadmap
// decision).
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
