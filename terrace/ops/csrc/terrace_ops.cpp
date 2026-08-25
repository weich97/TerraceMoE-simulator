/**
 * terrace_ops -- torch.library binding: torch.ops.terrace.* -> aclnn two-phase calls.
 *
 * Compiled on the cluster only (python csrc/build_ext.py; needs torch_npu plus
 * the installed opp vendor package). Structure follows CANN's official
 * PytorchInvocation route: schema registration (TORCH_LIBRARY) + PrivateUse1
 * (NPU) implementation + Meta implementation (for fake-tensor shape inference).
 *
 * The aclnn symbols (aclnnTerracePassthrough{GetWorkspaceSize,}) come from
 * libcust_opapi.so wrapped by the opp package; build_ext.py links it directly
 * -- a missing symbol fails at compile time, fail loud, no dlopen lazy binding.
 *
 * K1 (terrace::k1_arrival, landed 2026-08-20):
 *   - schema: k1_arrival(Tensor rx, Tensor rslot, Tensor rgate, int quota,
 *     int epr, int rpn, int my_local) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
 *     = (send_buf, gate_pairs, r_idx, slot_idx, i_send);
 *   - functional spec = a bit-for-bit replica of the current arrival-side
 *     composite chain, see the op_kernel/terrace_k1_arrival.cpp file header
 *     (two-pass method + the stable-ordering bit-for-bit argument);
 *   - send_buf/gate_pairs are differentiable (backward = the composite chain's
 *     scatter-add/gather, carried by a Python-side autograd.Function);
 *     r_idx/slot_idx/i_send are index/count planes, mark_non_differentiable on
 *     the Python side;
 *   - int64 rslot goes straight into the kernel (int32 lo/hi scalar access, no
 *     int64 vector instructions issued).
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
// Op-specific headers generated once the opp vendor package is installed
// (vendors/<vendor>/op_api/include).
#include "aclnn_terrace_passthrough.h"
#include "aclnn_terrace_k1_arrival.h"

// torch_npu: grab the current NPU stream. Header/library paths injected by
// build_ext.py.
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {

// at::Tensor -> aclTensor view (no data copy). The caller guarantees contiguous.
aclTensor *MakeAclTensor(const at::Tensor &t)
{
    static const std::map<at::ScalarType, aclDataType> kDtype = {
        {at::kHalf, ACL_FLOAT16},
        {at::kBFloat16, ACL_BF16},
        {at::kFloat, ACL_FLOAT},
        {at::kLong, ACL_INT64},        // K1's rslot / r_idx / slot_idx / i_send
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
                " (did host tiling reject this shape? the template requires 32B "
                "divisibility, header comment of the build script ascendc/build.sh)");

    // Workspace comes from torch's NPU caching allocator; no manual
    // aclrtMalloc / lifetime management.
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
    return at::empty_like(x);   // shape inference: output shape/dtype == input
}

// ======================================================================================
// K1: the arrival-side fused chain (functional spec and the bit-for-bit
// argument live in the op_kernel/terrace_k1_arrival.cpp file header).
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
    // Host tiling rejects row widths not divisible into 32B (same fail-loud as
    // the template); give the human-readable error here first.
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
    // i_send uses zeros: when P == 0 the kernel never launches (short-circuit
    // below) and the zero histogram is the correct answer.
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
    // aclnn generated signature order: input tensors -> int attributes ->
    // output tensors (the msopgen aclnn convention; if this drop's generated
    // header orders things differently, aclnn_terrace_k1_arrival.h is
    // authoritative -- a cluster-compile verification point; a signature
    // mismatch errors at compile time, fail loud).
    auto ret = aclnnTerraceK1ArrivalGetWorkspaceSize(
        rxAcl, rslotAcl, rgateAcl, quota, epr, rpn, my_local,
        sendAcl, gateAcl, ridxAcl, slotAcl, isendAcl, &workspaceSize, &executor);
    TORCH_CHECK(ret == ACL_SUCCESS,
                "aclnnTerraceK1ArrivalGetWorkspaceSize failed: ", ret,
                " (did host tiling reject this geometry? see "
                "op_host/terrace_k1_arrival.cpp)");

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

// Schema and implementations register separately: autograd is carried by the
// Python-side autograd.Function (terrace/ops/__init__.py), no Autograd key here
// -- the K1/K2 backward takes the composite chain, the kernel appears in
// forward only (decided in internal design records, not published with this
// repo).
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
