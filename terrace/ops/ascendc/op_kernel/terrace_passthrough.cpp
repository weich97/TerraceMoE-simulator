/**
 * terrace_passthrough -- AscendC kernel side (engineering-pipeline template, no compute logic).
 *
 * Semantics: y = x, element-for-element copy-out. Its only reason to exist is to compile
 * and run the chain
 *   msopgen project -> opp package -> aclnn -> torch.library
 * end to end on the cluster (see the header of the build script ascendc/build.sh).
 * **The K1/K2 kernel bodies are NOT here**; tomorrow they get their own files
 * (k1_arrival.cpp / k2_pack.cpp) following the post-C1 chain, and this file stays as the
 * full-chain smoke baseline.
 *
 * The structure follows the three-stage paradigm (CopyIn -> [Compute] -> CopyOut) of the
 * official Ascend AddCustom sample (gitee.com/ascend/samples, Apache-2.0,
 * Copyright Huawei Technologies Co., Ltd.); this implementation is written independently,
 * attribution in the NOTICE at the repo root. Three stages: double buffer; passthrough has
 * no Compute stage, data goes straight through GM -> UB -> GM.
 *
 * What to change here for the real K1 (compare as a whole, not a line-by-line patch):
 *   - inputs become the rmask bit plane (int64 GM); outputs become the three planes
 *     r_idx/slot_idx/i_send;
 *   - two-pass method: pass 1 popcount counting + inter-core prefix cursors (cursor table
 *     in workspace), pass 2 expands and writes rows by cursor -- bitwise identical to the
 *     existing composed chain _expand_arrival + stable owner sort (the nine-section
 *     mapping in internal design records (not published with this repo));
 *   - 910C has no int64 shifts / no cheap int64 sort: bit extraction takes the exact
 *     float path for slots<=24 and the per-bit mask path for >24; no int64 sorting at all
 *     (the cursor method eliminates it);
 *   - tiling gains slots/quota/epr/rpn fields (op_host side changes in lockstep).
 */
#include "kernel_operator.h"
// The tiling struct is a plain C struct shared by host and kernel (CANN 9.0.0 ASC system;
// see the header of terrace_passthrough_tiling.h).
#include "terrace_passthrough_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;   // double buffer: copy-in/copy-out pipeline

class KernelTerracePassthrough {
public:
    __aicore__ inline KernelTerracePassthrough() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                                uint32_t tileNum)
    {
        // Uniform per-core partition: host-side tiling guarantees totalLength is
        // divisible by (blockDim * tileNum * ALIGN) (see the alignment notes in op_host;
        // if unmet, this kernel is simply not dispatched and the torch side takes the
        // composed chain -- the template keeps it simple; the real K1 changes this to
        // handle arbitrary lengths with DataCopyPad / a tail-block branch).
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * GetBlockIdx(),
                            this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * GetBlockIdx(),
                            this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        // **The VECOUT queue is mandatory.** 2026-08-24: originally only inQueueX
        // existed and CopyOut issued MTE3 straight from the VECIN tensor -- the
        // MTE2/MTE3 barrier is inserted by the queue framework at EnQue/DeQue pairs, so
        // without this queue nobody inserts it, and what gets written out is data not
        // yet fully landed or already recycled.
        // Measured symptom: mostly-zero output (8x256, 2047/2048 wrong).
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute();    // The identity op's "compute" is just a move within UB.
                          // **This stage cannot be skipped** -- it is what pairs the
                          // VECIN and VECOUT queues so the barriers get inserted.
                          // The real K1 replaces this with bit extraction/counting/expansion.
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        DataCopy(yLocal, xLocal, this->tileLength);   // UB -> UB, identity
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength = 0;
    uint32_t tileNum = 0;
    uint32_t tileLength = 0;
};

// The entry symbol name must be the snake_case of the op type (msopgen skeleton
// convention); the DTYPE_X/DTYPE_Y macros are injected by the build system at per-dtype
// instantiation, so the kernel source hard-codes no dtype.
extern "C" __global__ __aicore__ void terrace_passthrough(GM_ADDR x, GM_ADDR y,
                                                          GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TerracePassthroughTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelTerracePassthrough op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
