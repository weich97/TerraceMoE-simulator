/**
 * terrace_passthrough -- AscendC kernel 侧(工程链路样板,无计算逻辑)。
 *
 * 语义:y = x,逐元素原样拷出。存在的唯一理由是把
 *   msopgen 工程 -> opp 包 -> aclnn -> torch.library
 * 这条链在集群上端到端编译并跑通(构建脚本 ascendc/build.sh 的头注)。**K1/K2 的 kernel 本体不在这里**,
 * 明天按 C1 落地后的链另起文件(k1_arrival.cpp / k2_pack.cpp),本文件保留作
 * 全链路冒烟基准。
 *
 * 结构遵循 Ascend 官方 AddCustom 样例(gitee.com/ascend/samples,Apache-2.0,
 * Copyright Huawei Technologies Co., Ltd.)的三段式范式(CopyIn -> [Compute] ->
 * CopyOut);本实现为独立编写,署名见仓库根 NOTICE。三段式:
 * double buffer;passthrough 没有 Compute 段,数据 GM -> UB -> GM 直通。
 *
 * K1 正式实现时改这里(整体对照,不是逐行 patch):
 *   - 输入换成 rmask 位平面(int64 GM),输出 r_idx/slot_idx/i_send 三平面;
 *   - 两遍法:第 1 遍 popcount 计数 + 核间前缀游标(workspace 里放游标表),
 *     第 2 遍按游标展开写行 —— 与现组合链 _expand_arrival + 稳定 owner 排序
 *     逐位一致(内部设计记录(未随仓发布)九段映射);
 *   - 910C 无 int64 移位/廉价 int64 sort:位抽取按 slots<=24 走 float 精确路径,
 *     >24 走逐位掩码,不做任何 int64 排序(排序用游标法免掉);
 *   - tiling 增加 slots/quota/epr/rpn 字段(op_host 侧同步改)。
 */
#include "kernel_operator.h"
// tiling 结构体是 host/kernel 共享的普通 C 结构体(CANN 9.0.0 ASC 体系;
// 见 terrace_passthrough_tiling.h 文件头)。
#include "terrace_passthrough_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;   // double buffer:搬入搬出流水

class KernelTerracePassthrough {
public:
    __aicore__ inline KernelTerracePassthrough() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                                uint32_t tileNum)
    {
        // 均匀按核切块:host 侧 tiling 保证 totalLength 能被
        // (blockDim * tileNum * ALIGN) 整除(见 op_host 的对齐说明;不满足直接
        // 不下发本 kernel,由 torch 侧走组合链 —— 样板从简,K1 正式实现时改这里:
        // 用 DataCopyPad / 尾块分支处理任意长度)。
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * GetBlockIdx(),
                            this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * GetBlockIdx(),
                            this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        // **必须有 VECOUT 队列。** 2026-08-24:原来只有 inQueueX,CopyOut 直接从
        // VECIN 的张量发 MTE3 —— MTE2/MTE3 之间的屏障是队列框架按 EnQue/DeQue
        // 配对插的,少了这条队列就没人插,写出去的是还没搬完或已被复用的内容。
        // 实测症状:输出大部分为零(8x256 错 2047/2048)。
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute();    // 恒等算子的"计算"就是 UB 内搬一下。**这一段不能省** ——
                          // 它把 VECIN 与 VECOUT 两条队列配对起来,屏障才有人插。
                          // K1 正式实现在这里换成位抽取/计数/展开。
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
        DataCopy(yLocal, xLocal, this->tileLength);   // UB -> UB,恒等
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

// 入口符号名必须是算子类型的 snake_case(msopgen 生成骨架的约定);DTYPE_X/DTYPE_Y
// 宏由构建系统按数据类型实例化时注入,kernel 源不写死 dtype。
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
