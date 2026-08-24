/**
 * terrace_k1_arrival -- AscendC kernel:T-A2A 到达侧融合链(K1,C1 quota 线格式)。
 *
 * ============================== 功能规格(逐位复刻)==============================
 *
 * 复刻 terrace/ta2a_fwd.py::ta2a_moe_forward 与 ta2a_dispatch.py::ta2a_permute 里
 * C1 之后、Hop B 之前的整段现组合链(quota 快路径分支):
 *
 *     r_idx, slot_idx = _expand_arrival_quota(rslot)      # 配对展开(槽号表即配对表)
 *     owner = slot_idx // epr                             # 目的本地 rank(桶号)
 *     ordo  = _stable_argsort_small(owner, rpn)           # 稳定升序(owner 桶排序)
 *     r_idx, slot_idx = r_idx[ordo], slot_idx[ordo]       # 配对重排
 *     i_send = torch.bincount(owner, minlength=rpn)       # 每 peer 行数(置换盲)
 *     send_buf   = rx[r_idx]                              # [P, H] 展开发送缓冲
 *     gate_pairs = rgate.reshape(-1)[ordo]                # 配对同序 gate(致密表平铺)
 *
 * 输入:rx [R, H](fp16/bf16/fp32)、rslot [R, quota](int64,C1 线格式:每行升序
 * 槽号)、rgate [R, quota](与 rx 同 dtype);属性 quota/epr/rpn/my_local。
 * 输出:send_buf [P, H]、gate_pairs [P]、r_idx [P] int64、slot_idx [P] int64、
 * i_send [rpn] int64,P = R*quota。my_local 本段数学不使用 —— 接口按 K1 规格
 * 预留给到达链 Hop B 之后的专家序整理半段(exp_j = my_slot - my_local*epr 的
 * 同构桶排序,后续若融合为同一算子的第二形态时启用),tiling 已携带。
 *
 * ========================= 两遍法与稳定序逐位一致的论证 =========================
 *
 * 现链的排序原语 _stable_argsort_small(owner, rpn) 是**稳定升序**:输出序 =
 * 按 owner 升序分桶、桶内按配对的平铺位置 p(= r*quota + i)升序。它靠复合键
 * pos + n*key 实现(整数 < 2^24 时 float32 精确,否则回退整数稳定排序),两条
 * 路径给出**同一个置换**,所以 kernel 只需复刻"稳定桶排序"这个数学对象本身。
 *
 * 两遍法(计数排序):
 *   第 1 遍(计数):对配对平铺序 p = 0..P-1 扫描 rslot,owner(p) = slot(p)/epr,
 *     得直方图 hist[rpn](== i_send,bincount 对置换盲)。多核下每核负责一段连续
 *     配对区间 [p0, p1),但**每核都完整扫全量 [0, P)**,顺带在 p == p0 处快照
 *     "本核区间之前"的部分直方图 mine[]。全量扫换来的是:每核独立算出全局桶基址
 *     base[b] = sum_{b'<b} hist[b'] 与本核游标起点 cur[b] = base[b] + mine[b],
 *     **零核间同步**(无 SyncAll / 无 workspace 游标表 —— 版本分歧面最小)。
 *     代价是 rslot 平面(P*8B,万级配对 = 几百 KB)每核多读一遍,相对 [P, H]
 *     载荷搬运(每行 4-16KB)是零头。
 *   第 2 遍(展开写行):每核按平铺序扫自己的 [p0, p1),dst = cur[owner(p)]++,写
 *     r_idx[dst] = p / quota、slot_idx[dst] = slot(p)、gate_pairs[dst] = rgate 平铺
 *     第 p 元、send_buf[dst, :] = rx[p / quota, :]。
 *
 * 逐位一致证明:对任意桶 b,写进它的配对是全体 owner==b 的配对;核间按 p0 升序
 * 划分连续区间、核内按 p 升序扫描,而某核在桶 b 的游标起点恰好越过了所有更小 p
 * 的同桶配对(mine[b] 计的就是 [0, p0) 内 owner==b 的个数),所以桶内落位次序
 * == 平铺位置 p 的升序 == 现链稳定序的桶内序;桶间按 base[] 升序排布 == owner
 * 升序。故 dst 置换与 ordo 逐元素相等,五个输出全部逐位一致。r_idx 的取值用
 * p / quota 而非查表:_expand_arrival_quota 的 r_idx 本就是 arange(R) 按 quota
 * 展开,第 p 个配对的行号恒等于 p / quota,ordo 重排后即 ordo[j] / quota。
 *
 * ================================ 910C 硬件边界 ================================
 *
 *   - 无 int64 向量算术/移位/廉价排序(内部工程记录;路线图"硬件负面"),本 kernel
 *     不发出任何 int64 **向量**指令:int64 平面(rslot/r_idx/slot_idx/i_send)全部
 *     以 int32 对(小端 lo/hi)做标量读写 —— 槽号 < epr*rpn <= 63、行号 < R < 2^31、
 *     计数 <= P < 2^31,高 32 位恒 0,读低词、写 (lo, 0) 即位级完整的非负 int64。
 *     标量单元本身是通用核,uint32/uint64 标量算术不受向量指令集限制。
 *   - 排序整体免掉(游标法),owner 只是标量除法。
 *   - 载荷行搬运走 DataCopy(GM->UB->GM,double buffer 队列,与 passthrough 样板
 *     同款);host tiling 保证 H*sizeof(dtype) 是 32B 的倍数(真实 hidden
 *     2048/7168 恒真),rowTile 对齐由 host 保证,故无 DataCopyPad 需求。
 *   - rslot 的扫描经 UB staging(int32 视图 DataCopy 进 UB 后标量读),尾部不足
 *     32B 对齐的 <4 个配对退化为 GM 标量读 —— 不越界读。
 *
 * 集群编译验证点(拿不准处一律选了保守写法,标注如下):
 *   [V1] PipeBarrier<PIPE_ALL>:staging 与标量读之间的重锤同步。若该重载在本
 *        CANN drop 不可用,换 SetFlag/WaitFlag<HardEvent::MTE2_S> 与 S_MTE2 对。
 *   [V2] GlobalTensor<int32_t>::GetValue/SetValue 标量 GM 访问;若个别 drop 要求
 *        显式 dcache 刷新才对 host 可见,在 Process() 末尾加
 *        DataCacheCleanAndInvalid(ENTIRE_DATA_CACHE)。
 *   [V3] 标量 GM 写与 MTE3 DataCopy 写不同地址区间,无别名冲突;若位级验证发现
 *        i_send/r_idx 偶发旧值,先查 [V2]。
 *   [V4] DTYPE_RGATE 为 bf16 时以 uint16 位搬(sizeof 分支),不触碰 bf16 标量
 *        算术语义 —— 纯 move,与 torch 的 gather 逐位同。
 */
#include "kernel_operator.h"
// tiling 结构体是 host/kernel 共享的普通 C 结构体(CANN 9.0.0 ASC 体系)。
// 少了这一行 GET_TILING_DATA 展开时 TerraceK1ArrivalTilingData 未定义,kernel
// 编译在 binary 子构建里失败,主日志只剩下游的 "Target path not found"。
#include "terrace_k1_arrival_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;      // 载荷行搬运 double buffer
constexpr uint32_t MAX_RPN = 64;       // 桶数上限:slots = epr*rpn <= 63(现链硬约束)

class KernelTerraceK1Arrival {
public:
    __aicore__ inline KernelTerraceK1Arrival() {}

    __aicore__ inline void Init(GM_ADDR rx, GM_ADDR rslot, GM_ADDR rgate,
                                GM_ADDR sendBuf, GM_ADDR gatePairs, GM_ADDR rIdx,
                                GM_ADDR slotIdx, GM_ADDR iSend,
                                uint32_t rows, uint32_t hidden, uint32_t quota,
                                uint32_t epr, uint32_t rpn, uint32_t pairCount,
                                uint32_t pairsPerCoreBase, uint32_t pairsRem,
                                uint32_t slotChunk, uint32_t rowTile)
    {
        this->hidden = hidden;
        this->quota = quota;
        this->epr = epr;
        this->rpn = rpn;
        this->pairCount = pairCount;
        this->slotChunk = slotChunk;
        this->rowTile = rowTile;

        // 本核负责的连续配对区间 [p0, p1):前 pairsRem 核各多 1(与 host 一致)。
        uint32_t idx = static_cast<uint32_t>(GetBlockIdx());
        this->p0 = idx * pairsPerCoreBase + (idx < pairsRem ? idx : pairsRem);
        this->p1 = this->p0 + pairsPerCoreBase + (idx < pairsRem ? 1u : 0u);

        rxGm.SetGlobalBuffer((__gm__ DTYPE_RX *)rx,
                             static_cast<uint64_t>(rows) * hidden);
        sendGm.SetGlobalBuffer((__gm__ DTYPE_SEND_BUF *)sendBuf,
                               static_cast<uint64_t>(pairCount) * hidden);
        // int64 平面的 int32 视图(小端 lo/hi 对,见文件头"硬件边界")。
        rslotGm32.SetGlobalBuffer((__gm__ int32_t *)rslot,
                                  static_cast<uint64_t>(pairCount) * 2);
        ridxGm32.SetGlobalBuffer((__gm__ int32_t *)rIdx,
                                 static_cast<uint64_t>(pairCount) * 2);
        slotIdxGm32.SetGlobalBuffer((__gm__ int32_t *)slotIdx,
                                    static_cast<uint64_t>(pairCount) * 2);
        isendGm32.SetGlobalBuffer((__gm__ int32_t *)iSend,
                                  static_cast<uint64_t>(rpn) * 2);
        // gate 平面按位宽以整数位型搬运([V4]):两套视图都建,运行期按 sizeof 选。
        rgateGm16.SetGlobalBuffer((__gm__ uint16_t *)rgate, pairCount);
        gateOutGm16.SetGlobalBuffer((__gm__ uint16_t *)gatePairs, pairCount);
        rgateGm32.SetGlobalBuffer((__gm__ uint32_t *)rgate, pairCount);
        gateOutGm32.SetGlobalBuffer((__gm__ uint32_t *)gatePairs, pairCount);

        pipe.InitBuffer(rowQueue, BUFFER_NUM,
                        this->rowTile * sizeof(DTYPE_RX));
        // **必须有 VECOUT 队列。** 2026-08-24 在 passthrough 上实测出来的:
        // `TQue` 的位置决定它同步哪两条流水 —— VECIN 的 EnQue/DeQue 配的是
        // MTE2->V,VECOUT 配的才是 V->MTE3。CopyRow 是 GM->UB->GM,若只走 VECIN,
        // 那条 MTE3 就没人拦,搬出去的是还没搬完或已被下一轮 AllocTensor 复用的内容。
        // passthrough 上的症状是**输出大部分为零**(8x256 错 2047/2048)。
        pipe.InitBuffer(rowOutQueue, BUFFER_NUM,
                        this->rowTile * sizeof(DTYPE_RX));
        pipe.InitBuffer(slotStage, this->slotChunk * 2 * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        uint32_t hist[MAX_RPN];    // 全局直方图(每核独立算出同一份)
        uint32_t mine[MAX_RPN];    // [0, p0) 的部分直方图(本核游标起点的桶内偏置)
        uint32_t cur[MAX_RPN];     // 本核写游标
        for (uint32_t b = 0; b < rpn; b++) {
            hist[b] = 0;
            mine[b] = 0;
        }

        // ---- 第 1 遍:全量计数 + 在 p == p0 处快照部分直方图(零核间同步)----
        bool snapped = (p0 == 0);
        for (uint32_t chunkStart = 0; chunkStart < pairCount;
             chunkStart += slotChunk) {
            uint32_t n = pairCount - chunkStart;
            if (n > slotChunk) {
                n = slotChunk;
            }
            StageSlots(chunkStart, n);
            for (uint32_t i = 0; i < n; i++) {
                uint32_t p = chunkStart + i;
                if (!snapped && p == p0) {
                    for (uint32_t b = 0; b < rpn; b++) {
                        mine[b] = hist[b];
                    }
                    snapped = true;
                }
                int32_t s = SlotAt(chunkStart, i);
                if (s < 0 || static_cast<uint32_t>(s) >= epr * rpn) {
                    continue;   // 越界槽号:现链会在 bincount/gather 大声死;
                }               // kernel 侧跳过以免写穿(输入损坏场景,不承诺位级)
                hist[static_cast<uint32_t>(s) / epr]++;
            }
        }
        if (!snapped) {          // p0 == pairCount:本核无配对,mine 仅形式取值
            for (uint32_t b = 0; b < rpn; b++) {
                mine[b] = hist[b];
            }
        }

        // 全局桶基址 + 本核游标起点(稳定序的关键:base[b] + mine[b],证明见文件头)。
        uint32_t base = 0;
        for (uint32_t b = 0; b < rpn; b++) {
            cur[b] = base + mine[b];
            base += hist[b];
        }

        // i_send = 直方图(bincount 置换盲)。单核写,避免多核同址标量写([V3])。
        if (GetBlockIdx() == 0) {
            for (uint32_t b = 0; b < rpn; b++) {
                isendGm32.SetValue(2 * b, static_cast<int32_t>(hist[b]));
                isendGm32.SetValue(2 * b + 1, 0);
            }
        }

        // ---- 第 2 遍:本核区间 [p0, p1) 按平铺序展开写行 ----
        // 分块沿全局对齐网格(slotChunk 的整倍数,slotChunk 是 4 的倍数):p0 本身
        // 不保证 4 对齐,直接从 p0 stage 会让 DataCopy 的 GM 源地址落在 8B 而非
        // 32B 边界。网格化后每块的 GM 起点 = chunkStart*8B,chunkStart % 4 == 0
        // 恒真,32B 对齐恒真;本核只消费块内 [p0, p1) 的交集。
        for (uint32_t chunkStart = (p0 / slotChunk) * slotChunk; chunkStart < p1;
             chunkStart += slotChunk) {
            uint32_t n = pairCount - chunkStart;
            if (n > slotChunk) {
                n = slotChunk;
            }
            StageSlots(chunkStart, n);
            uint32_t iLo = (p0 > chunkStart) ? (p0 - chunkStart) : 0u;
            uint32_t iHi = (p1 - chunkStart < n) ? (p1 - chunkStart) : n;
            for (uint32_t i = iLo; i < iHi; i++) {
                uint32_t p = chunkStart + i;
                int32_t s = SlotAt(chunkStart, i);
                if (s < 0 || static_cast<uint32_t>(s) >= epr * rpn) {
                    continue;   // 与第 1 遍同一跳过准则,两遍一致故游标不漏不重
                }
                uint32_t b = static_cast<uint32_t>(s) / epr;
                uint32_t dst = cur[b]++;
                uint32_t srcRow = p / quota;
                // int64 输出 = (lo, hi=0) int32 对([V2])。
                ridxGm32.SetValue(2 * dst, static_cast<int32_t>(srcRow));
                ridxGm32.SetValue(2 * dst + 1, 0);
                slotIdxGm32.SetValue(2 * dst, s);
                slotIdxGm32.SetValue(2 * dst + 1, 0);
                if (sizeof(DTYPE_RGATE) == 2) {          // [V4] 纯位搬运
                    gateOutGm16.SetValue(dst, rgateGm16.GetValue(p));
                } else {
                    gateOutGm32.SetValue(dst, rgateGm32.GetValue(p));
                }
                CopyRow(srcRow, dst);
            }
        }
        // [V2] 若位级验证发现标量写平面偶发旧值,在此加 dcache 刷新。
    }

private:
    // rslot 配对块 [chunkStart, chunkStart+n) 进 UB:32B 对齐部分 DataCopy,
    // 尾部 <4 配对留给 SlotAt 的 GM 标量读。int32 视图下 4 配对 = 8 int32 = 32B。
    __aicore__ inline void StageSlots(uint32_t chunkStart, uint32_t n)
    {
        stagedAligned = n & ~3u;
        if (stagedAligned > 0) {
            LocalTensor<int32_t> sl = slotStage.Get<int32_t>();
            // [V1] 复用缓冲:上一块的标量读必须先于本次 MTE2 覆写完成。
            PipeBarrier<PIPE_ALL>();
            DataCopy(sl, rslotGm32[static_cast<uint64_t>(chunkStart) * 2],
                     stagedAligned * 2);
            // [V1] 标量读必须等 MTE2 落定。
            PipeBarrier<PIPE_ALL>();
        }
    }

    // 第 chunkStart+i 个配对的槽号(int64 低词;高词恒 0,见文件头)。
    __aicore__ inline int32_t SlotAt(uint32_t chunkStart, uint32_t i)
    {
        if (i < stagedAligned) {
            LocalTensor<int32_t> sl = slotStage.Get<int32_t>();
            return sl.GetValue(2 * i);
        }
        return rslotGm32.GetValue((static_cast<uint64_t>(chunkStart) + i) * 2);
    }

    // rx 第 srcRow 行 -> send_buf 第 dst 行:GM->UB->GM,rowTile 分段,double buffer。
    // **两条队列,不是一条**:VECIN 管 MTE2->V,VECOUT 管 V->MTE3。原来只有一条
    // VECIN,那道 MTE3 屏障没人插(见 InitBuffer 处的注释)。
    // 中间那次 UB->UB 拷贝不是浪费 —— 它是把两条队列配起来的那一环;
    // 而且 double buffer 下 tile i+1 的 MTE2 能与 tile i 的 MTE3 重叠,
    // 比用 PipeBarrier<PIPE_ALL> 串起来快(StageSlots 那处用 PipeBarrier 是因为
    // 它是 TBuf、且只搬一小块,不在带宽路径上)。
    __aicore__ inline void CopyRow(uint32_t srcRow, uint32_t dst)
    {
        uint64_t srcBase = static_cast<uint64_t>(srcRow) * hidden;
        uint64_t dstBase = static_cast<uint64_t>(dst) * hidden;
        for (uint32_t t = 0; t < hidden; t += rowTile) {
            uint32_t len = hidden - t;
            if (len > rowTile) {
                len = rowTile;
            }
            LocalTensor<DTYPE_RX> inBuf = rowQueue.AllocTensor<DTYPE_RX>();
            DataCopy(inBuf, rxGm[srcBase + t], len);
            rowQueue.EnQue(inBuf);                  // MTE2 -> V

            inBuf = rowQueue.DeQue<DTYPE_RX>();
            LocalTensor<DTYPE_RX> outBuf = rowOutQueue.AllocTensor<DTYPE_RX>();
            DataCopy(outBuf, inBuf, len);           // UB -> UB
            rowOutQueue.EnQue(outBuf);              // V -> MTE3
            rowQueue.FreeTensor(inBuf);

            outBuf = rowOutQueue.DeQue<DTYPE_RX>();
            DataCopy(sendGm[dstBase + t], outBuf, len);
            rowOutQueue.FreeTensor(outBuf);
        }
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> rowQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rowOutQueue;
    TBuf<TPosition::VECCALC> slotStage;

    GlobalTensor<DTYPE_RX> rxGm;
    GlobalTensor<DTYPE_SEND_BUF> sendGm;
    GlobalTensor<int32_t> rslotGm32, ridxGm32, slotIdxGm32, isendGm32;
    GlobalTensor<uint16_t> rgateGm16, gateOutGm16;
    GlobalTensor<uint32_t> rgateGm32, gateOutGm32;

    uint32_t hidden = 0, quota = 1, epr = 1, rpn = 1;
    uint32_t pairCount = 0, slotChunk = 0, rowTile = 0;
    uint32_t p0 = 0, p1 = 0;
    uint32_t stagedAligned = 0;
};

// 入口符号:算子类型 TerraceK1Arrival 的 snake_case(msopgen 骨架约定)。
// DTYPE_RX / DTYPE_RGATE / DTYPE_SEND_BUF 宏由构建系统按 dtype 组合实例化注入。
extern "C" __global__ __aicore__ void terrace_k1_arrival(
    GM_ADDR rx, GM_ADDR rslot, GM_ADDR rgate, GM_ADDR send_buf, GM_ADDR gate_pairs,
    GM_ADDR r_idx, GM_ADDR slot_idx, GM_ADDR i_send, GM_ADDR workspace,
    GM_ADDR tiling)
{
    // ASC 体系要求 kernel 显式声明本算子的 tiling 结构体,GET_TILING_DATA 的宏
    // 展开(构建期按 SOC 现生成)才知道往哪个类型反序列化。
    REGISTER_TILING_DEFAULT(TerraceK1ArrivalTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.pairCount == 0) {
        return;                       // R == 0:五个输出全空/全零,host 已按形状分配
    }
    KernelTerraceK1Arrival op;
    op.Init(rx, rslot, rgate, send_buf, gate_pairs, r_idx, slot_idx, i_send,
            tilingData.rows, tilingData.hidden, tilingData.quota, tilingData.epr,
            tilingData.rpn, tilingData.pairCount, tilingData.pairsPerCoreBase,
            tilingData.pairsRem, tilingData.slotChunk, tilingData.rowTile);
    op.Process();
}
