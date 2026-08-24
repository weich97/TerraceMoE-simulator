/**
 * terrace_k2_pack -- AscendC kernel:T-A2A 发送侧融合打包链(K2,C1 quota 线格式)。
 *
 * ============================== 功能规格(逐位复刻)==============================
 *
 * 复刻 quota 快路径(等额配额,groups_m=M)下、Hop A 计数交换之前的整段现组合链
 * (terrace/ta2a.py::plan_ta2a 快路径 + terrace/ta2a_fwd.py 的发送段;
 * ta2a_moe_forward 与 ta2a_dispatch.ta2a_permute[_overlap] 同段同构):
 *
 *     u_src, u_node, node_counts, inverse = plan_ta2a(expert_idx, world,
 *                                                     n_experts, rpn, groups_m=M)
 *     payload = hidden[u_src]                       # 去重 gather,节点主序
 *     mask, gate_rows = _pack_quota_wire(expert_idx, gates, inverse, payload,
 *                                        n_rows, slots, quota, n_experts)
 *
 * 输入:hidden [T, H](fp16/bf16/fp32)、expert_idx [T, k](int64)、gates [T, k]
 * (与 hidden 同 dtype —— C1 圆整点契约,gate 平面从 payload 派生,失配在 torch
 * 侧大声死);属性 world / n_experts / rpn / groups_m。
 * 输出:payload [T*M, H]、mask [T*M, quota](升序槽号表,int64)、gate_rows
 * [T*M, quota]、u_src [T*M] int64、node_counts [n_nodes] int64;quota = k/M。
 * u_node 不输出:三处调用方都不用它(ta2a_permute 丢弃、overlap 半边用 `_` 接、
 * 融合前向不再读);inverse 不输出:quota 支路上它只被打包消费,kernel 吸掉了。
 *
 * 行内无升序担保(kernel 内建行排序,见下),所以同一枚 kernel 同时逐位覆盖
 * _pack_quota_wire 的 sorted_rows=True(接缝,行升序时行排序恒等)与 =False
 * (融合前向,行内一次排序)两个入口。
 *
 * ========================= 两遍法与稳定序逐位一致的论证 =========================
 *
 * 现链的行序由 plan_ta2a 固定:occ [T, n_nodes] 占用表按**节点主序**平铺
 * (node_first = occ.t().reshape(-1)),置位的升序位置即发送行 —— 行按
 * (node 升序, token 升序) 排布;u_src = sel % T,counts[n] = 触达 n 的 token 数。
 * quota 支路的 searchsorted 压缩与通用支路的 argsort 压缩逐位同
 * (tests/test_ta2a.py::test_fastpath_sortfree_construction_is_bitwise_equal)。
 *
 * kernel 的同一数学对象,枚举单位是 run(等额配额下 token 的 k 个专家升序后
 * 恰分成 M 段,每段 quota 个、整段落在同一节点 —— plan_ta2a 首调 + 每 256 调
 * 验证的不变量):
 *   run 枚举序 p = t*M + j(t 升序、行内段号 j 升序);run 的目的节点
 *   d(t,j) = e_sorted[t, j*quota] / slots,随 j 严格升序(行升序 => 节点非降,
 *   等额 => 每节点恰一段 => 严格升序)。
 *
 *   第 1 遍(计数):对**槽位平面**全量扫描(f = 0..T*k-1),slotHist[e/slots]++,
 *     在 f == t0*k 处快照本核区间之前的部分直方图 mineSlots[]。等额配额下节点 d
 *     的槽位数 = quota * (touched-token 数),故 hist[d] = slotHist[d]/quota 与
 *     mine[d] = mineSlots[d]/quota 的除法**精确**(整除),且不需要行排序 ——
 *     第 1 遍零排序、零核间同步(K1 同款:全量扫换每核独立算出全局桶基址
 *     base[d] 与本核游标起点 cur[d] = base[d] + mine[d],无 SyncAll/无 workspace
 *     游标表)。代价是 expert_idx 平面(T*k*8B,万级 = 几百 KB)每核多读一遍,
 *     相对 [T*M, H] 载荷搬运(每行 4-16KB)是零头。
 *   第 2 遍(打包写行):每核按 token 序扫自己的 [t0, t1),行内插入排序
 *     (k <= 64,标量;专家号行内互异,置换唯一 —— 与 _pack_quota_wire 的
 *     float32-key argsort 同一置换,见 ta2a_fwd 的 <2^24 精确性论证),对每个
 *     run:dst = cur[d]++,写 u_src[dst] = t、mask[dst, i] = e_sorted[jq+i] %
 *     slots(升序,C1 线上契约)、gate_rows[dst, i] = gates 行内同置换第 jq+i
 *     元(纯位搬运)、payload[dst, :] = hidden[t, :]。
 *
 * 逐位一致证明:对任意节点桶 d,写进它的 run 是全体 d(t,j)==d 的 run;核间按
 * token 升序划分连续区间、核内按 (t, j) 升序扫描,而某核在桶 d 的游标起点恰好
 * 越过了所有更小 t 的同桶 run(mine[d] 计的就是 token < t0 的部分);等额配额下
 * 每 token 对每节点至多一个 run,故桶内落位次序 == token 升序 == plan 的桶内序;
 * 桶间按 base[] 升序 == 节点升序。故 dst 置换与 plan 的 sel 枚举逐元素相等 =>
 * payload/u_src/node_counts 逐位同。mask/gate_rows:_pack_quota_wire 的行散射
 * 目标 rof[t*M+j] = inverse[t, order[j*quota]] 恰是 (t, d(t,j)) 的发送行 ==
 * dst(t,j),散射值 = 行升序置换后的槽号/gate 段 == kernel 的写值;gate 全程
 * 无浮点算术(位搬运),与 torch 的 gather 逐位同。
 *
 * ================================ 910C 硬件边界 ================================
 *
 *   - 无 int64 向量算术/移位/廉价排序(内部工程记录;路线图"硬件负面"),本 kernel
 *     不发出任何 int64 **向量**指令:int64 平面(expert_idx/mask/u_src/node_counts)
 *     全部以 int32 对(小端 lo/hi)做标量读写 —— 专家号 < n_experts < 2^31(host
 *     拦截;现链上游 float32 键还限 < 2^24)、token 号 < T < 2^31、计数 <= T*M <
 *     2^31,高 32 位恒 0,读低词、写 (lo, 0) 即位级完整的非负 int64。
 *   - 排序仅行内 k 元素插入排序(标量单元,k <= 64);桶排序整体免掉(游标法)。
 *   - 载荷行搬运走 DataCopy(GM->UB->GM,double buffer 队列,与 K1/passthrough
 *     同款);host tiling 保证 H*sizeof(dtype) 是 32B 的倍数(真实 hidden
 *     2048/7168 恒真),故无 DataCopyPad 需求。
 *   - 第 1 遍 expert_idx 的扫描经 UB staging(int32 视图 DataCopy 进 UB 后标量
 *     读),尾部不足 32B 对齐的 <4 个 int64 退化为 GM 标量读 —— 不越界读。
 *     第 2 遍只读本核区间(1/usedCores 的平面),直接 GM 标量读,不与第 1 遍的
 *     staging 缓冲打架。
 *
 * 损坏输入的收容(不承诺位级,只承诺不写穿):专家号越界(<0 或 >= n_experts)
 * 两遍同判据跳过;等额配额不变量漂移(plan_ta2a 每 256 调之间未被验证的窗口)
 * 下游标可能越过桶界,dst >= nRows 的写整行跳过 —— torch 侧 mask/gate_rows/
 * u_src/node_counts 以 zeros 分配(_pack_quota_wire 的 zeros 收容原文:缺行 ==
 * 槽 0/gate 0,形状无害),现组合链在同窗口则是 searchsorted 越界行号在下游
 * gather 处大声死 —— 两种收容都不静默出假数,过闸以正常输入的逐位等价为准。
 *
 * 集群编译验证点([V1]-[V4] 与 K1 完全同款,拿不准处一律保守写法):
 *   [V1] PipeBarrier<PIPE_ALL>:staging 与标量读之间的重锤同步。若该重载在本
 *        CANN drop 不可用,换 SetFlag/WaitFlag<HardEvent::MTE2_S> 与 S_MTE2 对。
 *   [V2] GlobalTensor<int32_t>::GetValue/SetValue 标量 GM 访问;若个别 drop 要求
 *        显式 dcache 刷新才对 host 可见,在 Process() 末尾加
 *        DataCacheCleanAndInvalid(ENTIRE_DATA_CACHE)。
 *   [V3] 标量 GM 写与 MTE3 DataCopy 写不同地址区间,无别名冲突;若位级验证发现
 *        u_src/mask/node_counts 偶发旧值,先查 [V2]。
 *   [V4] DTYPE_GATES 为 bf16 时以 uint16 位搬(sizeof 分支),不触碰 bf16 标量
 *        算术语义 —— 纯 move,与 torch 的 gather 逐位同。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;      // 载荷行搬运 double buffer
constexpr uint32_t MAX_NODES = 256;    // 节点桶上限(host tiling 同值拦截)
constexpr uint32_t MAX_K = 64;         // 行内排序数组上限(host tiling 同值拦截)

class KernelTerraceK2Pack {
public:
    __aicore__ inline KernelTerraceK2Pack() {}

    __aicore__ inline void Init(GM_ADDR hidden, GM_ADDR expertIdx, GM_ADDR gates,
                                GM_ADDR payload, GM_ADDR mask, GM_ADDR gateRows,
                                GM_ADDR uSrc, GM_ADDR nodeCounts,
                                uint32_t tokens, uint32_t hiddenW, uint32_t topk,
                                uint32_t quota, uint32_t groupsM, uint32_t slots,
                                uint32_t nNodes, uint32_t nExperts, uint32_t nRows,
                                uint32_t tokensPerCoreBase, uint32_t tokensRem,
                                uint32_t idxChunk, uint32_t rowTile)
    {
        this->tokens = tokens;
        this->hiddenW = hiddenW;
        this->topk = topk;
        this->quota = quota;
        this->groupsM = groupsM;
        this->slots = slots;
        this->nNodes = nNodes;
        this->nExperts = nExperts;
        this->nRows = nRows;
        this->idxChunk = idxChunk;
        this->rowTile = rowTile;

        // 本核负责的连续 token 区间 [t0, t1):前 tokensRem 核各多 1(与 host 一致)。
        uint32_t idx = static_cast<uint32_t>(GetBlockIdx());
        this->t0 = idx * tokensPerCoreBase + (idx < tokensRem ? idx : tokensRem);
        this->t1 = this->t0 + tokensPerCoreBase + (idx < tokensRem ? 1u : 0u);

        hiddenGm.SetGlobalBuffer((__gm__ DTYPE_HIDDEN *)hidden,
                                 static_cast<uint64_t>(tokens) * hiddenW);
        payloadGm.SetGlobalBuffer((__gm__ DTYPE_PAYLOAD *)payload,
                                  static_cast<uint64_t>(nRows) * hiddenW);
        // int64 平面的 int32 视图(小端 lo/hi 对,见文件头"硬件边界")。
        idxGm32.SetGlobalBuffer((__gm__ int32_t *)expertIdx,
                                static_cast<uint64_t>(tokens) * topk * 2);
        maskGm32.SetGlobalBuffer((__gm__ int32_t *)mask,
                                 static_cast<uint64_t>(nRows) * quota * 2);
        usrcGm32.SetGlobalBuffer((__gm__ int32_t *)uSrc,
                                 static_cast<uint64_t>(nRows) * 2);
        countsGm32.SetGlobalBuffer((__gm__ int32_t *)nodeCounts,
                                   static_cast<uint64_t>(nNodes) * 2);
        // gate 平面按位宽以整数位型搬运([V4]):两套视图都建,运行期按 sizeof 选。
        uint64_t flat = static_cast<uint64_t>(tokens) * topk;
        gatesGm16.SetGlobalBuffer((__gm__ uint16_t *)gates, flat);
        gateOutGm16.SetGlobalBuffer((__gm__ uint16_t *)gateRows, flat);
        gatesGm32.SetGlobalBuffer((__gm__ uint32_t *)gates, flat);
        gateOutGm32.SetGlobalBuffer((__gm__ uint32_t *)gateRows, flat);

        pipe.InitBuffer(rowQueue, BUFFER_NUM,
                        this->rowTile * sizeof(DTYPE_HIDDEN));
        // **必须有 VECOUT 队列。** 2026-08-24 在 passthrough 上实测:`TQue` 的位置
        // 决定它同步哪两条流水 —— VECIN 配 MTE2->V,VECOUT 才配 V->MTE3。
        // CopyRow 是 GM->UB->GM,只走 VECIN 时那道 MTE3 没人拦,搬出去的是脏数据。
        // 这个错沿注释链 passthrough -> K1 -> K2 传了三份,全部**静默**:
        // 编译过、加载过、执行过,只有逐位比对才现形(8x256 错 2047/2048)。
        pipe.InitBuffer(rowOutQueue, BUFFER_NUM,
                        this->rowTile * sizeof(DTYPE_HIDDEN));
        pipe.InitBuffer(idxStage, this->idxChunk * 2 * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        uint32_t slotHist[MAX_NODES];  // 全局槽位直方图(每核独立算出同一份)
        uint32_t mineSlots[MAX_NODES]; // token < t0 的部分槽位直方图
        uint32_t cur[MAX_NODES];       // 本核写游标(run 粒度)
        for (uint32_t b = 0; b < nNodes; b++) {
            slotHist[b] = 0;
            mineSlots[b] = 0;
        }

        // ---- 第 1 遍:槽位平面全量计数 + 在 f == t0*topk 处快照(零核间同步,
        // 零排序 —— 等额配额下 run 计数 = 槽位计数 / quota,整除,见文件头)----
        uint32_t flatTotal = tokens * topk;
        uint32_t f0 = t0 * topk;
        bool snapped = (f0 == 0);
        for (uint32_t chunkStart = 0; chunkStart < flatTotal;
             chunkStart += idxChunk) {
            uint32_t n = flatTotal - chunkStart;
            if (n > idxChunk) {
                n = idxChunk;
            }
            StageIdx(chunkStart, n);
            for (uint32_t i = 0; i < n; i++) {
                uint32_t f = chunkStart + i;
                if (!snapped && f == f0) {
                    for (uint32_t b = 0; b < nNodes; b++) {
                        mineSlots[b] = slotHist[b];
                    }
                    snapped = true;
                }
                int32_t e = IdxAt(chunkStart, i);
                if (e < 0 || static_cast<uint32_t>(e) >= nExperts) {
                    continue;   // 越界专家号:现链在 plan 的 gather/scatter 处大声
                }               // 死;kernel 侧跳过以免写穿(损坏输入,不承诺位级)
                slotHist[static_cast<uint32_t>(e) / slots]++;
            }
        }
        if (!snapped) {          // t0 == tokens:本核无 token,mine 仅形式取值
            for (uint32_t b = 0; b < nNodes; b++) {
                mineSlots[b] = slotHist[b];
            }
        }

        // 全局桶基址 + 本核游标起点(稳定序的关键:base[b] + mine[b],证明见文件
        // 头)。node_counts = run 直方图,单核写([V3] 同 K1 的 i_send)。
        uint32_t base = 0;
        for (uint32_t b = 0; b < nNodes; b++) {
            uint32_t h = slotHist[b] / quota;          // 等额配额下整除,精确
            cur[b] = base + mineSlots[b] / quota;
            if (GetBlockIdx() == 0) {
                countsGm32.SetValue(2 * b, static_cast<int32_t>(h));
                countsGm32.SetValue(2 * b + 1, 0);
            }
            base += h;
        }

        // ---- 第 2 遍:本核区间 [t0, t1) 按 token 序行排序 + 逐 run 打包写行 ----
        uint32_t eVal[MAX_K];    // 行内专家号(int64 低词;高词恒 0,见文件头)
        uint32_t gBits[MAX_K];   // 行内 gate 原始位(16/32 位,纯搬运 [V4])
        bool eOk[MAX_K];         // 行内逐元素越界标记(两遍同判据)
        for (uint32_t t = t0; t < t1; t++) {
            uint64_t rowBase = static_cast<uint64_t>(t) * topk;
            for (uint32_t j = 0; j < topk; j++) {
                int32_t lo = idxGm32.GetValue((rowBase + j) * 2);
                eOk[j] = !(lo < 0 || static_cast<uint32_t>(lo) >= nExperts);
                eVal[j] = static_cast<uint32_t>(lo);
                gBits[j] = (sizeof(DTYPE_GATES) == 2)
                               ? static_cast<uint32_t>(gatesGm16.GetValue(rowBase + j))
                               : gatesGm32.GetValue(rowBase + j);
            }
            // 行内插入排序,升序按专家号(行内互异 => 置换唯一,== 现链
            // argsort(float32) 的置换;插入排序稳定,重复号的损坏输入下仍确定)。
            for (uint32_t a = 1; a < topk; a++) {
                uint32_t ev = eVal[a];
                uint32_t gv = gBits[a];
                bool ok = eOk[a];
                int32_t b = static_cast<int32_t>(a) - 1;
                while (b >= 0 && eVal[b] > ev) {
                    eVal[b + 1] = eVal[b];
                    gBits[b + 1] = gBits[b];
                    eOk[b + 1] = eOk[b];
                    b--;
                }
                eVal[b + 1] = ev;
                gBits[b + 1] = gv;
                eOk[b + 1] = ok;
            }
            for (uint32_t j = 0; j < groupsM; j++) {
                uint32_t lead = j * quota;
                if (!eOk[lead]) {
                    continue;    // run 首元越界:与第 1 遍同判据(该槽位没进直方图)
                }
                uint32_t d = eVal[lead] / slots;       // < nNodes(lead < nExperts)
                uint32_t dst = cur[d]++;
                if (dst >= nRows) {
                    continue;    // 配额不变量漂移的收容:跳过写,不写穿(文件头)
                }
                usrcGm32.SetValue(2 * static_cast<uint64_t>(dst),
                                  static_cast<int32_t>(t));
                usrcGm32.SetValue(2 * static_cast<uint64_t>(dst) + 1, 0);
                for (uint32_t i = 0; i < quota; i++) {
                    uint32_t p = lead + i;
                    uint64_t o = static_cast<uint64_t>(dst) * quota + i;
                    // 越界成员写槽 0/gate 0(zeros 收容,_pack_quota_wire 原文)。
                    int32_t s = eOk[p] ? static_cast<int32_t>(eVal[p] % slots) : 0;
                    maskGm32.SetValue(2 * o, s);
                    maskGm32.SetValue(2 * o + 1, 0);
                    if (sizeof(DTYPE_GATES) == 2) {    // [V4] 纯位搬运
                        gateOutGm16.SetValue(o, eOk[p]
                            ? static_cast<uint16_t>(gBits[p]) : uint16_t(0));
                    } else {
                        gateOutGm32.SetValue(o, eOk[p] ? gBits[p] : 0u);
                    }
                }
                CopyRow(t, dst);
            }
        }
        // [V2] 若位级验证发现标量写平面偶发旧值,在此加 dcache 刷新。
    }

private:
    // expert_idx 平铺块 [chunkStart, chunkStart+n) 进 UB:32B 对齐部分 DataCopy,
    // 尾部 <4 个 int64 留给 IdxAt 的 GM 标量读。int32 视图下 4 个 int64 = 8 int32
    // = 32B。仅第 1 遍使用(全量扫);第 2 遍只读本核 1/usedCores 区间,GM 标量读。
    __aicore__ inline void StageIdx(uint32_t chunkStart, uint32_t n)
    {
        stagedAligned = n & ~3u;
        stagedBase = chunkStart;
        if (stagedAligned > 0) {
            LocalTensor<int32_t> sl = idxStage.Get<int32_t>();
            // [V1] 复用缓冲:上一块的标量读必须先于本次 MTE2 覆写完成。
            PipeBarrier<PIPE_ALL>();
            DataCopy(sl, idxGm32[static_cast<uint64_t>(chunkStart) * 2],
                     stagedAligned * 2);
            // [V1] 标量读必须等 MTE2 落定。
            PipeBarrier<PIPE_ALL>();
        }
    }

    // 第 chunkStart+i 个平铺槽位的专家号(int64 低词;高词恒 0,见文件头)。
    __aicore__ inline int32_t IdxAt(uint32_t chunkStart, uint32_t i)
    {
        if (i < stagedAligned) {
            LocalTensor<int32_t> sl = idxStage.Get<int32_t>();
            return sl.GetValue(2 * i);
        }
        return idxGm32.GetValue((static_cast<uint64_t>(chunkStart) + i) * 2);
    }

    // hidden 第 srcTok 行 -> payload 第 dst 行:GM->UB->GM,rowTile 分段,double
    // **两条队列**:VECIN 管 MTE2->V,VECOUT 管 V->MTE3。原先照抄的
    // 「K1/passthrough 样板」只有一条 VECIN —— 那个样板本身是错的。
    __aicore__ inline void CopyRow(uint32_t srcTok, uint32_t dst)
    {
        uint64_t srcBase = static_cast<uint64_t>(srcTok) * hiddenW;
        uint64_t dstBase = static_cast<uint64_t>(dst) * hiddenW;
        for (uint32_t t = 0; t < hiddenW; t += rowTile) {
            uint32_t len = hiddenW - t;
            if (len > rowTile) {
                len = rowTile;
            }
            LocalTensor<DTYPE_HIDDEN> inBuf = rowQueue.AllocTensor<DTYPE_HIDDEN>();
            DataCopy(inBuf, hiddenGm[srcBase + t], len);
            rowQueue.EnQue(inBuf);                  // MTE2 -> V

            inBuf = rowQueue.DeQue<DTYPE_HIDDEN>();
            LocalTensor<DTYPE_HIDDEN> outBuf = rowOutQueue.AllocTensor<DTYPE_HIDDEN>();
            DataCopy(outBuf, inBuf, len);           // UB -> UB
            rowOutQueue.EnQue(outBuf);              // V -> MTE3
            rowQueue.FreeTensor(inBuf);

            outBuf = rowOutQueue.DeQue<DTYPE_HIDDEN>();
            DataCopy(payloadGm[dstBase + t], outBuf, len);
            rowOutQueue.FreeTensor(outBuf);
        }
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> rowQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rowOutQueue;
    TBuf<TPosition::VECCALC> idxStage;

    GlobalTensor<DTYPE_HIDDEN> hiddenGm;
    GlobalTensor<DTYPE_PAYLOAD> payloadGm;
    GlobalTensor<int32_t> idxGm32, maskGm32, usrcGm32, countsGm32;
    GlobalTensor<uint16_t> gatesGm16, gateOutGm16;
    GlobalTensor<uint32_t> gatesGm32, gateOutGm32;

    uint32_t tokens = 0, hiddenW = 0, topk = 1, quota = 1, groupsM = 1;
    uint32_t slots = 1, nNodes = 1, nExperts = 1, nRows = 0;
    uint32_t idxChunk = 0, rowTile = 0;
    uint32_t t0 = 0, t1 = 0;
    uint32_t stagedAligned = 0, stagedBase = 0;
};

// 入口符号:算子类型 TerraceK2Pack 的 snake_case(msopgen 骨架约定)。
// DTYPE_HIDDEN / DTYPE_GATES / DTYPE_PAYLOAD 宏由构建系统按 dtype 组合实例化注入。
extern "C" __global__ __aicore__ void terrace_k2_pack(
    GM_ADDR hidden, GM_ADDR expert_idx, GM_ADDR gates, GM_ADDR payload,
    GM_ADDR mask, GM_ADDR gate_rows, GM_ADDR u_src, GM_ADDR node_counts,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.nRows == 0) {
        return;                       // T == 0:五个输出全空/全零,host 已按形状分配
    }
    KernelTerraceK2Pack op;
    op.Init(hidden, expert_idx, gates, payload, mask, gate_rows, u_src, node_counts,
            tilingData.tokens, tilingData.hidden, tilingData.topk, tilingData.quota,
            tilingData.groupsM, tilingData.slots, tilingData.nNodes,
            tilingData.nExperts, tilingData.nRows, tilingData.tokensPerCoreBase,
            tilingData.tokensRem, tilingData.idxChunk, tilingData.rowTile);
    op.Process();
}
