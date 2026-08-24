/**
 * terrace_passthrough -- tiling 数据定义(工程链路样板)。
 *
 * **放在 op_kernel/ 而不是 op_host/**:CANN 9.0.0 的 ASC 构建体系里,tiling 结构体
 * 是 host 与 kernel 的**共享普通 C 结构体**,由两侧同时 include —— host 侧
 * `context->GetTilingData<T>()` 直接往 tiling buffer 里填,kernel 侧
 * `REGISTER_TILING_DEFAULT(T)` + `GET_TILING_DATA` 反序列化回同一个 T。
 * msopgen 生成的骨架就把它放在 op_kernel/,host stub 用 "../op_kernel/xxx_tiling.h"
 * 反向 include。本文件与 op_host/、op_kernel/ 的同名 stub 一一对应。
 *
 * 旧写法(CANN 8.x 的 BEGIN_TILING_DATA_DEF / TILING_DATA_FIELD_DEF /
 * REGISTER_TILING_DATA_CLASS,头放 op_host/)在 9.0.0 上**编不出 kernel**:
 * kernel 侧拿不到结构体定义,GET_TILING_DATA 展开即失败,而失败日志被 binary
 * 子构建吞掉,主日志只剩下游的 "The Target path not found: .../binary/ascend910_93"
 * —— 见 本文件头的踩坑记录 坑 4。
 *
 * 字段随 kernel 的均匀切块模型走:总元素数 + 每核 tile 数。
 */
#ifndef TERRACE_PASSTHROUGH_TILING_H
#define TERRACE_PASSTHROUGH_TILING_H

#include <cstdint>

// 全局命名空间(与 msopgen 骨架一致):host 侧在 namespace optiling 里按非限定名
// 引用即可命中;放进 optiling 会让 kernel 侧(不引 optiling)找不到。
struct TerracePassthroughTilingData {
    uint32_t totalLength;   // 输入张量总元素数
    uint32_t tileNum;       // 每核 tile 数(double buffer 前)
};

#endif  // TERRACE_PASSTHROUGH_TILING_H
