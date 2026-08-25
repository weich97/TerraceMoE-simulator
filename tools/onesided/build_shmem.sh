#!/bin/bash
# 编译 HyperParallel 的单边通信(Symmetric Memory)—— 只编这一块,不装框架。
#
# 为什么:PyTorch 跨设备 `tensor.copy_()` 每次调用约 257-268 µs 固定开销,
# 用它测单边带宽得到的只是「字节 ÷ 常数」—— 不是测量。aclshmem 的 `put_mem`
# 走另一条路:直接发 AscendC kernel,没有 aclnn、没有 host tiling,
# kernel 内拿远端 GM 指针 MTE 直搬。这才是能测单边的量具。
#
# 用法(先 source CANN set_env.sh 与你的 python 环境):
#   bash tools/onesided/build_shmem.sh /path/to/hyper-parallel-master
#
# 前置(在我们的验证机上确认过的组合):
#   gcc ∈ [7.3.0, 11.3.0];cmake ≥ 3.18;bisheng 在 $ASCEND_HOME_PATH/bin;
#   torch 2.9(hyper-parallel [torch] extra 的默认档)
#   **离线集群注意**:3rdparty/shmem 必须预先放好(见下),自动 clone 需要外网
set -eu

HP="${1:-}"
[ -n "$HP" ] && [ -d "$HP" ] || { echo "用法: bash $0 /path/to/hyper-parallel-master" >&2; exit 2; }
HP=$(cd "$HP" && pwd)
say(){ echo "[build_shmem] $*"; }

# 前置:调用方自己 source CANN 环境(set_env.sh)与 python 环境;
# 本脚本只查不设 —— 环境是机器的事,不是脚本的事。

# ---------------------------------------------------------------- 前置检查
say "前置检查"
gv=$(gcc -dumpfullversion 2>/dev/null || gcc -dumpversion)
case "$gv" in
  7.3.*|8.*|9.*|10.*|11.0.*|11.1.*|11.2.*|11.3.*) say "  gcc $gv OK" ;;
  *) echo "!! gcc $gv 不在 [7.3.0, 11.3.0]" >&2; exit 3 ;;
esac
command -v cmake >/dev/null || { echo "!! 没有 cmake" >&2; exit 3; }
say "  cmake $(cmake --version | head -1 | awk '{print $3}')"
BISHENG="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/bin/bisheng"
[ -x "$BISHENG" ] || command -v bisheng >/dev/null \
  || { echo "!! 找不到 bisheng(试过 $BISHENG)" >&2; exit 3; }
say "  bisheng OK"
export PATH="$(dirname "$BISHENG"):$PATH"

# **离线依赖**:build_symmetric_memory.sh:272 是
#     [[ ! -d "shmem" ]] && git clone --depth 1 https://gitcode.com/cann/shmem.git -b v1.3.0
# 目录已存在就跳过克隆 —— 这是唯一能在无外网集群上走通的路。
if [ ! -d "$HP/3rdparty/shmem" ]; then
  cat >&2 <<'MSG'
!! 缺 3rdparty/shmem。

   离线集群上自动 clone 走不通,需要把
   https://gitcode.com/cann/shmem.git @ v1.3.0 离线搬进:
       <hyper-parallel-master>/3rdparty/shmem
   目录一旦存在,build 脚本就会跳过 clone。

   注意 CANN 9.0.0 里**没有** aclshmem(找到的 svm_shmem_* 是驱动内部的,不是这个),
   仓库也不 vendor、无预编译包 —— 必须搬源码。
MSG
  exit 4
fi
say "  3rdparty/shmem 在(离线依赖已就位)"

# ---------------------------------------------------------------- 两处补丁
# 都是审计查出来的、会让读数作废的东西。**幂等**:改过就跳过。
cd "$HP"

P1=hyper_parallel/core/symmetric_memory/ops/put_mem/host/put_mem.cpp
if grep -q "TERRACE_PATCH_STATIC_SYNC_V3" "$P1" 2>/dev/null; then
  say "补丁 1 已在(V2)"
else
  # V2(2026-08-25)整文件替换。V1 的字符串手术打出两处坏账,作废:
  #   (a) 错误路径裸 return(函数返回 int,编不过);
  #   (b) replace(...,1) 把增长路径里自己插的 free 注释掉了,
  #       尾部真正要删的 aclrtFree(UAF 根源)原样活着 —— 裸 return 一修反而成真 UAF。
  # 教训:对 2 KB 的文件做字符串手术是拿确定性换省事。
  say "补丁 1(V2 整文件替换):malloc/free 提出热路径"
  [ -f "$P1.orig" ] || cp "$P1" "$P1.orig"
  cat > "$P1" <<'CPPEOF'
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstddef>
#include <iostream>

#include "acl/acl.h"

#include "shmem.h"
#include "include/shmem_kernel.h"

namespace ShmemKernel {

extern void put_mem(uint32_t block_dim, void *stream, uint64_t elementSize, uint8_t *target, uint8_t *target_offset,
                    uint8_t *src, uint8_t *src_offset, uint8_t *size, int64_t target_pe, bool non_blocking,
                    uint8_t *Syncmem);

// TERRACE_PATCH_STATIC_SYNC_V3(2026-08-25):
// 原版每次调用 aclrtMalloc + aclrtMemset + aclrtFree 一块 sync 区,
// aclrtFree 通常隐含设备同步 —— 单这一条就足以作废整组带宽读数;
// 且 kernel 里 SyncAll 用的正是这块被 free 掉的内存(UAF)。
// 改法:缓冲静态复用、按需增长、增长时旧块不 free(可能仍有在飞 kernel 引用;
// 增长只在 block_dim 变大时发生,泄漏上限 32B x 64 = 2 KB,可忽略)。
// **清零保留但改流内异步(V3)**:原版 host 同步 memset 之所以安全,是因为每次
// malloc 的是新缓冲;复用静态缓冲后,host memset 不排流序,会清掉上一发在飞
// kernel 的 SyncAll flag -> 永久自旋(实测:连发 put 卡死 3/8 张卡,
// 其余 rank 集合通信超时)。aclrtMemsetAsync 同流入队 = 严格排在上一发 kernel
// 之后、下一发之前,host 全程不同步。
// **约束**:静态缓冲按「单流使用」设计;多流并发 put 共享它仍会竞态(参考实现的
// alltoall 是每对端一条流)。本量具单流,够用;多流要一流一缓冲,那是后话。
// 错误路径不学原版的 aclFinalize()(库函数里拆全局运行时),直接返 -1。
int aclshmem_put_mem(uint32_t block_dim, aclrtStream stream, uint64_t elementSize, void *target, void *target_offset,
                     void *src, void *src_offset, void *size, int64_t target_pe, bool non_blocking) {
  static void *sync_mem_device = nullptr;
  static size_t sync_cap = 0;
  const size_t need = 8 * block_dim * sizeof(int32_t);
  aclError ret = ACL_SUCCESS;
  if (sync_cap < need) {
    sync_mem_device = nullptr;
    ret = aclrtMalloc(&sync_mem_device, need, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      std::cerr << "aclrtMalloc failed: " << ret << std::endl;
      sync_cap = 0;
      return -1;
    }
    sync_cap = need;
  }
  ret = aclrtMemsetAsync(sync_mem_device, need, 0, need, stream);
  if (ret != ACL_SUCCESS) {
    std::cerr << "aclrtMemsetAsync failed: " << ret << std::endl;
    return -1;
  }
  int status = 0;
  // put_mem
  put_mem(block_dim, stream, elementSize, (uint8_t *)target, (uint8_t *)target_offset, (uint8_t *)src,
          (uint8_t *)src_offset, (uint8_t *)size, target_pe, non_blocking, (uint8_t *)sync_mem_device);
  return status;
}

}  // namespace ShmemKernel
CPPEOF
  grep -q "TERRACE_PATCH_STATIC_SYNC_V3" "$P1" || { echo "!! 补丁 1 没生效" >&2; exit 5; }
fi

P3=hyper_parallel/core/symmetric_memory/ops/put_mem_signal/host/put_mem_signal.cpp
if grep -q "TERRACE_PATCH_POOL_SYNC" "$P3" 2>/dev/null; then
  say "补丁 3 已在"
else
  say "补丁 3(整文件替换):put_mem_signal 的 sync 区改 64 槽轮转池(多流安全)"
  [ -f "$P3.orig" ] || cp "$P3" "$P3.orig"
  cat > "$P3" <<'CPPEOF'
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <atomic>
#include <cstddef>
#include <iostream>

#include "acl/acl.h"
#include "shmem.h"
#include "include/shmem_kernel.h"

namespace ShmemKernel {

extern void put_mem_signal(uint32_t block_dim, void *stream, uint64_t elementSize, uint8_t *target,
                           uint8_t *target_offset, uint8_t *src, uint8_t *src_offset, uint8_t *size, uint8_t *signal,
                           uint8_t *signal_offset, uint8_t *signal_value, int64_t signal_op, int64_t target_pe,
                           bool non_blocking, uint8_t *Syncmem);

// TERRACE_PATCH_POOL_SYNC(2026-08-25):与 put_mem 的 V3 同因(原版每调用
// malloc/memset/free,free 隐含设备同步 -> 串行化 + UAF),但这条被官方
// alltoall 以「每对端一条流」连发,单块静态缓冲会跨流竞态。改 64 槽轮转池:
// 每次调用取下一槽,在**本调用的流**上异步清零。同一槽两次发放之间隔 63 次
// 调用,远大于在飞窗口(bench 每次交换 <=8 发在飞,步间有 signal_wait +
// synchronize)。这是量具级修法,不是通用库级修法(通用要一流一缓冲)。
static constexpr int SYNC_POOL = 64;

int aclshmem_put_mem_signal(uint32_t block_dim, aclrtStream stream, uint64_t elementSize, void *target,
                            void *target_offset, void *src, void *src_offset, void *size, void *signal,
                            void *signal_offset, void *signal_value, int64_t signal_op, int64_t target_pe,
                            bool non_blocking) {
  static void *pool[SYNC_POOL] = {};
  static size_t caps[SYNC_POOL] = {};
  static std::atomic<unsigned> next{0};
  const unsigned slot = next.fetch_add(1) % SYNC_POOL;
  const size_t need = 8 * block_dim * sizeof(int32_t);
  aclError ret = ACL_SUCCESS;
  if (caps[slot] < need) {
    pool[slot] = nullptr;
    ret = aclrtMalloc(&pool[slot], need, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      std::cerr << "aclrtMalloc failed: " << ret << std::endl;
      caps[slot] = 0;
      return -1;
    }
    caps[slot] = need;
  }
  ret = aclrtMemsetAsync(pool[slot], need, 0, need, stream);
  if (ret != ACL_SUCCESS) {
    std::cerr << "aclrtMemsetAsync failed: " << ret << std::endl;
    return -1;
  }
  int status = 0;
  // put_mem_signal
  put_mem_signal(block_dim, stream, elementSize, (uint8_t *)target, (uint8_t *)target_offset, (uint8_t *)src,
                 (uint8_t *)src_offset, (uint8_t *)size, (uint8_t *)signal, (uint8_t *)signal_offset,
                 (uint8_t *)signal_value, signal_op, target_pe, non_blocking, (uint8_t *)pool[slot]);
  return status;
}

}  // namespace ShmemKernel
CPPEOF
  grep -q "TERRACE_PATCH_POOL_SYNC" "$P3" || { echo "!! 补丁 3 没生效" >&2; exit 5; }
fi

P2=hyper_parallel/core/symmetric_memory/platform/torch/torch_bindings.cpp
if grep -q "TERRACE_PATCH_BLOCK_DIM" "$P2" 2>/dev/null; then
  say "补丁 2 已在"
else
  say "补丁 2:block_dim 可由环境变量调(默认仍是 1,行为不变)"
  python3 - "$P2" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8').read()
old = "static constexpr uint32_t DEFAULT_BLOCK_DIM = 1;"
assert old in s, "torch_bindings.cpp 的 DEFAULT_BLOCK_DIM 没匹配上"
new = ('''// TERRACE_PATCH_BLOCK_DIM(2026-08-24):原来写死 1 且无 setter,
// TorchScript 注册里也没暴露。kernel 本身按多核写好了
// (size_per_core = size_ / aiv_num_),但上游只给 1 个 block ——
// **这是带宽天花板,不是发射开销**:单核 MTE 撑不起 die-to-die 链路。
// 改成读环境变量,**默认仍是 1**(行为逐字不变,需要显式索取才变) ——
// 教训(本仓 docs/04 也有同款):能力一旦可用就自动启用,
// 等于把"能编译"当成"行为正确"的证据。
static uint32_t terrace_block_dim() {
  const char *e = std::getenv("TERRACE_SHMEM_BLOCK_DIM");
  if (e == nullptr) {
    return 1;
  }
  long v = std::strtol(e, nullptr, 10);
  if (v < 1 || v > 64) {
    return 1;
  }
  return static_cast<uint32_t>(v);
}
static const uint32_t DEFAULT_BLOCK_DIM = terrace_block_dim();''')
s = s.replace(old, new, 1)
if "#include <cstdlib>" not in s:
    i = s.index("#include")
    s = s[:i] + "#include <cstdlib>\n" + s[i:]
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("  torch_bindings.cpp 已改")
PY
  grep -q "TERRACE_PATCH_BLOCK_DIM" "$P2" || { echo "!! 补丁 2 没生效" >&2; exit 5; }
fi

# ---------------------------------------------------------------- 编译
# 默认是 --multicore mindspore --shmem all --custom-ops on,对 torch-only 环境三个全错。
say "编译(只要 torch 后端的 shmem)"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:?先 source CANN 的 set_env.sh}"
./build.sh --shmem torch --multicore off --custom-ops off --strict on

WHL=$(ls -t dist/hyper_parallel-*.whl 2>/dev/null | head -1)
[ -n "$WHL" ] || { echo "!! 没产出 wheel" >&2; exit 6; }
say "产出 $WHL"
# **不能 editable 安装**:.so 是 setup.py 的 BuildPy 从 build/lib 拷进 wheel 的,
# editable 装完 hyper_parallel/lib/shmem/ 不存在,import 会硬 raise FileNotFoundError。
pip install --force-reinstall --no-deps "$WHL"

# 按路径验,不 import —— 顶层 __init__ 是重初始化 + 无条件 monkey patch,
# bench 本来就绕开它(见 tools/onesided/bench_onesided.py 文档),验证器学它 import 是自相矛盾。
SO=$(python - <<'VPY'
import os, site, sysconfig
cands = site.getsitepackages() + [sysconfig.get_paths()["purelib"]]
for sp in cands:
    q = os.path.join(sp, "hyper_parallel", "platform", "torch",
                     "symmetric_memory", "libaclshmem_torch.so")
    if os.path.isfile(q):
        print(q)
        break
VPY
)
if [ -n "$SO" ] && [ -f "$SO" ]; then
  say "**成功**:$SO"
  say "下一步:torchrun --nnodes=1 --nproc_per_node=8 tools/onesided/bench_onesided.py"
else
  echo "!! wheel 装了但找不到 libaclshmem_torch.so —— 编译多半没带上 shmem" >&2
  exit 7
fi
