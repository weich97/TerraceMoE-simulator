#!/bin/bash
# Build HyperParallel's one-sided communication (Symmetric Memory) -- just this piece, without installing the framework.
#
# Why: PyTorch's cross-device `tensor.copy_()` carries a fixed ~257-268 µs overhead per
# call; "bandwidth" measured with it is just bytes ÷ a constant -- not a measurement.
# aclshmem's `put_mem` takes a different path: it launches an AscendC kernel directly,
# no aclnn, no host tiling; the kernel grabs a remote GM pointer and MTE moves the data
# directly. That is the instrument that can actually measure one-sided.
#
# Usage (source CANN's set_env.sh and your python environment first):
#   bash tools/onesided/build_shmem.sh /path/to/hyper-parallel-master
#
# Prerequisites (combination confirmed on our validation machine):
#   gcc ∈ [7.3.0, 11.3.0]; cmake ≥ 3.18; bisheng under $ASCEND_HOME_PATH/bin;
#   torch 2.9 (the default tier of hyper-parallel's [torch] extra)
#   **Offline clusters**: 3rdparty/shmem must be pre-staged (see below); the automatic clone needs internet access
set -eu

HP="${1:-}"
[ -n "$HP" ] && [ -d "$HP" ] || { echo "usage: bash $0 /path/to/hyper-parallel-master" >&2; exit 2; }
HP=$(cd "$HP" && pwd)
say(){ echo "[build_shmem] $*"; }

# Prerequisite: the caller sources the CANN environment (set_env.sh) and the python
# environment themselves; this script only checks, never sets -- the environment belongs
# to the machine, not to the script.

# ---------------------------------------------------------------- preflight checks
say "preflight checks"
gv=$(gcc -dumpfullversion 2>/dev/null || gcc -dumpversion)
case "$gv" in
  7.3.*|8.*|9.*|10.*|11.0.*|11.1.*|11.2.*|11.3.*) say "  gcc $gv OK" ;;
  *) echo "!! gcc $gv outside [7.3.0, 11.3.0]" >&2; exit 3 ;;
esac
command -v cmake >/dev/null || { echo "!! cmake not found" >&2; exit 3; }
say "  cmake $(cmake --version | head -1 | awk '{print $3}')"
BISHENG="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/bin/bisheng"
[ -x "$BISHENG" ] || command -v bisheng >/dev/null \
  || { echo "!! bisheng not found (tried $BISHENG)" >&2; exit 3; }
say "  bisheng OK"
export PATH="$(dirname "$BISHENG"):$PATH"

# **Offline dependency**: build_symmetric_memory.sh:272 reads
#     [[ ! -d "shmem" ]] && git clone --depth 1 https://gitcode.com/cann/shmem.git -b v1.3.0
# An existing directory skips the clone -- the only path that works on a cluster with no
# internet access.
if [ ! -d "$HP/3rdparty/shmem" ]; then
  cat >&2 <<'MSG'
!! 3rdparty/shmem is missing.

   The automatic clone does not work on an offline cluster; move
   https://gitcode.com/cann/shmem.git @ v1.3.0 in offline, to:
       <hyper-parallel-master>/3rdparty/shmem
   Once the directory exists, the build script skips the clone.

   Note: CANN 9.0.0 ships **no** aclshmem (the svm_shmem_* symbols you may find are
   driver-internal, not this), the repo does not vendor it, and there is no prebuilt
   package -- the source must be moved in.
MSG
  exit 4
fi
say "  3rdparty/shmem present (offline dependency staged)"

# ---------------------------------------------------------------- two patches
# All found by audit; each one would void the readings. **Idempotent**: skip if already applied.
cd "$HP"

P1=hyper_parallel/core/symmetric_memory/ops/put_mem/host/put_mem.cpp
if grep -q "TERRACE_PATCH_STATIC_SYNC_V3" "$P1" 2>/dev/null; then
  say "patch 1 already in (V2)"
else
  # V2 (2026-08-25) replaces the whole file. V1's string surgery produced two bad debts; retracted:
  #   (a) a bare return on the error path (the function returns int; does not compile);
  #   (b) replace(...,1) commented out the free it had itself inserted on the growth path,
  #       while the trailing aclrtFree that actually needed deleting (the UAF root cause)
  #       stayed alive as-is -- fixing the bare return would have made it a real UAF.
  # Lesson: string surgery on a 2 KB file trades determinism for convenience.
  say "patch 1 (V2 whole-file replacement): hoist malloc/free out of the hot path"
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

// TERRACE_PATCH_STATIC_SYNC_V3 (2026-08-25):
// The original did aclrtMalloc + aclrtMemset + aclrtFree on a sync region every call;
// aclrtFree usually implies a device synchronize -- that alone is enough to void the
// whole set of bandwidth readings; and the kernel's SyncAll uses exactly the memory
// that was freed (UAF).
// Fix: reuse a static buffer, grow on demand, never free the old block on growth
// (in-flight kernels may still reference it; growth only happens when block_dim
// increases, leak bound 32B x 64 = 2 KB, negligible).
// **Zeroing kept, but made in-stream async (V3)**: the original's host-synchronous
// memset was only safe because every call malloc'd a fresh buffer; with a reused
// static buffer, a host memset is not stream-ordered and wipes the SyncAll flag of
// the previous in-flight kernel -> permanent spin (observed: back-to-back puts hang
// 3/8 dies, the remaining ranks time out in collectives). aclrtMemsetAsync enqueued
// on the same stream = strictly after the previous kernel and before the next one,
// with the host never synchronizing.
// **Constraint**: the static buffer is designed for single-stream use; concurrent
// multi-stream puts sharing it still race (the reference implementation's alltoall
// uses one stream per peer). This instrument is single-stream, good enough;
// multi-stream needs one buffer per stream -- a later problem.
// The error path does not copy the original's aclFinalize() (tearing down the global
// runtime inside a library function); it returns -1 directly.
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
  grep -q "TERRACE_PATCH_STATIC_SYNC_V3" "$P1" || { echo "!! patch 1 did not take effect" >&2; exit 5; }
fi

P3=hyper_parallel/core/symmetric_memory/ops/put_mem_signal/host/put_mem_signal.cpp
if grep -q "TERRACE_PATCH_POOL_SYNC" "$P3" 2>/dev/null; then
  say "patch 3 already in"
else
  say "patch 3 (whole-file replacement): put_mem_signal sync region becomes a 64-slot rotating pool (multi-stream safe)"
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

// TERRACE_PATCH_POOL_SYNC (2026-08-25): same root cause as put_mem's V3 (the original
// did malloc/memset/free per call; free implies a device synchronize -> serialization
// + UAF), but this entry point gets fired back-to-back by the official alltoall with
// one stream per peer, so a single static buffer would race across streams. Switch to
// a 64-slot rotating pool: each call takes the next slot and zeroes it asynchronously
// on **this call's stream**. Two grants of the same slot are 63 calls apart, far
// beyond the in-flight window (the bench keeps <=8 puts in flight per exchange, with
// signal_wait + synchronize between steps). This is an instrument-grade fix, not a
// general library-grade fix (a general one needs one buffer per stream).
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
  grep -q "TERRACE_PATCH_POOL_SYNC" "$P3" || { echo "!! patch 3 did not take effect" >&2; exit 5; }
fi

P2=hyper_parallel/core/symmetric_memory/platform/torch/torch_bindings.cpp
if grep -q "TERRACE_PATCH_BLOCK_DIM" "$P2" 2>/dev/null; then
  say "patch 2 already in"
else
  say "patch 2: block_dim tunable via env var (default stays 1; behavior unchanged)"
  python3 - "$P2" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8').read()
old = "static constexpr uint32_t DEFAULT_BLOCK_DIM = 1;"
assert old in s, "DEFAULT_BLOCK_DIM in torch_bindings.cpp did not match"
new = ('''// TERRACE_PATCH_BLOCK_DIM (2026-08-24): originally hard-coded to 1 with no setter,
// and not exposed in the TorchScript registration either. The kernel itself is
// written for multi-core (size_per_core = size_ / aiv_num_), but upstream only ever
// hands it 1 block -- **this is a bandwidth ceiling, not launch overhead**: a single
// core's MTE cannot carry the die-to-die link.
// Changed to read an env var, **default stays 1** (behavior verbatim-unchanged;
// only changes when explicitly requested) -- lesson (this repo's docs/04 has the
// same one): a capability that auto-enables as soon as it is available amounts to
// treating "it compiles" as evidence of "it behaves correctly".
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
print("  torch_bindings.cpp patched")
PY
  grep -q "TERRACE_PATCH_BLOCK_DIM" "$P2" || { echo "!! patch 2 did not take effect" >&2; exit 5; }
fi

# ---------------------------------------------------------------- build
# Defaults are --multicore mindspore --shmem all --custom-ops on; all three are wrong for a torch-only environment.
say "building (only the shmem for the torch backend)"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:?source the CANN set_env.sh first}"
./build.sh --shmem torch --multicore off --custom-ops off --strict on

WHL=$(ls -t dist/hyper_parallel-*.whl 2>/dev/null | head -1)
[ -n "$WHL" ] || { echo "!! no wheel produced" >&2; exit 6; }
say "produced $WHL"
# **No editable install**: the .so is copied into the wheel from build/lib by setup.py's
# BuildPy; after an editable install, hyper_parallel/lib/shmem/ does not exist and the
# import hard-raises FileNotFoundError.
pip install --force-reinstall --no-deps "$WHL"

# Verify by path, without importing -- the top-level __init__ does heavy initialization
# plus an unconditional monkey patch; the bench bypasses it by design (see the
# tools/onesided/bench_onesided.py docstring), so a verifier that imports it would
# contradict itself.
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
  say "**success**: $SO"
  say "next: torchrun --nnodes=1 --nproc_per_node=8 tools/onesided/bench_onesided.py"
else
  echo "!! wheel installed but libaclshmem_torch.so not found -- the build most likely skipped shmem" >&2
  exit 7
fi
