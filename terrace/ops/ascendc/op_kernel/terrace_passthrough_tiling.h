/**
 * terrace_passthrough -- tiling data definition (engineering-pipeline template).
 *
 * **Lives in op_kernel/, not op_host/**: in the CANN 9.0.0 ASC build system, the tiling
 * struct is a **plain C struct shared** by host and kernel, included from both sides --
 * the host side fills the tiling buffer directly via `context->GetTilingData<T>()`, and
 * the kernel side deserializes back into the same T via `REGISTER_TILING_DEFAULT(T)` +
 * `GET_TILING_DATA`. The msopgen-generated skeleton puts it in op_kernel/, and the host
 * stub includes it backwards as "../op_kernel/xxx_tiling.h". This file corresponds
 * one-to-one with the same-named stubs in op_host/ and op_kernel/.
 *
 * The old style (CANN 8.x's BEGIN_TILING_DATA_DEF / TILING_DATA_FIELD_DEF /
 * REGISTER_TILING_DATA_CLASS with the header in op_host/) **fails to build the kernel**
 * on 9.0.0: the kernel side never sees the struct definition, GET_TILING_DATA fails right
 * at expansion, the failure log gets swallowed by the binary sub-build, and the main log
 * only shows the downstream "The Target path not found: .../binary/ascend910_93"
 * -- see pitfall 4 of the pitfall notes in this file's header.
 *
 * Fields follow the kernel's uniform partition model: total element count + tiles per core.
 */
#ifndef TERRACE_PASSTHROUGH_TILING_H
#define TERRACE_PASSTHROUGH_TILING_H

#include <cstdint>

// Global namespace (matching the msopgen skeleton): the host side, inside namespace
// optiling, hits it via unqualified lookup; putting it inside optiling would make it
// invisible to the kernel side (which never opens optiling).
struct TerracePassthroughTilingData {
    uint32_t totalLength;   // total element count of the input tensor
    uint32_t tileNum;       // tiles per core (before double buffering)
};

#endif  // TERRACE_PASSTHROUGH_TILING_H
