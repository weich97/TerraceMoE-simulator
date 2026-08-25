#!/usr/bin/env bash
# terrace ops (terrace_passthrough + terrace_k1_arrival): one-shot cluster-side build
# (msopgen skeleton -> overlay sources -> compile -> install opp package).
#
# Cluster only (no CANN locally). Prerequisite:
#   source $HOME/Ascend/ascend-toolkit/set_env.sh    # note: do not source under set -u
# Copy-paste-runnable steps, every pitfall and its fix: see the pitfall notes in this
# file's header.
#
# Tunable environment (all with defaults):
#   SOC_VERSION   msopgen's -c compile target (default: auto-detect via
#                 acl.get_soc_name(); measured Ascend910_9392 on 910C/CANN 9.0.0)
#   VENDOR_NAME   opp package vendor name (default terrace)
#   GEN_DIR       generated project dir (default /tmp/terrace_ops_build; safe to delete
#                 wholesale and start over)
#   TERRACE_OPS   list of ops to build (default "terrace_passthrough terrace_k1_arrival")
#
# ======================== Design points (pitfalls we hit) ========================
#
# 1. **The project MUST live under /tmp**: msopgen regex-validates the -out path; repo
#    paths (D:/... mounts, strings like hyphenated-repo-name) get rejected. That is why
#    GEN_DIR defaults to /tmp; do not change it back to ${HERE}/build_gen.
# 2. **Generate one op at a time, never merge IRs**: feeding a multi-op IR to a single
#    msopgen call triggers interactive prompts, which in a non-interactive setting is an
#    instant EOFError. Here we generate one skeleton per op, use the first skeleton as
#    the project base, and overlay our handwritten sources wholesale -- multi-op
#    collection is handled by CMake's aux_source_directory (op_host) and --op-type=ALL
#    (op_kernel); msopgen only ever sees a single op.
# 3. **The tiling header lives in op_kernel/**: in the CANN 9.0.0 ASC system the tiling
#    struct is a plain C struct shared by host and kernel. The old op_host/ +
#    BEGIN_TILING_DATA_DEF style makes the kernel compile fail, with the failure message
#    swallowed by the binary sub-build; the main log only shows the downstream
#    "The Target path not found: .../binary/ascend910_93" (pitfall notes).
# 4. **Do not vendor the full msopgen-generated CMake project**: the cmake machinery is
#    tightly coupled to the toolkit version (9.0.0 moved from CANN 8.x's cmake/util to
#    find_package(ASC) + npu_op_* functions). Let the current toolkit generate the
#    skeleton every time and only overlay the handwritten sources -- smallest
#    version-divergence surface.
# 5. **Read back and verify after every file write**: the SOC-string substitution and
#    source overlays are all grep-verified by reading back, never trusting the return
#    code alone.
set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd)"

# Op list: explicit enumeration, no *.json glob -- the directory still holds the
# unfinished terrace_k2_pack (still the old CANN 8.x layout, tiling header in op_host/),
# and a glob would drag it into the build.
OPS="${TERRACE_OPS:-terrace_passthrough terrace_k1_arrival}"
GEN_DIR="${GEN_DIR:-/tmp/terrace_ops_build}"
PROJ="${GEN_DIR}/TerraceOps"
VENDOR_NAME="${VENDOR_NAME:-terrace}"

case "${GEN_DIR}" in
    /tmp/*) ;;
    *) echo "[build.sh] GEN_DIR must live under /tmp (msopgen's path regex rejects repo paths)" >&2
       exit 1 ;;
esac

# ---- 0. Toolchain self-check ----------------------------------------------------
# **Set up the environment yourself; never assume the caller sourced it first.**
# 2026-08-23: the unattended scheduler hit this line and died three times in a row --
# when running by hand I had always sourced set_env.sh first, so this precondition never
# surfaced. **An unattended arm must not depend on "a human did a step first"**; and it
# failed cleanly (nonzero exit code, one log line), dragging all three of
# k1-rebuild / k1-verify / k1-verify2 into failed.
if ! command -v msopgen >/dev/null 2>&1; then
    for _env in "${ASCEND_TOOLKIT_HOME:-}/../set_env.sh"                 $HOME/Ascend/ascend-toolkit/set_env.sh                 /usr/local/Ascend/ascend-toolkit/set_env.sh; do
        if [ -f "$_env" ]; then
            echo "[build.sh] msopgen not on PATH, auto-sourcing $_env"
            set +u; . "$_env"; set -u        # note: set_env.sh blows up under set -u
            break
        fi
    done
fi
if ! command -v msopgen >/dev/null 2>&1; then
    echo "[build.sh] msopgen not found, and auto-sourcing did not bring it back." >&2
    echo "[build.sh]   tried: \$ASCEND_TOOLKIT_HOME/../set_env.sh, $HOME/Ascend/..., /usr/local/Ascend/..." >&2
    exit 1
fi
echo "[build.sh] msopgen: $(command -v msopgen)"
# msopgen is a python shell; its package lives in CANN's python/site-packages, not in
# the conda env. With PYTHONPATH missing it hits ModuleNotFoundError, yet reports it as
# "The path ... is not valid" -- two layers away from the root cause. So we add the
# path proactively and self-check with an import.
for _cannroot in "${ASCEND_HOME_PATH:-}" "${ASCEND_TOOLKIT_HOME:-}" \
                 "$(dirname "$(dirname "$(command -v msopgen)")")"; do
    [ -n "${_cannroot}" ] && [ -d "${_cannroot}/python/site-packages/msopgen" ] && {
        export PYTHONPATH="${_cannroot}/python/site-packages:${PYTHONPATH:-}"; break; }
done
python3 -c "import msopgen" 2>/dev/null || {
    echo "[build.sh] msopgen package import failed -- check whether CANN python/site-packages is on PYTHONPATH" >&2
    exit 1
}

if [ -z "${SOC_VERSION:-}" ]; then
    SOC_VERSION="$(python3 -c 'import acl; print(acl.get_soc_name())' 2>/dev/null || true)"
fi
if [ -z "${SOC_VERSION}" ]; then
    echo "[build.sh] acl.get_soc_name() probe failed and no SOC_VERSION given -- stopping" >&2
    echo "[build.sh] specify it manually, e.g.: SOC_VERSION=Ascend910_9392 bash build.sh" >&2
    exit 1
fi
echo "[build.sh] SOC_VERSION=${SOC_VERSION}  (measured on 910C: Ascend910_9392)"

# ---- 1. Generate skeletons per op (single op, never merge IRs) --------------------
# Skeleton generation is idempotent: if GEN_DIR exists, generation is skipped.
# **After the op set changes you MUST rm -rf ${GEN_DIR}**.
mkdir -p "${GEN_DIR}"
FIRST_OP=""
for op in ${OPS}; do
    [ -z "${FIRST_OP}" ] && FIRST_OP="${op}"
    json="${HERE}/${op}.json"
    # **Copy the IR to /tmp before feeding it to msopgen.** 2026-08-24: passing the repo
    # path directly got rejected --
    #   [ERROR] The path <repo>/.../terrace_passthrough.json is not valid
    # The file exists and the JSON is valid; but $HOME is a **symlink** to the real home
    # directory, and msopgen regex-validates paths (pitfall 1 already noted: -out via a
    # repo path gets rejected) -- the input IR path goes through the same validation.
    # Same medicine: land it on a real path under /tmp first.
    mkdir -p "$GEN_DIR/ir"
    cp -f "$json" "$GEN_DIR/ir/$(basename "$json")"
    json="$GEN_DIR/ir/$(basename "$json")"
    [ -f "${json}" ] || { echo "[build.sh] missing IR definition ${json}" >&2; exit 1; }
    skel="${GEN_DIR}/skel_${op}"
    if [ ! -d "${skel}" ]; then
        echo "[build.sh] msopgen gen: ${op}"
        # -f pytorch only affects the framework plugin stub; aclnn artifacts are
        # unaffected. Some msopgen versions lack the -f flag; on failure retry without
        # it. </dev/null: any interactive prompt fails immediately with EOF instead of
        # hanging there waiting for input.
        msopgen gen -i "${json}" -f pytorch -c "ai_core-${SOC_VERSION}" \
            -lan cpp -out "${skel}" </dev/null \
        || msopgen gen -i "${json}" -c "ai_core-${SOC_VERSION}" \
            -lan cpp -out "${skel}" </dev/null
    fi
    [ -f "${skel}/op_host/${op}.cpp" ] || {
        echo "[build.sh] skeleton ${skel} has no op_host/${op}.cpp -- layout does not match expectations" >&2
        exit 1; }
done

# Project base = the first op's skeleton (CMakeLists / CMakePresets.json / build.sh are op-independent).
if [ ! -d "${PROJ}" ]; then
    cp -r "${GEN_DIR}/skel_${FIRST_OP}" "${PROJ}"
fi

# ---- 2. SOC config string: grab the authoritative value from the skeleton stub -----
# The AddConfig("...") in the stub is this toolkit's authoritative spelling for this SOC
# (ascend910_93 on 910C/9.0.0 -- note it is **NOT the same string** as
# acl.get_soc_name()'s Ascend910_9392).
SOC_CFG="$(grep -o 'AddConfig("[^"]*")' "${GEN_DIR}/skel_${FIRST_OP}/op_host/${FIRST_OP}.cpp" \
           | head -1 | sed 's/AddConfig("\(.*\)")/\1/')"
if [ -z "${SOC_CFG}" ]; then
    echo "[build.sh] could not grab the AddConfig SOC string from the generated stub -- layout does not match expectations, stopping" >&2
    exit 1
fi
echo "[build.sh] stub AddConfig SOC string: ${SOC_CFG}"

# ---- 3. Overlay handwritten sources over the generated stubs (kernel / shared tiling header / host prototype) ----
for op in ${OPS}; do
    for f in "op_kernel/${op}.cpp" "op_kernel/${op}_tiling.h"; do
        [ -f "${HERE}/${f}" ] || { echo "[build.sh] missing source file ${HERE}/${f}" >&2; exit 1; }
        cp -f "${HERE}/${f}" "${PROJ}/${f}"
    done
    # Host source: substitute the SOC placeholder, write to disk, then read back to
    # verify (a leftover placeholder = broken substitution logic, stop).
    sed "s/AddConfig(\"__TERRACE_SOC__\")/AddConfig(\"${SOC_CFG}\")/" \
        "${HERE}/op_host/${op}.cpp" > "${PROJ}/op_host/${op}.cpp"
    if grep -q "__TERRACE_SOC__" "${PROJ}/op_host/${op}.cpp"; then
        echo "[build.sh] ${op}: SOC placeholder substitution failed, placeholder still present -- stopping" >&2
        exit 1
    fi
    grep -q "AddConfig(\"${SOC_CFG}\")" "${PROJ}/op_host/${op}.cpp" || {
        echo "[build.sh] ${op}: read-back did not find AddConfig(\"${SOC_CFG}\") -- stopping" >&2
        exit 1; }
done
# The skeleton base ships the first op's stub; the other ops' stubs never entered PROJ,
# so there is nothing left over.
echo "[build.sh] op sources in the project:"
ls -1 "${PROJ}/op_host" "${PROJ}/op_kernel"

# ---- 4. Project config: CANN package path + vendor name --------------------------
# msopgen fills CMakePresets.json's ASCEND_CANN_PACKAGE_PATH from the current
# environment; vendor_name defaults to "customize" -- change it to ours so we don't
# collide with someone else's custom op package in the same vendors directory.
if [ -f "${PROJ}/CMakePresets.json" ]; then
    python3 - "${PROJ}/CMakePresets.json" "${VENDOR_NAME}" "${ASCEND_HOME_PATH:-}" <<'PYEOF'
import json, sys
path, vendor, cann = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
def patch(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "vendor_name" and isinstance(v, dict):
                v["value"] = vendor
            elif k == "ASCEND_CANN_PACKAGE_PATH" and isinstance(v, dict) and cann:
                v["value"] = cann
            else:
                patch(v)
    elif isinstance(node, list):
        for v in node:
            patch(v)
patch(data)
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(data, f, indent=4)
PYEOF
    grep -q "\"${VENDOR_NAME}\"" "${PROJ}/CMakePresets.json" || {
        echo "[build.sh] read-back did not find vendor_name=${VENDOR_NAME} in CMakePresets.json -- stopping" >&2
        exit 1; }
    echo "[build.sh] vendor_name=${VENDOR_NAME} written into CMakePresets.json"
fi

# ---- 5. Compile + package -------------------------------------------------------
# On failure the kernel's real compile errors are not on stdout but in the sub-build
# logs under build_out -- see the log-locating commands in the pitfall notes.
( cd "${PROJ}" && bash build.sh )

RUN_PKG="$(ls "${PROJ}"/build_out/custom_opp_*.run 2>/dev/null | head -1)"
if [ -z "${RUN_PKG}" ]; then
    echo "[build.sh] no custom_opp_*.run under build_out -- compile failed" >&2
    echo "[build.sh] find the real errors here:" >&2
    echo "  grep -rn 'error:' ${PROJ}/build_out --include=*.log | head -40" >&2
    exit 1
fi

# ---- 6. Install into opp vendors ------------------------------------------------
# </dev/null: if the installer script ever tries to ask something, it fails immediately
# with EOF instead of hanging for input.
echo "[build.sh] installing ${RUN_PKG}"
bash "${RUN_PKG}" </dev/null

# ---- 7. Post-install read-back check: both ops' aclnn headers must be present -----
OPP_VENDOR="${ASCEND_OPP_PATH:-${ASCEND_HOME_PATH}/opp}/vendors/${VENDOR_NAME}"
MISSING=0
for op in ${OPS}; do
    if [ ! -f "${OPP_VENDOR}/op_api/include/aclnn_${op}.h" ]; then
        echo "[build.sh] missing ${OPP_VENDOR}/op_api/include/aclnn_${op}.h after install" >&2
        MISSING=1
    fi
done
[ -f "${OPP_VENDOR}/op_api/lib/libcust_opapi.so" ] || {
    echo "[build.sh] missing ${OPP_VENDOR}/op_api/lib/libcust_opapi.so after install" >&2; MISSING=1; }
[ "${MISSING}" -eq 0 ] || exit 1

echo "[build.sh] done. vendor package: ${OPP_VENDOR}"
echo "[build.sh] next step: python ../csrc/build_ext.py  (torch bindings)"
