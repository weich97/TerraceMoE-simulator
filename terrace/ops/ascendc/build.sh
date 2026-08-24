#!/usr/bin/env bash
# terrace ops(terrace_passthrough + terrace_k1_arrival):集群侧一键构建
# (msopgen 骨架 -> 覆盖源 -> 编译 -> 装 opp 包)。
#
# 只在集群跑(本地无 CANN)。前置:
#   source $HOME/Ascend/ascend-toolkit/set_env.sh    # 注意:别在 set -u 下 source
# 完整照抄可执行的步骤、每个坑与修法见同目录 本文件头的踩坑记录。
#
# 可调环境(全部有缺省):
#   SOC_VERSION   msopgen 的 -c 编译目标(缺省自动探测 acl.get_soc_name();
#                 910C/CANN 9.0.0 实测 Ascend910_9392)
#   VENDOR_NAME   opp 包 vendor 名(缺省 terrace)
#   GEN_DIR       生成的工程目录(缺省 /tmp/terrace_ops_build,可整目录删掉重来)
#   TERRACE_OPS   要构建的算子列表(缺省 "terrace_passthrough terrace_k1_arrival")
#
# ============================ 设计要点(踩过的坑)============================
#
# 1. **工程必须建在 /tmp 下**:msopgen 对 -out 路径有正则校验,仓库路径
#    (D:/... 挂载、含 hyphenated-repo-name 这类串)会被拒。GEN_DIR 缺省因此指向 /tmp,
#    不要改回 ${HERE}/build_gen。
# 2. **单算子生成,绝不合并 IR**:多算子 IR 喂给一次 msopgen 会触发交互提问,
#    非交互场景下直接 EOFError。这里对每个算子各生成一份骨架,再把第一个骨架
#    当工程底座,把我们手写的源整体盖进去 —— 多算子靠 CMake 的
#    aux_source_directory(op_host)与 --op-type=ALL(op_kernel)自动收集,
#    msopgen 全程只见过单算子。
# 3. **tiling 头在 op_kernel/**:CANN 9.0.0 的 ASC 体系里 tiling 结构体是
#    host/kernel 共享的普通 C 结构体。旧的 op_host/ + BEGIN_TILING_DATA_DEF 写法
#    会让 kernel 编译失败,而失败信息被 binary 子构建吞掉,主日志只剩下游的
#    "The Target path not found: .../binary/ascend910_93"(踩坑记录)。
# 4. **不 vendor msopgen 生成的完整 CMake 工程**:cmake 机器与 toolkit 版本强耦合
#    (9.0.0 已从 CANN 8.x 的 cmake/util 换成 find_package(ASC) + npu_op_* 函数)。
#    每次让当前 toolkit 自己生成骨架,只把手写源盖进去 —— 版本分歧面最小。
# 5. **每次写文件后回读校验**:soc 串替换、源文件覆盖都 grep 回读,不只看 rc。
set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd)"

# 算子列表:显式枚举,不用 *.json 通配 —— 目录里还躺着未完工的 terrace_k2_pack
# (仍是 CANN 8.x 旧布局,tiling 头在 op_host/),通配会把它拖进来编。
OPS="${TERRACE_OPS:-terrace_passthrough terrace_k1_arrival}"
GEN_DIR="${GEN_DIR:-/tmp/terrace_ops_build}"
PROJ="${GEN_DIR}/TerraceOps"
VENDOR_NAME="${VENDOR_NAME:-terrace}"

case "${GEN_DIR}" in
    /tmp/*) ;;
    *) echo "[build.sh] GEN_DIR 必须在 /tmp 下(msopgen 路径正则会拒仓库路径)" >&2
       exit 1 ;;
esac

# ---- 0. 工具链自检 -------------------------------------------------------------
# **自己把环境备齐,别指望调用者先 source 过。**
# 2026-08-23:无人值守调度器(无人值守)连续三次点这个脚本都死在这一行 —— 我手工跑时
# 总是先 source 过 set_env.sh,于是这个前置条件从没暴露过。**一个无人值守的臂
# 不能依赖"人先做了一步"**;而且它失败得很干净(退出码非零、日志一行),
# 于是连着把 k1-rebuild / k1-verify / k1-verify2 三条都拖成 failed。
if ! command -v msopgen >/dev/null 2>&1; then
    for _env in "${ASCEND_TOOLKIT_HOME:-}/../set_env.sh"                 $HOME/Ascend/ascend-toolkit/set_env.sh                 /usr/local/Ascend/ascend-toolkit/set_env.sh; do
        if [ -f "$_env" ]; then
            echo "[build.sh] msopgen 不在 PATH,自动 source $_env"
            set +u; . "$_env"; set -u        # 注意:set_env.sh 在 set -u 下会炸
            break
        fi
    done
fi
if ! command -v msopgen >/dev/null 2>&1; then
    echo "[build.sh] 找不到 msopgen,且自动 source 也没找回来。" >&2
    echo "[build.sh]   试过:\$ASCEND_TOOLKIT_HOME/../set_env.sh、$HOME/Ascend/...、/usr/local/Ascend/..." >&2
    exit 1
fi
echo "[build.sh] msopgen: $(command -v msopgen)"
# msopgen 是 python 壳,其包在 CANN 的 python/site-packages,不在 conda env 里。
# 缺 PYTHONPATH 时它 ModuleNotFoundError,却把错报成 "The path ... is not valid"
# —— 报错离根因两层远。这里主动补路径并 import 自检。
for _cannroot in "${ASCEND_HOME_PATH:-}" "${ASCEND_TOOLKIT_HOME:-}" \
                 "$(dirname "$(dirname "$(command -v msopgen)")")"; do
    [ -n "${_cannroot}" ] && [ -d "${_cannroot}/python/site-packages/msopgen" ] && {
        export PYTHONPATH="${_cannroot}/python/site-packages:${PYTHONPATH:-}"; break; }
done
python3 -c "import msopgen" 2>/dev/null || {
    echo "[build.sh] msopgen 包 import 失败 —— 检查 CANN python/site-packages 是否在 PYTHONPATH" >&2
    exit 1
}

if [ -z "${SOC_VERSION:-}" ]; then
    SOC_VERSION="$(python3 -c 'import acl; print(acl.get_soc_name())' 2>/dev/null || true)"
fi
if [ -z "${SOC_VERSION}" ]; then
    echo "[build.sh] acl.get_soc_name() 探测失败,且没给 SOC_VERSION —— 停" >&2
    echo "[build.sh] 手动指定,例:SOC_VERSION=Ascend910_9392 bash build.sh" >&2
    exit 1
fi
echo "[build.sh] SOC_VERSION=${SOC_VERSION}  (910C 实测值:Ascend910_9392)"

# ---- 1. 逐算子生成骨架(单算子,绝不合并 IR)-------------------------------------
# 骨架幂等:GEN_DIR 存在即跳过生成。**算子集合变化后必须 rm -rf ${GEN_DIR}**。
mkdir -p "${GEN_DIR}"
FIRST_OP=""
for op in ${OPS}; do
    [ -z "${FIRST_OP}" ] && FIRST_OP="${op}"
    json="${HERE}/${op}.json"
    # **把 IR 拷到 /tmp 再喂给 msopgen。** 2026-08-24:直接传仓库路径被拒 ——
    #   [ERROR] The path <repo>/.../terrace_passthrough.json is not valid
    # 文件在、JSON 合法;但 $HOME 是指向 真实主目录 的**符号链接**,
    # 而 msopgen 对路径有正则校验(坑 1 已记:-out 走仓库路径会被拒)——
    # 输入 IR 路径吃同一套校验。同样的药:先落到 /tmp 下的实路径。
    mkdir -p "$GEN_DIR/ir"
    cp -f "$json" "$GEN_DIR/ir/$(basename "$json")"
    json="$GEN_DIR/ir/$(basename "$json")"
    [ -f "${json}" ] || { echo "[build.sh] 缺 IR 定义 ${json}" >&2; exit 1; }
    skel="${GEN_DIR}/skel_${op}"
    if [ ! -d "${skel}" ]; then
        echo "[build.sh] msopgen gen: ${op}"
        # -f pytorch 只影响 framework 插件桩,aclnn 产物不受影响;个别 msopgen 版本
        # 无 -f 旗标,失败则去掉重试。</dev/null:任何交互提问都当即 EOF 失败,
        # 不要挂在那儿等输入。
        msopgen gen -i "${json}" -f pytorch -c "ai_core-${SOC_VERSION}" \
            -lan cpp -out "${skel}" </dev/null \
        || msopgen gen -i "${json}" -c "ai_core-${SOC_VERSION}" \
            -lan cpp -out "${skel}" </dev/null
    fi
    [ -f "${skel}/op_host/${op}.cpp" ] || {
        echo "[build.sh] 骨架 ${skel} 里没有 op_host/${op}.cpp —— 布局与预期不符" >&2
        exit 1; }
done

# 工程底座 = 第一个算子的骨架(CMakeLists / CMakePresets.json / build.sh 与算子无关)。
if [ ! -d "${PROJ}" ]; then
    cp -r "${GEN_DIR}/skel_${FIRST_OP}" "${PROJ}"
fi

# ---- 2. soc 配置串:从骨架 stub 抓权威值 -----------------------------------------
# stub 里的 AddConfig("...") 是本 toolkit 对本 SOC 的权威写法(910C/9.0.0 上是
# ascend910_93,注意与 acl.get_soc_name() 的 Ascend910_9392 **不是同一个串**)。
SOC_CFG="$(grep -o 'AddConfig("[^"]*")' "${GEN_DIR}/skel_${FIRST_OP}/op_host/${FIRST_OP}.cpp" \
           | head -1 | sed 's/AddConfig("\(.*\)")/\1/')"
if [ -z "${SOC_CFG}" ]; then
    echo "[build.sh] 未能从生成 stub 抓到 AddConfig soc 串 -- 布局与预期不符,停" >&2
    exit 1
fi
echo "[build.sh] stub AddConfig soc 串: ${SOC_CFG}"

# ---- 3. 手写源覆盖生成 stub(kernel / 共享 tiling 头 / host 原型)----------------
for op in ${OPS}; do
    for f in "op_kernel/${op}.cpp" "op_kernel/${op}_tiling.h"; do
        [ -f "${HERE}/${f}" ] || { echo "[build.sh] 缺源文件 ${HERE}/${f}" >&2; exit 1; }
        cp -f "${HERE}/${f}" "${PROJ}/${f}"
    done
    # host 源:替换 soc 占位符后落盘,再回读校验(占位符残留 = 替换逻辑失灵,停)。
    sed "s/AddConfig(\"__TERRACE_SOC__\")/AddConfig(\"${SOC_CFG}\")/" \
        "${HERE}/op_host/${op}.cpp" > "${PROJ}/op_host/${op}.cpp"
    if grep -q "__TERRACE_SOC__" "${PROJ}/op_host/${op}.cpp"; then
        echo "[build.sh] ${op}: soc 占位符替换失败,占位符仍在 —— 停" >&2
        exit 1
    fi
    grep -q "AddConfig(\"${SOC_CFG}\")" "${PROJ}/op_host/${op}.cpp" || {
        echo "[build.sh] ${op}: 回读没找到 AddConfig(\"${SOC_CFG}\") —— 停" >&2
        exit 1; }
done
# 骨架底座自带第一个算子的 stub;其余算子的 stub 从没进过 PROJ,不会有残留。
echo "[build.sh] 工程内算子源:"
ls -1 "${PROJ}/op_host" "${PROJ}/op_kernel"

# ---- 4. 工程配置:CANN 包路径 + vendor 名 ---------------------------------------
# CMakePresets.json 的 ASCEND_CANN_PACKAGE_PATH 由 msopgen 按当前环境填;vendor_name
# 缺省 "customize",改成我们的,免得和别人的自定义算子包撞同一个 vendors 目录。
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
        echo "[build.sh] CMakePresets.json 里没回读到 vendor_name=${VENDOR_NAME} —— 停" >&2
        exit 1; }
    echo "[build.sh] vendor_name=${VENDOR_NAME} 已写入 CMakePresets.json"
fi

# ---- 5. 编译 + 打包 -------------------------------------------------------------
# 失败时 kernel 的真实编译错误不在 stdout,而在 build_out 下的子构建日志里 ——
# 见 踩坑记录 的日志定位命令。
( cd "${PROJ}" && bash build.sh )

RUN_PKG="$(ls "${PROJ}"/build_out/custom_opp_*.run 2>/dev/null | head -1)"
if [ -z "${RUN_PKG}" ]; then
    echo "[build.sh] build_out 下没有 custom_opp_*.run -- 编译失败" >&2
    echo "[build.sh] 真实错误找这里:" >&2
    echo "  grep -rn 'error:' ${PROJ}/build_out --include=*.log | head -40" >&2
    exit 1
fi

# ---- 6. 安装到 opp vendors ------------------------------------------------------
# </dev/null:装包脚本万一想问什么,当即 EOF 失败,不挂着等输入。
echo "[build.sh] 安装 ${RUN_PKG}"
bash "${RUN_PKG}" </dev/null

# ---- 7. 装完回读校验:两个算子的 aclnn 头都得在 -----------------------------------
OPP_VENDOR="${ASCEND_OPP_PATH:-${ASCEND_HOME_PATH}/opp}/vendors/${VENDOR_NAME}"
MISSING=0
for op in ${OPS}; do
    if [ ! -f "${OPP_VENDOR}/op_api/include/aclnn_${op}.h" ]; then
        echo "[build.sh] 装包后缺 ${OPP_VENDOR}/op_api/include/aclnn_${op}.h" >&2
        MISSING=1
    fi
done
[ -f "${OPP_VENDOR}/op_api/lib/libcust_opapi.so" ] || {
    echo "[build.sh] 装包后缺 ${OPP_VENDOR}/op_api/lib/libcust_opapi.so" >&2; MISSING=1; }
[ "${MISSING}" -eq 0 ] || exit 1

echo "[build.sh] 完成。vendor 包: ${OPP_VENDOR}"
echo "[build.sh] 下一步: python ../csrc/build_ext.py  (torch 绑定)"
