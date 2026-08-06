#!/bin/sh
# 在给定目录下逐个执行可执行文件（默认参数，无额外 CLI）
# 本机与板端同一用法：sh test_run.sh /abs/or/rel/path/to/tests
# 需要给 ELF 传参时，请直接单独运行该 ELF，不要经本脚本。
# 依赖 install 布局自洽：RPATH=$ORIGIN/../lib，fixture 经 /proc/self/exe 解析。
set -eu

usage() {
    echo "用法: sh $0 <dir>" >&2
    echo "  逐个执行 <dir> 下的可执行文件（无额外参数）" >&2
    echo "  示例: sh $0 /customer/tmp/deploy_percept/tests" >&2
}

if [ $# -ne 1 ]; then
    usage
    exit 1
fi

DIR=$1
case "${DIR}" in
    /*) ;;
    *)
        DIR=$(CDPATH= cd "${DIR}" && pwd)
        ;;
esac

if [ ! -d "${DIR}" ]; then
    echo "错误: 不是目录: ${DIR}" >&2
    exit 1
fi

cd "${DIR}"

echo "=== test_run dir=${DIR} ==="

fail=0
passed=0
failed=0
skipped=0
found=0

for f in *; do
    # 无匹配时字面量 "*"
    [ -e "${f}" ] || continue
    [ -f "${f}" ] && [ -x "${f}" ] || continue
    found=1
    echo ">>> ${DIR}/${f}"
    set +e
    "./${f}"
    rc=$?
    set -e
    if [ "${rc}" -eq 0 ]; then
        echo "    PASS"
        passed=$((passed + 1))
    else
        echo "    FAIL (rc=${rc})"
        failed=$((failed + 1))
        fail=1
    fi
done

if [ "${found}" -eq 0 ]; then
    echo "skip: 目录下无可执行文件: ${DIR}"
    skipped=1
fi

echo "passed=${passed} failed=${failed} skipped=${skipped}"
exit "${fail}"
