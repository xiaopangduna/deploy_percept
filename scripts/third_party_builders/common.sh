#!/bin/bash
# 第三方库构建器公共函数

# 需先设置 PROJECT_ROOT
init_modules_tmp() {
    TMP_MODULES_DIR="${PROJECT_ROOT}/tmp/modules"
    mkdir -p "${TMP_MODULES_DIR}" "${PROJECT_ROOT}/third_party"
}

setup_git_safe_directories() {
    git config --global --add safe.directory "${PROJECT_ROOT}/tmp" 2>/dev/null || true
    git config --global --add safe.directory "${TMP_MODULES_DIR}" 2>/dev/null || true
    local git_dir
    for git_dir in "${TMP_MODULES_DIR}/"*/; do
        if [ -d "${git_dir}.git" ]; then
            git config --global --add safe.directory "${git_dir}" 2>/dev/null || true
        fi
    done
}
