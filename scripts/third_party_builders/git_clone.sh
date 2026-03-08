#!/bin/bash
# 0.0.2 2026.03.08

# 用法: ./git_clone.sh "命令1" "命令2" "命令3"
# 示例: ./git_clone.sh \
#       "git clone -b v1.14.0 https://github.com/google/googletest.git" \
#       "git clone https://gitee.com/mirrors/googletest.git"

for cmd in "$@"; do
    echo "尝试执行: $cmd"
    
    if eval "$cmd" ; then
        echo "✅ 成功：$cmd"
        exit 0
    else
        echo "❌ 失败，尝试下一个命令..."
        # 如果创建了目录但克隆失败，清理掉（提取目录名）
        dir=$(echo "$cmd" | grep -o '[^ ]*\.git' | sed 's/\.git$//' | xargs basename)
        [ -n "$dir" ] && [ -d "$dir" ] && rm -rf "$dir"
    fi
done

echo "所有命令都执行失败 ❌"
exit 1