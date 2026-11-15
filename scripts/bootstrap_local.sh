#!/bin/sh
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

python3 -m pip install -e '.[video]'

if [ ! -f config/local.toml ]; then
    python3 -m voca init-config
fi

python3 -m voca doctor

cat <<'EOF'

本地命令已安装。请编辑 config/local.toml 中的远程主机和项目目录，
然后运行：python3 -m voca doctor --remote --profile mutsumi_excited_gsv
EOF
