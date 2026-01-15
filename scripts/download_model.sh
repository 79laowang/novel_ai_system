#!/bin/bash
#
# HuggingFace 模型下载脚本 (Shell 包装器)
# 使用官方 huggingface_hub + 镜像源
#
# 用法:
#   ./scripts/download_model.sh Qwen/Qwen2.5-7B-Instruct
#   ./scripts/download_model.sh --embedding
#

set -e

# 颜色定义
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  HuggingFace 模型下载工具${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# 运行 Python 脚本
python3 scripts/download_hf_model.py "$@"

echo ""
echo -e "${GREEN}✓ 完成!${NC}"
