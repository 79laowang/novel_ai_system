#!/bin/bash
#
# 将训练好的 LoRA adapter 转换为 GGUF 格式
# 用法: ./scripts/convert_lora_to_gguf.sh <base_model> <lora_path> [output_dir]
#
# 示例:
#   ./scripts/convert_lora_to_gguf.sh Qwen/Qwen2.5-7B-Instruct ./checkpoints/final_model
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 参数
BASE_MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
LORA_PATH="${2:-./checkpoints/final_model}"
OUTPUT_DIR="${3:-./models/lora-gguf}"
LLAMA_CPP_DIR="./third_party/llama.cpp"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  LoRA → GGUF 转换脚本${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# 检查 LoRA 路径
if [ ! -d "$LORA_PATH" ]; then
    echo -e "${RED}错误: LoRA 路径不存在: $LORA_PATH${NC}"
    exit 1
fi

# 检查 llama.cpp
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo -e "${YELLOW}克隆 llama.cpp 仓库...${NC}"
    mkdir -p ./third_party
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo -e "${CYAN}LoRA 转换配置:${NC}"
echo -e "  基础模型: $BASE_MODEL"
echo -e "  LoRA 路径: $LORA_PATH"
echo -e "  输出目录: $OUTPUT_DIR"
echo ""

# 检查是否有转换脚本
CONVERT_SCRIPT="$LLAMA_CPP_DIR/convert_lora_to_gguf.py"

if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo -e "${RED}错误: llama.cpp 中没有 convert_lora_to_gguf.py${NC}"
    echo -e "${YELLOW}请更新 llama.cpp 到最新版本${NC}"
    echo -e "  cd $LLAMA_CPP_DIR && git pull${NC}"
    exit 1
fi

# 执行转换
echo -e "${CYAN}开始转换 LoRA...${NC}"

# 提取 LoRA 名称用于输出文件
LORA_NAME=$(basename "$LORA_PATH")
OUTPUT_FILE="$OUTPUT_DIR/${LORA_NAME}-gguf.pth"

python "$CONVERT_SCRIPT" \
    --base "$BASE_MODEL" \
    "$LORA_PATH" \
    --outfile "$OUTPUT_FILE"

echo ""
echo -e "${GREEN}✓ LoRA 转换完成${NC}"
echo ""

# 显示结果
echo -e "${CYAN}生成的文件:${NC}"
ls -lh "$OUTPUT_DIR"/*.gguf 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  (未找到 GGUF 文件)"
echo ""

echo -e "更新 config.py 中的路径:"
echo -e "  ${YELLOW}llama_cpp_lora_path: str = \"$OUTPUT_DIR\"${NC}"
echo ""

# 使用说明
echo -e "${CYAN}使用方法:${NC}"
echo -e "  1. 确保 GGUF 基础模型已转换"
echo -e "  2. 在代码中加载:"
echo -e "     ${YELLOW}from llama_cpp import Llama${NC}"
echo -e "     ${YELLOW}llm = Llama(model_path=..., lora_path=\"$OUTPUT_DIR\")${NC}"
echo ""
