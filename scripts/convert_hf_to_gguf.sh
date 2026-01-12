#!/bin/bash
#
# 将 Hugging Face 模型转换为 GGUF 格式
# 用法: ./scripts/convert_hf_to_gguf.sh <model_name> [quant_type]
#
# 示例:
#   ./scripts/convert_hf_to_gguf.sh Qwen/Qwen2.5-7B-Instruct Q5_K_M
#   ./scripts/convert_hf_to_gguf.sh Qwen/Qwen2.5-7B-Instruct Q8_0
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 默认参数
MODEL_NAME="${1:-Qwen/Qwen2.5-7B-Instruct}"
QUANT_TYPE="${2:-Q5_K_M}"
OUTPUT_DIR="./models"
LLAMA_CPP_DIR="./third_party/llama.cpp"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Hugging Face → GGUF 转换脚本${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# 检查输出目录
mkdir -p "$OUTPUT_DIR"

# 步骤 1: 检查/克隆 llama.cpp
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo -e "${YELLOW}克隆 llama.cpp 仓库...${NC}"
    mkdir -p ./third_party
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi

# 步骤 2: 编译 llama.cpp (如果需要)
if [ ! -f "$LLAMA_CPP_DIR/llama-quantize" ]; then
    echo -e "${YELLOW}编译 llama.cpp...${NC}"
    cd "$LLAMA_CPP_DIR"
    make
    cd -
fi

# 提取模型名称 (用于文件名)
MODEL_FILE_NAME=$(echo "$MODEL_NAME" | sed 's/\//_/g' | sed 's/-/_/g')

# 步骤 3: 转换为 FP16 GGUF 格式
F16_OUTPUT="$OUTPUT_DIR/${MODEL_FILE_NAME}-f16.gguf"
echo -e "${CYAN}步骤 1: 转换 Hugging Face 格式到 GGUF (FP16)${NC}"
echo -e "  模型: $MODEL_NAME"
echo -e "  输出: $F16_OUTPUT"

python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    --model "$MODEL_NAME" \
    --outfile "$F16_OUTPUT" \
    --outtype f16

echo -e "${GREEN}✓ FP16 转换完成${NC}"
echo ""

# 步骤 4: 量化到目标精度
QUANT_OUTPUT="$OUTPUT_DIR/${MODEL_FILE_NAME}-${QUANT_TYPE}.gguf"
echo -e "${CYAN}步骤 2: 量化到 $QUANT_TYPE${NC}"
echo -e "  输入: $F16_OUTPUT"
echo -e "  输出: $QUANT_OUTPUT"

"$LLAMA_CPP_DIR/llama-quantize" \
    "$F16_OUTPUT" \
    "$QUANT_OUTPUT" \
    "$QUANT_TYPE"

echo -e "${GREEN}✓ 量化完成${NC}"
echo ""

# 显示结果
echo -e "${CYAN}========================================${NC}"
echo -e "${GREEN}转换完成！${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "生成的文件:"
ls -lh "$QUANT_OUTPUT" "$F16_OUTPUT" | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo -e "更新 config.py 中的路径:"
echo -e "  ${YELLOW}llama_cpp_model_path: str = \"$QUANT_OUTPUT\"${NC}"
echo ""

# 可选: 删除 FP16 文件以节省空间
read -p "是否删除 FP16 文件以节省空间? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm "$F16_OUTPUT"
    echo -e "${GREEN}✓ 已删除 FP16 文件${NC}"
fi
