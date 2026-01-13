#!/bin/bash
# 模型打包脚本 - CPU 推理版本 (GGUF 格式)
# 打包基础 GGUF 模型 + LoRA GGUF 适配器，保留目录结构
#
# 用法:
#   ./package_model.sh [输出名称]
#
# 示例:
#   ./package_model.sh urban_model

set -e

# 颜色输出
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ============================================
# 配置
# ============================================

# 默认 GGUF 模型路径
GGUF_BASE_MODEL="./models/qwen2.5-7b-q5_k_m.gguf"
GGUF_LORA_DIR="./models/lora-gguf"

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================
# 函数定义
# ============================================

print_step() {
    echo -e "\n${CYAN}[$1] $2${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# 自动查找可用的 LoRA GGUF 文件
find_lora_gguf() {
    if [ -d "$GGUF_LORA_DIR" ]; then
        local lora_files=$(find "$GGUF_LORA_DIR" -name "*.gguf" -type f 2>/dev/null)
        if [ -n "$lora_files" ]; then
            # 返回最新的文件
            echo "$lora_files" | xargs ls -t | head -n 1
        fi
    fi
}

# 从 LoRA 文件名提取模型名称
extract_model_name() {
    local lora_path="$1"
    basename "$lora_path" | sed 's/-gguf\.gguf$//' | sed 's/-/_/g'
}

# ============================================
# 参数解析
# ============================================

OUTPUT_NAME="$1"

# 如果没有指定输出名称，自动推导
if [ -z "$OUTPUT_NAME" ]; then
    LORA_FILE=$(find_lora_gguf)
    if [ -n "$LORA_FILE" ]; then
        OUTPUT_NAME=$(extract_model_name "$LORA_FILE")
        echo -e "${YELLOW}自动推导输出名称: $OUTPUT_NAME${NC}"
    else
        OUTPUT_NAME="novel_model"
    fi
fi

# ============================================
# 主流程
# ============================================

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  模型打包脚本 (GGUF CPU 推理版)${NC}"
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}输出名称: ${OUTPUT_NAME}${NC}"
echo -e "${CYAN}项目根目录: ${PROJECT_ROOT}${NC}"
echo ""

# 1. 检查基础 GGUF 模型
print_step "1/4" "检查基础 GGUF 模型..."
if [ ! -f "$PROJECT_ROOT/$GGUF_BASE_MODEL" ]; then
    print_error "基础 GGUF 模型不存在: $GGUF_BASE_MODEL"
    echo ""
    echo -e "${YELLOW}请先转换基础模型:${NC}"
    echo "  python start.py convert hf-to-gguf --model Qwen/Qwen2.5-7B-Instruct --quant Q5_K_M"
    exit 1
fi
BASE_MODEL_SIZE=$(du -h "$PROJECT_ROOT/$GGUF_BASE_MODEL" | cut -f1)
print_success "基础模型: $GGUF_BASE_MODEL ($BASE_MODEL_SIZE)"

# 2. 检查 LoRA GGUF 文件
print_step "2/4" "检查 LoRA GGUF 适配器..."
LORA_FILE=$(find_lora_gguf)
if [ -z "$LORA_FILE" ]; then
    print_error "未找到 LoRA GGUF 文件: $GGUF_LORA_DIR/*.gguf"
    echo ""
    echo -e "${YELLOW}请先转换 LoRA 适配器:${NC}"
    echo "  bash ./scripts/convert_lora_to_gguf.sh ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* ./checkpoints/<model_name>/final_model"
    exit 1
fi
LORA_SIZE=$(du -h "$LORA_FILE" | cut -f1)
LORA_FILENAME=$(basename "$LORA_FILE")
print_success "LoRA 适配器: $LORA_FILENAME ($LORA_SIZE)"

# 3. 准备打包目录 (保留目录结构)
print_step "3/4" "准备打包目录..."
TEMP_DIR="$PROJECT_ROOT/packages/${OUTPUT_NAME}_deployment"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# 创建目录结构
mkdir -p "$TEMP_DIR/models"
mkdir -p "$TEMP_DIR/config"

# 复制基础 GGUF 模型
echo -e "${YELLOW}复制基础 GGUF 模型...${NC}"
cp "$PROJECT_ROOT/$GGUF_BASE_MODEL" "$TEMP_DIR/models/"
print_success "基础模型已复制"

# 复制 LoRA GGUF 文件
echo -e "${YELLOW}复制 LoRA GGUF 适配器...${NC}"
mkdir -p "$TEMP_DIR/models/lora-gguf"
cp "$LORA_FILE" "$TEMP_DIR/models/lora-gguf/"
print_success "LoRA 适配器已复制"

# 4. 创建部署文件
print_step "4/4" "创建部署文件..."

# 配置文件
cat > "$TEMP_DIR/config/model_config.py" << EOF
"""
模型配置 - 部署环境
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ModelConfig:
    """模型配置 - CPU 推理 (GGUF)"""
    # 推理后端选择
    inference_backend: str = "llama_cpp"

    # 基础模型选择 (仅用于训练参考)
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # llama.cpp 配置 (CPU 推理)
    llama_cpp_model_path: str = "./models/qwen2.5-7b-q5_k_m.gguf"
    llama_cpp_lora_path: Optional[str] = "./models/lora-gguf/${LORA_FILENAME}"
    llama_cpp_n_ctx: int = 32768
    llama_cpp_n_threads: int = 6
    llama_cpp_n_batch: int = 512
    llama_cpp_use_mmap: bool = True
    llama_cpp_use_mlock: bool = False
    llama_cpp_gpu_layers: int = 0

    # 其他配置保持默认...
EOF

# 部署说明
cat > "$TEMP_DIR/DEPLOY.md" << 'EOF'
# 模型部署说明

## 模型信息
- **包类型**: CPU 推理完整包 (GGUF 格式)
- **基础模型**: Qwen2.5-7B-Instruct (Q5_K_M 量化)
- **LoRA 适配器**: 见 models/lora-gguf/ 目录

## 目录结构
```
<package_name>_deployment/
├── models/
│   ├── qwen2.5-7b-q5_k_m.gguf     # 基础模型 (5.1GB)
│   └── lora-gguf/
│       └── *.gguf                  # LoRA 适配器
├── config/
│   └── model_config.py             # 模型配置
├── deploy.sh                       # 一键部署脚本
├── start.sh                        # 启动脚本
└── DEPLOY.md                       # 本文件
```

## 一键部署

### 方法 1: 使用部署脚本 (推荐)
```bash
# 解压
tar -xzf <package-name>.tar.gz
cd <package-name>_deployment

# 运行部署脚本
bash deploy.sh
```

### 方法 2: 手动部署
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 将配置文件复制到项目
cp config/model_config.py <project_root>/config_model.py

# 3. 启动服务
bash start.sh
```

## 硬件要求
| 配置 | 内存需求 |
|------|----------|
| 最小配置 | 8 GB RAM |
| 推荐配置 | 16 GB RAM |
| 上下文长度 | 32768 tokens |

## 依赖安装
```bash
pip install -r requirements.txt
```

主要依赖:
- llama-cpp-python
- torch
- transformers
- gradio
- chromadb
- sentence-transformers

## 故障排除
- **内存不足**: 减小上下文长度 (llama_cpp_n_ctx)
- **速度慢**: 增加 CPU 线程数 (llama_cpp_n_threads)
- **找不到模型**: 检查 config.py 中的路径是否正确
EOF

# 部署脚本
cat > "$TEMP_DIR/deploy.sh" << 'EOF'
#!/bin/bash
# 一键部署脚本

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  模型部署脚本${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_NAME="$(basename "$SCRIPT_DIR")"

# 1. 检查项目目录
echo -e "${CYAN}[1/4] 检查项目环境...${NC}"
if [ -f "../start.py" ]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    echo -e "${GREEN}✓ 找到项目根目录: $PROJECT_ROOT${NC}"
elif [ -f "../../start.py" ]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    echo -e "${GREEN}✓ 找到项目根目录: $PROJECT_ROOT${NC}"
else
    echo -e "${YELLOW}警告: 未找到项目根目录${NC}"
    echo -e "${YELLOW}请确保将本目录放在项目根目录的 packages/ 文件夹中${NC}"
    PROJECT_ROOT="$SCRIPT_DIR"
fi

# 2. 复制模型文件
echo -e "${CYAN}[2/4] 复制模型文件...${NC}"
mkdir -p "$PROJECT_ROOT/models"
cp -n "$SCRIPT_DIR/models/"*.gguf "$PROJECT_ROOT/models/" 2>/dev/null || true
cp -rn "$SCRIPT_DIR/models/lora-gguf" "$PROJECT_ROOT/models/" 2>/dev/null || true
echo -e "${GREEN}✓ 模型文件已复制${NC}"

# 3. 更新配置
echo -e "${CYAN}[3/4] 更新配置文件...${NC}"
if [ -f "$PROJECT_ROOT/config.py" ]; then
    # 备份原配置
    cp "$PROJECT_ROOT/config.py" "$PROJECT_ROOT/config.py.bak.$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}已备份原配置: config.py.bak.*${NC}"
fi

# 读取模型文件名
LORA_FILE=$(ls "$SCRIPT_DIR/models/lora-gguf/"*.gguf 2>/dev/null | head -n 1)
LORA_FILENAME=$(basename "$LORA_FILE" 2>/dev/null || echo "urban_model-gguf.gguf")

# 更新 config.py 中的路径
sed -i "s|llama_cpp_lora_path: Optional\[str\] = \".*\"|llama_cpp_lora_path: Optional[str] = \"./models/lora-gguf/$LORA_FILENAME\"|g" "$PROJECT_ROOT/config.py"
sed -i 's|inference_backend: str = "vllm"|inference_backend: str = "llama_cpp"|g' "$PROJECT_ROOT/config.py"
echo -e "${GREEN}✓ 配置已更新${NC}"

# 4. 安装依赖
echo -e "${CYAN}[4/4] 检查依赖...${NC}"
if command -v python &> /dev/null; then
    echo -e "${GREEN}✓ Python 已安装${NC}"
else
    echo -e "${YELLOW}警告: 未找到 Python${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  部署完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${CYAN}启动方法:${NC}"
echo -e "  cd $PROJECT_ROOT"
echo -e "  python start.py webui"
echo ""
echo -e "${CYAN}或使用启动脚本:${NC}"
echo -e "  bash $SCRIPT_DIR/start.sh"
echo ""
EOF
chmod +x "$TEMP_DIR/deploy.sh"

# 启动脚本
cat > "$TEMP_DIR/start.sh" << 'EOF'
#!/bin/bash
# 启动脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 查找项目根目录
if [ -f "../start.py" ]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [ -f "../../start.py" ]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    echo "错误: 未找到项目根目录"
    echo "请确保将本目录放在项目根目录的 packages/ 文件夹中"
    exit 1
fi

cd "$PROJECT_ROOT"
echo "启动 WebUI..."
echo "项目目录: $PROJECT_ROOT"
python start.py webui
EOF
chmod +x "$TEMP_DIR/start.sh"

# 包信息
cat > "$TEMP_DIR/README.txt" << EOF
========================================
  中文小说写作AI系统 - CPU 推理模型包
========================================

基础模型: Qwen2.5-7B-Instruct (GGUF Q5_K_M)
LoRA 适配器: ${LORA_FILENAME}
打包时间: $(date)

一键部署: bash deploy.sh
直接启动: bash start.sh
详细说明: 查看 DEPLOY.md

========================================
EOF

print_success "部署文件已创建"

# 压缩打包
TOTAL_SIZE=$(du -sh "$TEMP_DIR" | cut -f1)
echo -e "\n${CYAN}压缩打包中... (总大小: ${TOTAL_SIZE})${NC}"

mkdir -p "$PROJECT_ROOT/packages"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_FILE="$PROJECT_ROOT/packages/${OUTPUT_NAME}_cpu_${TIMESTAMP}.tar.gz"

tar -czf "$PACKAGE_FILE" -C "$PROJECT_ROOT/packages" "${OUTPUT_NAME}_deployment"

# 清理临时目录
rm -rf "$TEMP_DIR"

PACKAGE_SIZE=$(du -sh "$PACKAGE_FILE" | cut -f1)

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  打包完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${CYAN}文件位置: ${PACKAGE_FILE}${NC}"
echo -e "${CYAN}文件大小: ${PACKAGE_SIZE} (压缩后)${NC}"
echo -e "${CYAN}原始大小: ${TOTAL_SIZE}${NC}"
echo ""
echo -e "${YELLOW}部署方法:${NC}"
echo -e "  1. 复制到目标机器:"
echo -e "     scp ${PACKAGE_FILE} user@host:/path/"
echo ""
echo -e "  2. 在目标机器解压并部署:"
echo -e "     tar -xzf $(basename $PACKAGE_FILE)"
echo -e "     cd ${OUTPUT_NAME}_deployment"
echo -e "     bash deploy.sh"
echo ""
