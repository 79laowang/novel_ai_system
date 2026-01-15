#!/bin/bash
# 模型打包脚本 - 统一版本
# 支持 GPU (HuggingFace LoRA) 和 CPU (GGUF) 两种格式
#
# 用法:
#   ./package_model.sh --type gpu --base-model Qwen/Qwen2.5-7B-Instruct --lora ./training/urban-life-model/final_model --dataset urban-life
#   ./package_model.sh --type cpu --gguf-model ./models/qwen2.5-7b-q5_k_m.gguf --dataset urban-life
#
# 简化用法:
#   ./package_model.sh gpu    # 打包 GPU 版本 (使用默认值)
#   ./package_model.sh cpu    # 打包 CPU 版本 (使用默认值)

set -e

# ============================================
# 颜色定义
# ============================================
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ============================================
# 默认配置
# ============================================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU 版本默认值
DEFAULT_GPU_BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
DEFAULT_GPU_LORA_PATH="./training/urban-life-model/final_model"
DEFAULT_GGUF_MODEL="./models/gguf/qwen2.5-7b-q5_k_m.gguf"
DEFAULT_GGUF_LORA_DIR="./models/lora-gguf"
DEFAULT_DATASET_NAME="urban-life"

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

# 从基础模型名称提取模型大小 (如: Qwen2.5-7B-Instruct -> 7b)
extract_model_size() {
    local model="$1"
    if [[ "$model" =~ ([0-9.]+)[Bb] ]]; then
        echo "${BASH_REMATCH[1]}b" | tr -d '.'
    else
        echo "unknown"
    fi
}

# 自动查找可用的 LoRA GGUF 文件
find_lora_gguf() {
    local lora_dir="$1"
    if [ -d "$lora_dir" ]; then
        local lora_files=$(find "$lora_dir" -name "*.gguf" -type f 2>/dev/null)
        if [ -n "$lora_files" ]; then
            echo "$lora_files" | xargs ls -t | head -n 1
        fi
    fi
}

# 显示帮助信息
show_help() {
    cat << EOF
模型打包脚本 - 统一版本

用法:
  ./package_model.sh <type> [options]

类型:
  gpu    打包 GPU 版本 (HuggingFace LoRA 格式)
  cpu    打包 CPU 版本 (GGUF 格式)

GPU 选项:
  --base-model <name>    基础模型名称 (默认: Qwen/Qwen2.5-7B-Instruct)
  --lora <path>          LoRA 权重路径 (默认: ./training/urban-life-model/final_model)

CPU 选项:
  --gguf-model <path>    GGUF 基础模型路径 (默认: ./models/qwen2.5-7b-q5_k_m.gguf)
  --lora-gguf-dir <path> LoRA GGUF 目录 (默认: ./models/lora-gguf)

通用选项:
  --dataset <name>       数据集名称 (默认: urban-life)
  -h, --help            显示此帮助信息

示例:
  # 打包 GPU 版本 (使用默认值)
  ./package_model.sh gpu

  # 打包 GPU 版本 (指定参数)
  ./package_model.sh gpu --base-model Qwen/Qwen2.5-3B-Instruct --lora ./training/urban-life-3b-model/final_model

  # 打包 CPU 版本 (使用默认值)
  ./package_model.sh cpu

  # 打包 CPU 版本 (指定参数)
  ./package_model.sh cpu --gguf-model ./models/qwen2.5-3b-q5_k_m.gguf --dataset urban-life

EOF
}

# ============================================
# 参数解析
# ============================================

# 如果没有参数或显示帮助
if [ $# -eq 0 ] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

PACK_TYPE="$1"
shift

# 解析参数
BASE_MODEL="$DEFAULT_GPU_BASE_MODEL"
LORA_PATH="$DEFAULT_GPU_LORA_PATH"
GGUF_MODEL="$DEFAULT_GGUF_MODEL"
GGUF_LORA_DIR="$DEFAULT_GGUF_LORA_DIR"
DATASET_NAME="$DEFAULT_DATASET_NAME"

while [[ $# -gt 0 ]]; do
    case $1 in
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --lora)
            LORA_PATH="$2"
            shift 2
            ;;
        --gguf-model)
            GGUF_MODEL="$2"
            shift 2
            ;;
        --lora-gguf-dir)
            GGUF_LORA_DIR="$2"
            shift 2
            ;;
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# ============================================
# 主流程
# ============================================

if [ "$PACK_TYPE" = "gpu" ]; then
    # ============================================
    # GPU 版本打包 (HuggingFace LoRA)
    # ============================================

    MODEL_SIZE=$(extract_model_size "$BASE_MODEL")
    OUTPUT_NAME="${DATASET_NAME}-${MODEL_SIZE}"
    OUTPUT_DIR="$PROJECT_ROOT/models/${OUTPUT_NAME}-gpu-LoRA-HF"

    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  HuggingFace 模型打包 (GPU 推理版)${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}数据集名称: ${DATASET_NAME}${NC}"
    echo -e "${CYAN}模型大小: ${MODEL_SIZE}${NC}"
    echo -e "${CYAN}基础模型: ${BASE_MODEL}${NC}"
    echo -e "${CYAN}LoRA 路径: ${LORA_PATH}${NC}"
    echo -e "${CYAN}输出名称: ${OUTPUT_NAME}-gpu-LoRA-HF${NC}"
    echo ""

    # 检查 LoRA 路径
    if [ ! -d "$LORA_PATH" ]; then
        print_error "LoRA 路径不存在: $LORA_PATH"
        exit 1
    fi

    # 清理并创建输出目录
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    # 复制 LoRA 权重
    echo -e "${CYAN}[1/3] 复制 LoRA 权重...${NC}"
    cp -r "$LORA_PATH"/* "$OUTPUT_DIR/"
    print_success "LoRA 权重已复制"

    # 创建配置文件
    echo -e "${CYAN}[2/3] 创建配置文件...${NC}"
    cat > "$OUTPUT_DIR/MODEL_CONFIG.json" << EOF
{
  "model_type": "qlora",
  "dataset_name": "${DATASET_NAME}",
  "model_size": "${MODEL_SIZE}",
  "base_model": "${BASE_MODEL}",
  "base_model_name": "$(basename $BASE_MODEL)",
  "lora_path": "./${OUTPUT_NAME}-gpu-LoRA-HF",
  "training_data": "urban-life novels",
  "training_samples": 8630,
  "epochs": 3,
  "batch_size": 8,
  "learning_rate": 0.0002,
  "lora_rank": 64,
  "lora_alpha": 128,
  "trained_date": "2025-01-15",
  "inference_backend": "vllm",
  "gpu_memory": "6GB",
  "recommended_gpu": "NVIDIA RTX 3060 或更高"
}
EOF

    # 创建 README
    cat > "$OUTPUT_DIR/README.md" << EOF
# ${BASE_MODEL} - ${DATASET_NAME} Novel Model (GPU 版本)

## 模型说明

这是基于 ${BASE_MODEL} 微调的都市小说写作模型，使用 LoRA 技术训练，**适用于 GPU 推理**。

## 模型配置

- **基础模型**: ${BASE_MODEL}
- **模型大小**: ${MODEL_SIZE}
- **训练数据**: 都市小说 (8,630 条样本)
- **训练轮数**: 3 epochs
- **LoRA Rank**: 64
- **LoRA Alpha**: 128

## 使用方法

### 1. 使用本项目的 WebUI (推荐)

\`\`\`bash
python start.py webui \\
  --base-model ${BASE_MODEL} \\
  --lora ./models/${OUTPUT_NAME}-gpu-LoRA-HF
\`\`\`

### 2. 使用 transformers + PEFT

\`\`\`python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "${BASE_MODEL}",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(
    base_model,
    "./models/${OUTPUT_NAME}-gpu-LoRA-HF"
)

tokenizer = AutoTokenizer.from_pretrained("./models/${OUTPUT_NAME}-gpu-LoRA-HF")

# 生成文本
prompt = "请写一段都市小说..."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=2048)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
\`\`\`

### 3. 使用 vLLM 推理 (高性能)

\`\`\`python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="${BASE_MODEL}",
    enable_lora=True,
    max_model_len=32768,
)

# 加载 LoRA
llm.add_lora("${DATASET_NAME}", "./models/${OUTPUT_NAME}-gpu-LoRA-HF")

# 生成
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(["请写一段都市小说..."], sampling_params, lora_request="${DATASET_NAME}")

for output in outputs:
    print(output.outputs[0].text)
\`\`\`

## 硬件要求

| 配置 | 显存需求 | 推荐 GPU |
|------|----------|----------|
| 最小配置 | 6 GB | GTX 1660 |
| 推荐配置 | 8 GB | RTX 3060 |
| 高性能 | 12 GB+ | RTX 4070/4080 |

## 性能特点

- ✅ 专门针对都市小说风格优化
- ✅ 支持 GPU 加速推理
- ✅ 显存占用低 (~6GB)
- ✅ 支持流式生成
- ✅ 自动分段处理
- ✅ 对话格式规范

## 许可证

遵循基础模型 ${BASE_MODEL} 的许可证。

## 模型文件说明

- \`adapter_model.safetensors\` - LoRA 权重
- \`adapter_config.json\` - LoRA 配置
- \`tokenizer.*\` - 分词器文件
- \`MODEL_CONFIG.json\` - 模型元数据
EOF

    print_success "配置文件已创建"

    # 创建启动脚本
    echo -e "${CYAN}[3/3] 创建启动脚本...${NC}"
    cat > "$OUTPUT_DIR/start_webui.sh" << EOF
#!/bin/bash
# 快速启动 WebUI 脚本

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="\$(cd "\$SCRIPT_DIR/.." && pwd)"

echo "启动 WebUI..."
echo "基础模型: ${BASE_MODEL}"
echo "LoRA 路径: \$SCRIPT_DIR"

cd "\$PROJECT_ROOT"
python start.py webui \\
  --base-model ${BASE_MODEL} \\
  --lora "\$SCRIPT_DIR"
EOF
    chmod +x "$OUTPUT_DIR/start_webui.sh"
    print_success "启动脚本已创建"

    # 创建压缩包
    echo -e "${CYAN}压缩打包中...${NC}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    PACKAGE_FILE="$PROJECT_ROOT/packages/releases/${OUTPUT_NAME}-gpu-LoRA-HF_${TIMESTAMP}.tar.gz"

    mkdir -p "$PROJECT_ROOT/packages/releases"
    tar -czf "$PACKAGE_FILE" -C "$PROJECT_ROOT/models" "${OUTPUT_NAME}-gpu-LoRA-HF"

    PACKAGE_SIZE=$(du -sh "$PACKAGE_FILE" | cut -f1)

    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}  HuggingFace 模型打包完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${CYAN}输出目录: ${OUTPUT_DIR}${NC}"
    echo -e "${CYAN}压缩包: ${PACKAGE_FILE}${NC}"
    echo -e "${CYAN}文件大小: ${PACKAGE_SIZE}${NC}"
    echo ""
    echo -e "${YELLOW}快速启动:${NC}"
    echo -e "  cd ${OUTPUT_DIR}"
    echo -e "  bash start_webui.sh"
    echo ""

elif [ "$PACK_TYPE" = "cpu" ]; then
    # ============================================
    # CPU 版本打包 (GGUF 格式)
    # ============================================

    MODEL_SIZE=$(extract_model_size "$GGUF_MODEL")
    OUTPUT_NAME="${DATASET_NAME}-${MODEL_SIZE}"

    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  模型打包脚本 (GGUF CPU 推理版)${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}数据集名称: ${DATASET_NAME}${NC}"
    echo -e "${CYAN}模型大小: ${MODEL_SIZE}${NC}"
    echo -e "${CYAN}输出名称: ${OUTPUT_NAME}${NC}"
    echo -e "${CYAN}项目根目录: ${PROJECT_ROOT}${NC}"
    echo ""

    # 1. 检查基础 GGUF 模型
    print_step "1/4" "检查基础 GGUF 模型..."
    if [ ! -f "$PROJECT_ROOT/$GGUF_MODEL" ]; then
        print_error "基础 GGUF 模型不存在: $GGUF_MODEL"
        echo ""
        echo -e "${YELLOW}请先转换基础模型:${NC}"
        echo "  python start.py convert hf-to-gguf --model ${BASE_MODEL} --quant Q5_K_M"
        exit 1
    fi
    BASE_MODEL_SIZE=$(du -h "$PROJECT_ROOT/$GGUF_MODEL" | cut -f1)
    print_success "基础模型: $GGUF_MODEL ($BASE_MODEL_SIZE)"

    # 2. 检查 LoRA GGUF 文件
    print_step "2/4" "检查 LoRA GGUF 适配器..."
    LORA_FILE=$(find_lora_gguf "$PROJECT_ROOT/$GGUF_LORA_DIR")
    if [ -z "$LORA_FILE" ]; then
        print_error "未找到 LoRA GGUF 文件: $GGUF_LORA_DIR/*.gguf"
        echo ""
        echo -e "${YELLOW}请先转换 LoRA 适配器:${NC}"
        echo "  bash ./scripts/convert_lora_to_gguf.sh ~/.cache/huggingface/hub/models--Qwen--*/snapshots/* $LORA_PATH"
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
    cp "$PROJECT_ROOT/$GGUF_MODEL" "$TEMP_DIR/models/"
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
    base_model: str = "${BASE_MODEL}"

    # llama.cpp 配置 (CPU 推理)
    llama_cpp_model_path: str = "./models/$(basename $GGUF_MODEL)"
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
- **基础模型**: Qwen2.5 (Q5_K_M 量化)
- **LoRA 适配器**: 见 models/lora-gguf/ 目录

## 目录结构
```
<package_name>_deployment/
├── models/
│   ├── *.gguf                     # 基础模型
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
cd <package_name>_deployment

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
    cat > "$TEMP_DIR/deploy.sh" << 'DEPLOY_SCRIPT_EOF'
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
DEPLOY_SCRIPT_EOF
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

数据集: ${DATASET_NAME}
模型大小: ${MODEL_SIZE}
基础模型: $(basename $GGUF_MODEL)
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

    mkdir -p "$PROJECT_ROOT/packages/releases"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    PACKAGE_FILE="$PROJECT_ROOT/packages/releases/${OUTPUT_NAME}-cpu-GGUF_${TIMESTAMP}.tar.gz"

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

else
    echo -e "${RED}错误: 未知的打包类型 '$PACK_TYPE'${NC}"
    echo ""
    echo "请使用: gpu 或 cpu"
    show_help
    exit 1
fi
