#!/bin/bash
# 模型打包脚本 - 统一版本
# 支持在线(仅LoRA)和离线(LoRA+基础模型)两种打包模式
#
# 用法:
#   在线模式 (仅LoRA): ./package_model.sh <模型路径> <输出名称>
#   离线模式 (完整包): ./package_model.sh <模型路径> <输出名称> --offline [基础模型名]
#
# 示例:
#   ./package_model.sh ./checkpoints/novel_model/final_model my_lora
#   ./package_model.sh ./checkpoints/novel_model/final_model my_full --offline Qwen/Qwen2.5-7B-Instruct

set -e

# 颜色输出
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 参数解析
MODEL_PATH=""
OUTPUT_NAME=""
OFFLINE_MODE=false
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --offline)
            OFFLINE_MODE=true
            shift
            ;;
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        *)
            if [ -z "$MODEL_PATH" ]; then
                MODEL_PATH="$1"
            elif [ -z "$OUTPUT_NAME" ]; then
                OUTPUT_NAME="$1"
            else
                echo "未知参数: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# 默认值
MODEL_PATH="${MODEL_PATH:-./checkpoints/final_model}"
OUTPUT_NAME="${OUTPUT_NAME:-novel_model}"

# 扩展 ~ 符号
MODEL_PATH="${MODEL_PATH/#\~/$HOME}"

# HuggingFace缓存目录 (离线模式使用)
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface/hub}"

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

# 查找HuggingFace缓存模型
find_cached_model() {
    local model_name="$1"
    local cache_dir="$2"

    local cache_name="models--$(echo $model_name | tr '/' '--')"
    echo -e "${YELLOW}正在查找缓存模型: $cache_name${NC}"

    local found_dirs=$(find "$cache_dir" -maxdepth 1 -type d -name "*$cache_name*" 2>/dev/null)

    if [ -z "$found_dirs" ]; then
        print_error "未找到缓存的基础模型!"
        echo ""
        echo -e "${YELLOW}请先运行以下命令下载模型:${NC}"
        echo "  python start.py webui --lora $MODEL_PATH"
        echo ""
        echo -e "${YELLOW}或者手动下载基础模型:${NC}"
        echo "  huggingface-cli download $BASE_MODEL --local-dir ./models/base_model"
        echo ""
        echo -e "${YELLOW}已缓存的模型列表:${NC}"
        find "$cache_dir" -maxdepth 1 -type d -name "models--*" 2>/dev/null | sed 's|.*/||' || echo "  (无)"
        return 1
    fi

    local model_dir=$(echo "$found_dirs" | head -n 1)
    local snapshot_dir=$(find "$model_dir/snapshots" -maxdepth 1 -type d | tail -n 1)

    if [ ! -d "$snapshot_dir" ]; then
        print_error "模型快照目录不存在: $snapshot_dir"
        return 1
    fi

    echo "$snapshot_dir"
}

# ============================================
# 主流程
# ============================================

# 打包模式信息
if [ "$OFFLINE_MODE" = true ]; then
    MODE_TEXT="离线完整包 (LoRA + 基础模型)"
else
    MODE_TEXT="在线包 (仅LoRA适配器)"
fi

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  模型打包脚本${NC}"
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}模式: ${MODE_TEXT}${NC}"
echo -e "${CYAN}模型路径: ${MODEL_PATH}${NC}"
if [ "$OFFLINE_MODE" = true ]; then
    echo -e "${CYAN}基础模型: ${BASE_MODEL}${NC}"
fi
echo -e "${CYAN}输出名称: ${OUTPUT_NAME}${NC}"
echo ""

# 1. 检查LoRA路径
print_step "1/5" "检查LoRA模型..."
if [ ! -d "$MODEL_PATH" ]; then
    print_error "LoRA路径不存在: $MODEL_PATH"
    exit 1
fi
LORA_SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
print_success "LoRA模型: $LORA_SIZE"

# 2. 离线模式：查找基础模型
BASE_MODEL_PATH=""
BASE_MODEL_SIZE=""
if [ "$OFFLINE_MODE" = true ]; then
    print_step "2/5" "查找基础模型缓存..."
    BASE_MODEL_PATH=$(find_cached_model "$BASE_MODEL" "$HF_CACHE_DIR")

    if [ $? -ne 0 ]; then
        exit 1
    fi

    BASE_MODEL_SIZE=$(du -sh "$BASE_MODEL_PATH" | cut -f1)
    print_success "基础模型: $BASE_MODEL_SIZE @ $BASE_MODEL_PATH"
else
    print_step "2/5" "在线模式 - 基础模型将自动下载..."
    echo -e "${YELLOW}基础模型将在首次运行时从HuggingFace自动下载${NC}"
fi

# 3. 创建打包目录
TOTAL_STEPS=5
print_step "3/$TOTAL_STEPS" "准备打包目录..."
TEMP_DIR="./packages/${OUTPUT_NAME}"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

if [ "$OFFLINE_MODE" = true ]; then
    mkdir -p "$TEMP_DIR/base_model"
    mkdir -p "$TEMP_DIR/lora_adapter"
else
    mkdir -p "$TEMP_DIR/model"
fi

# 4. 复制LoRA模型
print_step "4/$TOTAL_STEPS" "复制LoRA适配器..."
if [ "$OFFLINE_MODE" = true ]; then
    cp -r "$MODEL_PATH"/* "$TEMP_DIR/lora_adapter/"
else
    cp -r "$MODEL_PATH" "$TEMP_DIR/model"
fi
print_success "LoRA适配器已复制"

# 5. 离线模式：复制基础模型
if [ "$OFFLINE_MODE" = true ]; then
    print_step "5/$TOTAL_STEPS" "复制基础模型..."
    echo -e "${YELLOW}正在复制基础模型文件 (可能需要几分钟)...${NC}"

    if command -v rsync &> /dev/null; then
        rsync -a --progress "$BASE_MODEL_PATH"/ "$TEMP_DIR/base_model/" 2>&1 | grep -E "($|sent.*total)"
    else
        cp -r "$BASE_MODEL_PATH"/* "$TEMP_DIR/base_model/"
    fi

    print_success "基础模型已复制"
else
    print_step "5/$TOTAL_STEPS" "创建部署文件..."
fi

# ============================================
# 创建部署文件
# ============================================

if [ "$OFFLINE_MODE" = true ]; then
    # 离线模式配置文件
    cat > "$TEMP_DIR/model_config.json" << EOF
{
  "base_model": "base_model",
  "base_model_name": "$BASE_MODEL",
  "lora_adapter": "lora_adapter",
  "lora_adapter_name": "$(basename $MODEL_PATH)",
  "package_type": "full_offline",
  "created_at": "$(date -Iseconds)",
  "model_size": "$BASE_MODEL_SIZE + $LORA_SIZE"
}
EOF

    # 离线部署说明
    cat > "$TEMP_DIR/DEPLOY.md" << 'EOF'
# 离线部署说明 - 完整模型包

## 模型信息
- **包类型**: 完整离线包 (基础模型 + LoRA适配器)
- **基础模型**: 见 model_config.json
- **LoRA适配器**: 见 model_config.json

## 离线部署步骤

### 1. 解压模型包
```bash
tar -xzf <package-name>.tar.gz
cd <package-name>
```

### 2. 离线安装依赖
在有网络的机器上下载requirements.txt，然后传输到目标机器:

```bash
# 在有网络的机器
pip download -r requirements.txt -d ./packages

# 在目标机器
pip install --no-index --find-links=./packages -r requirements.txt
```

### 3. 配置离线模式

编辑项目根目录的 `config.py`:

```python
@dataclass
class ModelConfig:
    # 指定本地基础模型路径
    base_model: str = "/absolute/path/to/extracted/package/base_model"

    # 其他配置保持不变
    ...
```

### 4. 启动WebUI

```bash
# 修改config.py后启动
python start.py webui --lora /absolute/path/to/lora_adapter
```

## 目录结构
```
├── base_model/              # 基础模型
│   ├── config.json
│   ├── model-*.safetensors
│   ├── tokenizer.json
│   └── ...
├── lora_adapter/            # LoRA适配器
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
├── model_config.json        # 模型配置信息
└── DEPLOY.md                # 本文件
```

## 硬件要求
| 配置 | 显存需求 |
|------|----------|
| 4-bit量化 | ~6-8 GB |
| 8-bit量化 | ~10 GB |
| FP16全精度 | ~16 GB |

## 故障排除
- 找不到基础模型: 检查config.py中的base_model路径
- 显存不足: 启用4-bit量化 (load_in_4bit=True)
EOF

    # 离线启动脚本
    cat > "$TEMP_DIR/start.sh" << 'EOF'
#!/bin/bash
# 离线启动脚本

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BASE_MODEL_PATH="$SCRIPT_DIR/base_model"
LORA_PATH="$SCRIPT_DIR/lora_adapter"

if [ ! -f "$PROJECT_ROOT/start.py" ]; then
    echo "错误: 请将本模型包放在项目根目录的 packages/ 文件夹中"
    exit 1
fi

echo "启动离线WebUI..."
echo "基础模型: $BASE_MODEL_PATH"
echo "LoRA适配器: $LORA_PATH"

cd "$PROJECT_ROOT"
python start.py webui --lora "$LORA_PATH"
EOF
    chmod +x "$TEMP_DIR/start.sh"

    # 包信息
    cat > "$TEMP_DIR/README.txt" << EOF
========================================
  中文小说写作AI系统 - 离线模型包
========================================

基础模型: $BASE_MODEL
LoRA适配器: $(basename $MODEL_PATH)
打包时间: $(date)

快速启动: bash start.sh
详细说明: 查看 DEPLOY.md

========================================
EOF

else
    # 在线模式部署说明
    cat > "$TEMP_DIR/DEPLOY.md" << 'EOF'
# LoRA模型部署说明

## 模型信息
- **模型类型**: LoRA微调权重
- **基础模型**: 将从HuggingFace自动下载

## 快速部署

### 1. 解压模型
```bash
tar -xzf <package-name>.tar.gz
cd <package-name>
```

### 2. 依赖安装
```bash
pip install -r requirements.txt
```

### 3. 启动WebUI
```bash
# 在项目根目录运行
python start.py webui --lora /absolute/path/to/model
```

## 注意事项

1. **基础模型**: 首次运行时会自动从HuggingFace下载
2. **网络**: 确保能访问 huggingface.co 或使用镜像
3. **显存**: 4-bit量化约需6-8GB VRAM

## HuggingFace镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

EOF

    # 在线启动脚本
    cat > "$TEMP_DIR/start.sh" << 'EOF'
#!/bin/bash
# 快速启动脚本 - 需要在项目根目录下运行

if [ ! -f "start.py" ]; then
    echo "错误: 请在项目根目录下运行此脚本"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LORA_PATH="$SCRIPT_DIR/model"

echo "启动WebUI，使用LoRA模型: $LORA_PATH"
python start.py webui --lora "$LORA_PATH"
EOF
    chmod +x "$TEMP_DIR/start.sh"
fi

print_success "部署文件已创建"

# 压缩打包
TOTAL_SIZE=$(du -sh "$TEMP_DIR" | cut -f1)
echo -e "\n${CYAN}压缩打包中... (总大小: ${TOTAL_SIZE})${NC}"

mkdir -p ./packages
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ "$OFFLINE_MODE" = true ]; then
    PACKAGE_FILE="./packages/${OUTPUT_NAME}_offline_${TIMESTAMP}.tar.gz"
else
    PACKAGE_FILE="./packages/${OUTPUT_NAME}_${TIMESTAMP}.tar.gz"
fi

tar -czf "$PACKAGE_FILE" -C ./packages "$OUTPUT_NAME"

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
echo -e "  1. 复制到目标机器: scp ${PACKAGE_FILE} user@host:/path/"
echo -e "  2. 解压: tar -xzf $(basename $PACKAGE_FILE)"
echo -e "  3. 在项目根目录运行: bash ${OUTPUT_NAME}/start.sh"
