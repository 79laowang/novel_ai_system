#!/bin/bash
# 模型目录重组脚本 - 开发与部署分离
#
# 用法: bash scripts/reorganize_models.sh [--dry-run]

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}DRY RUN 模式 - 不会实际执行操作${NC}\n"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  模型目录重组 - 开发与部署分离${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# ============================================
# 1. 创建新的目录结构
# ============================================
echo -e "${CYAN}[1/5] 创建新的目录结构...${NC}"

mkdir -p models/base
mkdir -p models/gguf
mkdir -p models/lora-gguf
mkdir -p models/production/gpu
mkdir -p models/production/cpu
mkdir -p packages/releases
mkdir -p packages/archives
mkdir -p checkpoints/experiments

if [ "$DRY_RUN" = false ]; then
    echo -e "${GREEN}✓ 目录结构已创建${NC}"
else
    echo -e "${YELLOW}[DRY RUN] 将创建目录结构${NC}"
fi

# ============================================
# 2. 迁移 GGUF 基础模型
# ============================================
echo -e "\n${CYAN}[2/5] 整理 GGUF 基础模型...${NC}"

if [ -f "./models/qwen2.5-1.5b-q5_k_m.gguf" ] && [ ! -f "./models/gguf/qwen2.5-1.5b-q5_k_m.gguf" ]; then
    if [ "$DRY_RUN" = false ]; then
        mv "./models/qwen2.5-1.5b-q5_k_m.gguf" "./models/gguf/"
        echo -e "${GREEN}✓ 已移动: qwen2.5-1.5b-q5_k_m.gguf${NC}"
    else
        echo -e "${YELLOW}[DRY RUN] 将移动: qwen2.5-1.5b-q5_k_m.gguf -> models/gguf/${NC}"
    fi
fi

if [ -f "./models/qwen2.5-3b-q5_k_m.gguf" ] && [ ! -f "./models/gguf/qwen2.5-3b-q5_k_m.gguf" ]; then
    if [ "$DRY_RUN" = false ]; then
        mv "./models/qwen2.5-3b-q5_k_m.gguf" "./models/gguf/"
        echo -e "${GREEN}✓ 已移动: qwen2.5-3b-q5_k_m.gguf${NC}"
    else
        echo -e "${YELLOW}[DRY RUN] 将移动: qwen2.5-3b-q5_k_m.gguf -> models/gguf/${NC}"
    fi
fi

if [ -f "./models/qwen2.5-7b-q5_k_m.gguf" ] && [ ! -f "./models/gguf/qwen2.5-7b-q5_k_m.gguf" ]; then
    if [ "$DRY_RUN" = false ]; then
        mv "./models/qwen2.5-7b-q5_k_m.gguf" "./models/gguf/"
        echo -e "${GREEN}✓ 已移动: qwen2.5-7b-q5_k_m.gguf${NC}"
    else
        echo -e "${YELLOW}[DRY RUN] 将移动: qwen2.5-7b-q5_k_m.gguf -> models/gguf/${NC}"
    fi
fi

# ============================================
# 3. 创建基础模型符号链接
# ============================================
echo -e "\n${CYAN}[3/5] 创建基础模型符号链接...${NC}"

create_symlink() {
    local target="$1"
    local link_name="$2"
    local cache_path=$(find ~/.cache/huggingface/hub -name "*$target*" -type d 2>/dev/null | grep -v ".locks" | head -n 1)

    if [ -n "$cache_path" ] && [ -d "$cache_path" ]; then
        if [ "$DRY_RUN" = false ]; then
            ln -sfn "$cache_path" "$link_name"
            echo -e "${GREEN}✓ 已创建符号链接: $link_name -> $cache_path${NC}"
        else
            echo -e "${YELLOW}[DRY RUN] 将创建符号链接: $link_name -> $cache_path${NC}"
        fi
    else
        echo -e "${RED}✗ 未找到缓存: $target${NC}"
    fi
}

create_symlink "Qwen2.5-1.5B-Instruct" "./models/base/Qwen2.5-1.5B-Instruct"
create_symlink "Qwen2.5-3B-Instruct" "./models/base/Qwen2.5-3B-Instruct"
create_symlink "Qwen2.5-7B-Instruct" "./models/base/Qwen2.5-7B-Instruct"

# ============================================
# 4. 整理训练检查点
# ============================================
echo -e "\n${CYAN}[4/5] 整理训练检查点...${NC}"

# 移动旧的 urban_model 到 experiments
if [ -d "./checkpoints/urban_model" ]; then
    if [ "$DRY_RUN" = false ]; then
        mv "./checkpoints/urban_model" "./checkpoints/experiments/"
        echo -e "${GREEN}✓ 已移动: urban_model -> experiments/${NC}"
    else
        echo -e "${YELLOW}[DRY RUN] 将移动: urban_model -> experiments/${NC}"
    fi
fi

# ============================================
# 5. 归档旧的打包文件
# ============================================
echo -e "\n${CYAN}[5/5] 归档旧的打包文件...${NC}"

archive_count=0
for file in packages/*.tar.gz; do
    if [ -f "$file" ]; then
        if [ "$DRY_RUN" = false ]; then
            mv "$file" "packages/archives/"
            echo -e "${GREEN}✓ 已归档: $(basename $file)${NC}"
        else
            echo -e "${YELLOW}[DRY RUN] 将归档: $(basename $file)${NC}"
        fi
        ((archive_count++))
    fi
done

if [ $archive_count -eq 0 ]; then
    echo -e "${YELLOW}没有需要归档的文件${NC}"
fi

# ============================================
# 总结
# ============================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  重组完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${CYAN}新的目录结构:${NC}"
echo "models/"
echo "├── base/              # 基础模型 (符号链接到 HF 缓存)"
echo "├── gguf/              # GGUF 量化模型"
echo "├── lora-gguf/         # LoRA GGUF 适配器"
echo "└── production/        # 生产环境模型"
echo "    ├── gpu/           # GPU 推理模型"
echo "    └── cpu/           # CPU 推理模型"
echo ""
echo "checkpoints/"
echo "├── experiments/       # 实验性模型"
echo "├── urban-life-1.5b-model/"
echo "└── urban-life-model/"
echo ""
echo "packages/"
echo "├── releases/          # 正式发布版本"
echo "└── archives/          # 归档版本"
echo ""
echo -e "${YELLOW}注意: 请更新 config.py 中的路径配置${NC}"
echo ""
