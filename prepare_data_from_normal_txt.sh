#!/bin/bash
# 通用数据准备脚本：从指定目录复制 TXT/JSON/JSONL 文件并准备训练
# 用法: ./prepare_data_from_normal_txt.sh <源目录> [输出名称]
#
# 说明: 输出名称可选，默认使用源目录的 basename
#
# 示例:
#   ./prepare_data_from_normal_txt.sh ~/work/urban-novels
#   ./prepare_data_from_normal_txt.sh ~/work/wuxia-data wuxia

set -e

# 颜色输出
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ============================================
# 参数解析
# ============================================
SOURCE_DIR="${1:-}"
OUTPUT_NAME="${2:-}"

# 显示帮助信息
show_help() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  通用数据准备脚本${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo "用法: $0 <源目录> [输出名称]"
    echo ""
    echo "参数:"
    echo "  源目录    - 包含 TXT/JSON/JSONL 文件的目录路径"
    echo "  输出名称  - 数据集名称 (可选，默认使用源目录名)"
    echo ""
    echo "示例:"
    echo "  $0 ~/work/urban-novels"
    echo "  $0 ~/work/wuxia-data wuxia"
    echo "  $0 ~/data/fantasy-novels fantasy"
    echo ""
    echo "生成的目录结构:"
    echo "  ./data/raw_<输出名称>/     - 原始数据"
    echo "  ./data/train_<输出名称>/   - 训练数据"
    echo "  ./data/val_<输出名称>/     - 验证数据"
    echo ""
    echo "默认命名规则:"
    echo "  源目录 ~/work/urban-novels -> 输出名 urban-novels"
    echo "  源目录 ~/data/xianxia_data -> 输出名 xianxia_data"
    exit 0
}

# 检查参数
if [ -z "$SOURCE_DIR" ]; then
    echo -e "${RED}错误: 缺少源目录参数${NC}"
    echo ""
    show_help
fi

# 如果没有指定输出名称，从源目录推导
if [ -z "$OUTPUT_NAME" ]; then
    # 获取源目录的 basename，并移除可能的后缀如 -data, _raw 等
    DIR_NAME=$(basename "$SOURCE_DIR")

    # 清理目录名中的常见后缀
    OUTPUT_NAME=$(echo "$DIR_NAME" | sed -E 's/[-_]?(data|raw|files|novels|txt)?$//g' | sed 's/^-+//;s/-+$//')

    # 如果清理后为空，使用原目录名
    if [ -z "$OUTPUT_NAME" ]; then
        OUTPUT_NAME="$DIR_NAME"
    fi

    echo -e "${YELLOW}输出名称未指定，使用推导值: $OUTPUT_NAME${NC}"
fi

# 扩展 ~ 符号
SOURCE_DIR="${SOURCE_DIR/#\~/$HOME}"

# ============================================
# 配置参数 (可手动修改)
# ============================================
RAW_DIR="./data/raw_${OUTPUT_NAME}"         # 原始数据目录
TRAIN_DIR="./data/train_${OUTPUT_NAME}"     # 训练数据目录
VAL_DIR="./data/val_${OUTPUT_NAME}"         # 验证数据目录

CHUNK_SIZE=2048                        # 训练块大小
VAL_SPLIT=0.1                          # 验证集比例 (10%)
MIN_LENGTH=500                         # 最小文本长度

# ============================================
# 函数定义
# ============================================

print_header() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  通用数据准备脚本${NC}"
    echo -e "${CYAN}========================================${NC}"
}

print_step() {
    echo -e "\n${CYAN}[$1] $2${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ============================================
# 主流程
# ============================================

print_header
echo -e "${CYAN}源目录: ${SOURCE_DIR}${NC}"
echo -e "${CYAN}输出名称: ${OUTPUT_NAME}${NC}"
echo -e "${CYAN}输出目录: ${RAW_DIR}${NC}"

# 1. 检查源数据
print_step "1/5" "检查源数据目录..."
if [ ! -d "$SOURCE_DIR" ]; then
    print_error "源目录不存在: $SOURCE_DIR"
    exit 1
fi

FILE_COUNT=$(find "$SOURCE_DIR" -type f \( -name "*.txt" -o -name "*.json" -o -name "*.jsonl" \) 2>/dev/null | wc -l)
if [ "$FILE_COUNT" -eq 0 ]; then
    print_error "在 $SOURCE_DIR 中未找到数据文件"
    exit 1
fi
print_success "找到 $FILE_COUNT 个数据文件"

# 显示文件列表
echo -e "\n${YELLOW}文件列表:${NC}"
ls -lh "$SOURCE_DIR/"

# 2. 清理并创建目录
print_step "2/5" "准备数据目录..."

# 清空旧数据
rm -rf "$RAW_DIR" "$TRAIN_DIR" "$VAL_DIR"

# 创建新目录
mkdir -p "$RAW_DIR"
mkdir -p "$TRAIN_DIR"
mkdir -p "$VAL_DIR"

print_success "目录已创建"

# 3. 复制数据文件
print_step "3/5" "复制数据文件..."

cp -r "$SOURCE_DIR"/* "$RAW_DIR/" 2>/dev/null || true

COPIED_COUNT=$(find "$RAW_DIR" -type f | wc -l)
print_success "已复制 $COPIED_COUNT 个文件到 $RAW_DIR"

# 4. 准备训练数据 (使用 Python)
print_step "4/5" "生成训练数据..."

cat > /tmp/prepare_data_general.py << PYTHON_EOF
#!/usr/bin/env python3
import os
import sys
import json
import re
from pathlib import Path

# 配置
RAW_DIR = "./data/raw_${OUTPUT_NAME}"
TRAIN_DIR = "./data/train_${OUTPUT_NAME}"
VAL_DIR = "./data/val_${OUTPUT_NAME}"

CHUNK_SIZE = ${CHUNK_SIZE}
VAL_SPLIT = ${VAL_SPLIT}
MIN_LENGTH = ${MIN_LENGTH}
OVERLAP = 200

def clean_text(text: str) -> str:
    """清理文本"""
    # 移除特殊字符
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # 标准化空白
    text = re.sub(r'\s+', ' ', text)
    # 移除过多的换行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """将文本分割成训练块"""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        # 尝试在句子边界分割
        if end < text_len:
            last_period = chunk.rfind('。')
            last_newline = chunk.rfind('\n')
            split_pos = max(last_period, last_newline)

            if split_pos > chunk_size * 0.7:  # 至少保留70%
                chunk = text[start:start + split_pos + 1]
                end = start + split_pos + 1

        chunks.append(chunk.strip())
        start = end - overlap if end < text_len else end

    return [c for c in chunks if len(c) >= MIN_LENGTH]

def load_txt_file(file_path: Path) -> dict:
    """加载 TXT 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = clean_text(content)

    return {
        "title": file_path.stem,
        "content": content,
        "source": str(file_path)
    }

def main():
    raw_dir = Path(RAW_DIR)
    train_dir = Path(TRAIN_DIR)
    val_dir = Path(VAL_DIR)

    print(f"📁 扫描目录: {raw_dir}")

    # 查找所有文本文件
    txt_files = list(raw_dir.glob("*.txt"))
    json_files = list(raw_dir.glob("*.json"))
    jsonl_files = list(raw_dir.glob("*.jsonl"))

    all_files = txt_files + json_files + jsonl_files

    if not all_files:
        print("❌ 未找到任何数据文件!")
        return

    print(f"✓ 找到 {len(all_files)} 个文件")

    # 加载所有数据
    all_novels = []
    all_chunks = []

    for file_path in all_files:
        try:
            if file_path.suffix == '.txt':
                novel = load_txt_file(file_path)
                all_novels.append(novel)

                # 分块
                chunks = split_into_chunks(novel['content'])
                for chunk in chunks:
                    all_chunks.append({
                        "text": chunk,
                        "title": novel['title'],
                        "source": novel['source']
                    })

                print(f"  ✓ {file_path.name} -> {len(chunks)} 块")

            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 处理 JSON 数据...
                    print(f"  ⚠ JSON 文件: {file_path.name} (需手动处理)")

            elif file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            text = data.get('text', data.get('content', ''))
                            if text:
                                chunks = split_into_chunks(clean_text(text))
                                for chunk in chunks:
                                    all_chunks.append({
                                        "text": chunk,
                                        "title": data.get('title', file_path.stem),
                                        "source": str(file_path)
                                    })
                    print(f"  ✓ {file_path.name} -> JSONL 处理完成")

        except Exception as e:
            print(f"  ✗ {file_path.name}: {e}")

    if not all_chunks:
        print("❌ 没有生成任何训练块!")
        return

    print(f"\n📊 共生成 {len(all_chunks)} 个训练块")

    # 划分训练集和验证集
    import random
    random.shuffle(all_chunks)

    val_size = int(len(all_chunks) * VAL_SPLIT)
    train_chunks = all_chunks[val_size:]
    val_chunks = all_chunks[:val_size]

    print(f"  训练集: {len(train_chunks)} 条")
    print(f"  验证集: {len(val_chunks)} 条")

    # 保存数据
    train_file = train_dir / "train.jsonl"
    val_file = val_dir / "val.jsonl"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 保存数据...")

    with open(train_file, 'w', encoding='utf-8') as f:
        for chunk in train_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    with open(val_file, 'w', encoding='utf-8') as f:
        for chunk in val_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    print(f"✓ 训练集: {train_file}")
    print(f"✓ 验证集: {val_file}")

    # 显示数据统计
    print(f"\n📈 数据统计:")
    total_chars = sum(len(c['text']) for c in all_chunks)
    avg_chars = total_chars // len(all_chunks) if all_chunks else 0
    print(f"  总字符数: {total_chars:,}")
    print(f"  平均块大小: {avg_chars:,} 字符")

if __name__ == "__main__":
    main()
PYTHON_EOF

python3 /tmp/prepare_data_general.py

# 5. 更新配置
print_step "5/5" "生成配置信息..."

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  数据准备完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${CYAN}数据位置:${NC}"
echo -e "  原始数据: $RAW_DIR"
echo -e "  训练集:   $TRAIN_DIR/train.jsonl"
echo -e "  验证集:   $VAL_DIR/val.jsonl"
echo -e "\n${YELLOW}开始训练:${NC}"
echo -e "  python start.py train \\"
echo -e "    --train-data $TRAIN_DIR/train.jsonl \\"
echo -e "    --val-data $VAL_DIR/val.jsonl \\"
echo -e "    --output-dir ./checkpoints/${OUTPUT_NAME}_model"
