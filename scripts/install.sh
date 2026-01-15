#!/bin/bash
# 中文小说写作AI系统 - 自动安装脚本

set -e

echo "======================================"
echo "中文小说写作AI系统 - 安装脚本"
echo "======================================"
echo ""

# 检查Python版本
echo "检查 Python 版度..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✓ Python 版本: $python_version"
else
    echo "✗ Python 版本过低，需要 3.8+"
    exit 1
fi

# 检查CUDA
echo ""
echo "检查 CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ 检测到 NVIDIA GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ 未检测到 NVIDIA GPU，将使用 CPU (速度较慢)"
fi

# 创建虚拟环境 (可选)
read -p "是否创建虚拟环境? (y/n) [推荐]: " create_venv
if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ] || [ -z "$create_venv" ]; then
    echo ""
    echo "创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ 虚拟环境已激活"
fi

# 升级 pip
echo ""
echo "升级 pip..."
pip install --upgrade pip setuptools wheel

# 安装 PyTorch (根据 CUDA 版本)
echo ""
echo "安装 PyTorch..."
# 检测 CUDA 版本
cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1)
echo "检测到 CUDA $cuda_version"

# 根据版本选择安装方式
if [ -n "$cuda_version" ] && [ "$cuda_version" -ge 12 ]; then
    echo "使用 CUDA 12.x 安装源"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "使用默认安装源"
    pip install torch torchvision
fi

# 安装依赖
echo ""
echo "安装项目依赖..."
pip install -r requirements.txt

# 创建必要的目录
echo ""
echo "创建目录结构..."
mkdir -p data/raw data/train data/val data/chroma_db
mkdir -p checkpoints logs models
mkdir -p src/train src/inference src/memory src/webui src/data

# 修复 bashrc 错误 (可选)
echo ""
echo "检查 .bashrc..."
if grep -q ".claude_glm" ~/.bashrc 2>/dev/null; then
    echo "⚠ 检测到 .bashrc 中的错误，询问是否修复..."
    read -p "是否修复 .bashrc? (y/n): " fix_bashrc
    if [ "$fix_bashrc" = "y" ] || [ "$fix_bashrc" = "Y" ]; then
        sed -i '/\.claude_glm/d' ~/.bashrc
        echo "✓ .bashrc 已修复"
    fi
fi

# 创建示例数据
echo ""
read -p "是否创建示例数据? (y/n): " create_sample
if [ "$create_sample" = "y" ] || [ "$create_sample" = "Y" ]; then
    python3 start.py prepare --sample
fi

# 完成
echo ""
echo "======================================"
echo "✓ 安装完成！"
echo "======================================"
echo ""
echo "使用方法:"
echo "  1. 启动 WebUI:"
echo "     python3 start.py webui"
echo ""
echo "  2. 准备训练数据:"
echo "     python3 start.py prepare"
echo ""
echo "  3. 开始训练:"
echo "     python3 start.py train"
echo ""
echo "  4. 查看帮助:"
echo "     python3 start.py --help"
echo ""
echo "更多文档请查看 README.md"
echo ""
