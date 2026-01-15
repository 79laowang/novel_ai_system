#!/usr/bin/env python3
"""
中文小说写作AI系统 - 主启动入口

支持模式:
1. webui - 启动Web界面 (默认)
2. train - 启动训练
3. prepare - 准备训练数据
4. inference - 命令行推理测试
5. convert - 模型格式转换 (HF → GGUF, LoRA → GGUF)
"""
import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def launch_webui(args):
    """启动WebUI"""
    import asyncio
    from src.webui.app import launch_webui
    from config import config

    lora_path = args.lora if hasattr(args, 'lora') and args.lora else None

    # 应用命令行指定的基础模型
    if hasattr(args, 'base_model') and args.base_model:
        config.model.base_model = args.base_model

    # 应用 llama.cpp 模型格式配置
    if hasattr(args, 'model_format') and args.model_format:
        config.model.llama_cpp_model_format = args.model_format
        # 同时更新 HF 模型路径以匹配
        if args.model_format == "hf":
            config.model.llama_cpp_hf_model = args.base_model or config.model.base_model

    print("=" * 60)
    print("🚀 启动中文小说写作AI系统 - WebUI")
    print("=" * 60)
    print(f"📦 基础模型: {config.model.base_model}")

    asyncio.run(launch_webui(lora_path=lora_path))


def launch_train(args):
    """启动训练"""
    from src.train.train_lora import main as train_main
    from config import config

    # 应用命令行指定的基础模型
    if hasattr(args, 'base_model') and args.base_model:
        config.model.base_model = args.base_model

    print("=" * 60)
    print("🎯 启动模型微调训练")
    print("=" * 60)
    print(f"📦 基础模型: {config.model.base_model}")

    # 收集训练参数
    train_kwargs = {}
    if hasattr(args, 'resume') and args.resume:
        train_kwargs['resume_from_checkpoint'] = args.resume
    if hasattr(args, 'train_data') and args.train_data:
        train_kwargs['train_data_path'] = args.train_data
    if hasattr(args, 'val_data') and args.val_data:
        train_kwargs['val_data_path'] = args.val_data
    if hasattr(args, 'output_dir') and args.output_dir:
        train_kwargs['output_dir'] = args.output_dir
    if hasattr(args, 'checkpoint_dir') and args.checkpoint_dir:
        train_kwargs['checkpoint_dir'] = args.checkpoint_dir
    if hasattr(args, 'epochs') and args.epochs:
        train_kwargs['num_train_epochs'] = args.epochs
    if hasattr(args, 'batch_size') and args.batch_size:
        train_kwargs['per_device_train_batch_size'] = args.batch_size
    if hasattr(args, 'lr') and args.lr:
        train_kwargs['learning_rate'] = args.lr

    train_main(**train_kwargs)


def prepare_data(args):
    """准备数据"""
    from src.data.prepare_data import NovelDataPreparer

    print("=" * 60)
    print("📚 准备训练数据")
    print("=" * 60)

    preparer = NovelDataPreparer(args.data_dir or "./training/data")

    if args.sample:
        preparer.create_sample_data()
    else:
        stats = preparer.prepare_training_data(
            chunk_size=args.chunk_size,
            val_split=args.val_split,
            min_length=args.min_length,
        )
        print(f"\n✓ 完成! 训练集: {stats['train']} | 验证集: {stats['val']}")


def run_inference(args):
    """运行推理测试"""
    from config import config
    backend = config.model.inference_backend

    print("=" * 60)
    print("🤖 运行推理测试")
    print(f"📊 推理后端: {backend}")
    print("=" * 60)

    if backend == "vllm":
        import asyncio
        from src.inference.vllm_server import main as inference_main
        asyncio.run(inference_main())
    elif backend == "llama_cpp":
        from src.inference.llama_server import main as inference_main
        inference_main()
    else:
        print(f"❌ 不支持的推理后端: {backend}")
        print("支持的后端: vllm, llama_cpp")
        sys.exit(1)


def convert_model(args):
    """模型格式转换"""
    import subprocess
    from pathlib import Path

    script_dir = Path(__file__).parent / "scripts"

    print("=" * 60)
    print("🔄 模型格式转换")
    print("=" * 60)

    if args.convert_type == "hf-to-gguf":
        # Hugging Face → GGUF
        script = script_dir / "convert_hf_to_gguf.sh"
        model = args.model or "Qwen/Qwen2.5-7B-Instruct"
        quant = args.quant or "Q5_K_M"

        print(f"📦 转换: {model}")
        print(f"📊 量化类型: {quant}")
        print()

        subprocess.run([str(script), model, quant], check=True)

    elif args.convert_type == "lora-to-gguf":
        # LoRA → GGUF
        script = script_dir / "convert_lora_to_gguf.sh"
        base_model = args.base_model or "Qwen/Qwen2.5-7B-Instruct"
        lora_path = args.lora_path or "./training/final_model"
        output_dir = args.output_dir or "./models/lora-gguf"

        print(f"📦 基础模型: {base_model}")
        print(f"📦 LoRA 路径: {lora_path}")
        print(f"📁 输出目录: {output_dir}")
        print()

        subprocess.run([str(script), base_model, lora_path, output_dir], check=True)

    else:
        print(f"❌ 不支持的转换类型: {args.convert_type}")
        print("支持的类型: hf-to-gguf, lora-to-gguf")
        sys.exit(1)


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="中文小说写作AI系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 启动WebUI (默认模型)
  python start.py

  # 启动WebUI (指定Qwen基础模型)
  python start.py webui --base-model Qwen/Qwen2.5-7B-Instruct

  # 其他Qwen模型选项:
  # Qwen2.5系列: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
  python start.py webui --base-model Qwen/Qwen2.5-3B-Instruct
  python start.py webui --base-model Qwen/Qwen2.5-14B-Instruct
  python start.py webui --base-model Qwen/Qwen2.5-32B-Instruct
  python start.py webui --base-model Qwen/Qwen2.5-72B-Instruct

  # Qwen2系列
  python start.py webui --base-model Qwen/Qwen2-7B-Instruct
  python start.py webui --base-model Qwen/Qwen2-72B-Instruct

  # 启动WebUI (指定基础模型 + LoRA权重)
  python start.py webui --base-model Qwen/Qwen2.5-7B-Instruct --lora ./training/final_model

  # 准备训练数据
  python start.py prepare --sample

  # 开始训练 (默认模型)
  python start.py train

  # 开始训练 (指定基础模型)
  python start.py train --base-model Qwen/Qwen2.5-7B-Instruct

  # 开始训练 (指定数据和输出目录)
  # 示例: 使用 urban-novels 数据集
  # python start.py train \\
  #   --base-model Qwen/Qwen2.5-7B-Instruct \\
  #   --train-data ./data/train_urban-novels/train.jsonl \\
  #   --val-data ./data/val_urban-novels/val.jsonl \\
  #   --output-dir ./training/urban-novels_model \\
  #   --epochs 3

  # 推理测试
  python start.py inference

Qwen模型系列 (推荐):
  - Qwen/Qwen2.5-0.5B-Instruct   (最小, ~1GB显存)
  - Qwen/Qwen2.5-1.5B-Instruct   (小型, ~3GB显存)
  - Qwen/Qwen2.5-3B-Instruct    (中型, ~6GB显存)
  - Qwen/Qwen2.5-7B-Instruct    (推荐, ~14GB显存)
  - Qwen/Qwen2.5-14B-Instruct   (大型, ~28GB显存)
  - Qwen/Qwen2.5-32B-Instruct   (超大型, ~64GB显存)
  - Qwen/Qwen2.5-72B-Instruct   (最大, ~128GB显存)

CPU 推理 (llama.cpp) 模型格式:
  --model-format gguf  使用 GGUF 量化模型 (推荐, 内存占用小)
  --model-format hf    使用 Hugging Face 非量化模型 (精度高, 内存占用大)

  示例:
  # CPU 推理 - 使用量化模型 (默认)
  python start.py webui --model-format gguf

  # CPU 推理 - 使用非量化模型 (精度更高)
  python start.py webui --model-format hf --base-model Qwen/Qwen2.5-3B-Instruct
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    # WebUI模式
    webui_parser = subparsers.add_parser("webui", help="启动Web界面")
    webui_parser.add_argument("--base-model", type=str, default=None, help="基础模型名称 (如: Qwen/Qwen2.5-7B-Instruct)")
    webui_parser.add_argument("--model-format", type=str, default=None, choices=["gguf", "hf"], help="CPU推理模型格式: gguf(量化) 或 hf(非量化)")
    webui_parser.add_argument("--lora", type=str, default=None, help="LoRA权重路径")
    webui_parser.add_argument("--host", type=str, default=None, help="服务器地址")
    webui_parser.add_argument("--port", type=int, default=None, help="服务器端口")
    webui_parser.add_argument("--share", action="store_true", help="创建公共链接")

    # 训练模式
    train_parser = subparsers.add_parser("train", help="启动模型训练")
    train_parser.add_argument("--base-model", type=str, default=None, help="基础模型名称 (如: Qwen/Qwen2.5-7B-Instruct)")
    train_parser.add_argument("--data", type=str, default=None, help="训练数据路径")
    train_parser.add_argument("--train-data", type=str, default=None, help="训练数据路径 (JSONL)")
    train_parser.add_argument("--val-data", type=str, default=None, help="验证数据路径 (JSONL)")
    train_parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    train_parser.add_argument("--batch-size", type=int, default=None, help="批次大小")
    train_parser.add_argument("--lr", type=float, default=None, help="学习率")
    train_parser.add_argument("--resume", type=str, default=None, help="从checkpoint恢复训练")
    train_parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    train_parser.add_argument("--checkpoint-dir", type=str, default=None, help="检查点目录")

    # 数据准备模式
    prepare_parser = subparsers.add_parser("prepare", help="准备训练数据")
    prepare_parser.add_argument("--data-dir", type=str, default="./training/data", help="数据目录")
    prepare_parser.add_argument("--chunk-size", type=int, default=2048, help="训练块大小")
    prepare_parser.add_argument("--val-split", type=float, default=0.1, help="验证集比例")
    prepare_parser.add_argument("--min-length", type=int, default=500, help="最小文本长度")
    prepare_parser.add_argument("--sample", action="store_true", help="创建示例数据")

    # 推理模式
    inference_parser = subparsers.add_parser("inference", help="运行推理测试")

    # 转换模式
    convert_parser = subparsers.add_parser("convert", help="模型格式转换")
    convert_subparsers = convert_parser.add_subparsers(dest="convert_type", help="转换类型")

    # HF → GGUF 转换
    hf_gguf_parser = convert_subparsers.add_parser("hf-to-gguf", help="Hugging Face 模型转换为 GGUF 格式")
    hf_gguf_parser.add_argument("--model", type=str, default=None, help="Hugging Face 模型名称")
    hf_gguf_parser.add_argument("--quant", type=str, default=None, help="量化类型 (Q5_K_M, Q8_0, etc.)")

    # LoRA → GGUF 转换
    lora_gguf_parser = convert_subparsers.add_parser("lora-to-gguf", help="LoRA 权重转换为 GGUF 格式")
    lora_gguf_parser.add_argument("--base-model", type=str, default=None, help="基础模型名称")
    lora_gguf_parser.add_argument("--lora-path", type=str, default=None, help="LoRA 权重路径")
    lora_gguf_parser.add_argument("--output-dir", type=str, default=None, help="输出目录")

    args = parser.parse_args()

    # 默认启动WebUI
    if args.mode is None:
        args.mode = "webui"

    # 根据模式启动
    if args.mode == "webui":
        launch_webui(args)
    elif args.mode == "train":
        launch_train(args)
    elif args.mode == "prepare":
        prepare_data(args)
    elif args.mode == "inference":
        run_inference(args)
    elif args.mode == "convert":
        convert_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
