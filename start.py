#!/usr/bin/env python3
"""
中文小说写作AI系统 - 主启动入口

支持模式:
1. webui - 启动Web界面 (默认)
2. train - 启动训练
3. prepare - 准备训练数据
4. inference - 命令行推理测试
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

    lora_path = args.lora if hasattr(args, 'lora') and args.lora else None

    print("=" * 60)
    print("🚀 启动中文小说写作AI系统 - WebUI")
    print("=" * 60)

    asyncio.run(launch_webui(lora_path=lora_path))


def launch_train(args):
    """启动训练"""
    from src.train.train_lora import main as train_main

    print("=" * 60)
    print("🎯 启动模型微调训练")
    print("=" * 60)

    resume_from = getattr(args, 'resume', None)
    train_main(resume_from_checkpoint=resume_from)


def prepare_data(args):
    """准备数据"""
    from src.data.prepare_data import NovelDataPreparer

    print("=" * 60)
    print("📚 准备训练数据")
    print("=" * 60)

    preparer = NovelDataPreparer(args.data_dir or "./data")

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
    import asyncio
    from src.inference.vllm_server import main as inference_main

    print("=" * 60)
    print("🤖 运行推理测试")
    print("=" * 60)

    asyncio.run(inference_main())


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="中文小说写作AI系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 启动WebUI
  python start.py

  # 启动WebUI (带LoRA权重)
  python start.py webui --lora ./checkpoints/final_model

  # 准备训练数据
  python start.py prepare --sample

  # 开始训练
  python start.py train

  # 推理测试
  python start.py inference
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    # WebUI模式
    webui_parser = subparsers.add_parser("webui", help="启动Web界面")
    webui_parser.add_argument("--lora", type=str, default=None, help="LoRA权重路径")
    webui_parser.add_argument("--host", type=str, default=None, help="服务器地址")
    webui_parser.add_argument("--port", type=int, default=None, help="服务器端口")
    webui_parser.add_argument("--share", action="store_true", help="创建公共链接")

    # 训练模式
    train_parser = subparsers.add_parser("train", help="启动模型训练")
    train_parser.add_argument("--data", type=str, default=None, help="训练数据路径")
    train_parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    train_parser.add_argument("--batch-size", type=int, default=None, help="批次大小")
    train_parser.add_argument("--lr", type=float, default=None, help="学习率")
    train_parser.add_argument("--resume", type=str, default=None, help="从checkpoint恢复训练")

    # 数据准备模式
    prepare_parser = subparsers.add_parser("prepare", help="准备训练数据")
    prepare_parser.add_argument("--data-dir", type=str, default="./data", help="数据目录")
    prepare_parser.add_argument("--chunk-size", type=int, default=2048, help="训练块大小")
    prepare_parser.add_argument("--val-split", type=float, default=0.1, help="验证集比例")
    prepare_parser.add_argument("--min-length", type=int, default=500, help="最小文本长度")
    prepare_parser.add_argument("--sample", action="store_true", help="创建示例数据")

    # 推理模式
    inference_parser = subparsers.add_parser("inference", help="运行推理测试")

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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
