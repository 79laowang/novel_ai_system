#!/usr/bin/env python3
"""
HuggingFace 模型下载工具
使用官方 huggingface_hub + 镜像源，企业级稳定方案

用法:
    # 下载基础模型
    python scripts/download_hf_model.py Qwen/Qwen2.5-7B-Instruct

    # 下载到指定目录
    python scripts/download_hf_model.py sentence-transformers/all-MiniLM-L6-v2 --local-dir ./models/embedding

    # 使用镜像源
    python scripts/download_hf_model.py Qwen/Qwen2.5-7B-Instruct --endpoint https://hf-mirror.com

特点:
    ✔️ 不走 git-lfs
    ✔️ 不走 xethub
    ✔️ 完全可控
    ✔️ 支持断点续传
    ✔️ 企业 CI 友好
"""
import os
import sys
import argparse
from pathlib import Path


def download_model(
    repo_id: str,
    local_dir: str = None,
    endpoint: str = "https://hf-mirror.com",
    local_dir_use_symlinks: bool = False,
    resume: bool = True,
):
    """
    下载 HuggingFace 模型

    Args:
        repo_id: 模型 ID，如 "Qwen/Qwen2.5-7B-Instruct"
        local_dir: 本地保存目录
        endpoint: 镜像端点
        local_dir_use_symlinks: 是否使用符号链接
        resume: 是否断点续传
    """
    # 设置镜像端点
    os.environ["HF_ENDPOINT"] = endpoint

    from huggingface_hub import snapshot_download

    # 确定本地目录
    if local_dir is None:
        # 使用 models/ 目录，按模型名称组织
        model_name = repo_id.replace("/", "--")
        local_dir = f"./models/{model_name}"

    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"📦 下载模型: {repo_id}")
    print(f"📁 保存到: {local_path.absolute()}")
    print(f"🔗 镜像源: {endpoint}")
    print()

    try:
        # 下载模型
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_path),
            local_dir_use_symlinks=local_dir_use_symlinks,
            resume_download=resume,
        )

        print()
        print("✓ 下载完成!")
        print(f"📊 模型大小: {sum(f.stat().st_size for f in local_path.rglob('*') if f.is_file()) / 1024**3:.2f} GB")
        print()
        print("使用方法:")
        print(f"  Python: from transformers import AutoModel; AutoModel.from_pretrained('{local_path.absolute()}')")
        print(f"  或:    AutoModel.from_pretrained('{repo_id}')  # 会使用本地缓存")

    except Exception as e:
        print(f"✗ 下载失败: {e}")
        sys.exit(1)


def download_embedding_model(local_dir: str = "./models/embeddings"):
    """
    下载项目使用的 embedding 模型 (BAAI/bge-m3)
    """
    print("📥 下载 Embedding 模型 (BAAI/bge-m3)")
    print()

    download_model(
        repo_id="BAAI/bge-m3",
        local_dir=local_dir,
        endpoint="https://hf-mirror.com",
    )

    print()
    print("更新 config.py:")
    print(f'  memory_config.embedding_model = "{local_dir}"')


def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace 模型下载工具 (企业级稳定方案)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载基础模型
  %(prog)s Qwen/Qwen2.5-7B-Instruct

  # 下载到指定目录
  %(prog)s sentence-transformers/all-MiniLM-L6-v2 --local-dir ./models/embedding

  # 下载项目 embedding 模型
  %(prog)s --embedding

  # 使用不同镜像源
  %(prog)s Qwen/Qwen2.5-7B-Instruct --endpoint https://huggingface.co
        """,
    )

    parser.add_argument(
        "repo_id",
        nargs="?",
        help="模型 ID (如: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="本地保存目录 (默认: ./models/<model-name>)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="镜像端点 (默认: https://hf-mirror.com)",
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="下载项目 embedding 模型 (BAAI/bge-m3)",
    )
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        help="不使用符号链接",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不使用断点续传",
    )

    args = parser.parse_args()

    # 下载 embedding 模型
    if args.embedding:
        download_embedding_model(args.local_dir or "./models/embeddings")
        return

    # 需要 repo_id
    if not args.repo_id:
        parser.error("需要指定 repo_id 或使用 --embedding")

    # 下载指定模型
    download_model(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        endpoint=args.endpoint,
        local_dir_use_symlinks=not args.no_symlinks,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
