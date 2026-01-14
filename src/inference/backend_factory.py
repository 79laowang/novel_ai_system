"""
推理后端工厂 - 根据配置选择推理引擎
支持 vLLM (GPU) 和 llama.cpp (CPU)
根据硬件环境自动选择最佳后端
"""
from typing import Optional, Union
from pathlib import Path
import logging

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


def detect_hardware_backend() -> str:
    """
    根据硬件环境自动选择推理后端

    硬件检测逻辑:
    1. 检测是否有 CUDA GPU → 使用 vLLM
    2. 无 GPU → 使用 llama.cpp (CPU)

    Returns:
        推荐的后端名称: "vllm" 或 "llama_cpp"
    """
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"检测到 GPU: {gpu_name} ({gpu_count} 个设备)")
            logger.info("✓ 自动选择 vLLM 后端 (GPU 加速)")
            return "vllm"
        else:
            logger.info("未检测到 GPU，使用 llama.cpp 后端 (CPU 推理)")
            return "llama_cpp"
    except ImportError:
        logger.warning("PyTorch 未安装，默认使用 llama.cpp (CPU 推理)")
        return "llama_cpp"


def create_generator(config):
    """
    根据配置创建推理生成器

    Args:
        config: 系统配置对象

    Returns:
        推理生成器实例 (VLLMNovelGenerator 或 LlamaCppNovelGenerator)
    """
    backend = config.model.inference_backend.lower()

    # 自动检测硬件环境
    if backend == "auto":
        backend = detect_hardware_backend()

    if backend == "vllm":
        from src.inference.vllm_server import VLLMNovelGenerator
        return VLLMNovelGenerator(config)

    elif backend in ("llama_cpp", "llama-cpp", "llamacpp"):
        from src.inference.llama_server import LlamaCppNovelGenerator
        return LlamaCppNovelGenerator(config)

    else:
        raise ValueError(
            f"不支持的推理后端: {backend}\n"
            f"支持的后端: 'auto', 'vllm', 'llama_cpp'"
        )


async def get_generator(lora_path: Optional[str] = None):
    """
    获取已初始化的生成器单例

    Args:
        lora_path: LoRA 权重路径（可选）

    Returns:
        已初始化的推理生成器
    """
    from config import config

    generator = create_generator(config)

    # 初始化生成器
    if hasattr(generator, 'initialize'):
        # 检查是异步还是同步
        import inspect
        init_method = generator.initialize

        if inspect.iscoroutinefunction(init_method):
            await init_method(lora_path)
        else:
            init_method(lora_path)

    return generator


def get_generator_sync(lora_path: Optional[str] = None):
    """
    同步获取已初始化的生成器单例

    Args:
        lora_path: LoRA 权重路径（可选）

    Returns:
        已初始化的推理生成器
    """
    from config import config

    generator = create_generator(config)

    # 初始化生成器
    if hasattr(generator, 'initialize'):
        generator.initialize(lora_path)

    return generator
