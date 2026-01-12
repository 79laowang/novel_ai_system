"""
推理后端工厂 - 根据配置选择推理引擎
支持 vLLM (GPU) 和 llama.cpp (CPU)
"""
from typing import Optional, Union
from pathlib import Path

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_generator(config):
    """
    根据配置创建推理生成器

    Args:
        config: 系统配置对象

    Returns:
        推理生成器实例 (VLLMNovelGenerator 或 LlamaCppNovelGenerator)
    """
    backend = config.model.inference_backend.lower()

    if backend == "vllm":
        from src.inference.vllm_server import VLLMNovelGenerator
        return VLLMNovelGenerator(config)

    elif backend in ("llama_cpp", "llama-cpp", "llamacpp"):
        from src.inference.llama_server import LlamaCppNovelGenerator
        return LlamaCppNovelGenerator(config)

    else:
        raise ValueError(
            f"不支持的推理后端: {backend}\n"
            f"支持的后端: 'vllm', 'llama_cpp'"
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
