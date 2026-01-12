"""
llama.cpp 推理服务 - 支持 GGUF 模型和 LoRA adapter
适用于 CPU 推理或低资源设备
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Generator, Union
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich import print as rprint

console = Console()


class LlamaCppNovelGenerator:
    """基于llama.cpp的小说生成器 (支持 CPU 推理)"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.lora_path = None
        self._tokenizer = None

    def initialize(self, lora_path: Optional[str] = None):
        """初始化llama.cpp模型"""
        rprint("[bold cyan]正在初始化llama.cpp引擎...[/bold cyan]")

        try:
            from llama_cpp import Llama
        except ImportError:
            rprint("[red]错误: 未安装 llama-cpp-python[/red]")
            rprint("[yellow]请运行: pip install llama-cpp-python[/yellow]")
            raise

        # 检查模型文件
        model_path = Path(self.config.model.llama_cpp_model_path)
        if not model_path.exists():
            rprint(f"[red]错误: 模型文件不存在: {model_path}[/red]")
            rprint("[yellow]请先运行转换脚本将 Hugging Face 模型转换为 GGUF 格式[/yellow]")
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 设置 LoRA 路径
        self.lora_path = lora_path or self.config.model.llama_cpp_lora_path

        # 检查 LoRA 文件（如果指定）
        if self.lora_path:
            lora_path_obj = Path(self.lora_path)
            if lora_path_obj.exists():
                rprint(f"[green]✓ LoRA 路径: {self.lora_path}[/green]")
            else:
                rprint(f"[yellow]警告: LoRA 路径不存在: {self.lora_path}[/yellow]")
                self.lora_path = None

        # 初始化模型
        rprint(f"[cyan]加载模型: {model_path}[/cyan]")
        start_time = datetime.now()

        self.model = Llama(
            model_path=str(model_path),
            n_ctx=self.config.model.llama_cpp_n_ctx,
            n_threads=self.config.model.llama_cpp_n_threads,
            n_batch=self.config.model.llama_cpp_n_batch,
            use_mmap=self.config.model.llama_cpp_use_mmap,
            use_mlock=self.config.model.llama_cpp_use_mlock,
            n_gpu_layers=self.config.model.llama_cpp_gpu_layers,
            verbose=False,
        )

        # 加载 LoRA（如果存在）
        if self.lora_path:
            # llama-cpp-python 会自动加载目录中的 GGUF LoRA 文件
            rprint(f"[cyan]加载 LoRA: {self.lora_path}[/cyan]")

        load_time = (datetime.now() - start_time).total_seconds()
        rprint(f"[bold green]✓ llama.cpp 引擎初始化完成 (耗时: {load_time:.1f}秒)[/bold green]")

    def format_prompt(
        self,
        user_input: str,
        context: Optional[str] = None,
        memory: Optional[str] = None,
    ) -> str:
        """格式化prompt为Qwen对话格式"""
        # Qwen2.5 使用 ChatML 格式
        messages = []

        # 系统提示
        system_prompt = """你是一位专业的小说作家，擅长创作各种类型的中文小说。
你的写作风格优美流畅，人物刻画生动，情节引人入胜。
请根据用户的要求进行创作，保持故事的连贯性和逻辑性。"""

        # 添加上下文/记忆
        if memory:
            system_prompt += f"\n\n【之前的故事内容】\n{memory}\n"

        messages.append({"role": "system", "content": system_prompt})

        # 添加上下文
        if context:
            messages.append({"role": "user", "content": f"【故事背景】\n{context}"})
            messages.append({"role": "assistant", "content": "好的，我了解了故事背景，请继续。"})

        # 用户输入
        messages.append({"role": "user", "content": user_input})

        # 转换为 ChatML 格式 (Qwen2.5 使用)
        prompt = self._apply_chat_template(messages)

        return prompt

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """手动应用 ChatML 模板"""
        prompt = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        # 添加 assistant 开始标记
        prompt += "<|im_start|>assistant\n"

        return prompt

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """生成文本"""
        # 调用 llama.cpp 生成
        if stream:
            return self._generate_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop=stop,
            )
        else:
            return self._generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop=stop,
            )

    def _generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stop: Optional[List[str]],
    ) -> str:
        """非流式生成"""
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repetition_penalty,
            stop=stop or ["<|im_end|>"],
            echo=False,
        )
        return output["choices"][0]["text"]

    def _generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stop: Optional[List[str]],
    ) -> Generator[str, None, None]:
        """流式生成"""
        for chunk in self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repetition_penalty,
            stop=stop or ["<|im_end|>"],
            echo=False,
            stream=True,
        ):
            yield chunk["choices"][0]["text"]

    def generate_novel(
        self,
        user_input: str,
        context: Optional[str] = None,
        memory: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> str:
        """生成小说内容（带格式化）- 同步方法"""
        prompt = self.format_prompt(user_input, context, memory)
        result = self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        return result

    async def generate_novel_async(
        self,
        user_input: str,
        context: Optional[str] = None,
        memory: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> str:
        """异步生成小说内容（兼容接口）"""
        # llama-cpp-python 是同步的，这里直接调用同步方法
        return self.generate_novel(
            user_input=user_input,
            context=context,
            memory=memory,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 2048,
        **kwargs
    ) -> List[str]:
        """批量生成"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_tokens=max_tokens, **kwargs)
            results.append(result)
        return results

    @property
    def tokenizer(self):
        """获取 tokenizer（兼容接口）"""
        # llama.cpp 不需要单独的 tokenizer
        return self

    def shutdown(self):
        """关闭模型"""
        if self.model:
            del self.model
            self.model = None
            rprint("[yellow]llama.cpp 引擎已关闭[/yellow]")


# 全局单例
_generator: Optional[LlamaCppNovelGenerator] = None


def get_generator(lora_path: Optional[str] = None) -> LlamaCppNovelGenerator:
    """获取生成器单例"""
    global _generator
    if _generator is None:
        from config import config
        _generator = LlamaCppNovelGenerator(config)
        _generator.initialize(lora_path)
    return _generator


# CLI 接口
def main():
    """命令行测试"""
    from config import config

    generator = LlamaCppNovelGenerator(config)
    generator.initialize(lora_path=None)

    # 测试生成
    prompt = "请写一段武侠小说的开头，描述一位少年在雪山之巅练剑的场景。"
    console.print("\n[bold cyan]生成中...[/bold cyan]")

    result = generator.generate_novel(prompt)

    console.print("\n[bold green]生成结果:[/bold green]")
    console.print(result)


if __name__ == "__main__":
    main()
