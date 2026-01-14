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

        # 根据配置选择模型格式
        model_format = self.config.model.llama_cpp_model_format.lower()
        model_path = None

        if model_format == "gguf":
            # 使用 GGUF 量化模型
            model_path = Path(self.config.model.llama_cpp_gguf_model)
            rprint(f"[cyan]模型格式: GGUF (量化)[/cyan]")
            if not model_path.exists():
                rprint(f"[red]错误: GGUF 模型文件不存在: {model_path}[/red]")
                rprint("[yellow]请先运行转换脚本将 Hugging Face 模型转换为 GGUF 格式[/yellow]")
                rprint(f"[yellow]命令: python start.py convert hf-to-gguf --model {self.config.model.base_model}[/yellow]")
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

        elif model_format == "hf":
            # 使用 Hugging Face 非量化模型
            model_name = self.config.model.llama_cpp_hf_model
            rprint(f"[cyan]模型格式: Hugging Face (非量化)[/cyan]")
            rprint(f"[cyan]模型: {model_name}[/cyan]")
            # llama.cpp 支持直接从 Hugging Face 加载模型
            model_path = model_name

        else:
            rprint(f"[red]错误: 不支持的模型格式: {model_format}[/red]")
            rprint("[yellow]支持的格式: 'gguf' (量化), 'hf' (非量化)[/yellow]")
            raise ValueError(f"不支持的模型格式: {model_format}")

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

        # 准备初始化参数
        llama_kwargs = {
            "model_path": str(model_path),
            "n_ctx": self.config.model.llama_cpp_n_ctx,
            "n_threads": self.config.model.llama_cpp_n_threads,
            "n_batch": self.config.model.llama_cpp_n_batch,
            "use_mmap": self.config.model.llama_cpp_use_mmap,
            "use_mlock": self.config.model.llama_cpp_use_mlock,
            "n_gpu_layers": self.config.model.llama_cpp_gpu_layers,
            "verbose": False,
        }

        # 添加 LoRA 参数（如果存在）
        if self.lora_path:
            llama_kwargs["lora_path"] = self.lora_path
            rprint(f"[cyan]将加载 LoRA: {self.lora_path}[/cyan]")

        self.model = Llama(**llama_kwargs)

        # 确认 LoRA 已加载
        if self.lora_path:
            rprint(f"[green]✓ LoRA 已加载: {self.lora_path}[/green]")

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

【严格的格式要求 - 必须遵守】
1. 每段控制在2-4句话，绝对不能超过5句话
2. 每写完一段后必须换行
3. 对话必须单独成行，格式："对话内容"
4. 场景切换时必须空一行
5. 禁止出现超过100字的连续段落

错误示范（不要这样）：
一大段连续文字...（超过5句话连在一起）

正确示范（必须这样）：
第一段内容。

第二段内容。

"人物对话"
角色说道。

第三段内容。"""

        # 添加上下文/记忆
        if memory:
            system_prompt += f"\n\n【之前的故事内容】\n{memory}\n"

        messages.append({"role": "system", "content": system_prompt})

        # 添加上下文
        if context:
            messages.append({"role": "user", "content": f"【故事背景】\n{context}"})
            messages.append({"role": "assistant", "content": "好的，我了解了故事背景，请继续。"})

        # 用户输入 - 强调分段要求
        formatted_input = f"""请严格按照分段格式创作以下内容：

{user_input}

【输出格式】
第一段。

第二段。

"对话"
角色说。

第三段。

记住：每段2-4句话，必须频繁换行！"""
        messages.append({"role": "user", "content": formatted_input})

        # 转换为 ChatML 格式 (Qwen2.5 使用)
        prompt = self._apply_chat_template(messages)

        return prompt

    def _post_process_paragraph(self, text: str) -> str:
        """后处理：自动分段

        策略：
        1. 按句子结束符（。！？）分割
        2. 每2-4句合并成一段
        3. 对话单独成行
        4. 段落间保留空行
        """
        import re

        if not text:
            return text

        # 清理多余的空白
        text = re.sub(r'\s+', '', text)  # 去除所有空白
        text = re.sub(r'([。！？])', r'\1\n', text)  # 在句号后添加换行
        text = re.sub(r'("[^"]*")', r'\n\1\n', text)  # 对话单独成行

        # 分割成句子数组
        sentences = [s.strip() for s in text.split('\n') if s.strip()]

        # 重新组织段落
        paragraphs = []
        current_para = []
        sentences_in_para = 0

        for sentence in sentences:
            # 检查是否是对话
            is_dialogue = sentence.startswith('"') or sentence.startswith('"') or sentence.startswith('"')

            if is_dialogue:
                # 对话单独成行
                if current_para:
                    paragraphs.append(''.join(current_para))
                    current_para = []
                    sentences_in_para = 0
                paragraphs.append(sentence)
            else:
                # 普通句子
                current_para.append(sentence)
                sentences_in_para += 1

                # 每2-4句话成一段
                if sentences_in_para >= 3:  # 平均3句一段
                    paragraphs.append(''.join(current_para))
                    current_para = []
                    sentences_in_para = 0

        # 处理剩余内容
        if current_para:
            paragraphs.append(''.join(current_para))

        # 合并段落，段落间保留空行
        result = '\n\n'.join(paragraphs)

        return result

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
        """生成小说内容（带格式化和分段后处理）- 同步方法"""
        prompt = self.format_prompt(user_input, context, memory)
        result = self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        # 应用后处理：自动分段
        result = self._post_process_paragraph(result)
        return result

    def generate_novel_stream(
        self,
        user_input: str,
        context: Optional[str] = None,
        memory: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ):
        """流式生成小说内容"""
        prompt = self.format_prompt(user_input, context, memory)
        for chunk in self.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stream=True,
        ):
            yield chunk

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
