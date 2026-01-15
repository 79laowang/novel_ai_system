"""
vLLM 高性能推理服务
支持LoRA权重加载和批量推理
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Generator, Union
import asyncio
import json

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.lora.request import LoRARequest
from rich.console import Console
from rich import print as rprint

console = Console()


class VLLMNovelGenerator:
    """基于vLLM的小说生成器"""

    def __init__(self, config):
        self.config = config
        self.engine: Optional[AsyncLLMEngine] = None
        self.lora_path: Optional[str] = None

    async def initialize(self, lora_path: Optional[str] = None):
        """初始化vLLM引擎"""
        rprint("[bold cyan]正在初始化vLLM引擎...[/bold cyan]")

        # 引擎配置
        engine_args = AsyncEngineArgs(
            model=self.config.model.base_model,
            tensor_parallel_size=self.config.model.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.config.model.vllm_gpu_memory_utilization,
            max_model_len=self.config.model.vllm_max_model_len,
            dtype=self.config.model.vllm_dtype,
            trust_remote_code=True,
            enable_lora=bool(lora_path),
            max_loras=1,
            max_lora_rank=self.config.model.lora_r,
            # 启用前缀缓存加速
            enable_prefix_caching=True,
            # 块大小
            block_size=16,
        )

        # 创建引擎
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # 加载LoRA权重
        if lora_path and Path(lora_path).exists():
            self.lora_path = lora_path
            rprint(f"[green]✓ LoRA权重将按需加载: {lora_path}[/green]")

        rprint("[bold green]✓ vLLM引擎初始化完成[/bold green]")

    def create_sampling_params(
        self,
        max_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ) -> SamplingParams:
        """创建采样参数"""
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop or ["<|im_end|>"],
        )

    def format_prompt(
        self,
        user_input: str,
        context: Optional[str] = None,
        memory: Optional[str] = None,
    ) -> str:
        """格式化prompt为Qwen对话格式"""
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

        # 转换为Qwen格式
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return prompt

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """生成文本"""
        sampling_params = self.create_sampling_params(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        # LoRA请求
        lora_request = None
        if self.lora_path:
            lora_request = LoRARequest(
                lora_name="novel_lora",
                lora_int_id=1,
                lora_path=self.lora_path,
            )

        # 生成
        if stream:
            return self._generate_stream(prompt, sampling_params, lora_request)
        else:
            return await self._generate(prompt, sampling_params, lora_request)

    async def _generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        lora_request: Optional[LoRARequest],
    ) -> str:
        """非流式生成 - 在引擎的事件循环中执行"""
        import uuid
        import asyncio
        from datetime import datetime
        import traceback
        import sys

        log_file = '/home/kewang/work/novel_ai_system/logs/debug.log'

        # 使用唯一的request_id避免冲突
        request_id = f"novel_gen_{uuid.uuid4().hex[:8]}"

        def log_write(msg):
            """线程安全的日志写入"""
            with open(log_file, 'a') as f:
                f.write(f"[{datetime.now()}] {msg}\n")
                f.flush()

        log_write(f"[vLLM._generate] 开始生成，prompt长度: {len(prompt)}, request_id: {request_id}")
        log_write(f"[vLLM._generate] 当前事件循环: {id(asyncio.get_running_loop())}")

        # 获取引擎的事件循环（从模块全局变量）
        try:
            from src.webui.app import _engine_event_loop
            if _engine_event_loop is None:
                log_write("[vLLM._generate] 警告: 引擎事件循环未记录，使用当前循环")
                engine_loop = asyncio.get_running_loop()
            else:
                engine_loop = _engine_event_loop
                log_write(f"[vLLM._generate] 引擎事件循环: {id(engine_loop)}")
        except:
            engine_loop = asyncio.get_running_loop()
            log_write(f"[vLLM._generate] 无法获取引擎事件循环，使用当前: {id(engine_loop)}")

        # 定义生成协程（将在引擎的事件循环中运行）
        async def generate_in_engine_loop():
            log_write(f"[EngineLoop] 开始执行，事件循环: {id(asyncio.get_running_loop())}")
            outputs = []
            chunk_count = 0
            previous_len = 0

            log_write(f"[EngineLoop] 调用 self.engine.generate()...")
            gen = self.engine.generate(
                prompt,
                sampling_params,
                request_id=request_id,
                lora_request=lora_request,
            )
            log_write(f"[EngineLoop] engine.generate() 返回，开始迭代...")

            async for request_output in gen:
                chunk_count += 1
                text = request_output.outputs[0].text
                new_text = text[previous_len:]
                outputs.append(new_text)
                previous_len = len(text)

                if chunk_count % 10 == 0 or chunk_count == 1:
                    log_write(f"[EngineLoop] 收到chunk #{chunk_count}, 新增: {len(new_text)}, 总: {len(text)}")

            result = "".join(outputs)
            log_write(f"[EngineLoop] 生成完成，共 {chunk_count} 块，长度: {len(result)}")
            return result

        try:
            # 检查是否在同一个事件循环中
            current_loop = asyncio.get_running_loop()
            if current_loop is engine_loop:
                log_write("[vLLM._generate] 在同一事件循环中，直接执行")
                return await generate_in_engine_loop()
            else:
                log_write(f"[vLLM._generate] 在不同事件循环中，使用 run_coroutine_threadsafe")
                # 在不同事件循环中，使用 run_coroutine_threadsafe
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(generate_in_engine_loop(), engine_loop)
                # 等待结果（同步）
                result = future.result(timeout=120)
                log_write(f"[vLLM._generate] run_coroutine_threadsafe 返回，长度: {len(result)}")
                return result

        except Exception as e:
            log_write(f"[vLLM._generate] 异常: {e}\n{traceback.format_exc()}")
            raise

    async def _generate_stream(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        lora_request: Optional[LoRARequest],
    ) -> Generator[str, None, None]:
        """流式生成"""
        import uuid
        request_id = f"novel_gen_stream_{uuid.uuid4().hex[:8]}"
        previous_text = ""
        async for request_output in self.engine.generate(
            prompt,
            sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        ):
            new_text = request_output.outputs[0].text[len(previous_text):]
            previous_text = request_output.outputs[0].text
            yield new_text

    async def generate_novel(
        self,
        user_input: str,
        context: Optional[str] = None,
        memory: Optional[str] = None,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """生成小说内容（带格式化）"""
        from datetime import datetime
        log_file = '/home/kewang/work/novel_ai_system/logs/debug.log'

        with open(log_file, 'a') as f:
            f.write(f"[{datetime.now()}] [generate_novel] 开始，用户输入长度: {len(user_input)}\n")
            f.write(f"[{datetime.now()}] [generate_novel] 上下文长度: {len(context) if context else 0}\n")

        try:
            prompt = self.format_prompt(user_input, context, memory)
            with open(log_file, 'a') as f:
                f.write(f"[{datetime.now()}] [generate_novel] 格式化后prompt长度: {len(prompt)}\n")
                f.write(f"[{datetime.now()}] [generate_novel] 调用generate方法...\n")

            result = await self.generate(prompt, max_tokens=max_tokens, **kwargs)

            with open(log_file, 'a') as f:
                f.write(f"[{datetime.now()}] [generate_novel] 完成，返回结果长度: {len(result)}\n")
            return result
        except Exception as e:
            import traceback
            with open(log_file, 'a') as f:
                f.write(f"[{datetime.now()}] [generate_novel] 错误: {e}\n")
                f.write(traceback.format_exc())
            raise

    def generate_novel_sync(
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
        """同步生成小说内容（用于Gradio）"""
        import asyncio
        from datetime import datetime

        log_file = '/home/kewang/work/novel_ai_system/logs/debug.log'
        with open(log_file, 'a') as f:
            f.write(f"[{datetime.now()}] [generate_novel_sync] 开始\n")

        try:
            # 获取或创建事件循环
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # 运行异步生成
            result = loop.run_until_complete(
                self.generate_novel(
                    user_input=user_input,
                    context=context,
                    memory=memory,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )
            )

            with open(log_file, 'a') as f:
                f.write(f"[{datetime.now()}] [generate_novel_sync] 完成，长度: {len(result)}\n")

            return result
        except Exception as e:
            with open(log_file, 'a') as f:
                f.write(f"[{datetime.now()}] [generate_novel_sync] 错误: {e}\n")
            import traceback
            with open(log_file, 'a') as f:
                f.write(traceback.format_exc())
            raise

    async def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 2048,
        **kwargs
    ) -> List[str]:
        """批量生成"""
        sampling_params = self.create_sampling_params(max_tokens=max_tokens, **kwargs)

        # 创建请求
        request_ids = [f"batch_{i}" for i in range(len(prompts))]

        # 并发生成
        tasks = []
        for req_id, prompt in zip(request_ids, prompts):
            task = self.engine.generate(
                prompt,
                sampling_params,
                request_id=req_id,
            )
            tasks.append(task)

        # 收集结果
        results = []
        async for task_outputs in asyncio.as_completed(tasks):
            output = await task_outputs
            results.append(output.outputs[0].text)

        return results

    @property
    def tokenizer(self):
        """获取tokenizer（懒加载）"""
        if not hasattr(self, '_tokenizer'):
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.base_model,
                trust_remote_code=True,
            )
        return self._tokenizer

    async def shutdown(self):
        """关闭引擎"""
        if self.engine:
            del self.engine
            self.engine = None
            rprint("[yellow]vLLM引擎已关闭[/yellow]")


# 全局单例
_generator: Optional[VLLMNovelGenerator] = None


async def get_generator(lora_path: Optional[str] = None) -> VLLMNovelGenerator:
    """获取生成器单例"""
    global _generator
    if _generator is None:
        from config import config
        _generator = VLLMNovelGenerator(config)
        await _generator.initialize(lora_path)
    return _generator


# CLI接口
async def main():
    """命令行测试"""
    from config import config

    generator = VLLMNovelGenerator(config)
    await generator.initialize(lora_path="./training/final_model")

    # 测试生成
    prompt = "请写一段武侠小说的开头，描述一位少年在雪山之巅练剑的场景。"
    result = await generator.generate_novel(prompt)

    console.print("\n[bold green]生成结果:[/bold green]")
    console.print(result)


if __name__ == "__main__":
    asyncio.run(main())
