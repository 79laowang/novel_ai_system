#!/usr/bin/env python3
"""
直接测试vLLM推理功能
不经过Gradio，直接调用底层推理
"""
import sys
import asyncio
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from rich.console import Console
from datetime import datetime
import uuid

console = Console()


async def test_direct_inference():
    """直接测试vLLM推理"""

    log_file = '/home/kewang/work/novel_ai_system/logs/debug.log'
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[{datetime.now()}] 开始直接推理测试\n")
        f.write(f"{'='*60}\n")

    console.print("[bold cyan]正在初始化vLLM引擎...[/bold cyan]")

    # 创建引擎
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=32768,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=True,
        block_size=16,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    console.print("[green]✓ vLLM引擎初始化完成[/green]")

    # 测试prompt
    prompt = """<|im_start|>system
你是一位专业的小说作家，擅长创作各种类型的中文小说。
你的写作风格优美流畅，人物刻画生动，情节引人入胜。
请根据用户的要求进行创作，保持故事的连贯性和逻辑性。<|im_end|>
<|im_start|>user
写一段关于年轻剑客在雪山练剑的小说开头<|im_end|>
<|im_start|>assistant
"""

    console.print(f"\n[bold]测试Prompt:[/bold]\n{prompt[:200]}...")

    # 创建采样参数
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        stop=["<|im_end|>"],
    )

    # 使用唯一request_id
    request_id = f"test_{uuid.uuid4().hex[:8]}"

    with open(log_file, 'a') as f:
        f.write(f"[{datetime.now()}] 开始推理，request_id: {request_id}\n")
        f.write(f"[{datetime.now()}] prompt长度: {len(prompt)}\n")
        f.write(f"[{datetime.now()}] 采样参数: max_tokens={sampling_params.max_tokens}, temp={sampling_params.temperature}\n")

    console.print(f"\n[bold yellow]开始生成... (request_id: {request_id})[/bold yellow]")

    # 生成文本
    outputs = []
    chunk_count = 0
    start_time = datetime.now()

    async for request_output in engine.generate(
        prompt,
        sampling_params,
        request_id=request_id,
    ):
        chunk_count += 1
        text = request_output.outputs[0].text

        with open(log_file, 'a') as f:
            f.write(f"[{datetime.now()}] 接收块 #{chunk_count}, 长度: {len(text)}\n")

        console.print(f"[dim]块 #{chunk_count}: {len(text)} 字符[/dim]")

        # 每5块显示一次预览
        if chunk_count % 5 == 0 and text:
            preview = text[-100:] if len(text) > 100 else text
            console.print(f"[dim]预览: ...{preview}[/dim]")

        outputs.append(text)

    end_time = datetime.now()
    result = "".join(outputs)
    duration = (end_time - start_time).total_seconds()

    with open(log_file, 'a') as f:
        f.write(f"[{datetime.now()}] 推理完成\n")
        f.write(f"[{datetime.now()}] 总块数: {chunk_count}\n")
        f.write(f"[{datetime.now()}] 总长度: {len(result)} 字符\n")
        f.write(f"[{datetime.now()}] 耗时: {duration:.2f} 秒\n")
        f.write(f"[{datetime.now()}] 生成内容:\n{result}\n")
        f.write(f"{'='*60}\n")

    # 显示结果
    console.print(f"\n[bold green]✓ 生成完成![/bold green]")
    console.print(f"[dim]耗时: {duration:.2f} 秒 | {chunk_count} 块 | {len(result)} 字符[/dim]")
    console.print(f"\n[bold]生成结果:[/bold]")
    console.print(result)

    return result


if __name__ == "__main__":
    try:
        result = asyncio.run(test_direct_inference())
        console.print("\n[bold green]✓ 测试成功![/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]✗ 测试失败: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)
