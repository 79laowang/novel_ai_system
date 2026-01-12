"""
Gradio WebUI - 中文小说写作AI系统交互界面
支持实时生成、记忆管理、参数调整
"""
import os
import sys
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gradio as gr
from rich.console import Console
from rich import print as rprint

# 导入本地模块
from src.inference.backend_factory import create_generator, get_generator, get_generator_sync
from src.memory.memory_manager import NovelMemoryManager, get_memory_manager

console = Console()


class NovelWebUI:
    """小说写作WebUI"""

    def __init__(self, config):
        self.config = config
        self.generator = None  # 类型根据后端动态确定
        self.memory_manager: Optional[NovelMemoryManager] = None
        self.current_session: List[Dict[str, str]] = []

    async def initialize(self, lora_path: Optional[str] = None):
        """初始化组件"""
        rprint("[bold cyan]正在初始化WebUI组件...[/bold cyan]")

        # 显示推理后端
        backend = self.config.model.inference_backend
        rprint(f"[cyan]推理后端: {backend}[/cyan]")

        # 初始化生成器（使用工厂函数）
        self.generator = create_generator(self.config)

        # 根据生成器类型选择初始化方式
        import inspect
        init_method = self.generator.initialize
        if inspect.iscoroutinefunction(init_method):
            await self.generator.initialize(lora_path)
        else:
            self.generator.initialize(lora_path)

        # 初始化记忆管理器
        self.memory_manager = NovelMemoryManager(self.config)
        self.memory_manager.initialize()

        rprint("[bold green]✓ WebUI组件初始化完成[/bold green]")

    def _create_generate_tab(self) -> gr.Blocks:
        """创建生成标签页"""
        with gr.Column() as tab:
            gr.Markdown("## 📝 小说创作")

            # 输入区域
            with gr.Row():
                with gr.Column(scale=3):
                    user_input = gr.Textbox(
                        label="创作要求",
                        placeholder="请描述你想要的小说内容...",
                        lines=3,
                    )
                with gr.Column(scale=1):
                    memory_toggle = gr.Checkbox(
                        label="使用记忆",
                        value=True,
                        info="使用之前的上下文",
                    )

            # 生成参数
            with gr.Accordion("生成参数", open=False):
                with gr.Row():
                    max_tokens = gr.Slider(
                        minimum=256,
                        maximum=4096,
                        value=2048,
                        step=256,
                        label="最大生成长度",
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        label="温度 (随机性)",
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        label="Top-P",
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=5,
                        label="Top-K",
                    )

            # 生成按钮
            with gr.Row():
                generate_btn = gr.Button("🚀 生成", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ 清空", variant="secondary")

            # 输出区域
            output = gr.Textbox(
                label="生成内容",
                lines=15,
            )

            # 生成历史
            history = gr.State([])

            # 事件绑定 - 使用async处理
            async def generate_novel_handler(input_text, toggle, max_tok, temp, top_p, top_k, hist):
                """异步生成小说（Gradio原生支持async）"""
                from datetime import datetime
                import traceback

                log_file = '/home/kewang/work/novel_ai_system/logs/debug.log'
                with open(log_file, 'a') as f:
                    f.write(f"\n[{datetime.now()}] [生成请求] 输入: {input_text[:100]}...\n")
                    f.write(f"[参数] max_tokens={max_tok}, temp={temp}, top_p={top_p}, top_k={top_k}\n")

                try:
                    # 直接调用async方法
                    result = await self.generator.generate_novel(
                        user_input=input_text,
                        max_tokens=max_tok,
                        temperature=temp,
                        top_p=top_p,
                        top_k=top_k,
                    )

                    with open(log_file, 'a') as f:
                        f.write(f"[生成完成] 结果长度: {len(result)}\n")

                    return result, hist
                except Exception as e:
                    error_msg = f"生成出错: {str(e)}\n{traceback.format_exc()}"
                    with open(log_file, 'a') as f:
                        f.write(f"[错误] {error_msg}\n")
                    return error_msg, hist

            generate_btn.click(
                fn=generate_novel_handler,
                inputs=[user_input, memory_toggle, max_tokens, temperature, top_p, top_k, history],
                outputs=[output, history],
            )
            clear_btn.click(
                fn=lambda: ("", []),
                outputs=[output, history],
            )

        return tab

    def _create_memory_tab(self) -> gr.Blocks:
        """创建记忆管理标签页"""
        with gr.Column() as tab:
            gr.Markdown("## 🧠 记忆管理")

            # 记忆类型选择
            memory_type = gr.Radio(
                choices=[
                    ("👤 人物记忆", "character"),
                    ("📖 情节记忆", "plot"),
                    ("🏞️ 环境设定", "setting"),
                    ("💬 重要对话", "dialogue"),
                    ("📝 故事上下文", "context"),
                ],
                value="character",
                label="记忆类型",
            )

            # 添加记忆表单
            with gr.Row():
                with gr.Column():
                    memory_name = gr.Textbox(label="名称 (仅人物)", visible=True)
                    memory_content = gr.Textbox(
                        label="记忆内容",
                        lines=3,
                        placeholder="输入记忆内容...",
                    )
                    add_memory_btn = gr.Button("➕ 添加记忆", variant="primary")

                with gr.Column():
                    memory_list = gr.Textbox(
                        label="现有记忆",
                        lines=10,
                        interactive=False,
                    )
                    refresh_memory_btn = gr.Button("🔄 刷新")

            # 检索相关记忆
            with gr.Row():
                search_query = gr.Textbox(label="搜索记忆", placeholder="输入关键词...")
                search_btn = gr.Button("🔍 搜索")
            search_results = gr.Textbox(label="搜索结果", lines=5, interactive=False)

            # 记忆操作
            with gr.Row():
                clear_memory_btn = gr.Button("🗑️ 清空当前类型记忆", variant="stop")
                export_memory_btn = gr.Button("📤 导出记忆", variant="secondary")
            export_output = gr.Textbox(label="导出结果", visible=False)

            # 事件绑定
            add_memory_btn.click(
                fn=self.add_memory_wrapper,
                inputs=[memory_type, memory_name, memory_content],
                outputs=[memory_list],
            )
            refresh_memory_btn.click(
                fn=lambda mem_type: self.get_memory_list(mem_type),
                inputs=[memory_type],
                outputs=[memory_list],
            )
            search_btn.click(
                fn=self.search_memory_wrapper,
                inputs=[search_query],
                outputs=[search_results],
            )
            clear_memory_btn.click(
                fn=self.clear_memory_wrapper,
                inputs=[memory_type],
                outputs=[memory_list],
            )
            export_memory_btn.click(
                fn=self.export_memory_wrapper,
                outputs=[export_output],
            )

        return tab

    def _create_train_tab(self) -> gr.Blocks:
        """创建训练标签页"""
        with gr.Column() as tab:
            gr.Markdown("## 🎯 模型微调")

            # 数据配置
            with gr.Row():
                train_data_path = gr.Textbox(
                    label="训练数据路径",
                    placeholder="./data/train",
                    value="./data/train",
                )
                val_data_path = gr.Textbox(
                    label="验证数据路径 (可选)",
                    placeholder="./data/val",
                )

            # 训练参数
            with gr.Accordion("训练参数", open=False):
                with gr.Row():
                    num_epochs = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="训练轮数",
                    )
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=2,
                        step=1,
                        label="批次大小",
                    )
                    learning_rate = gr.Slider(
                        minimum=1e-5,
                        maximum=1e-3,
                        value=2e-4,
                        label="学习率",
                    )
                    lora_r = gr.Slider(
                        minimum=8,
                        maximum=128,
                        value=64,
                        step=8,
                        label="LoRA Rank",
                    )

            # 训练控制
            with gr.Row():
                start_train_btn = gr.Button("🚀 开始训练", variant="primary")
                stop_train_btn = gr.Button("⏹️ 停止训练", variant="stop")

            # 训练状态
            train_status = gr.Textbox(
                label="训练状态",
                lines=5,
                interactive=False,
                value="等待开始训练...",
            )
            train_progress = gr.Progress()

            # 事件绑定
            start_train_btn.click(
                fn=self.start_training_wrapper,
                inputs=[train_data_path, val_data_path, num_epochs, batch_size, learning_rate, lora_r],
                outputs=[train_status],
            )

        return tab

    def _create_settings_tab(self) -> gr.Blocks:
        """创建设置标签页"""
        with gr.Column() as tab:
            gr.Markdown("## ⚙️ 系统设置")

            # 模型设置
            with gr.Group():
                gr.Markdown("### 🤖 模型配置")
                base_model = gr.Textbox(
                    label="基础模型",
                    value=self.config.model.base_model,
                )
                lora_path = gr.Textbox(
                    label="LoRA 权重路径 (可选)",
                    placeholder="./checkpoints/final_model",
                )
                reload_model_btn = gr.Button("🔄 重新加载模型", variant="primary")

            # 系统信息
            with gr.Group():
                gr.Markdown("### 📊 系统信息")
                system_info = gr.Textbox(
                    label="系统状态",
                    lines=8,
                    interactive=False,
                    value=self.get_system_info(),
                )
                refresh_info_btn = gr.Button("🔄 刷新信息")

            # 事件绑定
            refresh_info_btn.click(
                fn=lambda: self.get_system_info(),
                outputs=[system_info],
            )

        return tab

    def build_ui(self) -> gr.Blocks:
        """构建完整UI"""
        with gr.Blocks(
            title=self.config.webui.title,
        ) as app:
            # 标题和描述
            gr.Markdown(
                f"""
                # {self.config.webui.title}

                {self.config.webui.description}

                ---
                """
            )

            # 标签页
            with gr.Tabs():
                with gr.Tab("📝 创作"):
                    self._create_generate_tab()

                with gr.Tab("🧠 记忆"):
                    self._create_memory_tab()

                with gr.Tab("🎯 训练"):
                    self._create_train_tab()

                with gr.Tab("⚙️ 设置"):
                    self._create_settings_tab()

            # 页脚
            gr.Markdown(
                """
                ---

                💡 **提示**: 使用记忆功能可以让AI记住之前的创作内容，生成更连贯的故事。

                🚀 **Powered by**: vLLM + LoRA + ChromaDB
                """
            )

        return app

    def _get_custom_css(self) -> str:
        """获取自定义CSS"""
        return """
        .gradio-container {
            max-width: 1400px !important;
        }
        .generate-btn {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
        }
        """

    # Wrapper函数 (用于异步调用)
    async def generate_wrapper(
        self,
        user_input: str,
        use_memory: bool,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        history: List[Dict[str, str]],
    ) -> Tuple[str, List[Dict[str, str]]]:
        """生成包装器"""
        import logging
        logger = logging.getLogger(__name__)

        logger.info("[异步生成] 函数开始执行")
        if not user_input.strip():
            logger.warning("[异步生成] 输入为空")
            return "请输入创作要求", history

        # 获取记忆上下文
        memory_context = ""
        if use_memory and self.memory_manager:
            logger.info("[记忆系统] 获取记忆上下文...")
            memory_context = self.memory_manager.get_formatted_context(user_input)
            logger.info(f"[记忆系统] 上下文长度: {len(memory_context)} 字符")

        # 生成内容（支持异步和同步生成器）
        logger.info(f"[{self.config.model.inference_backend}] 开始生成...")

        # 检查生成器是否有异步方法
        if hasattr(self.generator, 'generate_novel_async'):
            result = await self.generator.generate_novel_async(
                user_input=user_input,
                context=memory_context,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        else:
            # 同步方法，直接调用
            result = self.generator.generate_novel(
                user_input=user_input,
                context=memory_context,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

        logger.info(f"[{self.config.model.inference_backend}] 生成完成，结果长度: {len(result)} 字符")

        # 保存到历史
        history.append({
            "user": user_input,
            "assistant": result,
            "timestamp": str(asyncio.get_event_loop().time()),
        })

        # 保存到记忆
        if self.memory_manager:
            logger.info("[记忆系统] 保存生成结果到记忆...")
            self.memory_manager.summarize_session(result)

        logger.info("[异步生成] 函数执行完成")
        return result, history

    def generate_wrapper_sync(
        self,
        user_input: str,
        use_memory: bool,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        history: List[Dict[str, str]],
    ) -> Tuple[str, List[Dict[str, str]]]:
        """同步包装器，用于Gradio"""
        import asyncio
        import logging
        from datetime import datetime

        # 配置调试日志
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/home/kewang/work/novel_ai_system/logs/debug.log'),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)

        logger.info(f"[生成请求] 输入: {user_input[:100]}...")
        logger.info(f"[生成参数] max_tokens={max_tokens}, temp={temperature}, top_p={top_p}, top_k={top_k}")

        # 获取或创建事件循环
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            logger.info("[事件循环] 开始运行异步生成...")
            start_time = datetime.now()

            # 运行异步函数
            result = loop.run_until_complete(
                self.generate_wrapper(
                    user_input, use_memory, max_tokens,
                    temperature, top_p, top_k, history
                )
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"[生成完成] 耗时: {duration:.2f}秒")
            logger.info(f"[生成结果] 长度: {len(result[0])} 字符")

            return result
        except Exception as e:
            logger.error(f"[生成错误] {type(e).__name__}: {e}", exc_info=True)
            return f"生成出错: {str(e)}\n\n请查看日志文件: /home/kewang/work/novel_ai_system/logs/debug.log", history

    def add_memory_wrapper(
        self,
        memory_type: str,
        memory_name: str,
        memory_content: str,
    ) -> str:
        """添加记忆包装器"""
        if not memory_content.strip():
            return "请输入记忆内容"

        if memory_type == "character":
            self.memory_manager.add_character(
                name=memory_name or "未命名",
                description=memory_content,
            )
        elif memory_type == "plot":
            self.memory_manager.add_plot(memory_content)
        elif memory_type == "setting":
            self.memory_manager.add_setting(memory_content)
        elif memory_type == "dialogue":
            self.memory_manager.add_dialogue(memory_content, speaker=memory_name)
        else:
            self.memory_manager.add_memory(memory_content, memory_type="context")

        return self.get_memory_list(memory_type)

    def get_memory_list(self, memory_type: str) -> str:
        """获取记忆列表"""
        memories = self.memory_manager.get_all_memories(memory_type)
        if not memories:
            return "暂无记忆"

        result = []
        for i, mem in enumerate(memories, 1):
            result.append(f"{i}. {mem.get('content', mem.get('description', ''))[:100]}")

        return "\n\n".join(result)

    def search_memory_wrapper(self, query: str) -> str:
        """搜索记忆包装器"""
        if not query.strip():
            return "请输入搜索关键词"

        results = self.memory_manager.retrieve_memory(query, top_k=5)
        if not results:
            return "未找到相关记忆"

        output = []
        for i, res in enumerate(results, 1):
            output.append(
                f"{i}. [{res['type']}] {res['content'][:100]}\n"
                f"   相关度: {1-res['distance']:.2f}"
            )

        return "\n\n".join(output)

    def clear_memory_wrapper(self, memory_type: str) -> str:
        """清除记忆包装器"""
        self.memory_manager.clear_memories(memory_type)
        return "已清空记忆"

    def export_memory_wrapper(self):
        """导出记忆包装器"""
        import json
        memories = self.memory_manager.get_all_memories()
        return json.dumps(memories, ensure_ascii=False, indent=2)

    def start_training_wrapper(
        self,
        train_data_path: str,
        val_data_path: str,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        lora_r: int,
    ) -> str:
        """开始训练包装器"""
        return "训练功能需要单独运行 train_lora.py\n\n" \
               f"配置:\n" \
               f"- 数据路径: {train_data_path}\n" \
               f"- 训练轮数: {num_epochs}\n" \
               f"- 批次大小: {batch_size}\n" \
               f"- 学习率: {learning_rate}\n" \
               f"- LoRA Rank: {lora_r}"

    def get_system_info(self) -> str:
        """获取系统信息"""
        import torch
        import subprocess

        info = []

        # GPU信息
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            info.append(f"GPU: {gpu_count} x {torch.cuda.get_device_name(0)}")
            info.append(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            info.append("GPU: 不可用")

        # 内存信息
        result = subprocess.run(['free', '-h'], capture_output=True, text=True)
        info.append(f"内存: {result.stdout.split()[7]}")

        # 模型信息
        info.append(f"基础模型: {self.config.model.base_model}")

        # 记忆统计
        if self.memory_manager:
            total_memories = sum(len(v) for v in self.memory_manager.long_term_memory.values())
            info.append(f"记忆数量: {total_memories} 条")

        return "\n".join(info)


# 全局实例
_webui: Optional[NovelWebUI] = None
_engine_event_loop = None  # 保存引擎的事件循环


async def launch_webui(lora_path: Optional[str] = None):
    """启动WebUI - 保持事件循环运行"""
    from config import config
    global _engine_event_loop

    # 获取当前事件循环（将用于引擎）
    _engine_event_loop = asyncio.get_running_loop()

    global _webui
    _webui = NovelWebUI(config)
    await _webui.initialize(lora_path)

    # 构建UI
    app = _webui.build_ui()

    # 启动 - 不阻塞线程，让事件循环继续运行
    app.launch(
        server_name=config.webui.host,
        server_port=config.webui.port,
        share=config.webui.share,
        show_error=True,
        prevent_thread_lock=True,  # 关键：不阻塞事件循环
        theme=gr.themes.Soft(),
        css=_webui._get_custom_css(),
    )

    # 保持事件循环运行
    try:
        # 无限等待，保持事件循环活跃
        await asyncio.Future()
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭...")


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="中文小说写作AI系统 - WebUI")
    parser.add_argument("--lora", type=str, default=None, help="LoRA权重路径")
    parser.add_argument("--host", type=str, default=None, help="服务器地址")
    parser.add_argument("--port", type=int, default=None, help="服务器端口")
    parser.add_argument("--share", action="store_true", help="创建公共链接")

    args = parser.parse_args()

    # 更新配置
    from config import config
    if args.host:
        config.webui.host = args.host
    if args.port:
        config.webui.port = args.port
    if args.share:
        config.webui.share = True

    # 启动
    asyncio.run(launch_webui(lora_path=args.lora))


if __name__ == "__main__":
    main()
