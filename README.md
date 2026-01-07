# 中文小说写作AI系统

基于 **vLLM + LoRA + ChromaDB** 的智能小说创作系统，支持训练、推理和记忆管理。

## ✨ 特性

- **高性能推理**: 基于 vLLM，支持快速文本生成
- **LoRA 微调**: QLoRA 4-bit 量化训练，降低显存需求
- **记忆功能**: 向量数据库存储，支持长期记忆和 RAG
- **WebUI 界面**: Gradio 构建，易于使用
- **中文优化**: 专为中文小说写作优化

## 📋 系统要求

- Python 3.8+
- CUDA 12.0+
- GPU: 建议 16GB+ 显存 (RTX 4090 / A100 等)
- 内存: 建议 64GB+

## 🚀 快速开始

### 1. 安装依赖

```bash
cd novel_ai_system
pip install -r requirements.txt
```

### 2. 准备训练数据

创建示例数据（用于测试）:
```bash
python start.py prepare --sample
```

或使用自己的小说数据:
```bash
# 将小说文件放入 data/raw/ 目录
# 支持 .txt, .json, .jsonl 格式
python start.py prepare
```

### 3. 启动 WebUI

```bash
python start.py webui
```

访问 `http://localhost:7860` 开始使用！

### 4. 训练模型 (可选)

```bash
python start.py train
```

训练完成后，使用 LoRA 权重启动:
```bash
python start.py webui --lora ./checkpoints/final_model
```

## 📖 使用指南

### WebUI 功能

#### 📝 创作标签页
- 输入创作要求，AI 自动生成小说内容
- 支持调整生成参数 (温度、top-p、top-k)
- 可启用/禁用记忆功能

#### 🧠 记忆标签页
- **人物记忆**: 添加和管理小说人物信息
- **情节记忆**: 记录故事情节发展
- **环境设定**: 存储场景和世界观设定
- **重要对话**: 保存关键对话内容

#### 🎯 训练标签页
- 配置训练参数
- 监控训练进度

#### ⚙️ 设置标签页
- 查看系统状态
- 配置模型路径

### CLI 命令

```bash
# 启动 WebUI
python start.py webui [OPTIONS]

  OPTIONS:
    --lora PATH     LoRA 权重路径
    --host ADDR     服务器地址 (默认: 0.0.0.0)
    --port PORT     端口 (默认: 7860)
    --share         创建公共链接

# 准备训练数据
python start.py prepare [OPTIONS]

  OPTIONS:
    --sample        创建示例数据
    --chunk-size    训练块大小 (默认: 2048)
    --val-split     验证集比例 (默认: 0.1)

# 启动训练
python start.py train [OPTIONS]

  OPTIONS:
    --data PATH     训练数据路径
    --epochs N      训练轮数 (默认: 3)
    --batch-size N  批次大小 (默认: 2)
    --lr FLOAT      学习率 (默认: 2e-4)

# 推理测试
python start.py inference
```

## 📁 项目结构

```
novel_ai_system/
├── config.py              # 配置文件
├── start.py               # 主入口
├── requirements.txt       # 依赖列表
├── README.md             # 说明文档
│
├── data/                  # 数据目录
│   ├── raw/              # 原始小说文件
│   ├── train/            # 训练数据
│   ├── val/              # 验证数据
│   └── chroma_db/        # 向量数据库
│
├── checkpoints/           # 训练检查点
│
├── logs/                  # 日志文件
│
├── models/                # 下载的模型
│
└── src/
    ├── train/            # 训练模块
    │   └── train_lora.py
    ├── inference/        # 推理模块
    │   └── vllm_server.py
    ├── memory/           # 记忆模块
    │   └── memory_manager.py
    ├── data/             # 数据处理
    │   └── prepare_data.py
    └── webui/            # Web界面
        └── app.py
```

## ⚙️ 配置说明

编辑 `config.py` 自定义配置:

```python
# 模型配置
model.base_model = "Qwen/Qwen2.5-7B-Instruct"  # 基础模型
model.load_in_4bit = True                       # 4-bit 量化
model.lora_r = 64                               # LoRA rank

# 训练配置
training.num_train_epochs = 3
training.per_device_train_batch_size = 2
training.gradient_accumulation_steps = 8
training.learning_rate = 2e-4

# 推理配置
model.vllm_max_model_len = 32768
model.vllm_gpu_memory_utilization = 0.85

# 记忆配置
memory.embedding_model = "BAAI/bge-m3"
memory.max_memory_items = 1000
```

## 🔧 技术架构

```
┌─────────────────────────────────────────────────────┐
│                     Gradio WebUI                     │
└─────────────────────┬───────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   ┌─────────┐  ┌─────────┐  ┌───────────┐
   │ vLLM    │  │ LoRA    │  │ ChromaDB  │
   │ 推理引擎 │  │ 微调    │  │ 记忆存储  │
   └─────────┘  └─────────┘  └───────────┘
        │             │
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │   Qwen2.5   │
        │  (或其他)   │
        └─────────────┘
```

## 📚 推荐基础模型

- **Qwen2.5-7B-Instruct**: 平衡性能和资源占用
- **Qwen2.5-14B-Instruct**: 更强性能，需要更多显存
- **Yi-1.5-9B-Chat**: 优秀的中文对话模型
- **DeepSeek-V3**: 最新的开源中文模型

## 🐛 常见问题

### 显存不足
- 使用 4-bit 量化: `model.load_in_4bit = True`
- 减小批次大小: `training.per_device_train_batch_size = 1`
- 减小最大序列长度: `training.max_seq_length = 2048`

### 训练速度慢
- 启用梯度检查点: `training.gradient_checkpointing = True`
- 使用 DeepSpeed 优化器
- 增加梯度累积步数

### 生成质量不佳
- 增加训练数据量
- 调整温度参数 (0.7-0.9)
- 使用更大的基础模型
- 训练更多轮次

## 🔍 问题排查与解决方案

### WebUI 生成无响应问题

#### 问题现象
- WebUI 界面可正常访问
- 点击生成按钮后无任何输出
- debug.log 显示调用链正常，但 `async for` 循环卡住

#### 根本原因
**事件循环不匹配**：Gradio 的 async handler 运行在它自己的事件循环中，而 vLLM 的 `AsyncLLMEngine` 在主事件循环中初始化。当在不同事件循环间调用时，`async for` 无法正确获取异步生成器的数据。

#### 解决方案

**1. 修改 `src/webui/app.py`**

```python
# 全局变量保存引擎的事件循环
_engine_event_loop = None

async def launch_webui(lora_path: Optional[str] = None):
    global _engine_event_loop
    # 保存引擎的事件循环
    _engine_event_loop = asyncio.get_running_loop()

    # 初始化引擎...
    await _webui.initialize(lora_path)

    # 构建UI
    app = _webui.build_ui()

    # 关键：prevent_thread_lock=True 不阻塞事件循环
    app.launch(
        prevent_thread_lock=True,  # 必须设置
        server_name=config.webui.host,
        server_port=config.webui.port,
        ...
    )

    # 保持事件循环持续运行
    try:
        await asyncio.Future()  # 无限等待
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭...")
```

**2. 修改 `src/inference/vllm_server.py`**

```python
async def _generate(self, prompt: str, sampling_params: SamplingParams, ...):
    # 获取引擎的事件循环
    from src.webui.app import _engine_event_loop
    engine_loop = _engine_event_loop
    current_loop = asyncio.get_running_loop()

    # 定义在引擎事件循环中运行的协程
    async def generate_in_engine_loop():
        outputs = []
        async for request_output in self.engine.generate(...):
            outputs.append(request_output.outputs[0].text)
        return "".join(outputs)

    # 检查是否在同一事件循环
    if current_loop is engine_loop:
        return await generate_in_engine_loop()
    else:
        # 跨事件循环调用
        future = asyncio.run_coroutine_threadsafe(
            generate_in_engine_loop(), engine_loop
        )
        return future.result(timeout=120)
```

#### 关键要点
1. **`prevent_thread_lock=True`**: 让 Gradio 不阻塞主事件循环
2. **`await asyncio.Future()`**: 保持主事件循环持续运行
3. **`asyncio.run_coroutine_threadsafe()`**: 跨事件循环安全地调用异步函数
4. **保存引擎事件循环**: 在全局变量中保存 `_engine_event_loop`

#### 调试日志示例
```
[vLLM._generate] 当前事件循环: 140427378303600
[vLLM._generate] 引擎事件循环: 140432959270800
[vLLM._generate] 在不同事件循环中，使用 run_coroutine_threadsafe
[EngineLoop] 收到chunk #1, 新增: 0, 总: 0
[EngineLoop] 收到chunk #10, 新增: 1, 总: 2
...
[EngineLoop] 生成完成，共 287 块，长度: 417
```

### GPU 内存占用问题

#### 现象
- 启动时提示 GPU 内存不足
- `ValueError: Free memory on device is less than desired GPU memory utilization`

#### 解决方案
```bash
# 查找并清理旧的 vLLM 进程
ps aux | grep -E "vllm|VLLM" | grep $USER
kill -9 <PID>

# 或使用一键清理
pkill -f "python3 start.py"
```

### vLLM 参数问题

#### 错误：stop 参数包含空字符串
```
ValueError: stop cannot contain an empty string.
```

**解决方案**：修改 `src/inference/vllm_server.py`
```python
# 错误写法
stop=stop or ["<|im_end|>", ""]

# 正确写法
stop=stop or ["<|im_end|>"]
```

### 依赖问题

#### torchaudio 安装失败
```bash
# 不安装 torchaudio，只安装 torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Gradio 6.0+ API 变化
- 移除 `show_copy_button=True` 参数
- 移除 `gr.Download` 组件，使用字符串返回代替

## 📝 数据格式

### TXT 格式
```
直接放入 data/raw/ 目录即可
```

### JSON 格式
```json
[
  {
    "title": "小说标题",
    "content": "小说内容...",
    "author": "作者",
    "genre": "类型",
    "tags": ["标签1", "标签2"]
  }
]
```

### JSONL 格式
```jsonl
{"title": "标题1", "content": "内容1..."}
{"title": "标题2", "content": "内容2..."}
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- [Qwen](https://github.com/QwenLM/Qwen) - 优秀的中文开源模型
- [vLLM](https://github.com/vllm-project/vllm) - 高性能推理引擎
- [Gradio](https://github.com/gradio-app/gradio) - WebUI 框架
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库

---

**🚀 开始创作你的小说之旅！**
