# 中文小说写作AI系统

基于 **vLLM/llama.cpp + LoRA + ChromaDB** 的智能小说创作系统，支持训练、推理和记忆管理。

## ✨ 特性

- **双推理后端**: 支持 GPU (vLLM) 和 CPU (llama.cpp) 推理
- **高性能推理**: GPU 上使用 vLLM，CPU 上使用 llama.cpp
- **LoRA 微调**: QLoRA 4-bit 量化训练，降低显存需求
- **记忆功能**: 向量数据库存储，支持长期记忆和 RAG
- **WebUI 界面**: Gradio 构建，易于使用
- **中文优化**: 专为中文小说写作优化

## 📋 系统要求

### GPU 训练机器
- Python 3.8+
- CUDA 12.0+
- GPU: 建议 16GB+ 显存 (RTX 4090 / A100 等)
- 内存: 建议 64GB+

### CPU 推理机器
- Python 3.8+
- 内存: 建议 22GB+ (7B 模型，Q5 量化)
- CPU: 建议 6 核心以上

## 🚀 快速开始

### 方式一：GPU 训练 + GPU 推理

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备训练数据
python start.py prepare --sample

# 3. 训练模型
python start.py train

# 4. 启动 WebUI (使用 vLLM)
python start.py webui
```

### 方式二：GPU 训练 + CPU 推理 ⭐ 推荐

```bash
# GPU 机器上：
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备训练数据
python start.py prepare --sample

# 3. 训练模型
python start.py train

# 4. 转换模型为 GGUF 格式
python start.py convert hf-to-gguf --model Qwen/Qwen2.5-7B-Instruct --quant Q5_K_M
python start.py convert lora-to-gguf --lora-path ./checkpoints/final_model

# 5. 将模型文件复制到 CPU 机器
# ./models/qwen2.5-7b-q5_k_m.gguf
# ./models/lora-gguf/

# CPU 机器上：
# 1. 安装依赖 (不需要 torch/vllm)
pip install llama-cpp-python gradio chromadb langchain sentence-transformers

# 2. 修改 config.py
# inference_backend: str = "llama_cpp"
# llama_cpp_model_path: str = "./models/qwen2.5-7b-q5_k_m.gguf"
# llama_cpp_lora_path: str = "./models/lora-gguf"

# 3. 启动 WebUI (使用 llama.cpp)
python start.py webui
```

## 🎓 训练完整指南

### 训练流程概览

```
┌─────────────────────────────────────────────────────────────┐
│                        训练流程                              │
└─────────────────────────────────────────────────────────────┘

   1. 准备数据          2. 配置参数          3. 开始训练
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │ 小说文件 │ ─────► │ config.py│ ─────► │ 训练中  │
   │.txt/json│         │ 调整参数 │         │ 监控loss│
   └─────────┘         └─────────┘         └────┬────┘
                                                    │
   5. 使用模型          4. 恢复训练              │
   ┌─────────┐         ┌─────────┐              │
   │ WebUI   │ ◄────── │中断后   │ ◄────────────┘
   │ 生成    │         │恢复训练 │
   └─────────┘         └─────────┘
```

### 步骤 1: 准备训练数据

#### 支持的数据格式

```bash
# 方式一：TXT 文本文件（最简单）
# 直接将小说文件放入 data/raw/ 目录
cp my_novel.txt data/raw/

# 方式二：JSON 格式
[
  {"text": "小说内容第一段..."},
  {"text": "小说内容第二段..."}
]

# 方式三：JSONL 格式（推荐大规模数据）
{"text": "内容1..."}
{"text": "内容2..."}
```

#### 数据准备命令

```bash
# 创建示例数据（用于测试）
python start.py prepare --sample

# 准备自定义数据
python start.py prepare --chunk-size 2048 --val-split 0.1

# 参数说明：
# --chunk-size    每个训练样本的最大token数 (默认: 2048)
# --val-split     验证集比例 (默认: 0.1，即10%)
# --min-length    最小文本长度，过滤太短的内容 (默认: 500)
```

#### 数据质量建议

| 指标 | 建议值 | 说明 |
|------|--------|------|
| 数据量 | 10MB+ | 越多越好，建议至少几MB纯文本 |
| 文本质量 | 高 | 去除HTML标签、乱码、无关内容 |
| 内容一致性 | 单一风格 | 同一风格/作者/题材效果更好 |
| 验证集比例 | 0.05-0.1 | 数据少时可降低比例 |

### 步骤 2: 配置训练参数

编辑 `config.py` 中的 `TrainingConfig` 部分：

```python
@dataclass
class TrainingConfig:
    # === 数据路径 ===
    train_data_path: str = "data/train/train.jsonl"  # 训练数据
    val_data_path: str = "data/val/val.jsonl"        # 验证数据
    max_seq_length: int = 2048                       # 最大序列长度

    # === 训练参数 ===
    num_train_epochs: int = 3                        # 训练轮数
    per_device_train_batch_size: int = 1             # 每GPU批次大小
    per_device_eval_batch_size: int = 1              # 评估批次大小
    gradient_accumulation_steps: int = 4             # 梯度累积步数
    learning_rate: float = 2e-4                      # 学习率
    warmup_steps: int = 100                          # 预热步数

    # === 保存和日志 ===
    logging_steps: int = 10                          # 日志记录频率
    save_steps: int = 500                            # checkpoint保存频率
    eval_steps: int = 500                            # 评估频率

    # === 优化器 ===
    optimizer: str = "paged_adamw_32bit"             # 优化器类型
    weight_decay: float = 0.01                       # 权重衰减
    max_grad_norm: float = 1.0                       # 梯度裁剪

    # === 其他 ===
    bf16: bool = True                                # 使用bfloat16
    fp16: bool = False                               # 使用float16
    gradient_checkpointing: bool = True              # 梯度检查点(省显存)
```

### 步骤 3: 参数调整指南

#### 根据显存调整参数

| 显存大小 | batch_size | gradient_accumulation | max_seq_length | 量化 |
|----------|------------|----------------------|----------------|------|
| 8GB      | 1          | 8                    | 1024           | 4-bit |
| 12GB     | 1          | 4                    | 2048           | 4-bit |
| 16GB     | 2          | 4                    | 2048           | 4-bit |
| 24GB     | 4          | 2                    | 4096           | 8-bit/无 |
| 40GB+    | 8          | 1                    | 8192           | 无 |

**有效批次大小计算**：
```
有效批次 = batch_size × gradient_accumulation × GPU数量
```

#### 根据数据量调整训练轮数

| 数据量 | 推荐轮数 | 说明 |
|--------|----------|------|
| < 1MB   | 5-10     | 数据少需要多轮次 |
| 1-10MB  | 3-5      | 标准 |
| 10-100MB| 2-3      | 数据充足 |
| > 100MB | 1-2      | 大数据集 |

#### 学习率调整

```python
# 默认学习率
learning_rate = 2e-4  # 适用于大多数情况

# 训练不稳定时（loss震荡）
learning_rate = 1e-4  # 降低学习率

# 训练太慢时
learning_rate = 5e-4  # 提高学习率（谨慎）

# 使用学习率调度器
warmup_steps = int(total_steps * 0.1)  # 预热10%的步数
```

#### LoRA 参数调整

```python
# config.py 中的 ModelConfig
lora_r: int = 64          # LoRA rank（越大效果越好但参数越多）
lora_alpha: int = 128     # LoRA alpha（通常 = r × 2）
lora_dropout: float = 0.1 # Dropout率

# rank 调整建议：
# r=16:  快速测试，显存占用少
# r=64:  标准配置（推荐）
# r=128: 高质量训练
# r=256: 最佳效果（需要更多显存）
```

### 步骤 4: 开始训练

#### 基础训练命令

```bash
# 使用默认配置训练
python start.py train

# 自定义参数训练
python start.py train --epochs 5 --batch-size 2 --lr 1e-4

# 后台训练（推荐）
nohup python start.py train > logs/train.log 2>&1 &

# 监控训练进度
tail -f logs/train.log

# 使用 TensorBoard 可视化
tensorboard --logdir logs/tensorboard
```

#### 从 checkpoint 恢复训练

```bash
# 训练中断后，从最新的 checkpoint 恢复
python start.py train --resume ./checkpoints/checkpoint-1000

# 或指定具体 checkpoint
python start.py train --resume ./checkpoints/checkpoint-5000
```

### 步骤 5: 监控训练状态

#### 训练输出解读

```
训练配置
┌────────────────┬──────────────────────────┐
│ 参数           │ 值                        │
├────────────────┼──────────────────────────┤
│ 基础模型       │ Qwen/Qwen2.5-7B-Instruct │
│ 训练轮数       │ 3                         │
│ 批次大小       │ 1                         │
│ 梯度累积       │ 4                         │
│ 学习率         │ 0.0002                    │
│ 有效批次大小   │ 4                         │
└────────────────┴──────────────────────────┘

{'loss': 2.8476, 'grad_norm': 1.234, 'learning_rate': 1.8e-5, 'epoch': 0.01}
  ↑              ↑            ↑                  ↑
  训练损失       梯度范数     当前学习率         当前进度

# 指标说明：
# - loss: 越低越好，应该稳定下降
# - grad_norm: 梯度范数，过大可能需要降低学习率
# - learning_rate: 实际学习率（经过warmup调整）
# - epoch: 已完成的训练轮数
```

#### 判断训练是否正常

✅ **训练正常的标志**：
- Loss 稳定下降
- 梯度范数 < 10
- 生成的文本质量逐渐提升
- 验证 Loss 不上升

❌ **训练异常的标志**：
- Loss 震荡或 NaN
- 梯度范数爆炸（>100）
- 验证 Loss 持续上升（过拟合）
- 显存溢出（OOM）

### 步骤 6: 使用训练好的模型

```bash
# 方式一：使用 final_model
python start.py webui --lora ./checkpoints/final_model

# 方式二：使用指定 checkpoint
python start.py webui --lora ./checkpoints/checkpoint-5000

# 启动后访问 http://localhost:7860
```

### 不同场景的配置方案

#### 场景 1: 快速测试（验证流程）

```python
# config.py
max_seq_length: int = 1024
per_device_train_batch_size: int = 1
gradient_accumulation_steps: int = 2
num_train_epochs: int = 1
save_steps: int = 100
```

```bash
# 使用少量数据测试
python start.py prepare --sample
python start.py train --epochs 1
```

#### 场景 2: 标准训练（推荐配置）

```python
# config.py
max_seq_length: int = 2048
per_device_train_batch_size: int = 1
gradient_accumulation_steps: int = 4
num_train_epochs: int = 3
learning_rate: float = 2e-4
save_steps: int = 500
```

#### 场景 3: 高质量训练（大数据量）

```python
# config.py
max_seq_length: int = 4096
per_device_train_batch_size: int = 2
gradient_accumulation_steps: int = 4
num_train_epochs: int = 2
learning_rate: float = 1e-4
lora_r: int = 128
lora_alpha: int = 256
save_steps: int = 1000
```

#### 场景 4: 低显存优化（< 12GB）

```python
# config.py
load_in_4bit: bool = True
max_seq_length: int = 1024
per_device_train_batch_size: int = 1
gradient_accumulation_steps: int = 8
gradient_checkpointing: bool = True
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
    --resume PATH   从 checkpoint 恢复

# 推理测试
python start.py inference

# 模型格式转换
python start.py convert [SUBCOMMAND]

  SUBCOMMANDS:
    hf-to-gguf      转换 Hugging Face 模型为 GGUF 格式
    lora-to-gguf    转换 LoRA 权重为 GGUF 格式
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
├── models/                # 模型文件 (GGUF)
│
├── scripts/               # 转换脚本
│   ├── convert_hf_to_gguf.sh
│   └── convert_lora_to_gguf.sh
│
└── src/
    ├── train/            # 训练模块
    │   └── train_lora.py
    ├── inference/        # 推理模块
    │   ├── backend_factory.py   # 后端工厂
    │   ├── vllm_server.py       # vLLM 推理
    │   └── llama_server.py      # llama.cpp 推理
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
# === 推理后端选择 ===
model.inference_backend = "llama_cpp"  # "vllm" (GPU) 或 "llama_cpp" (CPU)

# === vLLM 配置 (GPU 推理) ===
model.vllm_max_model_len = 32768
model.vllm_gpu_memory_utilization = 0.85

# === llama.cpp 配置 (CPU 推理) ===
model.llama_cpp_model_path = "./models/qwen2.5-7b-q5_k_m.gguf"
model.llama_cpp_lora_path = "./models/lora-gguf"
model.llama_cpp_n_ctx = 32768       # 上下文长度
model.llama_cpp_n_threads = 6       # CPU 线程数

# === 训练配置 ===
model.base_model = "Qwen/Qwen2.5-7B-Instruct"  # 基础模型 (Hugging Face 格式)
model.load_in_4bit = True                       # 4-bit 量化
model.lora_r = 64                               # LoRA rank

training.num_train_epochs = 3
training.per_device_train_batch_size = 2
training.gradient_accumulation_steps = 8
training.learning_rate = 2e-4

# === 记忆配置 ===
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
   │ Inference│  │ LoRA    │  │ ChromaDB  │
   │ Backend  │  │ 微调    │  │ 记忆存储  │
   │  Factory │  └─────────┘  └───────────┘
   └─────┬─────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌──────────┐
│ vLLM  │ │llama.cpp │
│ (GPU) │ │  (CPU)   │
└───────┘ └──────────┘
    │         │
    └────┬────┘
         ▼
  ┌─────────────┐
  │   Qwen2.5   │
  │ (HF/GGUF)   │
  └─────────────┘
```

### GPU 训练 + CPU 推理工作流

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GPU 机器 - 训练阶段                              │
│                                                                       │
│  Qwen/Qwen2.5-7B-Instruct (Hugging Face)                            │
│                │                                                    │
│                ├─► LoRA 训练 ──► adapter_model.safetensors           │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ 转换
┌─────────────────────────────────────────────────────────────────────┐
│                    模型格式转换                                     │
│                                                                       │
│  HF 模型 ──convert_hf_to_gguf──► FP16 GGUF ──quantize──► Q5 GGUF  │
│  LoRA ───convert-lora-to-gguf──► GGUF LoRA adapter                 │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CPU 机器 - 推理阶段                              │
│                                                                       │
│  llama.cpp 加载 GGUF 模型 + GGUF LoRA                               │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## 📚 推荐基础模型

- **Qwen2.5-7B-Instruct**: 平衡性能和资源占用
- **Qwen2.5-14B-Instruct**: 更强性能，需要更多显存
- **Yi-1.5-9B-Chat**: 优秀的中文对话模型
- **DeepSeek-V3**: 最新的开源中文模型

## 🐛 常见问题

### 推理后端选择
- **GPU 机器**: 使用 `inference_backend = "vllm"` 获得最佳性能
- **CPU 机器**: 使用 `inference_backend = "llama_cpp"` 进行 CPU 推理
- **GGUF 模型转换**: 使用 `python start.py convert hf-to-gguf` 转换模型

### GGUF 模型转换
```bash
# 转换基础模型 (一次即可)
python start.py convert hf-to-gguf --model Qwen/Qwen2.5-7B-Instruct --quant Q5_K_M

# 转换 LoRA 权重 (每次训练后)
python start.py convert lora-to-gguf --lora-path ./checkpoints/final_model
```

### llama.cpp CPU 推理性能
- **Q5_K_M 量化**: ~4-6 tokens/秒 (6核 CPU)
- **Q8_0 量化**: ~3-5 tokens/秒 (更高精度)
- 调整 `llama_cpp_n_threads` 以匹配 CPU 核心数

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
- [vLLM](https://github.com/vllm-project/vllm) - 高性能 GPU 推理引擎
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - 高效 CPU 推理引擎
- [Gradio](https://github.com/gradio-app/gradio) - WebUI 框架
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库
- [PEFT](https://github.com/huggingface/peft) - LoRA 微调库

---

**🚀 开始创作你的小说之旅！**
