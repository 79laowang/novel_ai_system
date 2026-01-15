# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Chinese Novel Writing AI System** - A sophisticated AI-powered novel writing system built with **vLLM/llama.cpp + LoRA + ChromaDB**, optimized for Chinese novel generation with training, inference, and memory management capabilities.

**Key Feature**: Supports dual inference backends - **vLLM for GPU** and **llama.cpp for CPU**.

## Common Commands

### Setup
```bash
# Automated setup (creates venv, installs dependencies, creates directories)
./scripts/install.sh

# Manual setup
pip install -r requirements.txt
```

### Data Preparation
```bash
# Create sample data for testing
python start.py prepare --sample

# Prepare custom data from ./data/raw/ (supports .txt, .json, .jsonl)
python start.py prepare --chunk-size 2048 --val-split 0.1
```

### Training
```bash
# Start LoRA training with default config
python start.py train

# With custom parameters
python start.py train --epochs 5 --batch-size 4 --lr 1e-4
```

### WebUI
```bash
# Basic launch
python start.py webui

# With LoRA weights
python start.py webui --lora ./training/final_model

# Custom host/port/share
python start.py webui --host 0.0.0.0 --port 7860 --share
```

### Testing/Debugging
```bash
# Direct inference test (uses configured backend)
python start.py inference

# Monitor debug logs
./scripts/watch_logs.sh
# or
tail -f logs/debug.log
```

### Model Conversion (GGUF Format)
```bash
# Convert Hugging Face model to GGUF (for llama.cpp)
python start.py convert hf-to-gguf --model Qwen/Qwen2.5-7B-Instruct --quant Q5_K_M

# Convert trained LoRA to GGUF format
python start.py convert lora-to-gguf --lora-path ./training/final_model
```

## Architecture

### High-Level Structure

```
Gradio WebUI (src/webui/app.py)
    │
    ├──► Inference Backend (src/inference/backend_factory.py)
    │    ├──► VLLM Server (src/inference/vllm_server.py) [GPU]
    │    │    └──► AsyncLLMEngine + LoRA weights
    │    │
    │    └──► Llama.cpp Server (src/inference/llama_server.py) [CPU]
    │         └──► GGUF model + GGUF LoRA adapter
    │
    ├──► Memory Manager (src/memory/memory_manager.py)
    │    └──► ChromaDB + Sentence Transformers (BGE-M3 embeddings)
    │
    └──► Training Module (src/train/train_lora.py)
         └──► QLoRA 4-bit quantization + PEFT
```

### Entry Point

**`start.py`** - CLI dispatcher with subcommands:
- `webui` (default) - Launch Gradio interface
- `train` - Start LoRA training
- `prepare` - Prepare training data
- `inference` - Run inference tests
- `convert` - Model format conversion (HF→GGUF, LoRA→GGUF)

### Core Components

#### 1. Configuration (`config.py`)
Centralized configuration using dataclasses:
- `ModelConfig` - Inference backend selector, vLLM/llama.cpp settings, LoRA parameters, quantization
- `TrainingConfig` - Training parameters, data paths
- `MemoryConfig` - ChromaDB settings, embedding model (BAAI/bge-m3)
- `WebUIConfig` - Interface settings, generation parameters
- `SystemConfig` - Paths, logging, system settings

**Inference Backend Selection**:
```python
# config.py
inference_backend: str = "llama_cpp"  # "vllm" for GPU, "llama_cpp" for CPU
```

#### 2. Inference Engines

**VLLM Server** (`src/inference/vllm_server.py`) - GPU inference:
- **Key Pattern**: Complex event loop synchronization between Gradio and vLLM
- Uses `AsyncLLMEngine` from vLLM 0.6+
- LoRA weight loading on demand via `LoRARequest`
- Critical async/sync wrapper for Gradio compatibility
- Event loop bridge using `asyncio.run_coroutine_threadsafe()`

**Llama.cpp Server** (`src/inference/llama_server.py`) - CPU inference:
- Uses `llama-cpp-python` bindings
- Loads GGUF format models (converted from Hugging Face)
- Supports GGUF LoRA adapters
- Synchronous API (simpler than vLLM)
- Configurable thread count and context length

**Backend Factory** (`src/inference/backend_factory.py`):
- Selects appropriate inference backend based on `config.model.inference_backend`
- Provides unified interface for both backends
- Handles async/sync compatibility

**Event Loop Architecture**: The system handles a complex async scenario where Gradio runs in its own event loop while vLLM's `AsyncLLMEngine` runs in the main event loop. Cross-loop communication uses:
- Global variable `_engine_event_loop` in `src/webui/app.py`
- `asyncio.run_coroutine_threadsafe()` for cross-loop calls
- `prevent_thread_lock=True` in Gradio launch to keep event loop alive

#### 3. Memory System (`src/memory/memory_manager.py`)
- **Memory Types**: context, character, plot, setting, dialogue
- Vector-based retrieval (RAG) using ChromaDB
- BGE-M3 embeddings for Chinese text
- Hybrid storage: ChromaDB for fast retrieval + JSON for persistence
- Long-term memory with automatic session summarization

#### 4. Training Module (`src/train/train_lora.py`)
- QLoRA 4-bit quantization via BitsAndBytes
- LoRA fine-tuning with configurable rank (default: r=64, alpha=128)
- Supports multiple data formats (TXT, JSON, JSONL)
- Automatic dataset preparation with sentence-boundary chunking
- Tensorboard logging integration

#### 5. Data Preparation (`src/data/prepare_data.py`)
- Multi-format support with automatic detection
- Text chunking with sentence boundary detection
- Train/validation split configuration
- Sample data generation for testing

### Key Technical Patterns

#### Memory-Augmented Generation Pipeline
1. User input → Gradio
2. Memory retrieval based on query (ChromaDB semantic search)
3. Prompt formatting with context/memories
4. Backend generation (vLLM or llama.cpp) with configurable sampling
5. Result post-processing and memory storage

#### Dual Backend Workflow (GPU Training + CPU Inference)

```
┌─────────────────────────────────────────────────────────────────────┐
│              GPU 机器 - 训练阶段                                      │
│                                                                       │
│  基础模型: Qwen/Qwen2.5-7B-Instruct  (Hugging Face 格式)             │
│                │                                                    │
│                ├─► LoRA 训练 (Transformers + PEFT)                  │
│                │                                                    │
│                ▼                                                    │
│  输出: ./training/final_model/                                   │
│        ├── adapter_config.json                                      │
│        └── adapter_model.safetensors  ← LoRA 权重                  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ 转换
┌─────────────────────────────────────────────────────────────────────┐
│              GPU 或 CPU 机器 - 模型转换                               │
│                                                                       │
│  1. 基础模型转换 (只需一次)                                          │
│     convert_hf_to_gguf.py → llama-quantize                          │
│     │                                                               │
│     ▼                                                               │
│     ./models/gguf/qwen2.5-7b-q5_k_m.gguf                            │
│                                                                       │
│  2. LoRA 转换 (每次训练后)                                           │
│     convert-lora-to-gguf.py                                         │
│     │                                                               │
│     ▼                                                               │
│     ./models/lora-gguf/                                             │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              CPU 机器 - 推理阶段                                      │
│                                                                       │
│  llama.cpp --model ./models/gguf/qwen2.5-7b-q5_k_m.gguf \            │
│            --lora ./models/lora-gguf                                 │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

#### Model Conversion Process
- **HF → GGUF**: Two-step process (format conversion → quantization)
- **LoRA → GGUF**: Direct conversion using llama.cpp tools
- See `scripts/convert_hf_to_gguf.sh` and `scripts/convert_lora_to_gguf.sh`

#### Event Loop Synchronization (Critical)
When modifying `src/inference/vllm_server.py` or `src/webui/app.py`:
- Gradio handlers run in Gradio's event loop
- vLLM engine runs in the main event loop
- Use `_engine_event_loop` global variable to track the engine's loop
- Use `asyncio.run_coroutine_threadsafe()` for cross-loop calls
- Never use `await` directly across event loop boundaries

## Dependencies

### Core ML Stack
- PyTorch 2.5.0+, Transformers 4.46.0+, PEFT 0.13.0+, BitsAndBytes 0.44.0+
- vLLM 0.6.0+ (GPU inference)
- llama-cpp-python 0.3.0+ (CPU inference)

### Memory & RAG
- ChromaDB 0.6.0+, Sentence Transformers 3.3.0+, LangChain 0.3.0+

### UI
- Gradio 4.44.0+

## Directory Layout

```
novel_ai_system/
├── config.py              # Central configuration (all settings here)
├── start.py               # CLI entry point
├── requirements.txt       # Python dependencies
│
├── scripts/               # Utility scripts
│   ├── install.sh         # Automated setup script
│   ├── watch_logs.sh      # Log monitoring
│   ├── package_model.sh   # Model packaging
│   ├── prepare_data_from_normal_txt.sh  # Data preparation
│   ├── convert_hf_to_gguf.sh
│   └── convert_lora_to_gguf.sh
│
├── data/
│   └── chroma_db/        # Vector database storage (runtime)
│
├── training/             # Training directory (reorganized structure)
│   ├── data/             # Training data
│   │   ├── raw/          # Source novel files
│   │   ├── train/        # Training datasets (JSONL)
│   │   └── val/          # Validation datasets (JSONL)
│   ├── checkpoints/      # Training checkpoints
│   │   ├── urban-life-*/ # Model checkpoints
│   │   └── experiments/  # Experimental models
│   ├── logs/             # Training logs
│   │   ├── tensorboard/
│   │   └── runs/
│
├── logs/                 # Debug logs
│
├── models/               # Models (reorganized structure)
│   ├── base/             # Base models (symlinks → HF cache)
│   ├── gguf/             # GGUF quantized models (CPU inference)
│   ├── lora-gguf/        # LoRA GGUF adapters
│   └── production/       # Production-ready models
│
├── packages/             # Package outputs
│   ├── releases/        # Official releases
│   └── archives/        # Archived versions
│
├── scripts/              # Conversion and utility scripts
│   ├── convert_hf_to_gguf.sh
│   ├── convert_lora_to_gguf.sh
│   └── reorganize_models.sh
│
└── src/
    ├── train/            # LoRA training
    ├── inference/        # vLLM/llama.cpp inference
    │   ├── backend_factory.py
    │   ├── vllm_server.py
    │   └── llama_server.py
    ├── memory/           # Memory management
    ├── data/             # Data processing
    └── webui/            # Gradio interface
```

## Important Notes

### Configuration
All configuration changes should be made in `config.py`. The system uses dataclasses for type-safe configuration.

### Event Loop Debugging
- Debug logs are written to `/logs/debug.log`
- Check event loop IDs when debugging async issues
- The README.md contains detailed troubleshooting for event loop conflicts

### GPU Memory Management
- 4-bit quantization is enabled by default to reduce VRAM usage
- Adjust `vllm_gpu_memory_utilization` if encountering OOM errors
- Kill old vLLM processes: `pkill -f "python3 start.py"`

### Model Support
The system is configured for Qwen2.5-7B-Instruct by default but supports any compatible model:
- Yi-1.5-9B-Chat
- GLM-4-9b-chat
- DeepSeek-V3

Change `model.base_model` in `config.py` to switch models.

### Data Formats
- **TXT**: Raw text files in `data/raw/`
- **JSON**: Array of objects with `text` field
- **JSONL**: One JSON object per line

### Testing
Always test with `python src/inference/test_inference.py` or `python start.py inference` before validating via WebUI.
