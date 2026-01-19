# WebUI 启动命令参考

## 动态模型切换 (推荐)

无需修改 `config.py`，直接通过命令行参数切换不同模型。

### 基础语法

```bash
python start.py webui --gguf-model <GGUF模型路径> --lora-path <LoRA路径> [其他选项]
```

### 快速启动示例

#### 1.5B 模型 - 最快启动
```bash
python start.py webui \
  --gguf-model ./models/qwen2.5-1.5b-q5_k_m.gguf \
  --lora-path ./models/lora-gguf/urban-life-1.5b-lora.pth
```
- 启动时间: ~0.9秒
- 内存占用: ~3GB
- 适用: 快速测试、开发调试

#### 3B 模型 - 性能平衡 (推荐)
```bash
python start.py webui \
  --gguf-model ./models/qwen2.5-3b-q5_k_m.gguf \
  --lora-path ./models/lora-gguf/urban-life-3b-lora.pth
```
- 启动时间: ~1.4秒
- 内存占用: ~5GB
- 适用: 日常使用、质量与速度平衡

#### 7B 模型 - 最佳质量
```bash
python start.py webui \
  --gguf-model ./models/qwen2.5-7b-q5_k_m.gguf \
  --lora-path ./models/lora-gguf/urban-life-7b-lora.pth
```
- 启动时间: ~2.5秒
- 内存占用: ~10GB
- 适用: 高质量生成

#### 不使用 LoRA (基础模型)
```bash
python start.py webui --gguf-model ./models/qwen2.5-1.5b-q5_k_m.gguf
```

## 完整命令选项

```bash
python start.py webui [OPTIONS]

OPTIONS:
  --gguf-model PATH       GGUF模型文件路径 (动态切换模型)
  --lora-path PATH        LoRA文件路径 (动态切换LoRA)
  --base-model MODEL      基础模型名称 (HuggingFace格式)
  --model-format FORMAT   CPU推理模型格式: gguf 或 hf
  --lora PATH             LoRA权重路径 (兼容旧参数)
  --host ADDR             服务器地址 (默认: 0.0.0.0)
  --port PORT             端口 (默认: 7860)
  --share                 创建公共链接
```

## 常用组合示例

### 结合 host 和 port
```bash
python start.py webui \
  --gguf-model ./models/qwen2.5-3b-q5_k_m.gguf \
  --lora-path ./models/lora-gguf/urban-life-3b-lora.pth \
  --host 0.0.0.0 \
  --port 7860
```

### 创建公共链接
```bash
python start.py webui \
  --gguf-model ./models/qwen2.5-1.5b-q5_k_m.gguf \
  --lora-path ./models/lora-gguf/urban-life-1.5b-lora.pth \
  --share
```

### 使用 HuggingFace 模型格式
```bash
python start.py webui \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --model-format hf
```

## 模型文件对应关系

| 模型大小 | GGUF 模型文件 | LoRA 文件 |
|---------|--------------|-----------|
| 1.5B    | `qwen2.5-1.5b-q5_k_m.gguf` | `urban-life-1.5b-lora.pth` |
| 3B      | `qwen2.5-3b-q5_k_m.gguf` | `urban-life-3b-lora.pth` |
| 7B      | `qwen2.5-7b-q5_k_m.gguf` | `urban-life-7b-lora.pth` |

## 文件路径说明

- GGUF 模型默认位置: `./models/`
- LoRA 适配器默认位置: `./models/lora-gguf/`
- 相对路径从项目根目录开始

## 查看帮助

```bash
# 查看完整的 webui 命令帮助
python start.py webui --help

# 查看所有命令的帮助
python start.py --help
```
