"""
配置文件 - 中文小说写作AI系统
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelConfig:
    """模型配置"""
    # 推理后端选择: "auto" (自动检测), "vllm" (GPU) 或 "llama_cpp" (CPU)
    inference_backend: str = "auto"

    # 基础模型选择 (推荐使用开源中文模型)
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"  # Hugging Face 格式，用于训练
    # 备选: "THUDM/glm-4-9b-chat", "01-ai/Yi-1.5-9B-Chat"

    # llama.cpp 配置 (CPU 推理)
    # 模型格式: "gguf" (量化, 推荐) 或 "hf" (非量化, Hugging Face 格式)
    llama_cpp_model_format: str = "gguf"
    # GGUF 模型路径 (量化模型, 内存占用小) - 重组后的路径
    llama_cpp_gguf_model: str = "./models/gguf/qwen2.5-7b-q5_k_m.gguf"
    # Hugging Face 模型路径 (非量化模型, 精度更高但内存占用大)
    llama_cpp_hf_model: str = "Qwen/Qwen2.5-7B-Instruct"
    # 量化级别 (仅当 model_format="gguf" 时有效): Q4_K_M, Q5_K_M, Q8_0, F16 等
    llama_cpp_quantization: str = "Q5_K_M"
    # LoRA 路径 - 重组后的路径
    llama_cpp_lora_path: Optional[str] = "./models/lora-gguf/urban-life-3b-lora.pth"
    llama_cpp_n_ctx: int = 16384  # 上下文长度（降低内存使用，提高响应速度）
    llama_cpp_n_threads: int = 4  # CPU 线程数（留2核给系统）
    llama_cpp_n_batch: int = 256  # 批处理大小（提高响应速度）
    llama_cpp_use_mmap: bool = True  # 使用内存映射
    llama_cpp_use_mlock: bool = False  # 使用内存锁定 (需要 root)
    llama_cpp_gpu_layers: int = 0  # GPU 层数 (0=全CPU, >0=部分GPU)

    # vLLM 推理配置 (GPU 推理)
    vllm_tensor_parallel_size: int = 1  # GPU数量
    vllm_gpu_memory_utilization: float = 0.90  # GPU显存使用率
    vllm_max_model_len: int = 32768  # 最大序列长度
    vllm_dtype: str = "bfloat16"  # 数据类型

    # LoRA 微调配置
    use_lora: bool = True
    lora_r: int = 64  # LoRA rank
    lora_alpha: int = 128  # LoRA alpha
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # 量化配置 (节省显存)
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据
    train_data_path: str = "training/data/train/train.jsonl"
    val_data_path: str = "training/data/val/val.jsonl"
    max_seq_length: int = 2048
    preprocessing_num_workers: int = 1

    # 训练参数
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # 优化器
    optimizer: str = "paged_adamw_32bit"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # 输出
    output_dir: str = "./training"
    checkpoint_dir: str = "./training"

    # 其他
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    ddp_find_unused_parameters: bool = False

@dataclass
class MemoryConfig:
    """记忆功能配置"""
    # 向量数据库
    chroma_persist_directory: str = "./data/chroma_db"
    embedding_model: str = "BAAI/bge-m3"  # 多语言embeddings

    # 记忆管理
    max_memory_items: int = 1000
    memory_retrieval_top_k: int = 5
    long_term_memory_path: str = "./data/long_term_memory.json"

    # RAG 配置
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50
    rag_top_k: int = 3

@dataclass
class WebUIConfig:
    """WebUI配置"""
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False
    auth: bool = False
    username: Optional[str] = None
    password: Optional[str] = None

    # 界面配置
    title: str = "中文小说写作AI系统"
    description: str = "基于vLLM + LoRA的智能小说创作助手"
    theme: str = "soft"

    # 生成参数
    max_new_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.15

@dataclass
class SystemConfig:
    """系统总配置"""
    # 路径配置
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    model_dir: Path = field(default_factory=lambda: Path("./models"))
    log_dir: Path = field(default_factory=lambda: Path("./logs"))

    # 子配置
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    webui: WebUIConfig = field(default_factory=WebUIConfig)

    # 系统设置
    seed: int = 42
    debug: bool = False
    log_level: str = "INFO"

    def __post_init__(self):
        """确保路径是绝对路径"""
        self.project_root = Path(self.project_root).absolute()
        self.data_dir = self.project_root / self.data_dir
        self.model_dir = self.project_root / self.model_dir
        self.log_dir = self.project_root / self.log_dir

        # 创建必要的目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

# 全局配置实例
config = SystemConfig()
