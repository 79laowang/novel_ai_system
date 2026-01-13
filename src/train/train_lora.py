"""
LoRA/QLoRA 微调训练脚本
支持中文小说写作模型的训练
"""
import os
import sys
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import json

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset, load_dataset
from bitsandbytes.optim import AdamW8bit

# 第三方库
import tqdm
from rich import print as rprint
from rich.console import Console
from rich.table import Table

console = Console()


class NovelTrainer:
    """小说模型训练器"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        rprint("[bold cyan]正在加载模型和tokenizer...[/bold cyan]")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model,
            trust_remote_code=True,
            use_fast=True,
        )
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 加载模型
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }

        # 量化配置
        if self.config.model.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type=self.config.model.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
        elif self.config.model.load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model,
            **model_kwargs
        )

        # 准备kbit训练
        if self.config.model.load_in_4bit or self.config.model.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        rprint(f"[green]✓ 模型加载完成: {self.config.model.base_model}[/green]")

    def setup_lora(self):
        """配置LoRA"""
        rprint("[bold cyan]正在配置LoRA...[/bold cyan]")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.model.lora_r,
            lora_alpha=self.config.model.lora_alpha,
            lora_dropout=self.config.model.lora_dropout,
            target_modules=self.config.model.lora_target_modules,
            bias="none",
            inference_mode=False,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        rprint(f"[green]✓ LoRA配置完成[/green]")

    def load_and_preprocess_data(self, data_path: str) -> Dataset:
        """加载和预处理数据"""
        rprint(f"[bold cyan]正在加载数据: {data_path}[/bold cyan]")

        # 支持多种数据格式
        path = Path(data_path)

        if path.is_file() and path.suffix == '.json':
            # JSON格式: {"text": "..."}
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                dataset = Dataset.from_list(data)
            else:
                dataset = Dataset.from_dict(data)
        elif path.is_file() and path.suffix == '.jsonl':
            # JSONL格式
            dataset = load_dataset('json', data_files=str(path), split='train')
        elif path.is_dir():
            # 目录模式: 读取所有txt文件
                all_texts = []
                for txt_file in path.glob('**/*.txt'):
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            all_texts.append({'text': text})
                dataset = Dataset.from_list(all_texts)
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")

        rprint(f"[green]✓ 数据加载完成: {len(dataset)} 条记录[/green]")
        return dataset

    def tokenize_function(self, examples):
        """数据tokenization"""
        # 构建对话格式的prompt
        texts = []
        for text in examples['text']:
            # 使用Qwen格式的prompt
            prompt = f"<|im_start|>user\n请继续创作以下小说内容<|im_end|>\n<|im_start|>assistant\n{text}<|im_end|>"
            texts.append(prompt)

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config.training.max_seq_length,
            padding=False,
            return_tensors=None,
        )

        return tokenized

    def prepare_datasets(self):
        """准备训练和验证数据集"""
        rprint("[cyan]正在tokenize训练数据...[/cyan]")
        train_dataset = self.load_and_preprocess_data(self.config.training.train_data_path)
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing training data",
        )

        # 如果有验证数据
        val_dataset = None
        if self.config.training.val_data_path and Path(self.config.training.val_data_path).exists():
            rprint("[cyan]正在tokenize验证数据...[/cyan]")
            val_dataset = self.load_and_preprocess_data(self.config.training.val_data_path)
            val_dataset = val_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=val_dataset.column_names,
                desc="Tokenizing validation data",
            )

        return train_dataset, val_dataset

    def setup_trainer(self, train_dataset, val_dataset=None):
        """设置训练器"""
        rprint("[bold cyan]正在配置训练器...[/bold cyan]")

        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            warmup_steps=self.config.training.warmup_steps,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            eval_steps=self.config.training.eval_steps if val_dataset else None,
            eval_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            save_total_limit=3,
            bf16=self.config.training.bf16,
            fp16=self.config.training.fp16,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            optim=self.config.training.optimizer,
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.max_grad_norm,
            report_to=["tensorboard"],
            logging_dir=str(self.config.log_dir / "tensorboard"),
            ddp_find_unused_parameters=self.config.training.ddp_find_unused_parameters,
            remove_unused_columns=False,
        )

        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )

        # 创建trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        rprint(f"[green]✓ 训练器配置完成[/green]")

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """开始训练"""
        rprint("[bold green]开始训练...[/bold green]")

        # 显示训练信息
        table = Table(title="训练配置")
        table.add_column("参数", style="cyan")
        table.add_column("值", style="green")
        table.add_row("基础模型", self.config.model.base_model)
        table.add_row("训练轮数", str(self.config.training.num_train_epochs))
        table.add_row("批次大小", str(self.config.training.per_device_train_batch_size))
        table.add_row("梯度累积", str(self.config.training.gradient_accumulation_steps))
        table.add_row("学习率", str(self.config.training.learning_rate))
        table.add_row("有效批次大小", str(self.config.training.per_device_train_batch_size * self.config.training.gradient_accumulation_steps * torch.cuda.device_count()))
        if resume_from_checkpoint:
            table.add_row("恢复检查点", resume_from_checkpoint)
        console.print(table)

        # 开始训练
        if resume_from_checkpoint:
            rprint(f"[yellow]从检查点恢复训练: {resume_from_checkpoint}[/yellow]")
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            self.trainer.train()

        rprint("[bold green]训练完成！[/bold green]")

    def save_model(self, output_dir: Optional[str] = None):
        """保存模型"""
        if output_dir:
            save_path = Path(output_dir)
        else:
            save_path = Path(self.config.training.output_dir) / "final_model"
        save_path.mkdir(parents=True, exist_ok=True)

        rprint(f"[bold cyan]正在保存模型到: {save_path}[/bold cyan]")

        # 保存LoRA权重
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        rprint(f"[green]✓ 模型已保存到: {save_path}[/green]")


def main(
    resume_from_checkpoint: Optional[str] = None,
    train_data_path: Optional[str] = None,
    val_data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    num_train_epochs: Optional[int] = None,
    per_device_train_batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
):
    """主函数

    Args:
        resume_from_checkpoint: 从checkpoint恢复训练
        train_data_path: 训练数据路径 (JSONL)
        val_data_path: 验证数据路径 (JSONL)
        output_dir: 输出目录
        checkpoint_dir: 检查点目录
        num_train_epochs: 训练轮数
        per_device_train_batch_size: 批次大小
        learning_rate: 学习率
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="从checkpoint恢复训练")
    parser.add_argument("--train-data", type=str, default=None, help="训练数据路径")
    parser.add_argument("--val-data", type=str, default=None, help="验证数据路径")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="检查点目录")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="批次大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    args, _ = parser.parse_known_args()

    # 命令行参数覆盖
    resume_from_checkpoint = args.resume or resume_from_checkpoint
    train_data_path = args.train_data or train_data_path
    val_data_path = args.val_data or val_data_path
    output_dir = args.output_dir or output_dir
    checkpoint_dir = args.checkpoint_dir or checkpoint_dir
    num_train_epochs = args.epochs or num_train_epochs
    per_device_train_batch_size = args.batch_size or per_device_train_batch_size
    learning_rate = args.lr or learning_rate

    from config import config

    # 覆盖配置值
    if train_data_path:
        config.training.train_data_path = train_data_path
        rprint(f"[cyan]使用训练数据: {train_data_path}[/cyan]")
    if val_data_path:
        config.training.val_data_path = val_data_path
        rprint(f"[cyan]使用验证数据: {val_data_path}[/cyan]")
    if output_dir:
        config.training.output_dir = output_dir
        rprint(f"[cyan]输出目录: {output_dir}[/cyan]")
    if checkpoint_dir:
        config.training.checkpoint_dir = checkpoint_dir
        rprint(f"[cyan]检查点目录: {checkpoint_dir}[/cyan]")
    if num_train_epochs:
        config.training.num_train_epochs = num_train_epochs
        rprint(f"[cyan]训练轮数: {num_train_epochs}[/cyan]")
    if per_device_train_batch_size:
        config.training.per_device_train_batch_size = per_device_train_batch_size
        rprint(f"[cyan]批次大小: {per_device_train_batch_size}[/cyan]")
    if learning_rate:
        config.training.learning_rate = learning_rate
        rprint(f"[cyan]学习率: {learning_rate}[/cyan]")

    # 创建训练器
    trainer = NovelTrainer(config)

    # 加载模型和tokenizer
    trainer.load_model_and_tokenizer()

    # 配置LoRA
    trainer.setup_lora()

    # 准备数据集
    train_dataset, val_dataset = trainer.prepare_datasets()

    # 设置训练器
    trainer.setup_trainer(train_dataset, val_dataset)

    # 开始训练
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 保存模型
    trainer.save_model()


if __name__ == "__main__":
    main()
