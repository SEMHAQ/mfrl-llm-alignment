"""MFRL DPO训练器

基于trl库的DPOTrainer，集成多形式反馈融合和课程学习策略。
支持LoRA高效微调，适配单卡3090环境。
"""

import os
import torch
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig


@dataclass
class MFRLConfig:
    """MFRL训练配置。"""

    # 模型配置
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_dtype: str = "bfloat16"
    trust_remote_code: bool = True

    # LoRA配置
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )

    # DPO配置
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"
    learning_rate: float = 5e-6
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    max_prompt_length: int = 256

    # 随机种子
    seed: int = 42

    # 课程学习
    curriculum_enabled: bool = True
    curriculum_warmup_epochs: float = 0.5

    # 输出
    output_dir: str = "outputs/mfrl"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"


class MFRLTrainer:
    """MFRL训练器。

    封装了模型加载、LoRA配置、DPO训练的完整流程。
    """

    def __init__(self, config: MFRLConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self):
        """加载模型和分词器，配置LoRA。"""
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=dtype_map.get(self.config.model_dtype, torch.bfloat16),
            trust_remote_code=self.config.trust_remote_code,
        )

        # 配置LoRA
        torch.manual_seed(self.config.seed)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def train(self, train_dataset, eval_dataset=None):
        """执行DPO训练。

        Args:
            train_dataset: 训练数据集，需包含 prompt, chosen, rejected 字段
            eval_dataset: 评估数据集（可选）
        """
        training_args = DPOConfig(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            beta=self.config.dpo_beta,
            loss_type=self.config.dpo_loss_type,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            eval_strategy="no",
            bf16=(self.config.model_dtype == "bfloat16"),
            fp16=(self.config.model_dtype == "float16"),
            gradient_checkpointing=True,
            data_seed=self.config.seed,
            report_to="none",
            remove_unused_columns=False,
        )

        self.trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        self.trainer.train()

    def save(self, path: Optional[str] = None):
        """保存模型。"""
        save_path = path or os.path.join(self.config.output_dir, "final")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
