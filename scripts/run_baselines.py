"""Baseline实验脚本

运行SFT、DPO、KTO等baseline实验用于对比。
评估时对比人工摘要，与MFRL使用相同的测试集。
"""

import os
import sys
import json
import argparse
import yaml
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig, KTOTrainer, KTOConfig
from datasets import Dataset
from src.eval.evaluator import Evaluator


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(data_path: str):
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_model(model_name, dtype="bfloat16"):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype_map.get(dtype, torch.bfloat16),
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def setup_lora(model, r=8, alpha=16, dropout=0.05):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    return get_peft_model(model, lora_config)


def run_sft_baseline(config, train_data, output_dir):
    """SFT基线：只用chosen数据训练。"""
    print("\n" + "="*50)
    print("Running SFT Baseline")
    print("="*50)

    model, tokenizer = setup_model(config["model"]["name"], config["model"]["dtype"])
    model = setup_lora(model, config["lora"]["r"], config["lora"]["alpha"])

    # 为SFT构造dummy rejected（长度为0的文本）
    sft_dpo_data = [
        {"prompt": item["prompt"], "chosen": item["chosen"], "rejected": ""}
        for item in train_data
    ]
    sft_dataset = Dataset.from_list(sft_dpo_data)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        learning_rate=config["dpo"]["learning_rate"],
        num_train_epochs=config["dpo"]["num_epochs"],
        per_device_train_batch_size=config["dpo"]["batch_size"],
        gradient_accumulation_steps=config["dpo"]["gradient_accumulation"],
        max_length=config["dpo"]["max_length"],
        max_prompt_length=config["dpo"]["max_prompt_length"],
        beta=0.0,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=(config["model"]["dtype"] == "bfloat16"),
        fp16=(config["model"]["dtype"] == "float16"),
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    return model, tokenizer


def run_dpo_baseline(config, train_data, output_dir):
    """标准DPO基线（无多形式反馈融合）。"""
    print("\n" + "="*50)
    print("Running Standard DPO Baseline")
    print("="*50)

    model, tokenizer = setup_model(config["model"]["name"], config["model"]["dtype"])
    model = setup_lora(model, config["lora"]["r"], config["lora"]["alpha"])

    dataset = Dataset.from_list(train_data)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        learning_rate=config["dpo"]["learning_rate"],
        num_train_epochs=config["dpo"]["num_epochs"],
        per_device_train_batch_size=config["dpo"]["batch_size"],
        gradient_accumulation_steps=config["dpo"]["gradient_accumulation"],
        max_length=config["dpo"]["max_length"],
        max_prompt_length=config["dpo"]["max_prompt_length"],
        beta=config["dpo"]["beta"],
        loss_type=config["dpo"]["loss_type"],
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=(config["model"]["dtype"] == "bfloat16"),
        fp16=(config["model"]["dtype"] == "float16"),
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    return model, tokenizer


def run_kto_baseline(config, train_data, output_dir):
    """KTO基线（只需要好/坏标签，不需要成对偏好）。"""
    print("\n" + "="*50)
    print("Running KTO Baseline")
    print("="*50)

    model, tokenizer = setup_model(config["model"]["name"], config["model"]["dtype"])
    model = setup_lora(model, config["lora"]["r"], config["lora"]["alpha"])

    # KTO数据格式：prompt + completion + label（True/False）
    kto_data = []
    for item in train_data:
        kto_data.append({
            "prompt": item["prompt"],
            "completion": item["chosen"],
            "label": True,
        })
        kto_data.append({
            "prompt": item["prompt"],
            "completion": item["rejected"],
            "label": False,
        })
    kto_dataset = Dataset.from_list(kto_data)

    kto_config = KTOConfig(
        output_dir=output_dir,
        learning_rate=config["dpo"]["learning_rate"],
        num_train_epochs=config["dpo"]["num_epochs"],
        per_device_train_batch_size=config["dpo"]["batch_size"],
        gradient_accumulation_steps=config["dpo"]["gradient_accumulation"],
        max_length=config["dpo"]["max_length"],
        max_prompt_length=config["dpo"]["max_prompt_length"],
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=(config["model"]["dtype"] == "bfloat16"),
        fp16=(config["model"]["dtype"] == "float16"),
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = KTOTrainer(
        model=model,
        args=kto_config,
        train_dataset=kto_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    model.save_pretrained(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    return model, tokenizer


def evaluate_all(config, baselines, test_data):
    """评估所有baseline，对比人工摘要。"""
    results = {}

    for name, (model, tokenizer) in baselines.items():
        print(f"\nEvaluating {name}...")
        evaluator = Evaluator(model, tokenizer)
        rouge_scores = evaluator.evaluate_rouge(test_data, reference_key="reference")
        results[name] = rouge_scores
        print(f"  ROUGE: {rouge_scores}")

    # 保存对比结果
    output_dir = config["output"]["dir"]
    with open(os.path.join(output_dir, "baseline_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印对比表
    print("\n" + "="*60)
    print("Baseline Comparison")
    print("="*60)
    print(f"{'Model':<20} {'ROUGE-1':<12} {'ROUGE-2':<12} {'ROUGE-L':<12}")
    print("-"*60)
    for name, scores in results.items():
        print(f"{name:<20} {scores['rouge1']:<12.4f} {scores['rouge2']:<12.4f} {scores['rougeL']:<12.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MFRL Baseline Experiments")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to preference data JSON (must contain reference field)")
    parser.add_argument("--baselines", nargs="+",
                        default=["sft", "dpo", "kto"],
                        choices=["sft", "dpo", "kto"])
    args = parser.parse_args()

    config = load_config(args.config)
    dpo_data = load_data(args.data)

    # 验证数据包含reference字段
    if "reference" not in dpo_data[0]:
        print("ERROR: preference data missing 'reference' field. Regenerate with updated run_experiment.py")
        return

    # 划分数据（与MFRL相同的80/20划分）
    test_size = config["data"]["max_test_samples"]
    train_data = dpo_data[:-test_size]
    test_data = dpo_data[-test_size:]

    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    baselines = {}

    if "sft" in args.baselines:
        sft_dir = os.path.join(config["output"]["dir"], "sft")
        os.makedirs(sft_dir, exist_ok=True)
        model, tokenizer = run_sft_baseline(config, train_data, sft_dir)
        baselines["sft"] = (model, tokenizer)

    if "dpo" in args.baselines:
        dpo_dir = os.path.join(config["output"]["dir"], "dpo")
        os.makedirs(dpo_dir, exist_ok=True)
        model, tokenizer = run_dpo_baseline(config, train_data, dpo_dir)
        baselines["dpo"] = (model, tokenizer)

    if "kto" in args.baselines:
        kto_dir = os.path.join(config["output"]["dir"], "kto")
        os.makedirs(kto_dir, exist_ok=True)
        model, tokenizer = run_kto_baseline(config, train_data, kto_dir)
        baselines["kto"] = (model, tokenizer)

    # 评估对比（对比人工摘要）
    evaluate_all(config, baselines, test_data)


if __name__ == "__main__":
    main()
