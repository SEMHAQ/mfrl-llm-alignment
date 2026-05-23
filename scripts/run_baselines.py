"""Baseline实验脚本

运行SFT、DPO、KTO等baseline实验用于对比。
评估时对比人工摘要，与MFRL使用相同的测试集。
"""

import os
import sys
import gc
import json
import argparse
import yaml
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig, KTOTrainer, KTOConfig, SFTTrainer, SFTConfig
from datasets import Dataset
from src.eval.evaluator import Evaluator

META_PROMPTS = {
    "lcsts": "请为以下文本生成简洁准确的摘要：\n{input}\n摘要：",
    "alpaca_chinese": "{input}\n请回答以上指令。",
}

def get_prompt(input_text: str, dataset_name: str = "lcsts") -> str:
    fmt = META_PROMPTS.get(dataset_name, META_PROMPTS["lcsts"])
    return fmt.format(input=input_text)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_model(model_name, dtype="float16"):
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
        dtype=dtype_map.get(dtype, torch.float16),
        device_map={"": 0},
        trust_remote_code=True,
    )
    return model, tokenizer


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


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


def run_baseline(config, train_data, test_data, output_dir, baseline_type):
    """运行单个baseline实验。"""
    print(f"\n{'='*50}")
    print(f"Running {baseline_type} Baseline")
    print(f"{'='*50}")

    model, tokenizer = setup_model(config["model"]["name"], config["model"]["dtype"])
    model = setup_lora(model, config["lora"]["r"], config["lora"]["alpha"])

    if baseline_type == "sft":
        sft_data = []
        for item in train_data:
            text = item["prompt"] + item["chosen"]
            sft_data.append({"text": text})
        dataset = Dataset.from_list(sft_data)
        sft_config = SFTConfig(
            output_dir=output_dir,
            learning_rate=config["dpo"]["learning_rate"],
            num_train_epochs=config["dpo"]["num_epochs"],
            per_device_train_batch_size=config["dpo"]["batch_size"],
            gradient_accumulation_steps=config["dpo"]["gradient_accumulation"],
            max_length=config["dpo"]["max_length"],
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="no",
            bf16=(config["model"]["dtype"] == "bfloat16"),
            fp16=(config["model"]["dtype"] == "float16"),
            gradient_checkpointing=True,
            report_to="none",
            dataset_text_field="text",
        )
        trainer = SFTTrainer(model=model, args=sft_config, train_dataset=dataset, processing_class=tokenizer)

    elif baseline_type == "dpo":
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
        trainer = DPOTrainer(model=model, args=dpo_config, train_dataset=dataset, processing_class=tokenizer)

    elif baseline_type == "kto":
        kto_data = []
        for item in train_data:
            kto_data.append({"prompt": item["prompt"], "completion": item["chosen"], "label": True})
            kto_data.append({"prompt": item["prompt"], "completion": item["rejected"], "label": False})
        dataset = Dataset.from_list(kto_data)
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
        trainer = KTOTrainer(model=model, args=kto_config, train_dataset=dataset, processing_class=tokenizer)

    trainer.train()

    # 评估
    print(f"\nEvaluating {baseline_type}...")
    evaluator = Evaluator(model, tokenizer)
    rouge_scores = evaluator.evaluate_rouge(test_data, reference_key="reference")
    print(f"  ROUGE: {rouge_scores}")

    # 释放显存
    unload_model(model, tokenizer)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return rouge_scores


def main():
    parser = argparse.ArgumentParser(description="MFRL Baseline Experiments")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--baselines", nargs="+",
                        default=["sft", "dpo", "kto"],
                        choices=["sft", "dpo", "kto"])
    args = parser.parse_args()

    config = load_config(args.config)

    # 加载原始数据（与MFRL相同的数据源）
    dataset_name = config["data"]["dataset"]
    local_json = os.path.join("data", f"{dataset_name}_train.json")
    with open(local_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    max_samples = config["data"]["max_train_samples"]
    if max_samples and len(raw_data) > max_samples:
        raw_data = raw_data[:max_samples]

    # 划分训练集和测试集（与MFRL相同的80/20划分）
    test_size = config["data"]["max_test_samples"]
    test_raw = raw_data[-test_size:]

    # 加载偏好数据（用于训练）
    preference_cache = os.path.join(config["output"]["dir"], "preference_data.json")
    with open(preference_cache, "r", encoding="utf-8") as f:
        dpo_data = json.load(f)

    print(f"Raw data: {len(raw_data)}, Train preference pairs: {len(dpo_data)}, Test: {len(test_raw)}")

    # 构造测试集（对比人工摘要）
    test_data = []
    for item in test_raw:
        prompt = get_prompt(item["input"], dataset_name)
        test_data.append({"prompt": prompt, "reference": item["reference"]})

    # 运行baselines
    output_dir = config["output"]["dir"]
    results = {}

    for baseline_type in args.baselines:
        baseline_dir = os.path.join(output_dir, baseline_type)
        os.makedirs(baseline_dir, exist_ok=True)
        rouge_scores = run_baseline(config, dpo_data, test_data, baseline_dir, baseline_type)
        results[baseline_type] = rouge_scores

    # 保存对比结果
    with open(os.path.join(output_dir, "baseline_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印对比表
    print(f"\n{'='*60}")
    print("Baseline Comparison")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'ROUGE-1':<12} {'ROUGE-2':<12} {'ROUGE-L':<12}")
    print("-"*60)
    for name, scores in results.items():
        print(f"{name:<20} {scores['rouge1']:<12.4f} {scores['rouge2']:<12.4f} {scores['rougeL']:<12.4f}")


if __name__ == "__main__":
    main()
