"""通宵实验脚本

运行全部实验，结果自动保存。预计8-9小时。

用法：
    python scripts/overnight_experiments.py
"""

import os
import sys
import gc
import json
import yaml
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.feedback.rule_feedback import RuleFeedback
from src.feedback.feedback_fusion import AdaptiveFeedbackFusion
from src.trainer.dpo_trainer import MFRLTrainer, MFRLConfig
from src.data.preprocess import generate_candidates
from src.eval.evaluator import Evaluator
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig, KTOTrainer, KTOConfig
from datasets import Dataset


def load_config():
    with open("configs/train.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(config):
    dataset_name = config["data"]["dataset"]
    local_json = os.path.join("data", f"{dataset_name}_train.json")
    with open(local_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    max_samples = config["data"]["max_train_samples"]
    if max_samples and len(raw_data) > max_samples:
        raw_data = raw_data[:max_samples]
    return raw_data


def setup_model(model_name, dtype="float16"):
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype_map.get(dtype, torch.float16),
        device_map="auto", trust_remote_code=True,
    )
    return model, tokenizer


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def setup_lora(model, config):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora"]["r"], lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"], bias="none",
    )
    return get_peft_model(model, lora_config)


def generate_preference_data(config, train_raw, feedback_type="rule"):
    """生成偏好数据。"""
    num_candidates = config["feedback"]["num_candidates"]
    gen_config = config.get("generation", {})
    temp = gen_config.get("temperature", 0.8)
    shared_cache = os.path.join("data", f"candidates_k{num_candidates}_t{temp}.json")

    inputs = [item["input"] for item in train_raw]
    references = [item["reference"] for item in train_raw]

    if os.path.exists(shared_cache):
        with open(shared_cache, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) >= len(inputs):
            candidates_list = cached[:len(inputs)]
        else:
            print(f"  Cache insufficient ({len(cached)} < {len(inputs)}), generating more...")
            model, tokenizer = setup_model(config["model"]["name"], config["model"]["dtype"])
            extra = generate_candidates(model, tokenizer, inputs[len(cached):],
                num_candidates=num_candidates, temperature=temp,
                top_p=gen_config.get("top_p", 0.95),
                max_new_tokens=gen_config.get("max_new_tokens", 128))
            unload_model(model, tokenizer)
            candidates_list = cached + extra
            with open(shared_cache, "w", encoding="utf-8") as f:
                json.dump(candidates_list, f, ensure_ascii=False, indent=2)
    else:
        print("  Generating candidates from scratch...")
        model, tokenizer = setup_model(config["model"]["name"], config["model"]["dtype"])
        candidates_list = generate_candidates(model, tokenizer, inputs,
            num_candidates=num_candidates, temperature=temp,
            top_p=gen_config.get("top_p", 0.95),
            max_new_tokens=gen_config.get("max_new_tokens", 128))
        unload_model(model, tokenizer)
        with open(shared_cache, "w", encoding="utf-8") as f:
            json.dump(candidates_list, f, ensure_ascii=False, indent=2)

    rule_feedback = RuleFeedback(
        metrics=config["feedback"]["rule"]["metrics"],
        weights=config["feedback"]["rule"]["weights"],
    )
    rule_pairs = rule_feedback.generate_preference_pairs(inputs, references, candidates_list)

    dpo_data = []
    for pair in rule_pairs:
        prompt = f"请为以下文本生成简洁准确的摘要：\n{pair['input']}\n摘要："
        dpo_data.append({
            "prompt": prompt, "chosen": pair["chosen"],
            "rejected": pair["rejected"], "reference": pair["reference"],
            "score_diff": pair.get("score_diff", 0.0),
        })

    min_diff = config["feedback"].get("min_score_diff", 0.0)
    if min_diff > 0:
        before = len(dpo_data)
        dpo_data = [d for d in dpo_data if d["score_diff"] >= min_diff]
        print(f"  Filtered: {before} -> {len(dpo_data)} pairs")

    return dpo_data


def train_and_evaluate(config, dpo_data, test_data, output_dir, label):
    """训练并评估。"""
    print(f"\n{'='*50}")
    print(f"Training: {label}")
    print(f"{'='*50}")

    split_idx = int(len(dpo_data) * 0.9)
    train_dataset = Dataset.from_list(dpo_data[:split_idx])
    eval_dataset = Dataset.from_list(dpo_data[split_idx:])

    mfrl_config = MFRLConfig(
        model_name=config["model"]["name"],
        model_dtype=config["model"]["dtype"],
        lora_r=config["lora"]["r"], lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        lora_target_modules=config["lora"]["target_modules"],
        dpo_beta=config["dpo"]["beta"], dpo_loss_type=config["dpo"]["loss_type"],
        learning_rate=config["dpo"]["learning_rate"],
        num_train_epochs=config["dpo"]["num_epochs"],
        per_device_train_batch_size=config["dpo"]["batch_size"],
        gradient_accumulation_steps=config["dpo"]["gradient_accumulation"],
        max_length=config["dpo"]["max_length"],
        max_prompt_length=config["dpo"]["max_prompt_length"],
        output_dir=output_dir,
    )

    trainer = MFRLTrainer(mfrl_config)
    trainer.setup_model()
    trainer.train(train_dataset, eval_dataset)
    trainer.save()

    print(f"\nEvaluating {label}...")
    evaluator = Evaluator(trainer.model, trainer.tokenizer)
    rouge_scores = evaluator.evaluate_rouge(test_data, reference_key="reference")
    print(f"  ROUGE: {rouge_scores}")

    unload_model(trainer.model, trainer.tokenizer)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return rouge_scores


def run_baseline(config, train_data, test_data, output_dir, baseline_type):
    """运行单个baseline。"""
    print(f"\n{'='*50}")
    print(f"Baseline: {baseline_type}")
    print(f"{'='*50}")

    model, tokenizer = setup_model(config["model"]["name"], config["model"]["dtype"])
    model = setup_lora(model, config)

    if baseline_type == "sft":
        sft_data = [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": ""}
                     for p in train_data]
        dataset = Dataset.from_list(sft_data)
        trainer_config = DPOConfig(
            output_dir=output_dir, learning_rate=config["dpo"]["learning_rate"],
            num_train_epochs=config["dpo"]["num_epochs"],
            per_device_train_batch_size=config["dpo"]["batch_size"],
            gradient_accumulation_steps=config["dpo"]["gradient_accumulation"],
            max_length=config["dpo"]["max_length"],
            max_prompt_length=config["dpo"]["max_prompt_length"],
            beta=0.0, logging_steps=10, save_strategy="epoch", eval_strategy="no",
            fp16=True, gradient_checkpointing=True, report_to="none",
        )
        trainer = DPOTrainer(model=model, args=trainer_config, train_dataset=dataset, processing_class=tokenizer)

    elif baseline_type == "dpo":
        dataset = Dataset.from_list(train_data)
        trainer_config = DPOConfig(
            output_dir=output_dir, learning_rate=config["dpo"]["learning_rate"],
            num_train_epochs=config["dpo"]["num_epochs"],
            per_device_train_batch_size=config["dpo"]["batch_size"],
            gradient_accumulation_steps=config["dpo"]["gradient_accumulation"],
            max_length=config["dpo"]["max_length"],
            max_prompt_length=config["dpo"]["max_prompt_length"],
            beta=config["dpo"]["beta"], loss_type=config["dpo"]["loss_type"],
            logging_steps=10, save_strategy="epoch", eval_strategy="no",
            fp16=True, gradient_checkpointing=True, report_to="none",
        )
        trainer = DPOTrainer(model=model, args=trainer_config, train_dataset=dataset, processing_class=tokenizer)

    elif baseline_type == "kto":
        kto_data = []
        for p in train_data:
            kto_data.append({"prompt": p["prompt"], "completion": p["chosen"], "label": True})
            kto_data.append({"prompt": p["prompt"], "completion": p["rejected"], "label": False})
        dataset = Dataset.from_list(kto_data)
        trainer_config = KTOConfig(
            output_dir=output_dir, learning_rate=config["dpo"]["learning_rate"],
            num_train_epochs=config["dpo"]["num_epochs"],
            per_device_train_batch_size=config["dpo"]["batch_size"],
            gradient_accumulation_steps=config["dpo"]["gradient_accumulation"],
            max_length=config["dpo"]["max_length"],
            max_prompt_length=config["dpo"]["max_prompt_length"],
            logging_steps=10, save_strategy="epoch", eval_strategy="no",
            fp16=True, gradient_checkpointing=True, report_to="none",
        )
        trainer = KTOTrainer(model=model, args=trainer_config, train_dataset=dataset, processing_class=tokenizer)

    trainer.train()

    print(f"\nEvaluating {baseline_type}...")
    evaluator = Evaluator(model, tokenizer)
    rouge_scores = evaluator.evaluate_rouge(test_data, reference_key="reference")
    print(f"  ROUGE: {rouge_scores}")

    unload_model(model, tokenizer)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return rouge_scores


def main():
    start_time = datetime.now()
    print(f"===== Overnight experiments started at {start_time} =====")

    config = load_config()
    raw_data = load_raw_data(config)
    test_size = config["data"]["max_test_samples"]
    train_raw = raw_data[:-test_size]
    test_raw = raw_data[-test_size:]

    test_data = []
    for item in test_raw:
        prompt = f"请为以下文本生成简洁准确的摘要：\n{item['input']}\n摘要："
        test_data.append({"prompt": prompt, "reference": item["reference"]})

    all_results = {}

    # ============================================================
    # Experiment 1: MFRL (no filtering, 3 epochs)
    # ============================================================
    print("\n\n" + "#"*60)
    print("# Experiment 1: MFRL (no filter, 3 epochs)")
    print("#"*60)
    config["feedback"]["min_score_diff"] = 0.0
    config["dpo"]["num_epochs"] = 3
    os.makedirs("outputs/overnight/mfrl_3ep", exist_ok=True)
    dpo_data = generate_preference_data(config, train_raw)
    all_results["mfrl_3ep"] = train_and_evaluate(
        config, dpo_data, test_data, "outputs/overnight/mfrl_3ep", "MFRL 3 epochs")

    # ============================================================
    # Experiment 2: MFRL (no filtering, 5 epochs)
    # ============================================================
    print("\n\n" + "#"*60)
    print("# Experiment 2: MFRL (no filter, 5 epochs)")
    print("#"*60)
    config["dpo"]["num_epochs"] = 5
    os.makedirs("outputs/overnight/mfrl_5ep", exist_ok=True)
    all_results["mfrl_5ep"] = train_and_evaluate(
        config, dpo_data, test_data, "outputs/overnight/mfrl_5ep", "MFRL 5 epochs")

    # ============================================================
    # Experiment 3: MFRL (no filtering, 5 epochs, higher LR)
    # ============================================================
    print("\n\n" + "#"*60)
    print("# Experiment 3: MFRL (no filter, 5 epochs, lr=1e-5)")
    print("#"*60)
    config["dpo"]["learning_rate"] = 1e-5
    os.makedirs("outputs/overnight/mfrl_5ep_lr1e5", exist_ok=True)
    all_results["mfrl_5ep_lr1e5"] = train_and_evaluate(
        config, dpo_data, test_data, "outputs/overnight/mfrl_5ep_lr1e5", "MFRL 5ep lr=1e-5")
    config["dpo"]["learning_rate"] = 5e-6

    # ============================================================
    # Experiment 4-6: Baselines
    # ============================================================
    for bt in ["sft", "dpo", "kto"]:
        print(f"\n\n" + "#"*60)
        print(f"# Baseline: {bt}")
        print("#"*60)
        os.makedirs(f"outputs/overnight/{bt}", exist_ok=True)
        all_results[bt] = run_baseline(
            config, dpo_data, test_data, f"outputs/overnight/{bt}", bt)

    # ============================================================
    # Experiment 7: Ablation - no curriculum
    # ============================================================
    print("\n\n" + "#"*60)
    print("# Ablation: no curriculum")
    print("#"*60)
    config["curriculum"]["enabled"] = False
    config["dpo"]["num_epochs"] = 3
    os.makedirs("outputs/overnight/ablation_no_curriculum", exist_ok=True)
    all_results["no_curriculum"] = train_and_evaluate(
        config, dpo_data, test_data, "outputs/overnight/ablation_no_curriculum", "w/o curriculum")
    config["curriculum"]["enabled"] = True

    # ============================================================
    # Experiment 8: Ablation - with filtering (min_score_diff=0.05)
    # ============================================================
    print("\n\n" + "#"*60)
    print("# Ablation: with filtering (min_score_diff=0.05)")
    print("#"*60)
    config["feedback"]["min_score_diff"] = 0.05
    os.makedirs("outputs/overnight/ablation_filter_005", exist_ok=True)
    dpo_data_filtered = [d for d in dpo_data if d["score_diff"] >= 0.05]
    all_results["filter_005"] = train_and_evaluate(
        config, dpo_data_filtered, test_data, "outputs/overnight/ablation_filter_005", "filter=0.05")

    # ============================================================
    # Save all results
    # ============================================================
    end_time = datetime.now()
    duration = end_time - start_time

    final_results = {
        "experiments": all_results,
        "duration_hours": duration.total_seconds() / 3600,
        "timestamp": end_time.isoformat(),
    }

    with open("results/overnight_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"\n\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'ROUGE-1':<12} {'ROUGE-2':<12} {'ROUGE-L':<12}")
    print("-"*60)
    for name, scores in all_results.items():
        print(f"{name:<25} {scores['rouge1']:<12.4f} {scores['rouge2']:<12.4f} {scores['rougeL']:<12.4f}")
    print(f"\nTotal duration: {duration}")


if __name__ == "__main__":
    main()
