"""用新max_new_tokens=50重新评估所有已有模型"""

import os
import sys
import gc
import json
import yaml
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.eval.evaluator import Evaluator


def load_config():
    with open("configs/train.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_model(model_name, dtype="float16"):
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype_map.get(dtype, torch.float16),
        device_map={"": 0}, trust_remote_code=True,
    )
    return model, tokenizer


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_model(model, tokenizer, test_data, output_path, label):
    print(f"\nEvaluating {label}...")
    evaluator = Evaluator(model, tokenizer, max_new_tokens=50)
    rouge_scores = evaluator.evaluate_rouge(test_data, reference_key="reference")
    print(f"  ROUGE: {rouge_scores}")

    results = {"rouge": rouge_scores}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return rouge_scores


def main():
    config = load_config()
    model_name = config["model"]["name"]

    # 加载测试数据
    dataset_name = config["data"]["dataset"]
    local_json = os.path.join("data", f"{dataset_name}_train.json")
    with open(local_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    test_size = config["data"]["max_test_samples"]
    test_raw = raw_data[-test_size:]

    test_data = []
    for item in test_raw:
        prompt = f"请为以下文本生成简洁准确的摘要：\n{item['input']}\n摘要："
        test_data.append({"prompt": prompt, "reference": item["reference"]})

    all_results = {}

    # 1. 基座模型
    print("=" * 50)
    print("Base model")
    print("=" * 50)
    model, tokenizer = setup_model(model_name, config["model"]["dtype"])
    all_results["base"] = evaluate_model(
        model, tokenizer, test_data, "results/base_eval.json", "Base model")
    unload_model(model, tokenizer)

    # 2. MFRL
    mfrl_path = "outputs/mfrl_v3/final"
    if os.path.exists(os.path.join(mfrl_path, "adapter_config.json")):
        print("=" * 50)
        print("MFRL")
        print("=" * 50)
        model, tokenizer = setup_model(model_name, config["model"]["dtype"])
        model = PeftModel.from_pretrained(model, mfrl_path)
        all_results["MFRL"] = evaluate_model(
            model, tokenizer, test_data, "results/mfrl_v3_eval.json", "MFRL")
        unload_model(model, tokenizer)

    # 3. SFT
    sft_path = "outputs/mfrl_v3/sft"
    if os.path.exists(os.path.join(sft_path, "adapter_config.json")):
        print("=" * 50)
        print("SFT")
        print("=" * 50)
        model, tokenizer = setup_model(model_name, config["model"]["dtype"])
        model = PeftModel.from_pretrained(model, sft_path)
        all_results["SFT"] = evaluate_model(
            model, tokenizer, test_data, "results/sft_eval.json", "SFT")
        unload_model(model, tokenizer)

    # 4. Ablation no_model
    ablation_path = "outputs/mfrl_v3/ablation_no_model/final"
    if os.path.exists(os.path.join(ablation_path, "adapter_config.json")):
        print("=" * 50)
        print("Ablation: no_model")
        print("=" * 50)
        model, tokenizer = setup_model(model_name, config["model"]["dtype"])
        model = PeftModel.from_pretrained(model, ablation_path)
        all_results["no_model"] = evaluate_model(
            model, tokenizer, test_data, "results/ablation_no_model_eval.json", "no_model")
        unload_model(model, tokenizer)

    # 打印汇总
    print(f"\n{'='*60}")
    print("RESULTS (max_new_tokens=50)")
    print(f"{'='*60}")
    header = "{:<20} {:<12} {:<12} {:<12}".format("Method", "ROUGE-1", "ROUGE-2", "ROUGE-L")
    print(header)
    print("-" * 56)
    for name, scores in all_results.items():
        line = "{:<20} {:.4f}       {:.4f}       {:.4f}".format(
            name, scores["rouge1"], scores["rouge2"], scores["rougeL"])
        print(line)


if __name__ == "__main__":
    main()
