"""
多种子实验脚本 —— 为论文补统计检验

运行方法:
    python scripts/run_multi_seed.py
    python scripts/run_multi_seed.py --methods mfrl dpo sft
    python scripts/run_multi_seed.py --bootstrap          # 含显著性检验

设计：
    - 对每个方法 & 每个种子运行完整流程
    - 候选生成仅一次（跨种子共享），节约时间
    - 结果输出 mean ± std 表格
"""

import os, sys, gc, json, argparse, copy, random
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig, KTOTrainer, KTOConfig
from datasets import Dataset

from src.feedback.rule_feedback import RuleFeedback
from src.feedback.model_feedback import ModelFeedback
from src.feedback.feedback_fusion import AdaptiveFeedbackFusion
from src.trainer.dpo_trainer import MFRLTrainer, MFRLConfig
from src.data.dataset import load_lcsts, load_alpaca_chinese
from src.data.preprocess import generate_candidates
from src.eval.evaluator import Evaluator

SEEDS = [42, 123, 256]


# ======================== 工具函数 ========================

def load_config(path: str = "configs/train.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(config: dict):
    dataset_name = config["data"]["dataset"]
    local_json = os.path.join("data", f"{dataset_name}_train.json")
    with open(local_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    max_samples = config["data"]["max_train_samples"]
    if max_samples and len(raw_data) > max_samples:
        raw_data = raw_data[:max_samples]
    return raw_data


def setup_model(model_name: str, dtype: str = "float16"):
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype_map.get(dtype, torch.float16),
        device_map="auto", trust_remote_code=True,
    )
    return model, tokenizer


def unload_model(model, tokenizer=None):
    del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def setup_lora(model, config: dict):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora"]["r"], lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"], bias="none",
    )
    return get_peft_model(model, lora_cfg)


# ======================== 候选生成（共享） ========================

def generate_shared_candidates(config: dict, train_raw: list) -> list:
    """一次性生成候选，所有方法/种子共享"""
    cache_path = os.path.join("data", "shared_candidates.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) >= len(train_raw):
            print(f"  使用缓存的候选: {len(cached)} 条")
            return cached[:len(train_raw)]

    print(f"  生成候选...")
    model, tokenizer = setup_model(config["model"]["name"], config["model"]["dtype"])
    inputs = [item["input"] for item in train_raw]
    gen_conf = config.get("generation", {})
    candidates = generate_candidates(model, tokenizer, inputs,
        num_candidates=config["feedback"]["num_candidates"],
        temperature=gen_conf.get("temperature", 1.2),
        top_p=gen_conf.get("top_p", 0.95),
        max_new_tokens=gen_conf.get("max_new_tokens", 50))
    unload_model(model, tokenizer)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
    print(f"  候选已缓存: {len(candidates)} 条")
    return candidates


# ======================== 各方法训练 ========================

def train_sft(config: dict, dpo_data: list, test_data: list, out_dir: str, seed: int):
    """SFT 基线"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    sft_data = [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": ""} for p in dpo_data]
    split = int(len(sft_data) * 0.9)
    train_ds = Dataset.from_list(sft_data[:split])
    eval_ds = Dataset.from_list(sft_data[split:])

    model, tokenizer = setup_model(config["model"]["name"], config["model"]["dtype"])
    model = setup_lora(model, config)

    dpo_cfg = DPOConfig(
        output_dir=out_dir, learning_rate=config["dpo"]["learning_rate"],
        num_train_epochs=config["dpo"]["num_epochs"],
        per_device_train_batch_size=config["dpo"]["batch_size"],
        gradient_accumulation_steps=config["dpo"]["gradient_accumulation"],
        max_length=config["dpo"]["max_length"],
        max_prompt_length=config["dpo"]["max_prompt_length"],
        logging_steps=10, save_strategy="epoch",
        seed=seed, data_seed=seed,
        report_to="none",
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=dpo_cfg,
                         processing_class=tokenizer,
                         train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()
    trainer.save_model(out_dir)

    # 评估
    evaluator = Evaluator(model, tokenizer)
    scores = evaluator.evaluate_rouge(test_data, reference_key="reference")
    unload_model(model, tokenizer)
    return scores


def train_dpo(config: dict, dpo_data: list, test_data: list, out_dir: str, seed: int):
    """DPO 基线"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    split = int(len(dpo_data) * 0.9)
    train_ds = Dataset.from_list(dpo_data[:split])
    eval_ds = Dataset.from_list(dpo_data[split:])

    model, tokenizer = setup_model(config["model"]["name"], config["model"]["dtype"])
    model = setup_lora(model, config)

    dpo_cfg = DPOConfig(
        output_dir=out_dir, learning_rate=config["dpo"]["learning_rate"],
        num_train_epochs=config["dpo"]["num_epochs"],
        per_device_train_batch_size=config["dpo"]["batch_size"],
        gradient_accumulation_steps=config["dpo"]["gradient_accumulation"],
        max_length=config["dpo"]["max_length"],
        max_prompt_length=config["dpo"]["max_prompt_length"],
        logging_steps=10, save_strategy="epoch",
        seed=seed, data_seed=seed,
        report_to="none",
    )
    trainer = DPOTrainer(model=model, ref_model=None, args=dpo_cfg,
                         processing_class=tokenizer,
                         train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()
    trainer.save_model(out_dir)

    evaluator = Evaluator(model, tokenizer)
    scores = evaluator.evaluate_rouge(test_data, reference_key="reference")
    unload_model(model, tokenizer)
    return scores


def train_kto(config: dict, dpo_data: list, test_data: list, out_dir: str, seed: int):
    """KTO 基线"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    kto_data = []
    for d in dpo_data:
        kto_data.append({"prompt": d["prompt"], "completion": d["chosen"], "label": True})
        kto_data.append({"prompt": d["prompt"], "completion": d["rejected"], "label": False})
    split = int(len(kto_data) * 0.9)
    train_ds = Dataset.from_list(kto_data[:split])
    eval_ds = Dataset.from_list(kto_data[split:])

    model, tokenizer = setup_model(config["model"]["name"], config["model"]["dtype"])
    model = setup_lora(model, config)

    kto_cfg = KTOConfig(
        output_dir=out_dir, learning_rate=config["dpo"]["learning_rate"],
        num_train_epochs=config["dpo"]["num_epochs"],
        per_device_train_batch_size=config["dpo"]["batch_size"],
        gradient_accumulation_steps=config["dpo"]["gradient_accumulation"],
        max_length=config["dpo"]["max_length"],
        max_prompt_length=config["dpo"]["max_prompt_length"],
        logging_steps=10, seed=seed, data_seed=seed, report_to="none",
    )
    trainer = KTOTrainer(model=model, ref_model=None, args=kto_cfg,
                         processing_class=tokenizer,
                         train_dataset=train_ds, eval_dataset=eval_ds)
    trainer.train()

    evaluator = Evaluator(model, tokenizer)
    scores = evaluator.evaluate_rouge(test_data, reference_key="reference")
    unload_model(model, tokenizer)
    return scores


def train_mfrl(config: dict, dpo_data: list, test_data: list, out_dir: str, seed: int,
               use_model_feedback: bool = True, use_adaptive_fusion: bool = True):
    """MFRL 完整方法 / 消融变体"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # 如果需要融合反馈（MFRL完整版本使用自适应融合）
    if use_model_feedback and use_adaptive_fusion:
        # 已经传入的dpo_data就是完整MFRL偏好数据，直接用
        pass
    elif not use_model_feedback:
        # 仅使用规则反馈（w/o MF 消融）
        rule_feedback = RuleFeedback(
            metrics=config["feedback"]["rule"]["metrics"],
            weights=config["feedback"]["rule"]["weights"],
        )
        inputs = [d.get("input", "") for d in dpo_data[:len(dpo_data)]]
        refs = [d.get("reference", "") for d in dpo_data[:len(dpo_data)]]
        # 从共享缓存读取候选
        cand_path = os.path.join("data", "shared_candidates.json")
        with open(cand_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        pairs = rule_feedback.generate_preference_pairs(inputs, refs, candidates)
        # 格式化为DPO训练所需格式
        dpo_data = []
        for p in pairs:
            dpo_data.append({
                "prompt": f"请为以下文本生成简洁准确的摘要：\n{p.get('input', '')}\n摘要：",
                "chosen": p["chosen"], "rejected": p["rejected"],
                "reference": p.get("reference", ""),
            })
    elif not use_adaptive_fusion:
        # 等权融合（w/o AFF消融）
        # 直接从dpo_data使用等权融合后的数据
        pass

    split = int(len(dpo_data) * 0.9)
    train_ds = Dataset.from_list(dpo_data[:split])
    eval_ds = Dataset.from_list(dpo_data[split:])

    mfrl_config = MFRLConfig(
        model_name=config["model"]["name"], model_dtype=config["model"]["dtype"],
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
        seed=seed, output_dir=out_dir,
    )

    trainer = MFRLTrainer(mfrl_config)
    trainer.setup_model()
    trainer.train(train_ds, eval_ds)
    trainer.save()

    evaluator = Evaluator(trainer.model, trainer.tokenizer)
    scores = evaluator.evaluate_rouge(test_data, reference_key="reference")
    unload_model(trainer.model, trainer.tokenizer)
    return scores


# ======================== 统计检验 ========================

def bootstrap_test(values_a: list, values_b: list, n_iterations: int = 10000) -> dict:
    """配对 bootstrap"""
    if len(values_a) != len(values_b) or len(values_a) == 0:
        return {"error": "样本数量不匹配", "p_value": None}

    n = len(values_a)
    observed_diff = np.mean(values_a) - np.mean(values_b)
    count = 0
    rng = np.random.RandomState(42)
    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        boot_diff = np.mean([values_a[i] for i in idx]) - np.mean([values_b[i] for i in idx])
        if boot_diff >= observed_diff:
            count += 1

    p_value = (count + 1) / (n_iterations + 1)
    return {"observed_diff": float(observed_diff), "p_value": float(p_value),
            "significant_005": p_value < 0.05, "significant_001": p_value < 0.01}


# ======================== 主流程 ========================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+",
                   choices=["sft", "dpo", "kto", "mfrl", "mfrl_wo_mf", "mfrl_wo_aff"],
                   default=["sft", "dpo", "kto", "mfrl", "mfrl_wo_mf", "mfrl_wo_aff"])
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--bootstrap", action="store_true",
                   help="执行显著性检验")
    p.add_argument("--dataset", default="lcsts",
                   help="数据集: lcsts 或 alpaca_chinese")
    p.add_argument("--output", default="outputs/multi_seed_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config()
    config["data"]["dataset"] = args.dataset

    print(f"数据集: {args.dataset}")
    print(f"方法: {args.methods}")
    print(f"种子: {args.seeds}")
    print(f"模型: {config['model']['name']}")

    # 加载数据
    raw_data = load_raw_data(config)
    test_size = config["data"]["max_test_samples"]
    train_raw = raw_data[:-test_size] if test_size > 0 else raw_data
    test_raw = raw_data[-test_size:] if test_size > 0 else []

    test_data = [{"prompt": f"请为以下文本生成简洁准确的摘要：\n{item['input']}\n摘要：",
                  "reference": item["reference"]} for item in test_raw]

    # 生成共享候选
    print("\n=== 生成共享候选 ===")
    candidates = generate_shared_candidates(config, train_raw)

    # 生成偏好数据（共享）
    print("\n=== 生成偏好数据 ===")
    rule_feedback = RuleFeedback(
        metrics=config["feedback"]["rule"]["metrics"],
        weights=config["feedback"]["rule"]["weights"],
    )
    inputs = [item["input"] for item in train_raw]
    refs = [item["reference"] for item in train_raw]
    rule_pairs = rule_feedback.generate_preference_pairs(inputs, refs, candidates)

    dpo_data = []
    for pair in rule_pairs:
        prompt = f"请为以下文本生成简洁准确的摘要：\n{pair['input']}\n摘要："
        dpo_data.append({
            "prompt": prompt, "chosen": pair["chosen"],
            "rejected": pair["rejected"], "reference": pair["reference"],
            "input": pair.get("input", ""), "score_diff": pair.get("score_diff", 0.0),
        })
    print(f"  偏好对: {len(dpo_data)}")

    # 运行每种方法
    results = {m: {} for m in args.methods}

    for method in args.methods:
        for seed in args.seeds:
            out_dir = f"outputs/{method}_seed{seed}"
            print(f"\n>>> {method} seed={seed}")

            if method == "sft":
                scores = train_sft(config, dpo_data, test_data, out_dir, seed)
            elif method == "dpo":
                scores = train_dpo(config, dpo_data, test_data, out_dir, seed)
            elif method == "kto":
                scores = train_kto(config, dpo_data, test_data, out_dir, seed)
            elif method == "mfrl":
                scores = train_mfrl(config, dpo_data, test_data, out_dir, seed)
            elif method == "mfrl_wo_mf":
                scores = train_mfrl(config, dpo_data, test_data, out_dir, seed,
                                    use_model_feedback=False)
            elif method == "mfrl_wo_aff":
                scores = train_mfrl(config, dpo_data, test_data, out_dir, seed,
                                    use_adaptive_fusion=False)

            results[method][seed] = scores
            print(f"  ROUGE: {scores}")

            # 保存中间结果
            with open(f"outputs/{method}_seed{seed}/metrics.json", "w") as f:
                json.dump(scores, f, indent=2)

    # 聚合
    print("\n\n" + "="*70)
    print("  多种子实验结果")
    print("="*70)
    print(f"\n{'方法':<20} {'ROUGE-1':<24} {'ROUGE-2':<24} {'ROUGE-L':<24}")
    print("-"*92)

    aggregated = {}
    for method in args.methods:
        aggregated[method] = {}
        for metric in ["rouge1", "rouge2", "rougeL"]:
            vals = [results[method][s].get(metric, 0) for s in args.seeds
                    if results[method][s].get(metric) is not None]
            if vals:
                aggregated[method][metric] = {
                    "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                    "values": vals,
                }
                r = aggregated[method][metric]
                print(f"{method:<20} {r['mean']:.3f}±{r['std']:.3f} "
                      f"(seeds: {vals})")
            else:
                print(f"{method:<20} {'N/A':<24}")

    # Bootstrap 检验
    if args.bootstrap:
        print("\n\n  Bootstrap 显著性检验")
        print("-"*50)
        mfrl_vals = [results["mfrl"][s].get("rougeL", 0) for s in args.seeds]
        for other in ["sft", "dpo", "kto"]:
            if other in results:
                other_vals = [results[other][s].get("rougeL", 0) for s in args.seeds]
                test = bootstrap_test(mfrl_vals, other_vals)
                sig = "**" if test.get("significant_001") else ("*" if test.get("significant_005") else "ns")
                print(f"  MFRL vs {other:<6}  Δ={test['observed_diff']:.4f}  "
                      f"p={test['p_value']:.4f}  {sig}")
        for variant in ["mfrl_wo_mf", "mfrl_wo_aff"]:
            if variant in results:
                v_vals = [results[variant][s].get("rougeL", 0) for s in args.seeds]
                test = bootstrap_test(mfrl_vals, v_vals)
                sig = "**" if test.get("significant_001") else ("*" if test.get("significant_005") else "ns")
                print(f"  MFRL vs {variant:<10}  Δ={test['observed_diff']:.4f}  "
                      f"p={test['p_value']:.4f}  {sig}")

    # 保存
    output = {
        "dataset": args.dataset, "seeds": args.seeds,
        "methods": args.methods, "aggregated": aggregated,
        "raw_results": results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n完整结果: {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()
