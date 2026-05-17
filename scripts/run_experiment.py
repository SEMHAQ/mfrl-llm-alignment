"""MFRL完整实验脚本

执行流程：
1. 加载数据集
2. 生成候选输出
3. 计算反馈分数
4. 融合反馈构建偏好对
5. DPO训练
6. 评估

使用方法：
    python scripts/run_experiment.py --config configs/train.yaml
    python scripts/run_experiment.py --mode ablation --ablation_type no_rule
"""

import os
import sys
import json
import argparse
import yaml
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.feedback.rule_feedback import RuleFeedback
from src.feedback.model_feedback import ModelFeedback
from src.feedback.feedback_fusion import AdaptiveFeedbackFusion
from src.trainer.dpo_trainer import MFRLTrainer, MFRLConfig
from src.trainer.curriculum import CurriculumScheduler
from src.data.dataset import load_lcsts, load_alpaca_chinese, PreferenceDataset
from src.data.preprocess import generate_candidates
from src.eval.evaluator import Evaluator


def load_config(config_path: str) -> dict:
    """加载YAML配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(model_name: str, dtype: str = "bfloat16"):
    """加载模型和分词器。"""
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


def prepare_preference_data(
    model,
    tokenizer,
    raw_data,
    config,
    feedback_type="both",
):
    """准备偏好数据。

    Args:
        model: 用于生成候选的模型
        tokenizer: 分词器
        raw_data: 原始数据
        config: 配置
        feedback_type: 反馈类型 ("rule", "model", "both")
    """
    num_candidates = config["feedback"]["num_candidates"]
    inputs = [item["input"] for item in raw_data]
    references = [item["reference"] for item in raw_data]

    # 生成候选
    print(f"Generating {num_candidates} candidates per input...")
    candidates_list = generate_candidates(
        model, tokenizer, inputs, num_candidates=num_candidates
    )

    pairs_by_source = {}

    # 规则反馈
    if feedback_type in ("rule", "both") and config["feedback"]["rule"]["enabled"]:
        print("Computing rule-based feedback...")
        rule_feedback = RuleFeedback(
            metrics=config["feedback"]["rule"]["metrics"],
            weights=config["feedback"]["rule"]["weights"],
        )
        rule_pairs = rule_feedback.generate_preference_pairs(
            inputs, references, candidates_list
        )
        pairs_by_source["rule"] = rule_pairs
        print(f"  Rule pairs: {len(rule_pairs)}")

    # 模型反馈
    if feedback_type in ("model", "both") and config["feedback"]["model"]["enabled"]:
        print("Computing model-based feedback...")
        model_feedback = ModelFeedback(
            model_name=config["feedback"]["model"]["model_name"]
        )
        model_pairs = model_feedback.generate_preference_pairs(
            inputs, references, candidates_list
        )
        pairs_by_source["model"] = model_pairs
        print(f"  Model pairs: {len(model_pairs)}")

    # 融合
    if len(pairs_by_source) > 1:
        print("Fusing feedback signals...")
        fusion = AdaptiveFeedbackFusion(num_feedback_sources=len(pairs_by_source))
        final_pairs = fusion.fuse_preference_pairs(pairs_by_source)
    elif len(pairs_by_source) == 1:
        final_pairs = list(pairs_by_source.values())[0]
    else:
        raise ValueError(f"No feedback enabled for type: {feedback_type}")

    # 格式化为DPO格式
    dpo_data = []
    for pair in final_pairs:
        prompt = f"请为以下文本生成简洁准确的摘要：\n{pair['input']}\n摘要："
        dpo_data.append({
            "prompt": prompt,
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
            "score_diff": pair.get("score_diff", pair.get("fused_score_diff", 0.0)),
        })

    return dpo_data


def train_mfrl(dpo_data, config, output_dir):
    """执行MFRL训练。"""
    from datasets import Dataset

    # 划分训练/验证集
    split_idx = int(len(dpo_data) * 0.9)
    train_data = dpo_data[:split_idx]
    eval_data = dpo_data[split_idx:]

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    # 配置训练
    mfrl_config = MFRLConfig(
        model_name=config["model"]["name"],
        model_dtype=config["model"]["dtype"],
        lora_r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        lora_target_modules=config["lora"]["target_modules"],
        dpo_beta=config["dpo"]["beta"],
        dpo_loss_type=config["dpo"]["loss_type"],
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

    return trainer


def evaluate(model, tokenizer, test_data, output_dir):
    """评估模型。"""
    evaluator = Evaluator(model, tokenizer)

    # ROUGE评估
    rouge_scores = evaluator.evaluate_rouge(test_data)
    print(f"\nROUGE Scores:")
    for metric, score in rouge_scores.items():
        print(f"  {metric}: {score:.4f}")

    # 保存结果
    results = {"rouge": rouge_scores}
    save_path = os.path.join(output_dir, "eval_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def run_ablation(config, ablation_type):
    """运行消融实验。"""
    print(f"\n{'='*50}")
    print(f"Running ablation: {ablation_type}")
    print(f"{'='*50}\n")

    # 修改配置
    if ablation_type == "no_rule":
        config["feedback"]["rule"]["enabled"] = False
        config["feedback"]["model"]["enabled"] = True
        feedback_type = "model"
    elif ablation_type == "no_model":
        config["feedback"]["model"]["enabled"] = False
        feedback_type = "rule"
    elif ablation_type == "no_curriculum":
        config["curriculum"]["enabled"] = False
        feedback_type = "both"
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

    output_dir = os.path.join(config["output"]["dir"], f"ablation_{ablation_type}")
    os.makedirs(output_dir, exist_ok=True)

    return output_dir, feedback_type


def main():
    parser = argparse.ArgumentParser(description="MFRL Experiment")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "ablation", "eval_only"])
    parser.add_argument("--ablation_type", type=str, default=None,
                        choices=["no_rule", "no_model", "no_curriculum"])
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config["output"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print("Loading model...")
    model, tokenizer = load_model(config["model"]["name"], config["model"]["dtype"])

    # 加载数据集
    print("Loading dataset...")
    if config["data"]["dataset"] == "lcsts":
        dataset = load_lcsts(max_samples=config["data"]["max_train_samples"])
        raw_data = [
            {"input": item["source"], "reference": item["summary"]}
            for item in dataset
        ]
    else:
        dataset = load_alpaca_chinese(max_samples=config["data"]["max_train_samples"])
        raw_data = [
            {"input": item.get("instruction", "") + " " + item.get("input", ""),
             "reference": item["output"]}
            for item in dataset
        ]

    # 消融实验配置
    feedback_type = "both"
    if args.mode == "ablation" and args.ablation_type:
        output_dir, feedback_type = run_ablation(config, args.ablation_type)

    # 准备偏好数据
    print("Preparing preference data...")
    dpo_data = prepare_preference_data(
        model, tokenizer, raw_data, config, feedback_type
    )
    print(f"Total preference pairs: {len(dpo_data)}")

    # 保存偏好数据
    with open(os.path.join(output_dir, "preference_data.json"), "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)

    # 训练
    print("\nStarting MFRL training...")
    trainer = train_mfrl(dpo_data, config, output_dir)

    # 评估
    print("\nEvaluating...")
    test_data = dpo_data[-config["data"]["max_test_samples"]:]
    evaluate(trainer.model, trainer.tokenizer, test_data, output_dir)

    print(f"\nExperiment complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
