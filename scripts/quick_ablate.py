"""快速跑消融实验：重用已生成的偏好数据，只改训练参数"""
import sys, json, yaml, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trainer.dpo_trainer import MFRLTrainer, MFRLConfig
from datasets import Dataset


def train_from_prefs(pref_path, config, output_dir, curriculum_enabled):
    """用已有偏好数据训练，支持覆盖训练参数。"""
    with open(pref_path, "r", encoding="utf-8") as f:
        dpo_data = json.load(f)
    print(f"Loaded {len(dpo_data)} preference pairs from {pref_path}")

    split_idx = int(len(dpo_data) * 0.9)
    train_data = dpo_data[:split_idx]
    eval_data = dpo_data[split_idx:]
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

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
        curriculum_enabled=curriculum_enabled,
        curriculum_warmup_epochs=config["curriculum"]["warmup_ratio"],
        output_dir=output_dir,
    )

    trainer = MFRLTrainer(mfrl_config)
    trainer.setup_model()
    trainer.train(train_dataset, eval_dataset)
    trainer.save()


if __name__ == "__main__":
    # 用法: python scripts/quick_ablate.py <abl_type>
    #   abl_type: no_curriculum, no_model, no_filter
    abl_type = sys.argv[1] if len(sys.argv) > 1 else "no_curriculum"

    with open("configs/train.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pref_path = os.path.join(config["output"]["dir"], "preference_data.json")
    if not os.path.exists(pref_path):
        print(f"Preference data not found at {pref_path}")
        sys.exit(1)

    output_dir = os.path.join(config["output"]["dir"], f"ablation_{abl_type}")
    os.makedirs(output_dir, exist_ok=True)

    curriculum_enabled = True
    if abl_type == "no_curriculum":
        curriculum_enabled = False
        print(f"Ablation: no_curriculum (curriculum={curriculum_enabled})")
    else:
        print(f"Unknown ablation type: {abl_type}")
        sys.exit(1)

    train_from_prefs(pref_path, config, output_dir, curriculum_enabled)
    print(f"Done! Model saved to {output_dir}")
