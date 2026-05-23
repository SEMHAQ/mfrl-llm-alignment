"""快速 beta 敏感性实验：重用已有偏好数据，改 DPO beta 重新训练"""
import json, yaml, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.trainer.dpo_trainer import MFRLTrainer, MFRLConfig
from datasets import Dataset

beta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
beta_str = str(beta).replace(".", "")

with open("configs/train.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

pref_path = os.path.join(cfg["output"]["dir"], "preference_data.json")
with open(pref_path, encoding="utf-8") as f:
    dpo_data = json.load(f)

print(f"Training beta={beta}, {len(dpo_data)} preference pairs")

split = int(len(dpo_data) * 0.9)
train_ds = Dataset.from_list(dpo_data[:split])
eval_ds = Dataset.from_list(dpo_data[split:])

mfrl_cfg = MFRLConfig(
    model_name=cfg["model"]["name"], model_dtype=cfg["model"]["dtype"],
    lora_r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
    lora_dropout=cfg["lora"]["dropout"],
    lora_target_modules=cfg["lora"]["target_modules"],
    dpo_beta=beta, dpo_loss_type=cfg["dpo"]["loss_type"],
    learning_rate=cfg["dpo"]["learning_rate"],
    num_train_epochs=cfg["dpo"]["num_epochs"],
    per_device_train_batch_size=cfg["dpo"]["batch_size"],
    gradient_accumulation_steps=cfg["dpo"]["gradient_accumulation"],
    max_length=cfg["dpo"]["max_length"],
    max_prompt_length=cfg["dpo"]["max_prompt_length"],
    output_dir=f"outputs/ablation_beta{beta_str}",
    curriculum_enabled=False,
)
trainer = MFRLTrainer(mfrl_cfg)
trainer.setup_model()
trainer.train(train_ds, eval_ds)
trainer.save()
print(f"beta={beta} done")
