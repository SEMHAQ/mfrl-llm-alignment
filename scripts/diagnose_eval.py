"""诊断评估问题：看模型到底在生成什么"""

import torch, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.eval.metrics import compute_rouge


def main():
    model_name = "models/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    with open("data/lcsts_train.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    test_items = data[-200:]

    print("=" * 60)
    print("BASE MODEL (no fine-tuning) - 10 samples")
    print("=" * 60)

    predictions = []
    references = []
    for i in range(10):
        item = test_items[i]
        prompt = f"请为以下文本生成简洁准确的摘要：\n{item['input']}\n摘要："
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                temperature=1.0, pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        predictions.append(response)
        references.append(item["reference"])
        print(f"\n--- Sample {i} ---")
        print(f"Reference:  {item['reference']}")
        print(f"Generated:  {response[:150]}")
        print(f"Gen length: {len(response)} chars")

    rouge = compute_rouge(predictions, references)
    print(f"\nBase model ROUGE (10 samples): {rouge}")

    # 也测一下空输出的ROUGE
    empty_preds = [""] * len(references)
    empty_rouge = compute_rouge(empty_preds, references)
    print(f"Empty output ROUGE: {empty_rouge}")

    # 测一下随机乱码
    import random
    random_preds = ["".join([chr(random.randint(0x4e00, 0x9fff)) for _ in range(20)]) for _ in range(len(references))]
    random_rouge = compute_rouge(random_preds, references)
    print(f"Random Chinese ROUGE: {random_rouge}")

    # 测一下直接复制输入的前30字
    copy_preds = [item["input"][:30] for item in test_items[:10]]
    copy_rouge = compute_rouge(copy_preds, references)
    print(f"Copy input[:30] ROUGE: {copy_rouge}")

    # 也测试一下已训练的模型（如果存在）
    import gc
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    lora_path = "outputs/mfrl_v3"
    if os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        print("\n" + "=" * 60)
        print("FINE-TUNED MODEL (MFRL)")
        print("=" * 60)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()

        predictions_ft = []
        for i in range(10):
            item = test_items[i]
            prompt = f"请为以下文本生成简洁准确的摘要：\n{item['input']}\n摘要："
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=128, do_sample=False,
                    temperature=1.0, pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            predictions_ft.append(response)
            print(f"\n--- Sample {i} ---")
            print(f"Reference:  {item['reference']}")
            print(f"Generated:  {response[:150]}")

        rouge_ft = compute_rouge(predictions_ft, references)
        print(f"\nFine-tuned ROUGE (10 samples): {rouge_ft}")


if __name__ == "__main__":
    main()
