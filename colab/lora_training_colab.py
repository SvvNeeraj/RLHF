# Colab-compatible LoRA fine-tuning script (pure Python format)

import json
import os
import subprocess
import sys
import zipfile


def install_requirements() -> None:
    pkgs = [
        "transformers==4.48.3",
        "peft==0.14.0",
        "accelerate==1.3.0",
        "datasets==3.2.0",
        "bitsandbytes==0.45.2",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def main() -> None:
    install_requirements()

    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model

    base_model = "Qwen/Qwen1.5-1.8B-Chat"
    data_path = "/content/qa_dataset.jsonl"
    output_dir = "/content/lora_adapter"

    system_instruction = (
        "You are a professional B.Tech academic tutor. "
        "Always answer with summary, detailed explanation, examples, diagram guidance, exam points, "
        "advantages/disadvantages, and conclusion."
    )

    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    dataset = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    def preprocess(row):
        prompt = (
            f"Instruction: {system_instruction}\n"
            f"Question: {row['question']}\n"
            f"Answer: {row['answer']}"
        )
        tokens = tokenizer(prompt, truncation=True, max_length=700)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tok_ds = dataset.map(preprocess, remove_columns=dataset.column_names)

    args = TrainingArguments(
        output_dir="/content/outputs",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    zip_path = "/content/lora_adapter.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for name in files:
                path = os.path.join(root, name)
                arcname = os.path.relpath(path, "/content")
                zf.write(path, arcname)

    try:
        from google.colab import files  # type: ignore

        files.download(zip_path)
    except Exception:
        print(f"Adapter packaged at {zip_path}")


if __name__ == "__main__":
    main()
