# True RLHF (policy parameter update) script for Colab GPU
# --------------------------------------------------------
# This script performs:
# 1) Reward-model training from human ratings.
# 2) Policy-gradient updates on a LoRA adapter using reward-model scores.
#
# Usage in Colab:
# - Upload `policy_gradient_train.jsonl` (from scripts/export_rlhf_dataset.py)
# - Run this script
# - Download output adapter and use as local `adapters/lora_adapter`

import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def install_requirements() -> None:
    use_4bit = os.getenv("USE_4BIT", "0").strip().lower() in {"1", "true", "yes"}
    pkgs = [
        "torch>=2.2.0",
        "transformers==4.48.3",
        "accelerate==1.3.0",
        "datasets==3.2.0",
        "peft==0.14.0",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    # Optional: bitsandbytes can fail on some Colab Python/glibc images.
    # Keep it opt-in so default run stays stable on fp16.
    install_bnb = os.getenv("INSTALL_BNB", "0").strip().lower() in {"1", "true", "yes"}
    if install_bnb:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes==0.45.2"])
            print("bitsandbytes installed (optional path available).")
        except Exception:
            print("bitsandbytes install failed; continuing with fp16 path.")
    else:
        print("Skipping bitsandbytes install (INSTALL_BNB=0). Using fp16 path.")
        if not use_4bit:
            # Ensure no stale incompatible bitsandbytes binary remains in Colab session.
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "bitsandbytes"])
                print("Removed bitsandbytes from environment for stable fp16 run.")
            except Exception:
                pass


@dataclass
class CFG:
    base_model: str = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    train_jsonl: str = os.getenv("TRAIN_JSONL", "/content/policy_gradient_train.jsonl")
    reward_model_name: str = os.getenv("REWARD_MODEL_NAME", "distilbert-base-uncased")
    reward_out: str = os.getenv("REWARD_OUT", "/content/reward_model")
    adapter_out: str = os.getenv("ADAPTER_OUT", "/content/rlhf_lora_adapter")
    max_prompt_len: int = int(os.getenv("MAX_PROMPT_LEN", "384"))
    max_resp_len: int = int(os.getenv("MAX_RESP_LEN", "512"))
    rm_epochs: int = int(os.getenv("RM_EPOCHS", "2"))
    pg_epochs: int = int(os.getenv("PG_EPOCHS", "1"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "2"))
    lr_reward: float = float(os.getenv("LR_REWARD", "2e-5"))
    lr_policy: float = float(os.getenv("LR_POLICY", "2e-4"))
    kl_coef: float = float(os.getenv("KL_COEF", "0.05"))
    reward_clip: float = float(os.getenv("REWARD_CLIP", "1.0"))
    use_4bit: bool = os.getenv("USE_4BIT", "0").strip().lower() in {"1", "true", "yes"}


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _pack_zip(src_dir: str, zip_path: str) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for name in files:
                p = os.path.join(root, name)
                arc = os.path.relpath(p, os.path.dirname(src_dir))
                zf.write(p, arc)


def train_reward_model(cfg: CFG):
    import numpy as np
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )

    rows = load_jsonl(cfg.train_jsonl)
    if not rows:
        raise RuntimeError(f"No rows in {cfg.train_jsonl}")

    rm_rows = []
    for r in rows:
        q = str(r.get("query", "")).strip()
        a = str(r.get("response", "")).strip()
        reward = float(r.get("reward", 0.0))
        if not q or not a:
            continue
        rm_rows.append({"text": f"Question: {q}\nAnswer: {a}", "label": reward})
    ds = Dataset.from_list(rm_rows)
    split = ds.train_test_split(test_size=min(0.1, max(1, int(0.1 * len(ds))) / max(1, len(ds))), seed=42)

    tokenizer = AutoTokenizer.from_pretrained(cfg.reward_model_name, use_fast=True)

    def tok_fn(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=512)
        out["labels"] = [float(x) for x in batch["label"]]
        return out

    train_ds = split["train"].map(tok_fn, batched=True, remove_columns=split["train"].column_names)
    eval_ds = split["test"].map(tok_fn, batched=True, remove_columns=split["test"].column_names)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.reward_model_name, num_labels=1, problem_type="regression")

    args = TrainingArguments(
        output_dir="/content/rm_out",
        learning_rate=cfg.lr_reward,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=cfg.rm_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        fp16=True,
    )

    def metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.squeeze(preds)
        mse = float(np.mean((preds - labels) ** 2))
        mae = float(np.mean(np.abs(preds - labels)))
        return {"mse": mse, "mae": mae}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        compute_metrics=metrics,
    )
    trainer.train()
    model.save_pretrained(cfg.reward_out)
    tokenizer.save_pretrained(cfg.reward_out)
    return cfg.reward_out


def policy_gradient_update(cfg: CFG, reward_model_dir: str):
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Policy-gradient RLHF training requires GPU in Colab.")

    rows = load_jsonl(cfg.train_jsonl)
    rows = [r for r in rows if str(r.get("query", "")).strip() and str(r.get("response", "")).strip()]
    if not rows:
        raise RuntimeError("No valid rows for policy update.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _load_causal(model_name: str, as_reference: bool = False):
        # Stable default: fp16 (no bitsandbytes dependency)
        if not cfg.use_4bit:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            return model

        # Optional 4-bit path (only if explicitly enabled).
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            print(f"{'reference' if as_reference else 'policy'} model loaded in 4-bit mode.")
            return model
        except Exception as exc:
            print(f"4-bit load failed ({exc}); falling back to fp16.")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            return model

    policy = _load_causal(cfg.base_model, as_reference=False)

    def _find_linear_target_modules(model: nn.Module) -> list[str]:
        names = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                leaf = name.split(".")[-1]
                if leaf in {"lm_head"}:
                    continue
                names.add(leaf)
        preferred = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "fc1",
            "fc2",
            "Wqkv",
            "out_proj",
        ]
        hits = [p for p in preferred if p in names]
        if hits:
            return hits
        # fallback to first few linear leaf names if architecture differs
        return sorted(list(names))[:8]

    target_modules = _find_linear_target_modules(policy)
    print("LoRA target_modules:", target_modules)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(policy, lora_cfg)

    # Ensure LoRA weights are trainable even under mixed dispatch modes.
    trainable_names = []
    for n, p in policy.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
            trainable_names.append(n)
    print("Trainable LoRA tensors:", len(trainable_names))
    if hasattr(policy, "print_trainable_parameters"):
        policy.print_trainable_parameters()
    if len(trainable_names) == 0:
        raise RuntimeError("No trainable LoRA parameters found. Cannot run policy updates.")

    # Keep off by default to avoid no-grad edge cases in some Colab setups.
    if os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "0").strip() in {"1", "true", "yes"}:
        if hasattr(policy, "gradient_checkpointing_enable"):
            policy.gradient_checkpointing_enable()
    policy.train()

    # frozen reference model for KL-anchor
    reference = _load_causal(cfg.base_model, as_reference=True)
    reference.eval()

    rm_tok = AutoTokenizer.from_pretrained(reward_model_dir, use_fast=True)
    rm_model = AutoModelForSequenceClassification.from_pretrained(reward_model_dir).to(device)
    rm_model.eval()

    optim = AdamW(policy.parameters(), lr=cfg.lr_policy)

    def score_reward(question: str, answer: str) -> float:
        txt = f"Question: {question}\nAnswer: {answer}"
        enc = rm_tok(txt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = rm_model(**enc).logits.squeeze()
        reward = float(torch.tanh(out).item())
        reward = max(-cfg.reward_clip, min(cfg.reward_clip, reward))
        return reward

    system = (
        "You are a strict Academic AI Tutor for BTech students. "
        "Use only academic engineering context and produce clear, topic-focused responses."
    )

    def build_prompt(q: str) -> str:
        return f"System: {system}\nQuestion: {q}\nAnswer:"

    for epoch in range(cfg.pg_epochs):
        total_loss = 0.0
        update_steps = 0
        for idx, row in enumerate(rows):
            q = str(row["query"]).strip()
            prompt = build_prompt(q)

            enc_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_prompt_len).to(device)
            with torch.no_grad():
                gen_ids = policy.generate(
                    **enc_prompt,
                    max_new_tokens=220,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_text = tokenizer.decode(gen_ids[0][enc_prompt["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            if not gen_text:
                continue

            reward = score_reward(q, gen_text)

            full_text = prompt + " " + gen_text
            enc_full = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=cfg.max_prompt_len + cfg.max_resp_len).to(device)
            input_ids = enc_full["input_ids"]
            attn = enc_full["attention_mask"]

            labels = input_ids.clone()
            prompt_len = enc_prompt["input_ids"].shape[-1]
            labels[:, :prompt_len] = -100

            out = policy(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
            nll = out.loss
            if nll is None:
                continue
            if not nll.requires_grad:
                # This indicates trainable adapters were not attached for this forward path.
                # Skip sample instead of crashing; summary at epoch end will expose zero updates.
                continue

            with torch.no_grad():
                ref_out = reference(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
                ref_nll = ref_out.loss

            # True policy update signal from reward:
            # minimize reward * NLL  (positive reward => lower NLL, negative reward => higher NLL)
            # plus KL anchor to reference to avoid collapse.
            loss = (reward * nll) + cfg.kl_coef * (nll - ref_nll).pow(2)
            loss.backward()

            if (idx + 1) % cfg.batch_size == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                update_steps += 1

            total_loss += float(loss.detach().item())

        print(f"[epoch {epoch + 1}] policy_loss={total_loss / max(len(rows), 1):.6f} | updates={update_steps}")
        if update_steps == 0:
            raise RuntimeError(
                "No policy update steps executed (zero gradient updates). "
                "Re-run with a fresh runtime and USE_4BIT=0."
            )

    Path(cfg.adapter_out).mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(cfg.adapter_out)
    tokenizer.save_pretrained(cfg.adapter_out)
    return cfg.adapter_out


def main() -> None:
    install_requirements()
    cfg = CFG()
    rm_dir = train_reward_model(cfg)
    adapter_dir = policy_gradient_update(cfg, rm_dir)

    zip_path = "/content/rlhf_lora_adapter.zip"
    _pack_zip(adapter_dir, zip_path)
    print(f"Saved adapter zip: {zip_path}")

    try:
        from google.colab import files  # type: ignore

        files.download(zip_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
