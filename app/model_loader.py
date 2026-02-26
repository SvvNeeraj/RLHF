import os
import threading
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from app.config import settings
from app.utils.memory import clean_memory


class ModelRuntime:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._backend = "uninitialized"
        self._active_model_name = ""
        self._load_error = ""
        self._load_attempted = False

    def _load_base_4bit_cpu(self, model_name: str):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        return model, tokenizer

    def _load_base_dense_cpu(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        return model, tokenizer

    def _merge_adapter_if_available(self, model):
        if os.path.isdir(settings.adapter_path) and os.listdir(settings.adapter_path):
            try:
                peft_model = PeftModel.from_pretrained(model, settings.adapter_path)
                model = peft_model.merge_and_unload()
            except Exception as exc:
                # Adapter can be absent/incompatible for the chosen small model.
                self._load_error = f"{self._load_error} | adapter_merge_skipped: {exc}".strip(" |")
        return model

    def _build_fallback_answer(self, prompt: str) -> str:
        # Return empty text in CPU fallback mode so chat_service can force
        # context-grounded synthesis instead of template defaults.
        return ""

    def load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if self._load_attempted and self._backend == "retrieval_only_fallback":
            return
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return
            if self._load_attempted and self._backend == "retrieval_only_fallback":
                return
            self._load_attempted = True
            self._load_error = ""
            candidates = settings.parsed_model_candidates
            model = None
            tokenizer = None
            backend = ""

            for model_name in candidates:
                try:
                    model, tokenizer = self._load_base_4bit_cpu(model_name)
                    backend = "bnb_4bit_cpu"
                    self._active_model_name = model_name
                    break
                except Exception as exc_4bit:
                    self._load_error = f"{self._load_error} | {model_name}:4bit_failed:{exc_4bit}".strip(" |")
                    try:
                        model, tokenizer = self._load_base_dense_cpu(model_name)
                        backend = "dense_cpu_fp32"
                        self._active_model_name = model_name
                        break
                    except Exception as exc_dense:
                        self._load_error = f"{self._load_error} | {model_name}:dense_failed:{exc_dense}".strip(" |")
                        model = None
                        tokenizer = None
                        continue

            if model is None or tokenizer is None:
                self._backend = "retrieval_only_fallback"
                self._model = None
                self._tokenizer = None
                clean_memory()
                return

            model = self._merge_adapter_if_available(model)
            self._model = model.eval()
            self._tokenizer = tokenizer
            self._backend = backend
            clean_memory()

    @property
    def model(self):
        self.load()
        return self._model

    @property
    def tokenizer(self):
        self.load()
        return self._tokenizer

    @property
    def backend(self) -> str:
        self.load()
        return self._backend

    @property
    def active_model_name(self) -> str:
        self.load()
        return self._active_model_name or settings.model_name

    def generate(self, prompt: str, max_new_tokens: int = 700) -> str:
        model = self.model
        tokenizer = self.tokenizer
        if model is None or tokenizer is None:
            return self._build_fallback_answer(prompt)

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            except Exception:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
        else:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)

        if isinstance(inputs, torch.Tensor):
            model_inputs = {
                "input_ids": inputs,
                "attention_mask": torch.ones_like(inputs, dtype=torch.long),
            }
            input_len = int(inputs.shape[-1])
        else:
            model_inputs = inputs
            input_len = int(model_inputs["input_ids"].shape[-1]) if "input_ids" in model_inputs else 0
        with torch.no_grad():
            output = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                repetition_penalty=1.12,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = output[0][input_len:] if input_len > 0 else output[0]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        if not response:
            # Conservative fallback in case slicing fails for a model edge case.
            full_text = tokenizer.decode(output[0], skip_special_tokens=True)
            response = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text
        clean_memory()
        return response


runtime = ModelRuntime()
