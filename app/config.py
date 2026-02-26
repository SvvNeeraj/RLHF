import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    app_name: str = "Reinforcement-Aligned Academic Intelligence System"
    sqlite_path: str = os.getenv("SQLITE_PATH", "academic_ai.db")
    model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    model_candidates: str = os.getenv(
        "MODEL_CANDIDATES",
        "Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct",
    )
    adapter_path: str = os.getenv("ADAPTER_PATH", "adapters/lora_adapter")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    data_root: str = os.getenv("DATA_ROOT", "data")
    vector_store_root: str = os.getenv("VECTOR_STORE_ROOT", "vector_store")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "700"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    max_pdf_size_mb: int = int(os.getenv("MAX_PDF_SIZE_MB", "300"))
    max_pages_per_pdf: int = int(os.getenv("MAX_PAGES_PER_PDF", "120"))
    low_reward_threshold: float = float(os.getenv("LOW_REWARD_THRESHOLD", "0.0"))
    auth_secret: str = os.getenv("AUTH_SECRET", "change_me_strong_secret_key")
    auth_token_exp_minutes: int = int(os.getenv("AUTH_TOKEN_EXP_MINUTES", "720"))
    default_admin_user: str = os.getenv("DEFAULT_ADMIN_USER", "RLHF999")
    default_admin_password: str = os.getenv("DEFAULT_ADMIN_PASSWORD", "NT@1335")
    rlhf_feedback_jsonl_path: str = os.getenv("RLHF_FEEDBACK_JSONL_PATH", "data/rlhf/feedback_events.jsonl")
    rlhf_training_jsonl_path: str = os.getenv("RLHF_TRAINING_JSONL_PATH", "data/rlhf/training_examples.jsonl")
    rlhf_state_json_path: str = os.getenv("RLHF_STATE_JSON_PATH", "data/rlhf/online_state.json")
    rlhf_similarity_threshold: float = float(os.getenv("RLHF_SIMILARITY_THRESHOLD", "0.78"))
    rlhf_rewrite_attempts: int = int(os.getenv("RLHF_REWRITE_ATTEMPTS", "2"))

    @property
    def parsed_model_candidates(self) -> list[str]:
        raw = [x.strip() for x in str(self.model_candidates or "").split(",")]
        out = [x for x in raw if x]
        if self.model_name and self.model_name not in out:
            out.insert(0, self.model_name)
        return out or [self.model_name]

    @property
    def global_index_path(self) -> str:
        return str(Path(self.vector_store_root) / "global_index")

    @property
    def vector_store_metadata_path(self) -> str:
        return str(Path(self.vector_store_root) / "metadata.json")

    @property
    def keyword_lexicon_path(self) -> str:
        return str(Path(self.vector_store_root) / "keyword_lexicon.json")


settings = Settings()
