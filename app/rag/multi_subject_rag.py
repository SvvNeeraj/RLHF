import json
from functools import lru_cache
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import settings
from app.rag.folder_loader import discover_subject_folders, slugify


class MultiSubjectRAG:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self._current_subject: str | None = None
        self._current_store = None

    def available_subjects(self) -> list[str]:
        meta_file = Path(settings.vector_store_metadata_path)
        if meta_file.exists():
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            return [row["subject"] for row in data.get("subjects", [])]

        discovered = discover_subject_folders(settings.data_root)
        return [item.subject_name for item in discovered]

    def _subject_index_path(self, subject: str) -> Path:
        return Path(settings.vector_store_root) / slugify(subject)

    def _load_subject_store(self, subject: str) -> None:
        if self._current_subject == subject and self._current_store is not None:
            return

        path = self._subject_index_path(subject)
        if not path.exists():
            raise FileNotFoundError(
                f"Subject index not found for '{subject}'. Run /rebuild-index first. Expected: {path}"
            )

        self._current_store = FAISS.load_local(
            str(path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self._current_subject = subject

    def retrieve(self, query: str, subject: str, k: int = 3) -> tuple[str, list[str]]:
        self._load_subject_store(subject)
        docs = self._current_store.similarity_search(query, k=k)
        context = "\n\n".join(doc.page_content for doc in docs)
        sources = [doc.metadata.get("source", "unknown") for doc in docs]
        return context, sources


@lru_cache(maxsize=1)
def get_multi_subject_rag() -> MultiSubjectRAG:
    return MultiSubjectRAG()
