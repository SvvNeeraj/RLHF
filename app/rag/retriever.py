import os
from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import settings


class LocalRetriever:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self.store = None

    def load(self) -> None:
        if self.store is not None:
            return
        if not os.path.exists(settings.vector_store_path):
            raise FileNotFoundError(
                f"Vector store not found at {settings.vector_store_path}. Run scripts/build_faiss.py first."
            )
        self.store = FAISS.load_local(
            settings.vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, query: str, k: int = 3) -> tuple[str, list[str]]:
        self.load()
        docs = self.store.similarity_search(query, k=k)
        context = "\n\n".join(doc.page_content for doc in docs)
        sources = [doc.metadata.get("source", "unknown") for doc in docs]
        return context, sources


@lru_cache(maxsize=1)
def get_retriever() -> LocalRetriever:
    return LocalRetriever()
