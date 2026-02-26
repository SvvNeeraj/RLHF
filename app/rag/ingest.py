from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import settings


def build_vector_store() -> None:
    loader = DirectoryLoader(
        settings.docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=False,
    )
    docs = loader.load()
    if not docs:
        raise ValueError(f"No documents found in {settings.docs_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(settings.vector_store_path)


if __name__ == "__main__":
    build_vector_store()
    print("FAISS index built successfully.")
