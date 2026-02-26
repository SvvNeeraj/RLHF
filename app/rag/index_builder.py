import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import settings
from app.rag.pdf_loader import (
    cleanup_objects,
    discover_pdf_files,
    is_priority_subject_pdf,
    load_pdf_pages_limited,
    pages_limit_for_pdf,
)
from app.utils.memory import clean_memory

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "been", "have", "has", "had",
    "will", "shall", "can", "could", "would", "should", "into", "onto", "your", "their", "there", "about",
    "where", "when", "what", "which", "while", "also", "than", "then", "them", "they", "you", "our", "its",
    "not", "all", "any", "but", "how", "why", "who", "whom", "use", "used", "using", "such", "these", "those",
    "one", "two", "three", "four", "five", "more", "less", "very", "only", "each", "other", "same", "over",
    "under", "between", "within", "without", "through", "across", "before", "after", "during", "above", "below",
}


def _sanitize_chunks(chunks: list) -> list:
    cleaned = []
    for ch in chunks:
        text = ch.page_content
        if not isinstance(text, str):
            continue
        text = " ".join(text.split())
        if not text or len(text) < 20:
            continue
        ch.page_content = text
        cleaned.append(ch)
    return cleaned


def _metadata_path() -> Path:
    return Path(settings.vector_store_metadata_path)


def _keyword_lexicon_path() -> Path:
    return Path(settings.keyword_lexicon_path)


def _tokenize(text: str) -> list[str]:
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9/+\-.#]*", text.lower())
    out = []
    for tok in raw:
        tok = tok.strip("._-")
        if len(tok) < 3:
            continue
        if tok in STOPWORDS:
            continue
        out.append(tok)
    return out


def _update_keyword_counters(texts: list[str], unigrams: Counter, bigrams: Counter) -> None:
    for text in texts:
        toks = _tokenize(text)
        if not toks:
            continue
        unigrams.update(toks)
        bigrams.update(f"{a} {b}" for a, b in zip(toks, toks[1:]) if a not in STOPWORDS and b not in STOPWORDS)


def global_index_exists() -> bool:
    index_dir = Path(settings.global_index_path)
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


def get_document_metadata() -> dict:
    meta_path = _metadata_path()
    if not meta_path.exists():
        return {"documents": [], "total_pdfs": 0, "total_chunks": 0}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {"documents": [], "total_pdfs": 0, "total_chunks": 0}


def _save_keyword_lexicon(
    unigrams: Counter,
    bigrams: Counter,
    subjects_seen: set[str],
    subject_unigrams: dict[str, Counter],
) -> None:
    top_unigrams = [k for k, v in unigrams.most_common(8000) if v >= 2]
    top_bigrams = [k for k, v in bigrams.most_common(5000) if v >= 2]
    per_subject = {}
    for subject, ctr in subject_unigrams.items():
        if not subject:
            continue
        words = [k for k, v in ctr.most_common(1200) if v >= 2]
        if words:
            per_subject[subject] = words
    lexicon = {
        "keywords": sorted(set(top_unigrams + top_bigrams)),
        "subjects": sorted({s for s in subjects_seen if s}),
        "subject_keywords": per_subject,
        "stats": {
            "unigrams": len(top_unigrams),
            "bigrams": len(top_bigrams),
            "subjects": len(per_subject),
        },
    }
    _keyword_lexicon_path().write_text(json.dumps(lexicon, indent=2), encoding="utf-8")


def build_global_index(force_rebuild: bool = False) -> dict:
    index_dir = Path(settings.global_index_path)
    temp_index_dir = Path(settings.vector_store_root) / "_tmp_global_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    if global_index_exists() and not force_rebuild:
        return {"status": "skipped", "reason": "global index already exists", **get_document_metadata()}

    if temp_index_dir.exists():
        shutil.rmtree(temp_index_dir, ignore_errors=True)
    temp_index_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = discover_pdf_files(settings.data_root)
    if not pdf_files:
        out = {
            "status": "empty",
            "documents": [],
            "total_pdfs": 0,
            "total_chunks": 0,
            "index_path": str(index_dir),
        }
        _metadata_path().write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    store = None
    logs = []
    total_chunks = 0
    unigram_counter: Counter = Counter()
    bigram_counter: Counter = Counter()
    subjects_seen: set[str] = set()
    subject_unigrams: dict[str, Counter] = defaultdict(Counter)

    for pdf in pdf_files:
        size_mb = round(pdf.stat().st_size / (1024 * 1024), 2)

        try:
            max_pages = pages_limit_for_pdf(pdf, settings.max_pages_per_pdf)
            pages = load_pdf_pages_limited(pdf, max_pages=max_pages)
        except Exception as exc:
            logs.append(
                {
                    "file_name": pdf.name,
                    "file_path": str(pdf),
                    "subject": pdf.parent.name,
                    "size_mb": size_mb,
                    "status": f"skipped_error: {str(exc)[:120]}",
                    "page_count": 0,
                    "chunk_count": 0,
                }
            )
            continue

        chunks = splitter.split_documents(pages) if pages else []
        chunks = _sanitize_chunks(chunks)

        texts = [str(ch.page_content) for ch in chunks]
        metas = [dict(ch.metadata) for ch in chunks]
        _update_keyword_counters(texts, unigram_counter, bigram_counter)
        subject_name = str(metas[0].get("subject", pdf.parent.name)).strip() if metas else str(pdf.parent.name).strip()
        if subject_name:
            subjects_seen.add(subject_name)
            for text in texts:
                subject_unigrams[subject_name].update(_tokenize(text))

        indexed_here = 0
        if texts:
            try:
                if store is None:
                    store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metas)
                else:
                    store.add_texts(texts=texts, metadatas=metas)
                indexed_here = len(texts)
            except Exception:
                for text, meta in zip(texts, metas):
                    try:
                        if store is None:
                            store = FAISS.from_texts(texts=[text], embedding=embeddings, metadatas=[meta])
                        else:
                            store.add_texts(texts=[text], metadatas=[meta])
                        indexed_here += 1
                    except Exception:
                        continue

        total_chunks += indexed_here

        logs.append(
            {
                "file_name": pdf.name,
                "file_path": str(pdf),
                "subject": metas[0].get("subject", "General") if metas else pdf.parent.name,
                "size_mb": size_mb,
                "status": "indexed_partial" if size_mb > settings.max_pdf_size_mb else "indexed",
                "page_count": len(pages),
                "chunk_count": indexed_here,
            }
        )

        cleanup_objects(pages, chunks)
        clean_memory()

    if store is None:
        out = {
            "status": "empty_chunks",
            "documents": logs,
            "total_pdfs": len(logs),
            "total_chunks": 0,
            "index_path": str(index_dir),
        }
        _metadata_path().write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    store.save_local(str(temp_index_dir))

    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)
    temp_index_dir.rename(index_dir)

    cleanup_objects(store)
    clean_memory()

    _save_keyword_lexicon(unigram_counter, bigram_counter, subjects_seen, subject_unigrams)

    out = {
        "status": "ok",
        "documents": logs,
        "total_pdfs": len(logs),
        "total_chunks": total_chunks,
        "index_path": str(index_dir),
        "limits": {
            "max_pdf_size_mb": settings.max_pdf_size_mb,
            "max_pages_per_pdf": settings.max_pages_per_pdf,
        },
    }
    _metadata_path().write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
