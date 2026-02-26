from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.rag.pdf_loader import cleanup_objects, load_pdf_pages_limited
from app.utils.memory import clean_memory


def _load_metadata(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {"status": "ok", "documents": [], "total_pdfs": 0, "total_chunks": 0}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "ok", "documents": [], "total_pdfs": 0, "total_chunks": 0}


def _save_metadata(meta_path: Path, meta: dict) -> None:
    docs = meta.get("documents", [])
    meta["status"] = "ok"
    meta["total_pdfs"] = len(docs)
    meta["total_chunks"] = int(sum(int(d.get("chunk_count", 0)) for d in docs))
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally index specific subject folders into global FAISS.")
    parser.add_argument("--folders", nargs="+", required=True, help="Subject folders to index, e.g. ECE EEE 'Cyber Security'")
    parser.add_argument("--max-pages", type=int, default=80, help="Max pages per PDF for this incremental pass.")
    args = parser.parse_args()

    index_dir = Path(settings.global_index_path)
    if not index_dir.exists():
        raise SystemExit(f"Global index missing at {index_dir}. Run python scripts/build_faiss.py first.")

    meta_path = Path(settings.vector_store_metadata_path)
    meta = _load_metadata(meta_path)
    docs = meta.get("documents", [])
    indexed_paths = {str(d.get("file_path", "")) for d in docs if d.get("status") == "indexed"}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    store = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)

    new_logs = []
    total_new_chunks = 0

    pdf_files: list[Path] = []
    for folder in args.folders:
        root = Path(folder)
        if not root.exists() or not root.is_dir():
            print(f"skip missing folder: {folder}")
            continue
        pdf_files.extend(sorted(root.rglob("*.pdf"), key=lambda p: str(p).lower()))

    print(f"found pdfs={len(pdf_files)} in folders={args.folders}")

    for i, pdf in enumerate(pdf_files, start=1):
        file_path = str(pdf)
        if file_path in indexed_paths:
            continue

        try:
            pages = load_pdf_pages_limited(pdf, max_pages=args.max_pages)
        except Exception as exc:
            new_logs.append(
                {
                    "file_name": pdf.name,
                    "file_path": file_path,
                    "subject": pdf.parent.name,
                    "size_mb": round(pdf.stat().st_size / (1024 * 1024), 2),
                    "status": f"skipped_error: {str(exc)[:120]}",
                    "page_count": 0,
                    "chunk_count": 0,
                }
            )
            continue

        chunks = splitter.split_documents(pages) if pages else []
        cleaned = []
        for ch in chunks:
            text = " ".join(str(ch.page_content).split())
            if len(text) < 20:
                continue
            ch.page_content = text
            cleaned.append(ch)

        texts = [c.page_content for c in cleaned]
        metas = [dict(c.metadata) for c in cleaned]
        indexed_here = 0

        if texts:
            try:
                store.add_texts(texts=texts, metadatas=metas)
                indexed_here = len(texts)
            except Exception:
                for text, md in zip(texts, metas):
                    try:
                        store.add_texts(texts=[text], metadatas=[md])
                        indexed_here += 1
                    except Exception:
                        continue

        total_new_chunks += indexed_here
        new_logs.append(
            {
                "file_name": pdf.name,
                "file_path": file_path,
                "subject": metas[0].get("subject", pdf.parent.name) if metas else pdf.parent.name,
                "size_mb": round(pdf.stat().st_size / (1024 * 1024), 2),
                "status": "indexed" if indexed_here > 0 else "empty_chunks",
                "page_count": len(pages),
                "chunk_count": indexed_here,
            }
        )

        if i % 10 == 0:
            print(f"processed {i}/{len(pdf_files)}")

        cleanup_objects(pages, chunks, cleaned)
        clean_memory()

    store.save_local(str(index_dir))
    meta["documents"] = docs + new_logs
    _save_metadata(meta_path, meta)

    print(
        {
            "new_docs": len(new_logs),
            "new_chunks": total_new_chunks,
            "total_pdfs": meta["total_pdfs"],
            "total_chunks": meta["total_chunks"],
        }
    )


if __name__ == "__main__":
    main()
