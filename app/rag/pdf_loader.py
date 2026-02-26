import gc
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

EXCLUDED_TOP_DIRS = {
    "app",
    "scripts",
    "colab",
    "vector_store",
    "adapters",
    "__pycache__",
    ".git",
    ".venv",
}

PRIORITY_SUBJECT_DIRS = {
    "it",
    "eee",
    "ece",
    "cyber security",
    "cyber_security",
    "cybersecurity",
}


def _iter_pdf_files(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"), key=lambda p: str(p).lower())


def _is_subject_like_folder(name: str) -> bool:
    lname = name.strip().lower()
    if lname in PRIORITY_SUBJECT_DIRS:
        return True
    return lname not in EXCLUDED_TOP_DIRS


def discover_pdf_files(data_root: str) -> list[Path]:
    root = Path(data_root)
    files: set[Path] = set(_iter_pdf_files(root))

    # Also include subject folders placed directly at project root
    # (IT/EEE/ECE/Cyber security/etc.) even when data/ already has PDFs.
    cwd = Path(".")
    for entry in cwd.iterdir():
        if not entry.is_dir():
            continue
        if entry.resolve() == root.resolve():
            continue
        if not _is_subject_like_folder(entry.name):
            continue
        for pdf in _iter_pdf_files(entry):
            files.add(pdf)
    return sorted(files, key=lambda p: str(p).lower())


def _infer_subject(pdf_path: Path) -> str:
    parts = [p.strip() for p in pdf_path.parts if p and p not in (".",)]
    if not parts:
        return "General"

    lower_parts = [p.lower() for p in parts]
    if "data" in lower_parts:
        idx = lower_parts.index("data")
        if idx + 1 < len(parts):
            subject = parts[idx + 1]
            if subject.lower() != "docs":
                return subject

    return pdf_path.parent.name or "General"


def pages_limit_for_pdf(pdf_path: Path, default_max_pages: int) -> int:
    """Use deeper page limits for priority domains to improve source relevance."""
    subject = _infer_subject(pdf_path).strip().lower()
    if subject in PRIORITY_SUBJECT_DIRS:
        return 100000  # effectively "read full PDF" under lazy loading
    return default_max_pages


def is_priority_subject_pdf(pdf_path: Path) -> bool:
    subject = _infer_subject(pdf_path).strip().lower()
    if subject in PRIORITY_SUBJECT_DIRS:
        return True
    path_text = str(pdf_path).lower()
    return any(key in path_text for key in PRIORITY_SUBJECT_DIRS)


def load_pdf_pages_limited(pdf_path: Path, max_pages: int) -> list:
    loader = PyPDFLoader(str(pdf_path))
    docs = []
    subject = _infer_subject(pdf_path)

    # lazy_load keeps memory bounded on large PDFs
    for i, d in enumerate(loader.lazy_load()):
        if i >= max_pages:
            break
        d.metadata["file_name"] = pdf_path.name
        d.metadata["file_path"] = str(pdf_path)
        d.metadata["page_number"] = int(d.metadata.get("page", 0)) + 1
        d.metadata["subject"] = subject
        docs.append(d)

    return docs


def cleanup_objects(*objs) -> None:
    for obj in objs:
        del obj
    gc.collect()
