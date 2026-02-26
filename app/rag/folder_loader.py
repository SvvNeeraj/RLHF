import re
from dataclasses import dataclass
from pathlib import Path

IGNORED_DIRS = {
    "vector_store",
    "docs",
    "app",
    "scripts",
    "colab",
    "adapters",
    "__pycache__",
    ".git",
    ".venv",
}


@dataclass
class SubjectFolder:
    subject_name: str
    subject_slug: str
    folder_path: Path
    pdf_files: list[Path]


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return slug or "subject"


def _collect_pdf_subjects(root: Path) -> list[SubjectFolder]:
    subjects: list[SubjectFolder] = []
    if not root.exists() or not root.is_dir():
        return subjects

    for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir():
            continue
        if entry.name.lower() in IGNORED_DIRS:
            continue
        pdfs = sorted(entry.rglob("*.pdf"))
        if not pdfs:
            continue
        subjects.append(
            SubjectFolder(
                subject_name=entry.name,
                subject_slug=slugify(entry.name),
                folder_path=entry,
                pdf_files=pdfs,
            )
        )
    return subjects


def discover_subject_folders(data_root: str = "data") -> list[SubjectFolder]:
    root = Path(data_root)
    subjects = _collect_pdf_subjects(root)

    # Backward-compatible fallback for existing repos with subject folders at project root.
    if subjects:
        return subjects

    legacy_root = Path(".")
    return _collect_pdf_subjects(legacy_root)
