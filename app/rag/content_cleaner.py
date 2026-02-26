import re

NOISE_PATTERNS = [
    r"\bUNIT\s*[IVX0-9]+\b",
    r"\bCHAPTER\s*\d+\b",
    r"\bTABLE\s+OF\s+CONTENTS\b",
    r"\bSELF[-\s]*ASSESSMENT\b",
    r"\bEXERCISE\b",
    r"\bREVIEW\s+QUESTIONS\b",
    r"\bOBJECTIVE\s+QUESTIONS\b",
    r"\bLECTURE\b",
    r"\bTUTORIAL\b",
    r"\bDEPARTMENT\s+OF\b",
    r"\bWEEK\s*0?\d+\b",
    r"\bROYAL\s+INSTITUTE\b",
    r"\bNPTEL\b",
    r"^\s*\d+(\.\d+)+\s*$",
]

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _looks_noisy(text: str) -> bool:
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in NOISE_PATTERNS)


def _drop_noise_lines(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if len(line) < 20:
            continue
        upper_ratio = sum(c.isupper() for c in line) / max(len(line), 1)
        if upper_ratio > 0.6 and len(line.split()) <= 12:
            continue
        if _looks_noisy(line):
            continue
        if re.match(r"^(q\.?\s*\d+|question\s*\d+|ans\.?\s*\d+)", line, flags=re.IGNORECASE):
            continue
        lines.append(line)
    return " ".join(lines)


def clean_retrieved_text(text: str) -> str:
    cleaned = _drop_noise_lines(text)
    parts = SENTENCE_SPLIT.split(cleaned)

    kept = []
    for s in parts:
        sent = re.sub(r"\s+", " ", s).strip()
        if len(sent) < 35:
            continue
        if _looks_noisy(sent):
            continue
        if re.search(r"\b(mark\s+the\s+correct|choose\s+the\s+correct|fill\s+in\s+the\s+blank)\b", sent, flags=re.IGNORECASE):
            continue
        kept.append(sent)

    result = " ".join(kept)
    result = re.sub(r"\s+", " ", result).strip()
    return result
