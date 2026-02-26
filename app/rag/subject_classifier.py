from collections import defaultdict

DEFAULT_KEYWORDS = {
    "mechanical": ["mechanical", "thermo", "fluid", "strength", "machine", "mechatronics", "heat transfer"],
    "civil": ["civil", "survey", "geotechnical", "concrete", "structural", "environment", "transportation"],
    "cse_notes": ["cse", "dbms", "operating system", "os", "compiler", "dsa", "network", "java"],
    "ds_notes": ["data science", "data mining", "analytics", "data warehouse", "ds"],
    "aiml_notes": ["ai", "ml", "machine learning", "deep learning", "nlp", "computer vision", "neural"],
}


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def build_subject_keywords(subjects: list[str]) -> dict[str, list[str]]:
    keywords: dict[str, list[str]] = {}
    for subject in subjects:
        key = _normalize(subject)
        base_tokens = [t for t in key.replace("_", " ").split() if len(t) > 1]
        merged = list(dict.fromkeys(base_tokens + DEFAULT_KEYWORDS.get(key, [])))
        keywords[subject] = merged
    return keywords


def detect_subject(query: str, subjects: list[str]) -> str | None:
    if not subjects:
        return None

    q = query.lower()
    kw_map = build_subject_keywords(subjects)
    scores = defaultdict(int)

    for subject, keywords in kw_map.items():
        for kw in keywords:
            if kw and kw in q:
                scores[subject] += len(kw)

    if scores:
        return max(scores, key=scores.get)

    # Fallback: simple token overlap with subject names.
    q_tokens = set(q.replace("_", " ").split())
    best_subject = None
    best_overlap = 0
    for subject in subjects:
        s_tokens = set(subject.lower().replace("_", " ").split())
        overlap = len(q_tokens & s_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_subject = subject

    return best_subject if best_overlap > 0 else subjects[0]
