import json
import re
from functools import lru_cache
from pathlib import Path

from app.config import settings

# Core phrases by major B.Tech branches.
DOMAIN_CORE_PHRASES = {
    "it": {
        "information technology",
        "operating system",
        "process scheduling",
        "cpu scheduling",
        "deadlock",
        "memory management",
        "virtual memory",
        "dbms",
        "normalization",
        "acid properties",
        "indexing",
        "sql",
        "computer networks",
        "tcp/ip",
        "osi model",
        "routing",
        "software engineering",
        "web technology",
        "cloud computing",
        "data structures",
        "algorithms",
    },
    "cse": {
        "computer science engineering",
        "compiler design",
        "automata",
        "theory of computation",
        "discrete mathematics",
        "microprocessor",
        "object oriented programming",
        "java",
        "c++",
        "python",
        "operating systems",
        "dbms",
        "computer architecture",
        "computer networks",
        "software engineering",
        "distributed systems",
        "data structures",
        "design and analysis of algorithms",
    },
    "aiml": {
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "neural network",
        "cnn",
        "rnn",
        "transformer",
        "attention mechanism",
        "nlp",
        "computer vision",
        "reinforcement learning",
        "feature engineering",
        "overfitting",
        "underfitting",
        "model evaluation",
        "gradient descent",
        "supervised learning",
        "unsupervised learning",
    },
    "data_science": {
        "data science",
        "data analytics",
        "data mining",
        "data warehouse",
        "statistics",
        "probability",
        "regression",
        "classification",
        "clustering",
        "time series",
        "hypothesis testing",
        "eda",
        "feature selection",
        "python pandas",
        "numpy",
        "data visualization",
        "business intelligence",
    },
    "ece": {
        "electronics and communication engineering",
        "signals and systems",
        "analog electronics",
        "digital electronics",
        "communication systems",
        "control systems",
        "operational amplifier",
        "op-amp",
        "circuit theory",
        "sampling theorem",
        "aliasing",
        "modulation",
        "am fm pm",
        "adc",
        "dac",
        "semiconductor",
        "pn junction",
        "microcontrollers",
        "vlsi",
    },
    "eee": {
        "electrical and electronics engineering",
        "power systems",
        "power factor",
        "load flow",
        "power flow",
        "electrical machines",
        "induction motor",
        "synchronous machine",
        "transformer",
        "relay protection",
        "switchgear",
        "power electronics",
        "control systems",
        "stability analysis",
        "generation transmission distribution",
    },
    "cyber_security": {
        "cyber security",
        "network security",
        "information security",
        "cryptography",
        "encryption",
        "authentication",
        "authorization",
        "firewall",
        "ids",
        "ips",
        "phishing",
        "malware",
        "ransomware",
        "sql injection",
        "xss",
        "csrf",
        "incident response",
        "vulnerability assessment",
        "penetration testing",
        "ethical hacking",
    },
    "civil": {
        "civil engineering",
        "surveying",
        "structural analysis",
        "strength of materials",
        "geotechnical engineering",
        "soil mechanics",
        "fluid mechanics",
        "transportation engineering",
        "environmental engineering",
        "concrete technology",
        "construction management",
        "hydrology",
    },
    "mechanical": {
        "mechanical engineering",
        "thermodynamics",
        "heat transfer",
        "fluid mechanics",
        "machine design",
        "manufacturing process",
        "strength of materials",
        "theory of machines",
        "refrigeration and air conditioning",
        "internal combustion engine",
        "automobile engineering",
        "mechatronics",
    },
}

BASE_BTECH_TOPICS = {
    "engineering",
    "btech",
    "b.tech",
    "it",
    "cse",
    "aiml",
    "data science",
    "ece",
    "eee",
    "cyber security",
    "civil",
    "mechanical",
}.union(*DOMAIN_CORE_PHRASES.values())

CORE_BTECH_TOKENS = {
    "dbms",
    "sql",
    "os",
    "tcp",
    "osi",
    "routing",
    "firewall",
    "cryptography",
    "phishing",
    "cyber",
    "ml",
    "ai",
    "neural",
    "learning",
    "circuit",
    "signals",
    "control",
    "power",
    "ece",
    "eee",
    "cse",
    "it",
    "civil",
    "mechanical",
    "algorithm",
    "algorithms",
    "thermodynamics",
    "surveying",
    "vlsi",
    "opamp",
    "transformer",
    "motor",
    "encryption",
    "backpropagation",
    "gradient",
    "analytics",
    "database",
    "mysql",
    "postgresql",
    "firewalls",
    "intrusion",
    "compiler",
    "microprocessor",
    "microcontroller",
    "semiconductor",
    "modulation",
    "loadflow",
    "powerflow",
}

GENERIC_TOKENS = {
    "explain",
    "define",
    "compare",
    "difference",
    "example",
    "real",
    "world",
    "with",
    "and",
    "for",
    "the",
    "this",
    "that",
    "topic",
    "question",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _query_tokens(query: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9/+\-.#]*", query.lower())
    out = []
    for t in tokens:
        t = t.strip("._-")
        if len(t) < 3:
            continue
        out.append(t)
    return out


def _normalize_token(token: str) -> str:
    t = token.lower().strip("._-")
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("s") and len(t) > 4:
        return t[:-1]
    return t


@lru_cache(maxsize=1)
def _load_dynamic_keywords() -> set[str]:
    path = Path(settings.keyword_lexicon_path)
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        kws = payload.get("keywords", [])
        per_subject = payload.get("subject_keywords", {})
        flat_subject = []
        if isinstance(per_subject, dict):
            for _, vals in per_subject.items():
                if isinstance(vals, list):
                    flat_subject.extend(vals)
        raw = [*kws, *flat_subject]
        out = {str(k).strip().lower() for k in raw if str(k).strip()}
        # include normalized forms to reduce plural/morphology misses
        out.update({_normalize_token(k) for k in out})
        return out
    except Exception:
        return set()


def refresh_dynamic_keywords_cache() -> None:
    _load_dynamic_keywords.cache_clear()


def _dynamic_match(query: str) -> bool:
    kws = _load_dynamic_keywords()
    if not kws:
        return False

    tokens = _query_tokens(query)
    if not tokens:
        return False

    # unigram hits on meaningful tokens
    meaningful = [t for t in tokens if t not in GENERIC_TOKENS]
    normalized = [_normalize_token(t) for t in meaningful]
    if any(t in kws for t in meaningful) or any(t in kws for t in normalized):
        return True

    # bigram hits
    bigrams = [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
    if any(bg in kws for bg in bigrams):
        return True

    # relaxed substring match for technical terms (avoid trivial short tokens)
    for t in normalized:
        if len(t) < 5:
            continue
        if any((t in kw or kw in t) for kw in kws if len(kw) >= 5):
            return True

    return False


def is_btech_query(query: str) -> bool:
    q = _normalize_text(query)
    if not q:
        return False

    # fallback static
    if any(topic in q for topic in BASE_BTECH_TOPICS):
        return True

    # token fallback for short subject terms / acronyms
    tokens = set(_query_tokens(q))
    if tokens.intersection(CORE_BTECH_TOKENS):
        return True

    # primary dynamic document-driven matching
    return _dynamic_match(q)


def rejection_message() -> str:
    return "Invalid Question: This system only answers BTech academic and engineering-related queries."


def no_context_message() -> str:
    return "Invalid Question: No relevant academic context found."
