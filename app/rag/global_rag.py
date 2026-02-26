import re
import json
from functools import lru_cache
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import settings
from app.rag.content_cleaner import clean_retrieved_text
from app.rag.index_builder import global_index_exists, get_document_metadata


class GlobalRAG:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self.store = None

    def load(self) -> None:
        if self.store is not None:
            return
        if not global_index_exists():
            raise FileNotFoundError(
                "Global index not found. Run POST /rebuild-index or python scripts/build_faiss.py first."
            )

        self.store = FAISS.load_local(
            settings.global_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    @staticmethod
    def _detect_preferred_domains(query: str) -> set[str]:
        q = query.lower()
        domains: set[str] = set()

        if any(
            k in q
            for k in [
                "circuit",
                "circuit theory",
                "operational amplifier",
                "op amp",
                "op-amp",
                "sampling theorem",
                "aliasing",
                "signal",
                "control",
                "analog",
                "digital electronics",
                "communication",
                "am fm",
                "amplitude modulation",
                "frequency modulation",
                "phase modulation",
            ]
        ):
            domains.update({"ece", "eee"})
        if any(k in q for k in ["power system", "electrical machine", "power factor", "transformer", "relay"]):
            domains.add("eee")
        if any(k in q for k in ["network security", "sql injection", "xss", "malware", "cryptography", "cyber"]):
            domains.add("cyber security")
        if any(k in q for k in ["dbms", "operating system", "computer network", "data structure", "algorithm", "it"]):
            domains.add("it")

        return domains

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_subject_lexicon() -> dict[str, set[str]]:
        path = Path(settings.keyword_lexicon_path)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            per_subject = payload.get("subject_keywords", {})
            out: dict[str, set[str]] = {}
            for subject, kws in per_subject.items():
                key = str(subject).strip().lower()
                vals = {str(x).strip().lower() for x in (kws or []) if str(x).strip()}
                if key and vals:
                    out[key] = vals
            return out
        except Exception:
            return {}

    @staticmethod
    def _domain_alias(domain: str) -> str:
        d = str(domain).strip().lower()
        aliases = {
            "cybersecurity": "cyber security",
            "cyber_security": "cyber security",
            "data science": "ds",
            "data_science": "ds",
            "ds notes": "ds",
            "ds notess": "ds",
            "cse notes": "cse",
            "aiml notes": "aiml",
        }
        return aliases.get(d, d)

    def _lexicon_preferred_domains(self, query: str) -> set[str]:
        q_terms = self._query_terms(query)
        if not q_terms:
            return set()
        per_subject = self._load_subject_lexicon()
        if not per_subject:
            return set()

        hits: list[tuple[str, int]] = []
        for subject, kws in per_subject.items():
            score = sum(1 for t in q_terms if t in kws)
            if score > 0:
                hits.append((self._domain_alias(subject), score))
        if not hits:
            return set()
        hits.sort(key=lambda x: x[1], reverse=True)
        top = hits[0][1]
        return {subj for subj, sc in hits if sc >= max(1, top - 1)}

    @staticmethod
    def _source_domain(source: dict) -> str:
        subject = str(source.get("subject", "")).strip().lower()
        if subject:
            return GlobalRAG._domain_alias(subject)
        file_path = str(source.get("file_path", "")).lower()
        for key in ("ece", "eee", "it", "cyber security", "cyber_security", "cybersecurity", "cse", "aiml", "civil", "mechanical", "ds"):
            if key in file_path:
                return GlobalRAG._domain_alias(key)
        return ""

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#-]*", query.lower())
        stop = {
            "what", "is", "are", "the", "with", "and", "for", "how", "why", "can", "does",
            "explain", "define", "compare", "between", "difference", "describe", "importance",
            "advantages", "disadvantages", "example", "examples", "topic", "engineering", "system",
        }
        return {t for t in tokens if len(t) >= 4 and t not in stop}

    @staticmethod
    def _overlap_score(text: str, terms: set[str]) -> int:
        if not terms:
            return 0
        low = text.lower()
        return sum(1 for t in terms if t in low)

    @staticmethod
    def _hybrid_rank(row: tuple[str, dict], preferred_domains: set[str]) -> float:
        _, source = row
        sem_score = float(source.get("score", 10.0))
        sem_sim = 1.0 / (1.0 + max(0.0, sem_score))
        overlap = float(source.get("term_overlap", 0))
        domain = GlobalRAG._source_domain(source)
        domain_bonus = 0.22 if preferred_domains and domain in preferred_domains else 0.0
        overlap_bonus = min(0.40, overlap * 0.12)
        return sem_sim + overlap_bonus + domain_bonus

    def retrieve(self, query: str, k: int = 5) -> tuple[str, list[dict]]:
        self.load()
        fetch_k = max(k * 12, 60)
        scored_docs = self.store.similarity_search_with_score(query, k=fetch_k)
        scored_docs = sorted(scored_docs, key=lambda item: item[1])
        preferred_domains = self._detect_preferred_domains(query).union(self._lexicon_preferred_domains(query))
        query_terms = self._query_terms(query)

        all_rows = []
        for doc, score in scored_docs:
            cleaned = clean_retrieved_text(str(doc.page_content or ""))
            if not cleaned:
                continue
            overlap = self._overlap_score(cleaned, query_terms)
            source = {
                "file_name": doc.metadata.get("file_name", "unknown"),
                "file_path": doc.metadata.get("file_path", "unknown"),
                "page_number": doc.metadata.get("page_number", 0),
                "subject": doc.metadata.get("subject", ""),
                "score": float(score),
                "term_overlap": int(overlap),
            }
            all_rows.append((cleaned, source))

        # lexical prefilter keeps query-aligned rows before hybrid rerank
        candidate_rows = all_rows
        if query_terms:
            strong = [row for row in all_rows if int(row[1].get("term_overlap", 0)) >= 2]
            weak = [row for row in all_rows if int(row[1].get("term_overlap", 0)) >= 1]
            if strong:
                candidate_rows = strong
            elif weak:
                candidate_rows = weak

        ranked = sorted(candidate_rows, key=lambda row: self._hybrid_rank(row, preferred_domains), reverse=True)
        selected = []
        seen = set()
        for row in ranked:
            key = (row[1].get("file_path"), row[1].get("page_number"))
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            if len(selected) >= k:
                break

        context_chunks = [row[0] for row in selected]
        sources = [row[1] for row in selected]

        context = "\n\n".join(context_chunks)
        return context, sources

    def list_documents(self) -> dict:
        return get_document_metadata()


@lru_cache(maxsize=1)
def get_global_rag() -> GlobalRAG:
    return GlobalRAG()
