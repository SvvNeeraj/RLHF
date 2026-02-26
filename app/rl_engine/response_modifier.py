import re
from difflib import SequenceMatcher
from app.config import settings
from app.rl_engine.topic_templates import generate_template


class ResponseModifier:
    INVALID_DOMAIN_RESPONSE = "Invalid Question: This system only answers BTech academic and engineering-related queries."
    INVALID_CONTEXT_RESPONSE = "Invalid Question: No relevant academic context found."

    META_PATTERNS = [
        r"\bthis answer\b",
        r"\bretrieval[-\s]*grounded\b",
        r"\bknowledge base\b",
        r"\bas an ai\b",
        r"\bI am running\b",
    ]

    NOISE_PATTERNS = [
        r"\b\d+\.\d+\b",
        r"\bunit\s+[ivx0-9]+\b",
        r"\bchapter\s+\d+\b",
        r"\blecture\b",
        r"\btutorial\b",
        r"\bdepartment\s+of\b",
        r"\btable\s+of\s+contents\b",
        r"\bself[-\s]*assessment\b",
    ]

    GENERIC_BAD_PATTERNS = [
        r"begin by defining",
        r"this concept",
        r"engineering learning and practice",
        r"use a simple engineering scenario",
        r"topic for b\.tech learning",
    ]

    INSTRUCTION_PATTERNS = [
        r"\bfor exam\b",
        r"\bexam writing\b",
        r"\bwrite\b",
        r"\bscoring\b",
        r"\bstate why\b",
        r"\bmention\b",
        r"\bpresent .* as\b",
        r"\balso explain\b",
        r"\bfinally, include\b",
        r"\bcause-and-effect\b",
        r"\bcan be understood better when\b",
        r"\bmost effective when assumptions\b",
        r"\bfor better clarity\b",
        r"\bfocus on\b",
    ]

    TEMPLATE_LEAK_PATTERNS = [
        r"\bcan be explained through three parts\b",
        r"\bstart by defining\b",
        r"\bclear concept-first view\b",
        r"\bexam writing\b",
        r"\bthis concept\b",
        r"\bml workflow typically includes data collection\b",
        r"\bcore concepts include overfitting, underfitting\b",
        r"\bcore purpose and scope of\b",
    ]

    LOW_QUALITY_PATTERNS = [
        r"\bdissertation\b",
        r"\bph\.?\s*d\b",
        r"\bwww\.",
        r"https?://",
        r"\bdoi\b",
        r"\bisbn\b",
        r"\btable\s+\d+\b",
        r"\bfigure\s+\d+\b",
        r"\bcopyright\b",
        r"\bpp\.\s*\d+",
    ]

    REPETITIVE_GENERIC_PATTERNS = []

    @staticmethod
    def quality_gate(query: str, response: str, mode: str = "detailed") -> dict:
        text = ResponseModifier._sanitize_text(response or "")
        reasons: list[str] = []

        if not text:
            reasons.append("empty_response")
            return {"passed": False, "reasons": reasons}

        if ResponseModifier._is_generic_or_irrelevant(text, query):
            reasons.append("generic_or_irrelevant")

        if ResponseModifier._fails_noise_or_meta(text):
            reasons.append("noise_or_meta_detected")

        if mode == "short":
            if "short answer:" not in text.lower():
                reasons.append("short_header_missing")
            if "topic examples:" not in text.lower():
                reasons.append("short_examples_missing")
            short_section = ResponseModifier._section_text(text, "Short Answer:", ["Topic Examples:"])
            if len(ResponseModifier._split_sentences(short_section)) < 2:
                reasons.append("short_too_shallow")
        else:
            required = ["short summary:", "detailed explanation:", "real-life example:", "key points:", "conclusion:"]
            missing = [r for r in required if r not in text.lower()]
            if missing:
                reasons.append(f"missing_sections:{','.join(missing)}")
            if not ResponseModifier._key_points_relevant(query, text):
                reasons.append("key_points_not_relevant")
            detailed = ResponseModifier._section_text(
                text, "Detailed Explanation:", ["Real-Life Example:", "Key Points:", "Conclusion:"]
            )
            if len(ResponseModifier._split_sentences(detailed)) < 3:
                reasons.append("detailed_too_shallow")

        return {"passed": len(reasons) == 0, "reasons": reasons}

    @staticmethod
    def is_invalid_response(response: str) -> bool:
        txt = (response or "").strip()
        return txt in {ResponseModifier.INVALID_DOMAIN_RESPONSE, ResponseModifier.INVALID_CONTEXT_RESPONSE}

    @staticmethod
    def quality_gate_query_driven(query: str, response: str, requested_titles: list[str]) -> dict:
        text = ResponseModifier._sanitize_text(response or "")
        reasons: list[str] = []
        if not text:
            return {"passed": False, "reasons": ["empty_response"]}

        if ResponseModifier.is_invalid_response(text):
            return {"passed": True, "reasons": []}

        lower = text.lower()
        for forbidden in ["short summary:", "detailed explanation:", "real-life example:", "key points:", "conclusion:"]:
            if forbidden in lower and forbidden.split(":")[0].title() not in requested_titles:
                reasons.append("contains_legacy_fixed_template")
                break

        q_terms = ResponseModifier._query_terms(query)
        if q_terms:
            hit = sum(1 for t in set(q_terms) if t in lower)
            if hit == 0:
                reasons.append("query_terms_missing")

        headers_found = 0
        for title in requested_titles:
            if f"{title.lower()}:" in lower:
                headers_found += 1
        if requested_titles and headers_found < max(1, len(requested_titles) - 1):
            reasons.append("requested_headers_missing")

        if len(text.split()) < 20:
            reasons.append("too_short")
        if ResponseModifier._fails_noise_or_meta(text):
            reasons.append("noise_or_meta_detected")

        return {"passed": len(reasons) == 0, "reasons": reasons}

    @staticmethod
    def response_overlap_score(a: str, b: str) -> float:
        ta = " ".join(re.findall(r"[a-zA-Z0-9+#-]+", (a or "").lower()))
        tb = " ".join(re.findall(r"[a-zA-Z0-9+#-]+", (b or "").lower()))
        if not ta or not tb:
            return 0.0
        seq_ratio = SequenceMatcher(None, ta, tb).ratio()
        toks_a = set(ta.split())
        toks_b = set(tb.split())
        jaccard = len(toks_a & toks_b) / max(len(toks_a | toks_b), 1)
        return max(seq_ratio, jaccard)

    @staticmethod
    def context_relevance(query: str, context: str, sources: list[dict] | None = None) -> float:
        q_terms = ResponseModifier._query_terms(query)
        if not q_terms:
            return 0.0
        ctx = (context or "").lower()
        token_hits = sum(1 for t in set(q_terms) if t in ctx)
        token_ratio = token_hits / max(len(set(q_terms)), 1)
        source_overlap = 0.0
        if sources:
            source_overlap = max(float(s.get("term_overlap", 0.0)) for s in sources)
            source_overlap = min(1.0, source_overlap / 3.0)
        return max(token_ratio, source_overlap)

    @staticmethod
    def get_adaptation_flags(query_profile: dict) -> dict:
        avg_reward = float(query_profile.get("avg_reward", 0.0))
        total_feedback = int(query_profile.get("total_feedback", 0))
        low_count = int(query_profile.get("low_rating_count", 0))

        needs_improvement = total_feedback >= 1 and (avg_reward < settings.low_reward_threshold or low_count > 0)
        return {
            "needs_improvement": needs_improvement,
            "alternative_example": needs_improvement,
            "simpler_language": needs_improvement,
            "variation_id": (total_feedback + low_count) % 3,
        }

    @staticmethod
    def style_instruction(base_mode: str, avg_reward: float, flags: dict) -> str:
        style = "Use simple B.Tech-level language with clear logical paragraphs."
        if base_mode == "short":
            style += " Keep concise while preserving key meaning."
        else:
            style += " Keep detailed and structured with smooth flow."

        if flags["needs_improvement"]:
            style += " Previous rating was low; simplify wording, add an extra example, and improve logical flow."

        if avg_reward >= 0.4:
            style += " Maintain precision."
        return style

    @staticmethod
    def _sanitize_text(text: str) -> str:
        t = text
        t = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", t)
        t = t.replace("**", "")
        for pat in ResponseModifier.META_PATTERNS + ResponseModifier.NOISE_PATTERNS:
            t = re.sub(pat, "", t, flags=re.IGNORECASE)

        lines = []
        for line in t.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                lines.append(line)
        return "\n".join(lines).strip()

    @staticmethod
    def _topic_focus_ok(query: str, text: str) -> bool:
        q_tokens = [t for t in re.findall(r"[a-zA-Z]{4,}", query.lower()) if t not in {"explain", "define", "about", "with", "example"}]
        if not q_tokens:
            return True
        ltxt = text.lower()
        return any(tok in ltxt for tok in q_tokens)

    @staticmethod
    def _has_required_sections(text: str) -> bool:
        lowered = text.lower()
        required = ["short summary:", "detailed explanation:", "real-life example:", "key points:", "conclusion:"]
        return all(r in lowered for r in required)

    @staticmethod
    def _fails_noise_or_meta(text: str) -> bool:
        pats = ResponseModifier.META_PATTERNS + ResponseModifier.NOISE_PATTERNS + ResponseModifier.GENERIC_BAD_PATTERNS
        for pat in pats:
            if re.search(pat, text, flags=re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _section_text(text: str, header: str, next_headers: list[str]) -> str:
        start = re.search(re.escape(header), text, flags=re.IGNORECASE)
        if not start:
            return ""
        start_pos = start.end()
        tail = text[start_pos:]
        end_pos = len(tail)
        for nh in next_headers:
            m = re.search(re.escape(nh), tail, flags=re.IGNORECASE)
            if m:
                end_pos = min(end_pos, m.start())
        return tail[:end_pos].strip()

    @staticmethod
    def _topic_from_query(query: str) -> str:
        q = query.strip().rstrip("?.!")
        q = re.sub(
            r"^(what(?:\s+is)?|who(?:\s+is)?|define|explain|describe|compare|differentiate|tell\s+me\s+about)\s+",
            "",
            q,
            flags=re.IGNORECASE,
        )
        q = re.sub(
            r"^(advantages?\s+and\s+limitations?\s+of|advantages?\s+of|limitations?\s+of|disadvantages?\s+of|benefits?\s+of|merits?\s+of|drawbacks?\s+of)\s+",
            "",
            q,
            flags=re.IGNORECASE,
        )
        q = re.sub(
            r"\s+(with\s+example(?:s)?|and\s+example(?:s)?|with\s+real[-\s]*life\s+example(?:s)?)\b.*$",
            "",
            q,
            flags=re.IGNORECASE,
        )
        q = re.sub(r"^(the|a|an)\s+", "", q, flags=re.IGNORECASE)
        q = re.sub(r"\s+", " ", q).strip(" ,.-")
        return q or "the topic"

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        terms = re.findall(r"[a-zA-Z][a-zA-Z0-9+#-]*", query.lower())
        stop = {"what", "is", "are", "the", "with", "and", "for", "how", "why", "can", "does", "explain", "define", "compare"}
        out = []
        for t in terms:
            if len(t) < 3 or t in stop:
                continue
            out.append(t)
        return out

    @staticmethod
    def _salient_terms(sentences: list[str], query: str, limit: int = 8) -> list[str]:
        stop = {
            "this", "that", "these", "those", "using", "used", "system", "systems", "data", "process",
            "method", "methods", "approach", "based", "within", "between", "their", "there", "where",
            "which", "while", "from", "into", "also", "such", "very", "more", "most", "through",
            "explain", "defined", "definition", "engineering", "student", "students", "practical",
        }
        q = set(ResponseModifier._query_terms(query))
        counts: dict[str, int] = {}
        for s in sentences:
            for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9+#-]*", s.lower()):
                if len(tok) < 4:
                    continue
                if tok in stop or tok in q:
                    continue
                counts[tok] = counts.get(tok, 0) + 1
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in ranked[:limit]]

    @staticmethod
    def _is_generic_or_irrelevant(text: str, query: str) -> bool:
        if not text or len(text.split()) < 40:
            return True
        lower = text.lower()
        if any(re.search(pat, lower, flags=re.IGNORECASE) for pat in ResponseModifier.TEMPLATE_LEAK_PATTERNS):
            return True
        q_terms = ResponseModifier._query_terms(query)
        if not q_terms:
            return False
        hit = sum(1 for t in set(q_terms) if t in lower)
        return hit == 0

    @staticmethod
    def _context_sentences(context: str) -> list[str]:
        raw = ResponseModifier._split_sentences(context)
        out = []
        for s in raw:
            line = re.sub(r"[^\x20-\x7E]", " ", s)
            line = re.sub(r"\bT\\s*raining\b", "Training", line, flags=re.IGNORECASE)
            line = " ".join(line.split()).strip()
            if len(line) < 40:
                continue
            if len(line) > 260:
                continue
            if ResponseModifier._is_instruction_sentence(line):
                continue
            if re.search(r"\b(unit|chapter|exercise|self-assessment|objective questions|table|figure)\b", line, flags=re.IGNORECASE):
                continue
            if re.search(r"(.{20,90})\\1", line, flags=re.IGNORECASE):
                continue
            if len(re.findall(r"\b[A-Z][a-zA-Z]+\b", line)) > 12 and "," not in line:
                continue
            letters = len(re.findall(r"[A-Za-z]", line))
            if letters < 30:
                continue
            upper = len(re.findall(r"[A-Z]", line))
            if letters > 0 and (upper / letters) > 0.45:
                continue
            if ResponseModifier._is_low_quality_sentence(line):
                continue
            out.append(line)
        return out

    @staticmethod
    def _score_sentence(sentence: str, q_terms: list[str]) -> float:
        s = sentence.lower()
        if not q_terms:
            return float(len(sentence))
        overlap = sum(1 for t in set(q_terms) if t in s)
        dense = overlap / max(len(set(q_terms)), 1)
        length_bonus = min(len(sentence), 220) / 220.0
        return dense * 3.0 + length_bonus

    @staticmethod
    def _pick_best_sentences(context: str, query: str, n: int) -> list[str]:
        sents = ResponseModifier._context_sentences(context)
        if not sents:
            return []
        q_terms = ResponseModifier._query_terms(query)
        scored = sorted(sents, key=lambda x: ResponseModifier._score_sentence(x, q_terms), reverse=True)
        picked = []
        seen = set()
        for s in scored:
            norm = re.sub(r"\W+", "", s.lower())
            if norm in seen:
                continue
            seen.add(norm)
            picked.append(s)
            if len(picked) >= n:
                break
        return picked

    @staticmethod
    def _is_noisy_sentence(sentence: str) -> bool:
        s = sentence.strip()
        if not s:
            return True
        if re.search(r"\b\d+\.\s*[A-Za-z]\b", s):
            return True
        if re.search(r"[=]{1}|f\s*\(|\(\s*[a-z]\s*;", s):
            return True
        if len(re.findall(r"\b[A-Z]{2,}\b", s)) > 4:
            return True
        return False

    @staticmethod
    def _is_low_quality_sentence(sentence: str) -> bool:
        s = " ".join((sentence or "").split()).strip()
        if not s:
            return True
        if len(s) < 35:
            return True
        if len(s) > 240:
            return True
        if any(re.search(p, s, flags=re.IGNORECASE) for p in ResponseModifier.LOW_QUALITY_PATTERNS):
            return True
        if ResponseModifier._is_noisy_sentence(s):
            return True
        letters = len(re.findall(r"[A-Za-z]", s))
        digits = len(re.findall(r"\d", s))
        if letters and (digits / letters) > 0.18:
            return True
        return False

    @staticmethod
    def _not_found_response(query: str) -> str:
        topic = ResponseModifier._topic_from_query(query)
        return (
            "Short Summary:\n"
            f"Not found in knowledge base for {topic}.\n\n"
            "Detailed Explanation:\n"
            "The indexed documents do not contain enough clean, relevant context to answer this query reliably.\n\n"
            "Real-Life Example:\n"
            "- Please ask a narrower sub-topic or include exact chapter/topic keywords from your PDFs.\n"
            "- Rebuild index after adding notes that explicitly cover this concept.\n\n"
            "Key Points:\n"
            f"- Query topic: {topic}\n"
            "- No reliable context match was found.\n"
            "- Refine question with domain-specific keywords.\n"
            "- Re-index documents if new PDFs were added.\n\n"
            "Conclusion:\n"
            "Not found in knowledge base."
        )

    @staticmethod
    def build_context_grounded_response(query: str, context: str, flags: dict | None = None) -> str:
        flags = flags or {}
        topic = ResponseModifier._topic_from_query(query)
        picked = ResponseModifier._pick_best_sentences(context, query, n=10)
        if not picked:
            return ResponseModifier._sanitize_text(ResponseModifier._not_found_response(query))

        cleaned_picked = [s for s in picked if not ResponseModifier._is_noisy_sentence(s)]
        if len(cleaned_picked) >= 4:
            picked = cleaned_picked
        elif len(cleaned_picked) >= 2:
            picked = cleaned_picked + [s for s in picked if s not in cleaned_picked][:3]
        elif cleaned_picked:
            # keep at least a few less-clean lines when context is sparse, instead of template fallback
            picked = cleaned_picked + [s for s in picked if s not in cleaned_picked][:4]
        else:
            return ResponseModifier._sanitize_text(ResponseModifier._not_found_response(query))

        q_terms = ResponseModifier._query_terms(query)
        topic_sentences = []
        for s in picked:
            if ResponseModifier._is_low_quality_sentence(s):
                continue
            if not q_terms:
                topic_sentences.append(s)
                continue
            overlap = sum(1 for t in q_terms if t in s.lower())
            if overlap >= 1:
                topic_sentences.append(s)

        if len(topic_sentences) < 3:
            for s in picked:
                if s not in topic_sentences and not ResponseModifier._is_low_quality_sentence(s):
                    topic_sentences.append(s)
                if len(topic_sentences) >= 5:
                    break
        if not topic_sentences:
            return ResponseModifier._sanitize_text(ResponseModifier._not_found_response(query))

        def_like = [
            s for s in topic_sentences
            if re.search(r"\b(is|refers to|means|defined as)\b", s, flags=re.IGNORECASE)
            and (not q_terms or any(t in s.lower() for t in q_terms))
        ]
        topical_hits = sum(1 for s in topic_sentences if (not q_terms or any(t in s.lower() for t in q_terms)))
        if topical_hits < 2:
            return ResponseModifier._sanitize_text(ResponseModifier._not_found_response(query))
        anchor_terms = [t for t in q_terms if len(t) >= 6 and t not in {"system", "method", "methods", "theory"}]
        if anchor_terms:
            has_anchor = any(any(a in s.lower() for a in anchor_terms) for s in topic_sentences)
            if not has_anchor:
                return ResponseModifier._sanitize_text(ResponseModifier._not_found_response(query))
        salient = ResponseModifier._salient_terms(topic_sentences, query, limit=8)
        s1 = salient[0] if len(salient) > 0 else "core principles"
        s2 = salient[1] if len(salient) > 1 else "working flow"
        s3 = salient[2] if len(salient) > 2 else "performance behavior"
        s4 = salient[3] if len(salient) > 3 else "implementation constraints"

        if def_like:
            summary_sents = [
                ResponseModifier._trim_text(def_like[0], max_chars=200),
                f"It is mainly studied through {s1}, {s2}, and {s3} to understand practical output behavior.",
            ]
        else:
            summary_sents = [
                f"{topic.capitalize()} is an engineering concept that explains how a system behaves and produces measurable output.",
                f"It is mainly studied through {s1}, {s2}, and {s3} for exam and implementation clarity.",
            ]

        evidence = [ResponseModifier._trim_text(s, max_chars=165).rstrip(".") + "." for s in topic_sentences[:4]]
        detailed_sents = [
            f"The core idea of {topic} is connected to {s1}, {s2}, and {s3}, which define how inputs are processed and interpreted.",
            f"In practice, {topic} works by combining conceptual rules with measurable behavior, then validating outcomes against expected performance.",
            f"Important design considerations include {s4}, reliability, and consistency when operating conditions change.",
            f"The topic becomes clearer when each component is linked to its role in producing measurable outcomes.",
        ]
        if evidence:
            detailed_sents.extend(evidence[:2])

        # Prefer context-derived examples; fallback to topic-safe examples.
        example_sents = [
            s for s in topic_sentences
            if re.search(r"\b(example|for instance|in practice|application|used in)\b", s, flags=re.IGNORECASE)
        ]
        if len(example_sents) < 2:
            example_sents.extend(
                [
                    f"In a student lab, {topic} can be applied to observe how changes in input affect system output step by step.",
                    f"In industry workflows, {topic} is used to improve reliability, reduce errors, and optimize performance under constraints.",
                ]
            )
        example_sents = [ResponseModifier._trim_text(s, max_chars=180).rstrip(".") + "." for s in example_sents[:2]]

        key_points = []
        for s in detailed_sents:
            line = ResponseModifier._trim_text(s, max_chars=150).rstrip(".")
            if line not in key_points:
                key_points.append(line)
            if len(key_points) >= 4:
                break
        if len(key_points) < 4:
            key_points.extend([
                f"Core definition and purpose of {topic}.",
                f"Working mechanism of {topic} from input to output.",
                f"Practical use of {topic} in engineering systems.",
                f"One limitation or tradeoff while applying {topic}.",
            ])
            key_points = key_points[:4]

        conclusion = ResponseModifier._derive_conclusion_from_detailed(
            query=query,
            detailed_text=" ".join(detailed_sents),
            key_points_text=" ".join(key_points),
            existing_conclusion="",
        )

        response = (
            "Short Summary:\n"
            + " ".join(summary_sents[:2]).strip()
            + "\n\nDetailed Explanation:\n"
            + " ".join(detailed_sents).strip()
            + "\n\nReal-Life Example:\n"
            + f"- {example_sents[0]}\n- {example_sents[1]}"
            + "\n\nKey Points:\n"
            + "\n".join(f"- {kp}" for kp in key_points[:4])
            + "\n\nConclusion:\n"
            + conclusion
        )
        return ResponseModifier._sanitize_text(response)

    @staticmethod
    def _trim_text(text: str, max_chars: int = 170) -> str:
        t = " ".join(text.split()).strip()
        if len(t) <= max_chars:
            return t
        cut = t[:max_chars].rsplit(" ", 1)[0].strip()
        return (cut + "...") if cut else t[:max_chars]

    @staticmethod
    def _salient_from_text(text: str, limit: int = 3) -> list[str]:
        stop = {
            "this", "that", "these", "those", "there", "their", "where", "which", "while", "with",
            "from", "into", "also", "very", "more", "most", "only", "used", "using", "system",
            "systems", "engineering", "topic", "concept", "concepts", "input", "output", "model",
            "models", "process", "method", "methods", "result", "results", "practical", "applications",
            "explain", "definition", "summary",
        }
        counts: dict[str, int] = {}
        for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9+#-]*", (text or "").lower()):
            if len(tok) < 5 or tok in stop:
                continue
            counts[tok] = counts.get(tok, 0) + 1
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in ranked[:limit]]

    @staticmethod
    def _derive_conclusion_from_detailed(
        query: str,
        detailed_text: str = "",
        key_points_text: str = "",
        existing_conclusion: str = "",
    ) -> str:
        topic = ResponseModifier._topic_from_query(query)
        section_pat = r"(Short Summary:|Detailed Explanation:|Real-Life Example:|Key Points:|Conclusion:)"
        detailed_text = re.sub(section_pat, " ", detailed_text or "", flags=re.IGNORECASE)
        key_points_text = re.sub(section_pat, " ", key_points_text or "", flags=re.IGNORECASE)
        existing_conclusion = re.sub(section_pat, " ", existing_conclusion or "", flags=re.IGNORECASE)
        if "not found in knowledge base" in f"{detailed_text} {existing_conclusion}".lower():
            return "Not found in knowledge base."

        source_for_salient = f"{detailed_text} {key_points_text}".strip()

        def _token_set(text: str) -> set[str]:
            stop = {
                "the", "and", "for", "with", "that", "this", "from", "into", "over", "under", "through",
                "about", "using", "used", "where", "which", "while", "into", "real", "engineering",
                "practical", "system", "systems", "concept", "topic", "behavior",
            }
            toks = [t for t in re.findall(r"[a-zA-Z]{4,}", (text or "").lower()) if t not in stop]
            return set(toks)

        source_sents = ResponseModifier._non_instruction_sentences(source_for_salient, min_len=24)
        source_sets = [_token_set(s) for s in source_sents[-6:]]

        def _too_similar(candidate: str) -> bool:
            cand = _token_set(candidate)
            if not cand:
                return False
            for sset in source_sets:
                if not sset:
                    continue
                overlap = len(cand & sset) / max(len(cand), 1)
                if overlap >= 0.72:
                    return True
            return False

        first = (
            f"{topic.capitalize()} is fundamental for understanding how the underlying mechanism translates into reliable system outcomes."
        )
        second = (
            f"In practical applications, engineers apply {topic} by checking assumptions, inputs, and observed output behavior."
        )

        if _too_similar(first):
            first = f"{topic.capitalize()} remains important because it links theory with consistent, testable behavior in engineered systems."
        if _too_similar(second):
            second = f"In practical applications, engineers apply {topic} by checking assumptions, inputs, and observed output behavior."

        conclusion = f"{first} {second}".strip()
        for pat in ResponseModifier.REPETITIVE_GENERIC_PATTERNS:
            conclusion = re.sub(pat, "", conclusion, flags=re.IGNORECASE)
        conclusion = re.sub(r"\s+", " ", conclusion).strip()
        conclusion = re.sub(section_pat, " ", conclusion, flags=re.IGNORECASE)
        conclusion = re.sub(r"^(summary|explanation)\s*:\s*", "", conclusion, flags=re.IGNORECASE).strip()
        conclusion = " ".join(conclusion.split()).strip()

        sentences = ResponseModifier._split_sentences(conclusion)
        if not sentences:
            sentences = [f"{topic.capitalize()} is an important engineering concept with strong practical impact."]
        conclusion = " ".join(sentences[:2]).strip()
        if topic.lower() not in conclusion.lower():
            conclusion = f"{topic.capitalize()} is essential in engineering applications. {conclusion}".strip()
        return conclusion

    @staticmethod
    def _rebuild_response(
        text: str,
        summary_override: str | None = None,
        detailed_override: str | None = None,
        key_points_override: list[str] | None = None,
        conclusion_override: str | None = None,
    ) -> str:
        summary = ResponseModifier._section_text(
            text,
            "Short Summary:",
            ["Detailed Explanation:", "Real-Life Example:", "Key Points:", "Conclusion:"],
        )
        detailed = ResponseModifier._section_text(
            text,
            "Detailed Explanation:",
            ["Real-Life Example:", "Key Points:", "Conclusion:"],
        )
        real_example = ResponseModifier._section_text(text, "Real-Life Example:", ["Key Points:", "Conclusion:"])
        key_points = ResponseModifier._section_text(text, "Key Points:", ["Conclusion:"])
        conclusion = ResponseModifier._section_text(text, "Conclusion:", [])
        if not all([summary, detailed, real_example, key_points, conclusion]):
            return text

        if summary_override is not None:
            summary = summary_override.strip()
        if detailed_override is not None:
            detailed = detailed_override.strip()
        if key_points_override is not None:
            normalized_points = [kp.strip() for kp in key_points_override if kp.strip()]
            key_points = "\n".join(f"{i + 1}. {kp}" for i, kp in enumerate(normalized_points))
        if conclusion_override is not None:
            conclusion = conclusion_override.strip()

        return (
            "Short Summary:\n"
            f"{summary.strip()}\n\n"
            "Detailed Explanation:\n"
            f"{detailed.strip()}\n\n"
            "Real-Life Example:\n"
            f"{real_example.strip()}\n\n"
            "Key Points:\n"
            f"{key_points.strip()}\n\n"
            "Conclusion:\n"
            f"{conclusion.strip()}"
        ).strip()

    @staticmethod
    def _ensure_detailed_extra_sentences(query: str, text: str) -> str:
        detailed = ResponseModifier._section_text(
            text,
            "Detailed Explanation:",
            ["Real-Life Example:", "Key Points:", "Conclusion:"],
        )
        if not detailed:
            return text

        topic = ResponseModifier._topic_from_query(query)
        extra_1 = (
            f"{topic.capitalize()} can be understood better when the concept is connected to its underlying mechanism and output behavior."
        )
        extra_2 = (
            f"In practical systems, {topic} is most effective when assumptions, limits, and expected outcomes are clearly identified."
        )

        detailed_lower = detailed.lower()
        additions = []
        if extra_1.lower() not in detailed_lower:
            additions.append(extra_1)
        if extra_2.lower() not in detailed_lower:
            additions.append(extra_2)
        if not additions:
            return text

        detailed_new = (detailed + " " + " ".join(additions)).strip()
        return ResponseModifier._rebuild_response(text, detailed_override=detailed_new)

    @staticmethod
    def _generate_topic_key_points(query: str, text: str, limit: int = 4) -> list[str]:
        detailed = ResponseModifier._section_text(
            text,
            "Detailed Explanation:",
            ["Real-Life Example:", "Key Points:", "Conclusion:"],
        )
        topic = ResponseModifier._topic_from_query(query)
        q_tokens = [
            t
            for t in re.findall(r"[a-zA-Z]{4,}", query.lower())
            if t not in {"what", "define", "explain", "describe", "with", "example", "compare", "between"}
        ]

        salient_terms = ResponseModifier._salient_from_text(detailed, limit=10)
        topic_tokens = set(re.findall(r"[a-zA-Z]{4,}", topic.lower()))
        filtered_salient = [
            t
            for t in salient_terms
            if t.lower() not in topic_tokens
            and t.lower() not in {"engineering", "system", "systems", "concept", "process", "method", "analysis"}
        ]

        term_a = filtered_salient[0] if len(filtered_salient) > 0 else (q_tokens[0] if q_tokens else "core mechanism")
        topic_points = [
            f"{topic.capitalize()} explains how {term_a} influences system behavior and performance metrics under practical operating conditions.",
            f"The main logic of {topic} is the link between input conditions, internal processing, and final output behavior.",
        ]

        topic_l = topic.lower()
        if "deep learning" in topic_l:
            topic_points = [
                kp.replace("Machine learning", "Deep learning").replace("machine learning", "deep learning")
                for kp in topic_points
            ]
        if topic_points and topic_l not in " ".join(topic_points).lower():
            topic_points[0] = f"Core idea of {topic} and why it matters in practical engineering."

        writing_tips = [
            f"In exams, start with a one-line definition of {topic} before explaining the mechanism.",
            f"While writing, include one short example and one limitation of {topic} to improve scoring clarity.",
        ]

        key_points = (topic_points[:2] + writing_tips[:2])[:4]

        return key_points[:limit]

    @staticmethod
    def _enforce_topic_key_points(query: str, text: str) -> str:
        key_points = ResponseModifier._generate_topic_key_points(query, text, limit=4)
        if not key_points:
            return text
        return ResponseModifier._rebuild_response(text, key_points_override=key_points)

    @staticmethod
    def _post_process_detailed_response(query: str, text: str) -> str:
        out = ResponseModifier._normalize_summary_section(query, text)
        out = ResponseModifier._normalize_detailed_section(query, out)
        out = ResponseModifier._enforce_topic_key_points(query, out)
        if "not found in knowledge base" not in out.lower():
            detailed = ResponseModifier._section_text(
                out,
                "Detailed Explanation:",
                ["Real-Life Example:", "Key Points:", "Conclusion:"],
            )
            key_points = ResponseModifier._section_text(out, "Key Points:", ["Conclusion:"])
            existing_conclusion = ResponseModifier._section_text(out, "Conclusion:", [])
            out = ResponseModifier._rebuild_response(
                out,
                conclusion_override=ResponseModifier._derive_conclusion_from_detailed(
                    query=query,
                    detailed_text=detailed,
                    key_points_text=key_points,
                    existing_conclusion=existing_conclusion,
                ),
            )
        return ResponseModifier._sanitize_text(out)

    @staticmethod
    def _examples_depth_ok(text: str) -> bool:
        section = ResponseModifier._section_text(
            text,
            "Real-Life Example:",
            ["Key Points:", "Conclusion:"],
        )
        if not section:
            return False
        bullet_count = len(re.findall(r"(?m)^\s*-\s+", section))
        return bullet_count >= 2

    @staticmethod
    def _conclusion_depth_ok(text: str) -> bool:
        section = ResponseModifier._section_text(text, "Conclusion:", [])
        if not section:
            return False
        sentences = [s for s in re.split(r"[.!?]+", section) if s.strip()]
        return len(sentences) >= 2

    @staticmethod
    def _key_points_relevant(query: str, text: str) -> bool:
        section = ResponseModifier._section_text(text, "Key Points:", ["Conclusion:"])
        if not section:
            return False
        q_tokens = [
            t
            for t in re.findall(r"[a-zA-Z]{4,}", query.lower())
            if t not in {"explain", "define", "about", "with", "example", "compare"}
        ]
        if not q_tokens:
            return True
        sec = section.lower()
        return any(tok in sec for tok in q_tokens)

    @staticmethod
    def validate_response(query: str, text: str) -> bool:
        wc = len(text.split())
        if wc < 90:
            return False
        if not ResponseModifier._has_required_sections(text):
            return False
        if not ResponseModifier._key_points_relevant(query, text):
            return False
        if ResponseModifier._fails_noise_or_meta(text):
            return False
        if not ResponseModifier._topic_focus_ok(query, text):
            return False
        return True

    @staticmethod
    def _template_response(query: str, flags: dict) -> str:
        return generate_template(query, flags)

    @staticmethod
    def build_variant_response(query: str, flags: dict) -> str:
        repaired = ResponseModifier._template_response(query, flags)
        return ResponseModifier._sanitize_text(repaired)

    @staticmethod
    def _first_sentences(text: str, max_sentences: int = 3) -> list[str]:
        if not text:
            return []
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        return parts[:max_sentences]

    @staticmethod
    def _is_instruction_sentence(sentence: str) -> bool:
        s = sentence.strip().lower()
        if not s:
            return True
        return any(re.search(pat, s, flags=re.IGNORECASE) for pat in ResponseModifier.INSTRUCTION_PATTERNS)

    @staticmethod
    def _definition_fallback_sentences(topic: str) -> list[str]:
        return [
            f"{topic.capitalize()} is an engineering concept used to model behavior and solve technical problems.",
            f"It explains how defined inputs are transformed into meaningful outputs through a clear mechanism.",
            f"In practical systems, {topic} is applied to improve reliability, accuracy, and performance.",
        ]

    @staticmethod
    def _non_instruction_sentences(text: str, min_len: int = 30) -> list[str]:
        out = []
        for sent in ResponseModifier._split_sentences(text):
            s = " ".join(sent.split()).strip()
            if len(s) < min_len:
                continue
            if ResponseModifier._is_instruction_sentence(s):
                continue
            if ResponseModifier._is_low_quality_sentence(s):
                continue
            out.append(s)
        return out

    @staticmethod
    def _normalize_summary_section(query: str, text: str) -> str:
        summary = ResponseModifier._section_text(
            text,
            "Short Summary:",
            ["Detailed Explanation:", "Real-Life Example:", "Key Points:", "Conclusion:"],
        )
        detailed = ResponseModifier._section_text(
            text,
            "Detailed Explanation:",
            ["Real-Life Example:", "Key Points:", "Conclusion:"],
        )
        topic = ResponseModifier._topic_from_query(query)
        chosen = ResponseModifier._non_instruction_sentences(summary, min_len=24)[:2]
        if len(chosen) < 1:
            chosen = ResponseModifier._non_instruction_sentences(detailed, min_len=24)[:2]
        if len(chosen) < 1:
            chosen = ResponseModifier._definition_fallback_sentences(topic)[:2]
        normalized = " ".join(chosen[:2]).strip()
        return ResponseModifier._rebuild_response(text, summary_override=normalized)

    @staticmethod
    def _normalize_detailed_section(query: str, text: str) -> str:
        detailed = ResponseModifier._section_text(
            text,
            "Detailed Explanation:",
            ["Real-Life Example:", "Key Points:", "Conclusion:"],
        )
        if not detailed:
            return text
        topic = ResponseModifier._topic_from_query(query)
        kept = ResponseModifier._non_instruction_sentences(detailed, min_len=20)
        if not kept:
            kept = ResponseModifier._definition_fallback_sentences(topic)

        # Keep full explanation normally; only trim the last two sentences when it is long.
        if len(kept) >= 7:
            kept = kept[:-2]

        kept = [ResponseModifier._trim_text(s, max_chars=185).rstrip(".") + "." for s in kept[:6]]
        detailed_new = " ".join(kept).strip()
        return ResponseModifier._rebuild_response(text, detailed_override=detailed_new)

    @staticmethod
    def _extract_topic_examples(text: str, topic: str) -> list[str]:
        real_examples = ResponseModifier._section_text(
            text,
            "Real-Life Example:",
            ["Key Points:", "Conclusion:"],
        )
        bullets = re.findall(r"(?m)^\s*-\s+(.+)$", real_examples)
        cleaned = []
        for b in bullets:
            item = re.sub(r"^Example\s*\d+\s*:\s*", "", b, flags=re.IGNORECASE).strip()
            if item and not ResponseModifier._is_instruction_sentence(item):
                cleaned.append(item)
            if len(cleaned) >= 2:
                break
        if len(cleaned) < 2:
            defaults = [
                f"A student project where {topic} is used to solve a real technical problem.",
                f"An industry use case where {topic} improves reliability and performance.",
            ]
            for d in defaults:
                if len(cleaned) >= 2:
                    break
                cleaned.append(d)
        return cleaned[:2]

    @staticmethod
    def to_short_mode(text: str) -> str:
        cleaned = ResponseModifier._sanitize_text(text)
        summary = ResponseModifier._section_text(
            cleaned,
            "Short Summary:",
            ["Detailed Explanation:", "Real-Life Example:", "Key Points:", "Conclusion:"],
        )
        target_sentences = 3
        min_sentences = 2
        detailed = ResponseModifier._section_text(
            cleaned,
            "Detailed Explanation:",
            ["Real-Life Example:", "Key Points:", "Conclusion:"],
        )

        topic = "the topic"
        if summary:
            head_words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9+\-/]*", summary) if len(w) > 3]
            if head_words:
                topic = " ".join(head_words[:3]).lower()

        candidates = []
        for source in [summary, detailed, cleaned]:
            for sent in ResponseModifier._non_instruction_sentences(source, min_len=24):
                if sent not in candidates:
                    candidates.append(sent)
                if len(candidates) >= target_sentences:
                    break
            if len(candidates) >= target_sentences:
                break

        if len(candidates) < min_sentences:
            for s in ResponseModifier._definition_fallback_sentences(topic):
                if s not in candidates:
                    candidates.append(s)
                if len(candidates) >= target_sentences:
                    break

        short_answer_text = " ".join(candidates[:target_sentences]).strip()
        examples = ResponseModifier._extract_topic_examples(cleaned, topic)
        return (
            "Short Answer:\n"
            f"{short_answer_text}\n\n"
            "Topic Examples:\n"
            f"- {examples[0]}\n"
            f"- {examples[1]}"
        ).strip()

    @staticmethod
    def _section_from_response(text: str, title: str, all_titles: list[str]) -> str:
        start = re.search(rf"(?im)^\s*{re.escape(title)}\s*:\s*$", text or "")
        if not start:
            return ""
        tail = (text or "")[start.end():]
        end = len(tail)
        for other in all_titles:
            if other == title:
                continue
            m = re.search(rf"(?im)^\s*{re.escape(other)}\s*:\s*$", tail)
            if m:
                end = min(end, m.start())
        return tail[:end].strip()

    @staticmethod
    def _compose_section(title: str, body: str) -> str:
        return f"{title}:\n{(body or '').strip()}".strip()

    @staticmethod
    def _fallback_section_content(section: str, query: str, context_sents: list[str]) -> str:
        topic = ResponseModifier._topic_from_query(query)
        lower_sents = [s.lower() for s in context_sents]

        def pick_by_keywords(keywords: list[str], limit: int = 3) -> list[str]:
            out = []
            for s, sl in zip(context_sents, lower_sents):
                if any(k in sl for k in keywords):
                    out.append(ResponseModifier._trim_text(s, max_chars=190).rstrip(".") + ".")
                if len(out) >= limit:
                    break
            return out

        if section == "definition":
            defs = pick_by_keywords([" is ", " refers to ", " defined as ", " means "], limit=2)
            if defs:
                return " ".join(defs[:2]).strip()
            return f"{topic.capitalize()} is an engineering concept explained in the provided academic context."

        if section == "explanation":
            chunks = [ResponseModifier._trim_text(s, max_chars=190).rstrip(".") + "." for s in context_sents[:4]]
            return " ".join(chunks) if chunks else f"The provided context gives the conceptual explanation of {topic}."

        if section == "core_functions":
            lines = pick_by_keywords(["function", "responsible", "manages", "handles", "controls"], limit=4)
            if not lines:
                lines = [ResponseModifier._trim_text(s, max_chars=160).rstrip(".") + "." for s in context_sents[:3]]
            return "\n".join(f"- {ln}" for ln in lines[:4])

        if section == "working":
            lines = pick_by_keywords(["steps", "process", "flow", "works", "mechanism"], limit=4)
            if not lines:
                lines = [ResponseModifier._trim_text(s, max_chars=170).rstrip(".") + "." for s in context_sents[:3]]
            return "\n".join(f"- {ln}" for ln in lines[:4])

        if section == "example":
            lines = pick_by_keywords(["example", "for instance", "in practice", "application", "used in"], limit=2)
            if not lines:
                lines = [
                    f"A lab-level scenario where {topic} is applied to evaluate output behavior under different input conditions.",
                    f"A real engineering deployment where {topic} improves reliability or performance within system constraints.",
                ]
            return "\n".join(f"- {ln}" for ln in lines[:2])

        if section == "advantages":
            lines = pick_by_keywords(["advantage", "benefit", "improve", "efficient", "reliable"], limit=4)
            if not lines:
                lines = [f"{topic.capitalize()} improves correctness, consistency, and practical system performance."]
            return "\n".join(f"- {ln}" for ln in lines[:4])

        if section == "limitations":
            lines = pick_by_keywords(["limitation", "drawback", "tradeoff", "constraint", "challenge"], limit=4)
            if not lines:
                lines = [f"{topic.capitalize()} may involve tradeoffs in complexity, resource usage, or operating constraints."]
            return "\n".join(f"- {ln}" for ln in lines[:4])

        if section == "comparison":
            lines = pick_by_keywords(["difference", "compare", "versus", "vs", "whereas"], limit=4)
            if not lines:
                lines = [ResponseModifier._trim_text(s, max_chars=170).rstrip(".") + "." for s in context_sents[:3]]
            return "\n".join(f"- {ln}" for ln in lines[:4])

        if section == "applications":
            lines = pick_by_keywords(["application", "used", "deployment", "industry", "practical"], limit=4)
            if not lines:
                lines = [f"{topic.capitalize()} is applied in practical engineering systems where accurate and stable operation is required."]
            return "\n".join(f"- {ln}" for ln in lines[:4])

        return " ".join([ResponseModifier._trim_text(s, max_chars=190).rstrip(".") + "." for s in context_sents[:3]])

    @staticmethod
    def build_context_grounded_query_response(
        query: str,
        context: str,
        sections: list[str],
        section_titles: dict[str, str],
    ) -> str:
        if not (context or "").strip():
            return ResponseModifier.INVALID_CONTEXT_RESPONSE

        picked = ResponseModifier._pick_best_sentences(context, query, n=12)
        if not picked:
            return ResponseModifier.INVALID_CONTEXT_RESPONSE

        q_terms = ResponseModifier._query_terms(query)
        topic_sentences = []
        for s in picked:
            if ResponseModifier._is_low_quality_sentence(s):
                continue
            if not q_terms:
                topic_sentences.append(s)
                continue
            if any(t in s.lower() for t in q_terms):
                topic_sentences.append(s)
        if len(topic_sentences) < 2:
            topic_sentences = [s for s in picked if not ResponseModifier._is_low_quality_sentence(s)][:5]
        if len(topic_sentences) < 2:
            return ResponseModifier.INVALID_CONTEXT_RESPONSE

        parts = []
        all_titles = [section_titles[s] for s in sections]
        for sec in sections:
            title = section_titles[sec]
            body = ResponseModifier._fallback_section_content(sec, query, topic_sentences)
            if not body.strip():
                body = ResponseModifier._fallback_section_content(sec, query, picked)
            if body.strip():
                parts.append(ResponseModifier._compose_section(title, body))

        if not parts:
            return ResponseModifier.INVALID_CONTEXT_RESPONSE

        out = "\n\n".join(parts)
        return ResponseModifier._sanitize_text(out)

    @staticmethod
    def enforce_query_driven_response(
        raw_response: str,
        query: str,
        context: str,
        sections: list[str],
        section_titles: dict[str, str],
    ) -> str:
        cleaned = ResponseModifier._sanitize_text(raw_response or "")
        requested_titles = [section_titles[s] for s in sections]
        if ResponseModifier.is_invalid_response(cleaned):
            return cleaned

        if not cleaned:
            return ResponseModifier.build_context_grounded_query_response(query, context, sections, section_titles)

        existing = {}
        for title in requested_titles:
            existing[title] = ResponseModifier._section_from_response(cleaned, title, requested_titles)

        # If model did not follow headers, use full answer as explanation-like content when possible.
        if all(not v for v in existing.values()):
            if len(cleaned.split()) >= 25 and ResponseModifier._topic_focus_ok(query, cleaned):
                if sections == ["definition"]:
                    existing[requested_titles[0]] = " ".join(ResponseModifier._first_sentences(cleaned, 2))
                else:
                    # put free-form content under first requested section
                    existing[requested_titles[0]] = cleaned

        context_sents = ResponseModifier._pick_best_sentences(context, query, n=12)
        parts = []
        for sec in sections:
            title = section_titles[sec]
            body = (existing.get(title) or "").strip()
            if not body:
                body = ResponseModifier._fallback_section_content(sec, query, context_sents)
            parts.append(ResponseModifier._compose_section(title, body))

        final = "\n\n".join(parts).strip()
        if not final:
            return ResponseModifier.INVALID_CONTEXT_RESPONSE
        return ResponseModifier._sanitize_text(final)

    @staticmethod
    def enforce_structured_response(response: str, query: str, context: str, flags: dict) -> tuple[str, bool]:
        cleaned = ResponseModifier._sanitize_text(response)
        if ResponseModifier.validate_response(query, cleaned) and not ResponseModifier._is_generic_or_irrelevant(cleaned, query):
            cleaned = ResponseModifier._post_process_detailed_response(query, cleaned)
            return cleaned, flags["needs_improvement"]

        # Keep non-empty topic-focused LLM answers instead of forcing "not found"
        # when strict checks fail.
        if cleaned and not ResponseModifier._fails_noise_or_meta(cleaned) and ResponseModifier._topic_focus_ok(query, cleaned):
            if not ResponseModifier._has_required_sections(cleaned):
                topic = ResponseModifier._topic_from_query(query)
                sents = ResponseModifier._first_sentences(cleaned, max_sentences=3)
                summary = " ".join(sents[:2]).strip() if sents else f"{topic.capitalize()} is an important topic in engineering."
                detailed = cleaned
                wrapped = (
                    "Short Summary:\n"
                    f"{summary}\n\n"
                    "Detailed Explanation:\n"
                    f"{detailed}\n\n"
                    "Real-Life Example:\n"
                    f"- In a lab setting, {topic} can be used to observe how input changes affect output.\n"
                    f"- In practical systems, {topic} is applied to improve reliability and performance.\n\n"
                    "Key Points:\n"
                    f"- Core definition and purpose of {topic}.\n"
                    f"- Working mechanism of {topic} from input to output.\n"
                    f"- In exams, start with a one-line definition of {topic} before explaining the mechanism.\n"
                    f"- While writing, include one short example and one limitation of {topic} to improve scoring clarity.\n\n"
                    "Conclusion:\n"
                    f"{ResponseModifier._derive_conclusion_from_detailed(query, detailed, topic, '')}"
                )
                return ResponseModifier._post_process_detailed_response(query, wrapped), True
            return ResponseModifier._post_process_detailed_response(query, cleaned), True

        repaired = ResponseModifier.build_context_grounded_response(query, context, flags)
        repaired = ResponseModifier._post_process_detailed_response(query, repaired)
        return repaired, True

