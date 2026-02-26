import asyncio
import re
from app import db
from app.config import settings
from app.model_loader import runtime
from app.rag.global_rag import get_global_rag
from app.rag.prompt_builder import build_prompt
from app.rl_engine.online_rlhf import OnlineRLHFTrainer
from app.rl_engine.response_modifier import ResponseModifier
from app.rl_engine.reward_tracker import RewardTracker
from app.services.query_intent import SECTION_TITLES, parse_requested_sections, section_contract_text
from app.utils.guardrails import is_btech_query, no_context_message, rejection_message


def _query_domain_hints(query: str) -> set[str]:
    q = query.lower()
    hints = set()
    mapping = [
        (r"\bcivil\b", "civil"),
        (r"\bmechanical\b", "mechanical"),
        (r"\bece\b|\belectronics\b", "ece"),
        (r"\beee\b|\belectrical\b", "eee"),
        (r"\bcyber\s*security\b|\bcybersecurity\b|\bcyber\b", "cyber security"),
        (r"\baiml\b|\bmachine learning\b", "aiml"),
        (r"\bdata science\b|\bdata analytics\b|\bds\b", "ds"),
        (r"\bit\b|\binformation technology\b", "it"),
        (r"\bcse\b|\bcomputer science\b", "cse"),
    ]
    for pat, dom in mapping:
        if re.search(pat, q):
            hints.add(dom)
    return hints


async def generate_chat(user_id: str, query: str, short_answer: bool, detailed_explanation: bool) -> dict:
    guardrail_allowed = is_btech_query(query)

    user_pref = db.get_user_pref(user_id)
    query_profile = RewardTracker.get_query_profile(query, user_id=user_id)
    flags = ResponseModifier.get_adaptation_flags(query_profile)
    attempt_count = db.get_query_attempt_count(query)
    flags["variation_id"] = attempt_count % 3

    base_mode = "short" if short_answer else ("detailed" if detailed_explanation else user_pref["preferred_mode"])
    style_instruction = ResponseModifier.style_instruction(base_mode, user_pref["avg_reward"], flags)
    requested_sections = parse_requested_sections(query)
    requested_titles = [SECTION_TITLES[s] for s in requested_sections]
    contract_text = section_contract_text(requested_sections)
    request_style = "|".join(requested_sections)
    query_key = db.normalize_query(query)
    online_plan = OnlineRLHFTrainer.get_adaptation_plan(user_id=user_id, query_key=query_key)
    history_samples = db.get_user_query_feedback_samples(user_id=user_id, query=query, limit=8)

    adaptation_note = (
        "Previous ratings for this question were low. Use clearer structure, extra examples, and stronger exam focus."
        if flags["needs_improvement"]
        else "No special correction needed."
    )
    if online_plan.get("exists"):
        adaptation_note += (
            f" Dynamic RLHF history: total_feedback={online_plan.get('total_feedback', 0)}, "
            f"avg_reward={online_plan.get('avg_reward', 0.0):.2f}, "
            f"low_rating_count={online_plan.get('low_rating_count', 0)}."
        )

    avoid_snippets: list[str] = []
    preferred_snippets: list[str] = []
    for row in history_samples:
        if int(row.get("rating", 0)) <= 2 and row.get("response"):
            avoid_snippets.append(str(row["response"]))
        elif int(row.get("rating", 0)) >= 4 and row.get("response"):
            preferred_snippets.append(str(row["response"]))
    avoid_snippets.extend([str(x) for x in online_plan.get("avoid_snippets", []) if str(x).strip()])
    preferred_snippets.extend([str(x) for x in online_plan.get("preferred_snippets", []) if str(x).strip()])
    avoid_snippets = avoid_snippets[:6]
    preferred_snippets = preferred_snippets[:4]

    rlhf_constraints = []
    if avoid_snippets:
        rlhf_constraints.append(
            "RLHF corrective signal: prior same-user answers for this query were rated low. "
            "Generate a fully revised answer with significantly different wording and fresh examples."
        )
        for idx, snippet in enumerate(avoid_snippets[:2], start=1):
            cleaned = " ".join(snippet.split())[:220]
            rlhf_constraints.append(f"Avoid repeating low-rated pattern {idx}: {cleaned}")
    if preferred_snippets:
        rlhf_constraints.append(
            "Use the strengths of previously high-rated responses: clearer concept flow, concrete examples, and focused coverage."
        )
    rlhf_constraints_text = "\n".join(f"- {line}" for line in rlhf_constraints) if rlhf_constraints else "- No additional RLHF constraints."

    rag = get_global_rag()
    try:
        context, sources = await asyncio.to_thread(rag.retrieve, query, settings.retrieval_k)
    except FileNotFoundError:
        response = "Knowledge base index not built. Run POST /rebuild-index first."
        chat_id = db.save_chat(user_id, query, response, "", base_mode, request_style=request_style)
        return {
            "chat_id": chat_id,
            "response": response,
            "rejected": False,
            "sources": [],
            "response_improved": False,
            "improvement_reason": "index_missing",
            "response_variant": "",
        }

    hinted_domains = _query_domain_hints(query)
    if hinted_domains and sources:
        source_domains = {str(getattr(rag, "_source_domain")(s) or "").strip().lower() for s in sources}
        if source_domains and source_domains.isdisjoint(hinted_domains):
            context, sources = "", []

    if not guardrail_allowed:
        response = rejection_message()
        chat_id = db.save_chat(user_id, query, response, "", "rejected", request_style=request_style)
        return {
            "chat_id": chat_id,
            "response": response,
            "rejected": True,
            "sources": [],
            "response_improved": False,
            "improvement_reason": "out_of_domain",
            "response_variant": "",
        }

    relevance = ResponseModifier.context_relevance(query, context, sources)
    if relevance < 0.20:
        response = no_context_message()
        chat_id = db.save_chat(user_id, query, response, "", "rejected", request_style=request_style)
        return {
            "chat_id": chat_id,
            "response": response,
            "rejected": True,
            "sources": [],
            "response_improved": False,
            "improvement_reason": "no_relevant_context",
            "response_variant": "",
        }

    prompt = build_prompt(
        query,
        context,
        style_instruction,
        adaptation_note,
        requested_sections_text=contract_text,
        requested_section_titles=requested_titles,
        rlhf_constraints_text=rlhf_constraints_text,
    )
    if runtime.backend == "retrieval_only_fallback":
        raw_response = ResponseModifier.build_context_grounded_query_response(
            query,
            context,
            requested_sections,
            SECTION_TITLES,
        )
    else:
        raw_response = await asyncio.to_thread(runtime.generate, prompt, settings.max_new_tokens)
        if not str(raw_response or "").strip():
            raw_response = ResponseModifier.build_context_grounded_query_response(
                query,
                context,
                requested_sections,
                SECTION_TITLES,
            )

    final_response = ResponseModifier.enforce_query_driven_response(
        raw_response,
        query,
        context,
        requested_sections,
        SECTION_TITLES,
    )
    improved = bool(flags.get("needs_improvement"))
    force_rewrite = bool(flags.get("needs_improvement", False) or online_plan.get("low_rating_count", 0) > 0)
    if force_rewrite:
        low_candidates = [str(r.get("response", "")).strip() for r in history_samples if int(r.get("rating", 0)) <= 2 and str(r.get("response", "")).strip()]
        if online_plan.get("avoid_snippets"):
            low_candidates.extend([str(x) for x in online_plan.get("avoid_snippets", []) if str(x).strip()])
        low_candidates = low_candidates[:8]
        threshold = float(settings.rlhf_similarity_threshold)

        def _max_overlap(text: str, refs: list[str]) -> float:
            if not refs:
                return 0.0
            return max(ResponseModifier.response_overlap_score(text, r) for r in refs)

        overlap = _max_overlap(final_response, low_candidates)
        attempts = 0
        while overlap >= threshold and attempts < int(settings.rlhf_rewrite_attempts):
            attempts += 1
            rewrite_prompt = f"""
You are a strict Academic AI Tutor for BTech students.

You ONLY answer academic engineering questions from undergraduate curriculum.

Covered domains:
- Computer Science & Engineering
- Artificial Intelligence
- Information Technology
- Electronics & Communication Engineering
- Electrical & Electronics Engineering
- Mechanical Engineering
- Civil Engineering
- Artificial Intelligence & Data Science
- Cyber Security
- Emerging Technologies

STRICT RULES:

1. If the question is NOT related to BTech engineering subjects, respond EXACTLY with:
   "Invalid Question: This system only answers BTech academic and engineering-related queries."

2. If the provided textbook context does NOT contain relevant information for answering the question, respond EXACTLY with:
   "Invalid Question: No relevant academic context found."

3. Use ONLY the provided context to generate the answer.
4. Do NOT use external knowledge.
5. Do NOT answer general knowledge, personal advice, entertainment, shopping, or product queries.
6. Maintain formal academic tone suitable for BTech exams.
7. Ensure conceptual clarity and technical correctness.

Answer Guidelines:

1. Provide a structured academic answer.
2. Use proper headings and subheadings where necessary.
3. Use bullet points when suitable.
4. Keep clarity, technical correctness, and exam-oriented presentation.
5. Avoid repeating fixed formats.
6. Adapt answer depth based on question type.

CRITICAL RLHF REWRITE REQUIREMENT:
- Previous answer for this same user and query was low-rated.
- Return a fully revised answer, not minor edits.
- Keep same requested sections and topic coverage.
- Use different sentence structure and at least one different example.
- Avoid overlap with previously low-rated wording.

Requested sections:
{", ".join(requested_titles)}

Context:
{context}

Question:
{query}

Answer:
""".strip()
            if runtime.backend != "retrieval_only_fallback":
                retry_raw = await asyncio.to_thread(runtime.generate, rewrite_prompt, settings.max_new_tokens)
                candidate = ResponseModifier.enforce_query_driven_response(
                    retry_raw,
                    query,
                    context,
                    requested_sections,
                    SECTION_TITLES,
                )
            else:
                candidate = ResponseModifier.build_context_grounded_query_response(
                    query,
                    context,
                    requested_sections,
                    SECTION_TITLES,
                )
            cand_overlap = _max_overlap(candidate, low_candidates)
            if cand_overlap < overlap or cand_overlap < threshold:
                final_response = candidate
                overlap = cand_overlap
                improved = True

        # Last defensive pass if still too close.
        if overlap >= threshold:
            alt_context = ResponseModifier.build_context_grounded_query_response(
                query,
                context,
                requested_sections,
                SECTION_TITLES,
            )
            alt_overlap = _max_overlap(alt_context, low_candidates)
            if alt_overlap < overlap:
                final_response = alt_context
                improved = True

    # Hard quality gate for query-driven sections.
    gate = ResponseModifier.quality_gate_query_driven(query, final_response, requested_titles)
    if not gate.get("passed", False):
        if runtime.backend != "retrieval_only_fallback":
            retry_prompt = prompt + "\n\nReturn only the requested sections with exact headers and no extra sections."
            retry_raw = await asyncio.to_thread(runtime.generate, retry_prompt, settings.max_new_tokens)
            llm_repaired = ResponseModifier.enforce_query_driven_response(
                retry_raw,
                query,
                context,
                requested_sections,
                SECTION_TITLES,
            )
            gate_llm = ResponseModifier.quality_gate_query_driven(query, llm_repaired, requested_titles)
            if gate_llm.get("passed", False):
                final_response = llm_repaired
                improved = True
            else:
                final_response = ResponseModifier.build_context_grounded_query_response(
                    query,
                    context,
                    requested_sections,
                    SECTION_TITLES,
                )
                improved = True
        else:
            final_response = ResponseModifier.build_context_grounded_query_response(
                query,
                context,
                requested_sections,
                SECTION_TITLES,
            )
            improved = True

    source_text = ";".join(f"{s['file_name']}#p{s['page_number']}" for s in sources)
    chat_id = db.save_chat(
        user_id,
        query,
        final_response,
        source_text,
        base_mode,
        request_style=request_style,
    )

    return {
        "chat_id": chat_id,
        "response": final_response,
        "rejected": False,
        "sources": sources,
        "response_improved": improved,
        "improvement_reason": "low_reward_adaptation" if flags["needs_improvement"] else "standard_mode",
        "response_variant": f"v{int(flags.get('variation_id', 0)) + 1}",
    }


