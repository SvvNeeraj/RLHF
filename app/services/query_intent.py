import re


SECTION_ORDER = [
    "definition",
    "explanation",
    "core_functions",
    "working",
    "example",
    "advantages",
    "limitations",
    "comparison",
    "applications",
]

SECTION_TITLES = {
    "definition": "Definition",
    "explanation": "Explanation",
    "core_functions": "Core Functions",
    "working": "Working",
    "example": "Example",
    "advantages": "Advantages",
    "limitations": "Limitations",
    "comparison": "Comparison",
    "applications": "Applications",
}

SECTION_INSTRUCTIONS = {
    "definition": "Define the topic in clear academic terms in 2-4 sentences.",
    "explanation": "Explain the concept deeply with technical clarity and coherent flow.",
    "core_functions": "List and explain the core functions relevant to the asked topic.",
    "working": "Explain how the topic works step-by-step at concept level.",
    "example": "Provide one or two topic-relevant examples linked to the explanation.",
    "advantages": "Provide only topic-relevant advantages.",
    "limitations": "Provide only topic-relevant limitations/disadvantages.",
    "comparison": "Compare the asked items directly with concise technical differences.",
    "applications": "Provide key practical applications/use-cases of the topic.",
}


def parse_requested_sections(query: str) -> list[str]:
    q = (query or "").strip().lower()
    if not q:
        return ["explanation"]

    found: dict[str, int] = {}

    def mark(section: str, pattern: str) -> None:
        m = re.search(pattern, q, flags=re.IGNORECASE)
        if m:
            found[section] = min(found.get(section, 10_000), m.start())

    mark("definition", r"\b(what\s+is|define|meaning\s+of|definition\s+of)\b")
    mark("explanation", r"\b(explain|elaborate|describe|brief\s+about)\b")
    mark("core_functions", r"\b(core\s+functions?|main\s+functions?|functions?\s+of)\b")
    mark("working", r"\b(how\s+it\s+works?|working|workflow|mechanism)\b")
    mark("example", r"\b(example|for\s+instance|with\s+example|real[-\s]*life)\b")
    mark("advantages", r"\b(advantages?|benefits?|merits?)\b")
    mark("limitations", r"\b(limitations?|disadvantages?|demerits?|drawbacks?)\b")
    mark("comparison", r"\b(compare|comparison|difference\s+between|vs\.?|versus)\b")
    mark("applications", r"\b(applications?|use\s*cases?|where\s+it\s+is\s+used)\b")

    # Rule: "advantages and limitations" should return exactly those sections.
    if ("advantages" in found or re.search(r"\bpros\b", q)) and (
        "limitations" in found or re.search(r"\bcons\b", q)
    ):
        return ["advantages", "limitations"]

    # Rule: only definition when user asks only definition.
    definition_only = bool(re.search(r"^\s*(what\s+is|define)\b", q)) and len(found) == 1 and "definition" in found
    if definition_only:
        return ["definition"]

    if not found:
        return ["explanation"]

    ordered = [k for k, _ in sorted(found.items(), key=lambda kv: kv[1])]

    # For generic explain queries, prefer explanation first.
    if "explanation" in ordered and ordered[0] != "explanation":
        ordered.remove("explanation")
        ordered.insert(0, "explanation")

    # Deterministic final order by semantic section order.
    out = [s for s in SECTION_ORDER if s in ordered]
    return out or ["explanation"]


def section_contract_text(sections: list[str]) -> str:
    lines = []
    for sec in sections:
        title = SECTION_TITLES.get(sec, sec.replace("_", " ").title())
        ins = SECTION_INSTRUCTIONS.get(sec, "Provide topic-relevant content.")
        lines.append(f"- {title}: {ins}")
    return "\n".join(lines)

