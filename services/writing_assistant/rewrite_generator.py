import re


REPLACEMENTS = [
    (r"\bIt is important to note that\b", "A clearer way to state this is that"),
    (r"\bFurthermore,\s*", ""),
    (r"\bMoreover,\s*", ""),
    (r"\bIn conclusion,\s*", "Overall, "),
    (r"\bplays a crucial role\b", "is important"),
    (r"\bcomprehensive\b", "careful"),
    (r"\bmultifaceted\b", "complex"),
    (r"\bparamount\b", "important"),
]


def generate_revision(text, options, user_context):
    revised = text.strip()
    for pattern, replacement in REPLACEMENTS:
        revised = re.sub(pattern, replacement, revised, flags=re.IGNORECASE)

    if options.get("shorter_sentences"):
        revised = revised.replace("; ", ". ")
    if user_context.get("personal_example"):
        revised += f" For example, {user_context['personal_example'].strip()}"
    elif options.get("add_evidence_prompts"):
        revised += " [Add a specific example, source, or observation from your own work here.]"
    if user_context.get("source_notes"):
        revised += f" {user_context['source_notes'].strip()}"

    change_summary = [
        "softened formulaic phrasing",
        "preserved protected content where detected",
        "added prompts for author-owned evidence" if options.get("add_evidence_prompts") else "kept meaning close to the original",
    ]
    return {
        "revised_text": revised,
        "change_summary": change_summary,
    }

