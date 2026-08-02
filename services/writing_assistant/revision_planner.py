from .passage_analyzer import analyze_passage


def build_revision_plan(text, options):
    analysis = analyze_passage(text)
    preserve = ["main argument", "citations", "quotations", "numbers", "technical terms", "named entities"]
    improvements = []
    for issue in analysis["issues"]:
        if "generic" in issue:
            improvements.append("replace generic framing with direct, specific wording")
        elif "uniform" in issue:
            improvements.append("vary sentence length and structure")
        elif "evidence" in issue:
            improvements.append("ask author for a concrete example, source, or project detail")
        elif "vocabulary" in issue:
            improvements.append("reduce repeated wording without changing meaning")
        elif "openings" in issue:
            improvements.append("vary paragraph flow and sentence openings")

    if options.get("add_examples"):
        improvements.append("insert an author-provided example where available")
    if options.get("add_evidence_prompts"):
        improvements.append("mark places where evidence or citations are needed")

    return {
        "preserve": preserve,
        "improvements": improvements or ["light grammar and flow edit"],
        "detected_characteristics": analysis["issues"],
        "protected_content": analysis["protected_content"],
        "review_required": True,
    }

