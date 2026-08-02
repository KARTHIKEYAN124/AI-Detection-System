from .passage_analyzer import analyze_passage


def validate_citations(original, revised):
    original_items = analyze_passage(original)["protected_content"]
    revised_items = analyze_passage(revised)["protected_content"]
    missing = []
    for group in ("citations", "quotes", "numbers", "urls"):
        for item in original_items[group]:
            if item not in revised_items[group]:
                missing.append(item)
    return {
        "passed": len(missing) == 0,
        "missing_protected_content": missing,
    }

