import re


def check_factual_consistency(original, revised):
    original_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", original))
    revised_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", revised))
    new_numbers = sorted(revised_numbers - original_numbers)
    return {
        "status": "passed" if not new_numbers else "review",
        "new_numeric_claims": new_numbers,
        "note": "No new numeric claims detected." if not new_numbers else "Review new numeric claims before approval.",
    }

