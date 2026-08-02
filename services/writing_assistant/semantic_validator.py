from difflib import SequenceMatcher


def validate_similarity(original, revised):
    similarity = SequenceMatcher(None, original.lower(), revised.lower()).ratio()
    return {
        "semantic_similarity": round(similarity, 3),
        "passed": similarity >= 0.55,
        "warning": None if similarity >= 0.55 else "Revision may have changed the meaning too much.",
    }

