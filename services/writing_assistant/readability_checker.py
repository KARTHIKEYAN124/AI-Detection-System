import re


def check_readability(text):
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]
    words = re.findall(r"\b[\w'-]+\b", text)
    avg_sentence = len(words) / max(1, len(sentences))
    return {
        "average_sentence_words": round(avg_sentence, 1),
        "status": "passed" if avg_sentence <= 28 else "review",
        "note": "Readability improved or acceptable." if avg_sentence <= 28 else "Consider shorter sentences.",
    }

