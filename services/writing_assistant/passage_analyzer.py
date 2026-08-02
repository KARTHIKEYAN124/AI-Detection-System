import re


GENERIC_PHRASES = [
    "it is important to note",
    "furthermore",
    "moreover",
    "in conclusion",
    "in summary",
    "plays a crucial role",
    "it is essential",
    "comprehensive",
    "multifaceted",
    "in the context of",
]


def words(text):
    return re.findall(r"\b[\w'-]+\b", text)


def sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def analyze_passage(text):
    sentence_list = sentences(text)
    word_list = words(text)
    lengths = [len(words(sentence)) for sentence in sentence_list] or [0]
    avg = sum(lengths) / len(lengths)
    variance = sum((length - avg) ** 2 for length in lengths) / len(lengths)
    lower = text.lower()
    repeated_openings = {}
    for sentence in sentence_list:
        first = (words(sentence)[:1] or [""])[0].lower()
        if first:
            repeated_openings[first] = repeated_openings.get(first, 0) + 1

    issues = []
    if any(phrase in lower for phrase in GENERIC_PHRASES):
        issues.append("generic transition or framing phrases")
    if variance < 18 and len(sentence_list) >= 3:
        issues.append("highly uniform sentence lengths")
    if max(repeated_openings.values() or [0]) >= 3:
        issues.append("repeated sentence openings")
    if len(set(word.lower() for word in word_list)) / max(1, len(word_list)) < 0.48:
        issues.append("low vocabulary variation")
    if not re.search(r"\[[^\]]+\]|\([A-Z][A-Za-z]+,\s*\d{4}\)|\d+%|\bfor example\b|\bsuch as\b", text):
        issues.append("broad claims without concrete evidence")

    return {
        "word_count": len(word_list),
        "sentence_count": len(sentence_list),
        "issues": issues or ["no major revision issue detected"],
        "protected_content": {
            "citations": re.findall(r"\[[^\]]+\]|\([A-Z][A-Za-z]+,\s*\d{4}\)", text),
            "quotes": re.findall(r"\"[^\"]+\"", text),
            "numbers": re.findall(r"\b\d+(?:\.\d+)?%?\b", text),
            "urls": re.findall(r"https?://\S+", text),
        },
    }

