import re
from collections import Counter
from typing import List, Optional


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "being",
    "but", "by", "can", "could", "did", "do", "does", "for", "from", "had",
    "has", "have", "having", "he", "her", "his", "i", "if", "in", "into",
    "is", "it", "its", "may", "might", "more", "must", "not", "of", "on",
    "or", "our", "ours", "should", "so", "than", "that", "the", "their",
    "them", "there", "these", "they", "this", "those", "to", "too", "very",
    "was", "we", "were", "which", "who", "will", "with", "would", "you",
    "your",
    # Common debate filler verbs that usually hurt factual retrieval.
    "argue", "argues", "argument", "believe", "good", "bad", "better",
    "claim", "claims", "completely", "eventually", "government", "governments",
    "improve", "improves", "invest", "lowers", "need", "needs", "policy",
    "reduces", "think",
}


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extracts stable keyword tokens without external NLP downloads.
    It favors repeated terms, then preserves first-seen order for ties.
    """
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*", text.lower())

    first_seen = {}
    cleaned_tokens = []

    for token in tokens:
        token = token.strip("'")
        if len(token) <= 2 or token in STOPWORDS:
            continue

        if token not in first_seen:
            first_seen[token] = len(first_seen)
        cleaned_tokens.append(token)

    counts = Counter(cleaned_tokens)
    ranked = sorted(counts, key=lambda word: (-counts[word], first_seen[word]))

    return ranked[:max_keywords]


def build_retrieval_query(
    argument: str,
    topic: Optional[str] = None,
    max_keywords: int = 12
) -> str:
    """
    Turns a debate argument into a compact factual search query.
    Topic is kept separate and placed first when available.
    """
    keywords = extract_keywords(argument, max_keywords=max_keywords)
    query_parts = []

    if topic and topic.strip():
        query_parts.append(topic.strip())

    if keywords:
        query_parts.append(" ".join(keywords))

    return " ".join(query_parts) if query_parts else argument.strip()


def build_wikipedia_query(argument: str, topic: Optional[str] = None) -> str:
    """
    Wikipedia works better with a concise page-like query than a full sentence.
    """
    if topic and topic.strip():
        return topic.strip()

    keywords = extract_keywords(argument, max_keywords=5)
    return " ".join(keywords) if keywords else argument.strip()


def _unique(items: List[str]) -> List[str]:
    seen = set()
    unique_items = []

    for item in items:
        item = re.sub(r"\s+", " ", item.strip())
        key = item.lower()
        if not item or key in seen:
            continue

        seen.add(key)
        unique_items.append(item)

    return unique_items


def build_wikipedia_queries(
    argument: str,
    topic: Optional[str] = None,
    max_candidates: int = 8
) -> List[str]:
    """
    Builds multiple page-search candidates. Debate motions are often not exact
    Wikipedia titles, so this gives the retriever a few concise alternatives.
    """
    candidates = []

    if topic and topic.strip():
        candidates.append(topic.strip())
        topic_keywords = extract_keywords(topic, max_keywords=6)
        if topic_keywords:
            candidates.append(" ".join(topic_keywords))

    argument_keywords = extract_keywords(argument, max_keywords=8)
    if argument_keywords:
        candidates.append(" ".join(argument_keywords))

    combined_text = f"{topic or ''} {argument}"
    combined_keywords = extract_keywords(combined_text, max_keywords=10)
    combined_set = set(combined_keywords)

    if combined_keywords:
        candidates.append(" ".join(combined_keywords[:6]))

    # Small domain phrase hints for common debate topics. These still go through
    # Wikipedia search; they just give it page-like wording.
    if {"school", "schools"} & combined_set and {"smartphone", "smartphones", "phone", "phones"} & combined_set:
        candidates.extend([
            "mobile phone use in schools",
            "smartphones in schools",
        ])

    if "public" in combined_set and "transport" in combined_set:
        candidates.append("public transport")

    if "renewable" in combined_set and "energy" in combined_set:
        candidates.append("renewable energy")

    if "fossil" in combined_set and {"fuel", "fuels"} & combined_set:
        candidates.append("fossil fuel")

    if {"ai", "artificial", "intelligence"} & combined_set and "education" in combined_set:
        candidates.append("artificial intelligence in education")

    return _unique(candidates)[:max_candidates]


def expand_query(query: str, topic: Optional[str] = None) -> List[str]:
    base_query = build_retrieval_query(query, topic=topic)
    return [
        base_query,
        f"{base_query} facts evidence",
        f"{base_query} advantages disadvantages",
    ]
