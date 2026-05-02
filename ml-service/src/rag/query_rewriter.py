import re
from .query_expansion import extract_keywords


def detect_query_type(text: str) -> str:
    """
    Classifies argument intent
    """
    text = text.lower()

    research_triggers = ["research", "study", "studies", "data", "evidence", "analysis"]
    policy_triggers = ["should", "must", "ban", "allow", "policy"]

    if any(t in text for t in research_triggers):
        return "research"

    if any(t in text for t in policy_triggers):
        return "policy"

    return "general"


def rewrite_query(argument: str, topic: str = None) -> str:
    """
    Converts argument into retrieval-optimized query
    """

    # Extract keywords
    keywords = extract_keywords(argument, max_keywords=8)

    query_type = detect_query_type(argument)

    # =========================
    # Research-style queries
    # =========================
    if query_type == "research":
        base = " ".join(keywords)
        return f"{base} academic performance study research data"

    # =========================
    # Policy / debate queries
    # =========================
    if query_type == "policy":
        topic_part = topic if topic else ""
        base = " ".join(keywords)
        return f"{topic_part} {base} advantages disadvantages impact"

    # =========================
    # General queries
    # =========================
    base = " ".join(keywords)
    return base.strip()