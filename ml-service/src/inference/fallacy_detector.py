import re


def detect_fallacy(text):

    text = text.lower()

    # Circular reasoning
    if re.search(r"because .* (good|better|bad|true)", text):
        return "Circular Reasoning"

    # Hasty generalization
    if re.search(r"\bone\b.*\ball\b", text):
        return "Hasty Generalization"

    # Slippery slope
    if "soon" in text or "eventually" in text:
        return "Slippery Slope"

    # Personal preference
    if "i like" in text or "i prefer" in text:
        return "Personal Bias"

    # Weak reasoning
    if "i think" in text and "because" not in text:
        return "Unsupported Opinion"

    return "None"