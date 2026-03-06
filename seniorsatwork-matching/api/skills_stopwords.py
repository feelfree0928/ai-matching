"""
Stopwords and clean-skills helpers so we never treat words like "an", "der", "im"
as matched skills in rank explanation or scoring.
"""
# German stopwords (common articles, prepositions, pronouns, conjunctions)
SKILLS_STOPWORDS_DE = frozenset({
    "an", "der", "die", "das", "dem", "den", "des", "ein", "eine", "einer", "einem", "einen",
    "im", "in", "und", "oder", "mit", "für", "von", "zu", "zur", "zum", "auf", "aus", "bei",
    "bis", "durch", "nach", "über", "um", "als", "bis", "nach", "seit", "vom", "bei", "gegen",
    "ohne", "bis", "während", "wegen", "statt", "trotz", "innerhalb", "außerhalb",
    "sie", "er", "es", "wir", "ihr", "sie", "ich", "du", "sie", "sie", "ihnen", "ihm",
    "ist", "sind", "war", "waren", "werden", "wurde", "wurden", "hat", "haben", "hatte", "hatten",
    "kann", "können", "konnte", "konnten", "wird", "wurde", "wurden", "wird",
    "nicht", "auch", "noch", "nur", "schon", "sehr", "mehr", "immer", "noch", "doch",
    "aber", "denn", "sondern", "weil", "wenn", "dass", "ob", "falls", "sofern",
    "alle", "allem", "allen", "aller", "alles", "jeder", "jede", "jedes", "manche",
    "andere", "anderen", "beide", "einige", "mehrere", "viele", "wenige",
})

# English and other common stopwords that may appear in job skills text
SKILLS_STOPWORDS_EN = frozenset({
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "need", "dare", "ought", "shall",
    "this", "that", "these", "those", "it", "its", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "also",
})

# Combined set for "clean skills" filtering (display and any token-based scoring)
SKILLS_STOPWORDS = SKILLS_STOPWORDS_DE | SKILLS_STOPWORDS_EN

# Minimum character length for a token to count as a skill (avoids "an", "im", "zu")
MIN_SKILL_TOKEN_LEN = 3


def is_clean_skill_token(token: str) -> bool:
    """Return True if token should be treated as a skill (not a stopword, not too short)."""
    if not token or len(token) < MIN_SKILL_TOKEN_LEN:
        return False
    return token.lower().strip() not in SKILLS_STOPWORDS


def clean_skill_tokens(tokens: list[str]) -> list[str]:
    """Filter a list of tokens to only those that are valid skill terms."""
    return [t for t in tokens if is_clean_skill_token(t)]
