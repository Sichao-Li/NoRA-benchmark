from __future__ import annotations

import re
from functools import lru_cache


TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens."""

    return TOKEN_PATTERN.findall(text.lower())


def token_jaccard(text_a: str, text_b: str) -> float:
    """Compute token-set Jaccard similarity in [0, 1]."""

    set_a = set(tokenize(text_a))
    set_b = set(tokenize(text_b))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


@lru_cache(maxsize=1)
def _wordnet_available() -> bool:
    try:
        from nltk.corpus import wordnet as wn

        wn.synsets("test")
    except Exception:
        return False
    return True


@lru_cache(maxsize=2048)
def _token_synonyms(token: str) -> frozenset[str]:
    if not _wordnet_available():
        return frozenset({token})
    from nltk.corpus import wordnet as wn

    values = {token}
    for synset in wn.synsets(token):
        for lemma in synset.lemma_names():
            values.update(TOKEN_PATTERN.findall(lemma.lower().replace("_", " ")))
    return frozenset(values)


def synonym_jaccard(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity after expanding tokens with WordNet synonyms."""

    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    expanded_a: set[str] = set()
    expanded_b: set[str] = set()
    for token in tokens_a:
        expanded_a.update(_token_synonyms(token))
    for token in tokens_b:
        expanded_b.update(_token_synonyms(token))
    if not expanded_a and not expanded_b:
        return 1.0
    return len(expanded_a & expanded_b) / len(expanded_a | expanded_b)


def lexical_similarity(text_a: str, text_b: str, use_wordnet: bool = True) -> float:
    """Hybrid lexical overlap with optional WordNet expansion."""

    base = token_jaccard(text_a, text_b)
    if not use_wordnet or not _wordnet_available():
        return base
    return max(base, synonym_jaccard(text_a, text_b))


def wordnet_available() -> bool:
    """Public probe for optional WordNet support."""

    return _wordnet_available()
