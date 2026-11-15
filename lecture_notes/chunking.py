"""Split long transcripts into chunks that fit a small LLM's context window."""

from __future__ import annotations

import re

# Sentence enders for Latin, Devanagari (danda) and Gujarati text.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")


def chunk_text(text: str, max_chars: int = 3000) -> list[str]:
    """Greedily pack sentences into chunks of at most `max_chars` characters."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        # A single very long sentence (Whisper often omits punctuation): hard-split it.
        while len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks
