"""Small on-device LLM via llama.cpp (GGUF models). No network needed at run time."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .chunking import chunk_text

LANG_NAMES = {"gu": "Gujarati", "hi": "Hindi", "en": "English"}


class LocalLLM:
    def __init__(self, model_path: str | Path, n_ctx: int = 4096, n_threads: int | None = None):
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "llama-cpp-python is not installed. Run: pip install -r requirements.txt"
            ) from exc
        if not Path(model_path).exists():
            raise SystemExit(
                f"LLM model not found: {model_path}\n"
                "Download a small GGUF model (see README) and pass it with --llm."
            )
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )

    def _chat(self, system: str, user: str, max_tokens: int = 700, temperature: float = 0.2) -> str:
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return out["choices"][0]["message"]["content"].strip()

    def summarize(self, transcript: str, lang: str = "en", out_lang: str | None = None) -> str:
        """Map-reduce summary: summarize each chunk, then merge the partial summaries."""
        target = LANG_NAMES.get(out_lang or lang, "English")
        system = (
            "You are a careful note-taking assistant for university lectures. "
            f"The lecture is in {LANG_NAMES.get(lang, lang)}. Write in {target}. "
            "Use short bullet points. Do not invent facts that are not in the text."
        )
        partials = [
            self._chat(system, f"Summarize the key points of this lecture part:\n\n{chunk}")
            for chunk in chunk_text(transcript)
        ]
        if len(partials) == 1:
            return partials[0]
        merged = "\n\n".join(partials)
        return self._chat(
            system,
            "Combine these partial notes into one clean, well-organized set of lecture notes "
            f"with headings and bullet points:\n\n{merged}",
            max_tokens=1200,
        )

    def flashcards(self, transcript: str, lang: str = "en", per_chunk: int = 5) -> list[dict]:
        target = LANG_NAMES.get(lang, "English")
        system = (
            "You create study flashcards from lecture text. "
            f"Write questions and answers in {target}. "
            'Reply with ONLY a JSON array like [{"question": "...", "answer": "..."}].'
        )
        cards: list[dict] = []
        for chunk in chunk_text(transcript):
            raw = self._chat(
                system,
                f"Create {per_chunk} flashcards from this lecture text:\n\n{chunk}",
                max_tokens=800,
                temperature=0.3,
            )
            cards.extend(parse_cards(raw))
        return dedupe_cards(cards)


def parse_cards(raw: str) -> list[dict]:
    """Extract a JSON list of {question, answer} from possibly messy model output."""
    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    cards = []
    for item in data:
        if isinstance(item, dict) and item.get("question") and item.get("answer"):
            cards.append({"question": str(item["question"]).strip(), "answer": str(item["answer"]).strip()})
    return cards


def dedupe_cards(cards: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for card in cards:
        key = card["question"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(card)
    return unique
