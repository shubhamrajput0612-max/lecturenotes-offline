"""Command-line entry point: audio -> transcript -> summary + flashcards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .transcribe import segments_to_text, transcribe_audio


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lecture-notes",
        description="Offline Gujarati/Hindi lecture notes: Whisper + small on-device LLM.",
    )
    p.add_argument("audio", help="Lecture audio/video file (mp3, wav, m4a, mp4, ...)")
    p.add_argument("--lang", choices=["gu", "hi", "en"], default=None,
                   help="Lecture language. Omit to auto-detect.")
    p.add_argument("--whisper-model", default="small",
                   help="faster-whisper model size: tiny/base/small/medium/large-v3 (default: small)")
    p.add_argument("--llm", default="models/model.gguf",
                   help="Path to a GGUF model for llama.cpp (default: models/model.gguf)")
    p.add_argument("--out", default="output", help="Output folder (default: output)")
    p.add_argument("--summary-lang", choices=["gu", "hi", "en"], default=None,
                   help="Language for the summary (default: same as lecture)")
    p.add_argument("--transcript-only", action="store_true",
                   help="Skip the LLM step and only write the transcript")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.audio).stem

    print("[1/3] Transcribing with Whisper...")
    segments, lang = transcribe_audio(args.audio, args.whisper_model, args.lang)
    plain = segments_to_text(segments)
    (out / f"{stem}.transcript.txt").write_text(segments_to_text(segments, True), encoding="utf-8")
    print(f"      detected language: {lang}, {len(segments)} segments")

    if args.transcript_only:
        print(f"Done. Transcript saved in {out}/")
        return

    from .llm import LocalLLM  # imported lazily so --transcript-only works without llama.cpp

    llm = LocalLLM(args.llm)

    print("[2/3] Summarizing...")
    summary = llm.summarize(plain, lang=lang, out_lang=args.summary_lang)
    (out / f"{stem}.summary.md").write_text(f"# Lecture notes: {stem}\n\n{summary}\n", encoding="utf-8")

    print("[3/3] Creating flashcards...")
    cards = llm.flashcards(plain, lang=args.summary_lang or lang)
    (out / f"{stem}.flashcards.json").write_text(
        json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Anki-importable (tab-separated: front, back)
    with open(out / f"{stem}.flashcards.tsv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerows((c["question"], c["answer"]) for c in cards)

    print(f"Done. {len(cards)} flashcards. Files saved in {out}/")


if __name__ == "__main__":
    main()
