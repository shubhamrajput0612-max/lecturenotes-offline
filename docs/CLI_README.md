# Offline Lecture Notes (Gujarati / Hindi)

Turn a recorded lecture into **a transcript, a summary and flashcards**, entirely on your own machine.
No cloud API, no internet needed after the one-time model downloads.

```
lecture.mp3 ──► Whisper (faster-whisper) ──► transcript
                                              │
                          small on-device LLM (llama.cpp, GGUF)
                                              ├──► summary.md
                                              └──► flashcards.json / .tsv (Anki)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You also need `ffmpeg` installed for non-WAV audio.

### Models (download once)

- **Whisper**: downloaded automatically on first run (`small` by default; use `medium` or `large-v3` for better Gujarati accuracy).
- **LLM**: download any small instruction-tuned GGUF model and save it as `models/model.gguf`
  (or pass `--llm path`). A 1.5B–3B model in Q4 quantization runs on a normal laptop CPU.
  Models with good multilingual support (e.g. Qwen2.5 or Gemma instruct variants) work best for Hindi/Gujarati.

## Usage

```bash
# Full pipeline
python -m lecture_notes.cli lecture.mp3 --lang gu

# Hindi lecture, summary written in English
python -m lecture_notes.cli lecture.mp3 --lang hi --summary-lang en

# Transcript only (no LLM needed)
python -m lecture_notes.cli lecture.mp3 --transcript-only
```

Outputs go to `output/`:

| File | Content |
|------|---------|
| `<name>.transcript.txt` | Timestamped transcript |
| `<name>.summary.md` | Bullet-point lecture notes |
| `<name>.flashcards.json` | Question/answer cards |
| `<name>.flashcards.tsv` | Import directly into Anki |

## Tips

- Whisper accuracy for Gujarati is noticeably lower than for Hindi: use `--whisper-model medium` if your machine allows.
- Long lectures are split into chunks and summarized map-reduce style, so they fit a small model's context window.
- Always skim the output: small models can miss details or make mistakes.

## Tests

```bash
pytest
```

## License

MIT
