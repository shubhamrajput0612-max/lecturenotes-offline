"""
LectureNotes Offline: transcribe -> summary -> flashcards

Usage:
    python app.py --audio lecture.mp3 --lang gu --llm-model models/llama-3.2-3b-instruct.Q4_K_M.gguf

IMPORTANT (honest engine note):
    This reference build runs Whisper through faster-whisper (CPU) and the LLM
    through llama-cpp-python (CPU). That is the path I can run anywhere.
    The Snapdragon NPU path (Qualcomm AI Hub models + ONNX Runtime QNN EP) is
    NOT wired in here. report_engine() only *reports* whether the QNN provider
    is available on this machine; it does not claim NPU execution.
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

LANG_NAMES = {"gu": "Gujarati", "hi": "Hindi", "en": "English"}


# ----------------------------------------------------------------- engine info
def report_engine():
    """Print what hardware providers are available. Never claims NPU use."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_qnn = "QNNExecutionProvider" in providers
        print(f"[engine] ONNX Runtime providers: {providers}")
        print("[engine] QNN (Snapdragon NPU) provider available: "
              f"{has_qnn}. (This build still runs Whisper/LLM on CPU.)")
    except ImportError:
        print("[engine] onnxruntime not installed; running CPU-only path.")


# --------------------------------------------------------------- 1. transcribe
def transcribe(audio_path, lang, whisper_size):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("Missing dependency: pip install faster-whisper")

    print(f"[1/3] Loading Whisper ({whisper_size}) on CPU ...")
    model = WhisperModel(whisper_size, device="cpu", compute_type="int8")

    print("[1/3] Transcribing (this can take a while on long lectures) ...")
    t0 = time.time()
    segments, info = model.transcribe(
        audio_path,
        language=None if lang == "auto" else lang,
        vad_filter=True,          # skips silence, helps on long recordings
        beam_size=5,
    )
    lines = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            lines.append(text)
    elapsed = time.time() - t0
    print(f"[1/3] Done in {elapsed:.1f}s. Detected language: {info.language} "
          f"(p={info.language_probability:.2f})")
    return " ".join(lines), info.language


# ------------------------------------------------------------------- LLM utils
def load_llm(model_path, n_ctx=4096):
    try:
        from llama_cpp import Llama
    except ImportError:
        sys.exit("Missing dependency: pip install llama-cpp-python")
    if not Path(model_path).exists():
        sys.exit(f"LLM file not found: {model_path}\n"
                 "Download a Llama 3.2 3B Instruct GGUF and pass --llm-model.")
    print(f"[llm] Loading {model_path} on CPU ...")
    return Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)


def chat(llm, system, user, max_tokens=600, temperature=0.2):
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return out["choices"][0]["message"]["content"].strip()


def chunk_text(text, max_chars=1500):
    """Split on sentence-ish boundaries (includes Devanagari/Gujarati danda)."""
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > max_chars and cur:
            chunks.append(cur.strip())
            cur = s
        else:
            cur += " " + s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


# ------------------------------------------------------------------ 2. summary
def summarize(llm, transcript, lang_name):
    system = (
        f"You are a careful study assistant. Write in {lang_name}. "
        "Only use information present in the lecture text. "
        "Do not invent facts, formulas or numbers. Keep technical terms "
        "as spoken."
    )
    chunks = chunk_text(transcript)
    print(f"[2/3] Summarizing {len(chunks)} chunk(s) ...")
    partials = []
    for i, ch in enumerate(chunks, 1):
        print(f"      chunk {i}/{len(chunks)}")
        partials.append(chat(
            llm, system,
            "Summarize this part of a lecture in 3-5 short bullet points:\n\n"
            + ch,
            max_tokens=300,
        ))

    if len(partials) == 1:
        return partials[0]

    combined = "\n".join(partials)
    # Keep the merge step inside the context window.
    combined = combined[:6000]
    return chat(
        llm, system,
        "Below are bullet-point summaries of consecutive parts of one lecture. "
        "Merge them into revision notes: a 2-line overview, then key points "
        "as bullets, then a short list of important terms.\n\n" + combined,
        max_tokens=700,
    )


# ---------------------------------------------------------------- 3. flashcards
def parse_cards(raw):
    """Try JSON first, then fall back to 'Q: ... A: ...' parsing."""
    m = re.search(r"\[.*\]", raw, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            cards = [(str(c.get("question", "")).strip(),
                      str(c.get("answer", "")).strip())
                     for c in data if isinstance(c, dict)]
            cards = [c for c in cards if c[0] and c[1]]
            if cards:
                return cards
        except json.JSONDecodeError:
            pass
    cards = re.findall(r"Q[:.]\s*(.+?)\s*A[:.]\s*(.+?)(?=\n\s*Q[:.]|\Z)",
                       raw, re.S)
    return [(q.strip(), a.strip()) for q, a in cards]


def make_flashcards(llm, summary, lang_name, n_cards):
    system = (
        f"You create study flashcards in {lang_name}. Use ONLY the notes "
        "provided. Do not add outside facts."
    )
    print("[3/3] Generating flashcards ...")
    raw = chat(
        llm, system,
        f"From these notes, create {n_cards} flashcards. Reply ONLY with a "
        'JSON list like [{"question": "...", "answer": "..."}].\n\n'
        + summary,
        max_tokens=900,
        temperature=0.3,
    )
    cards = parse_cards(raw)
    if not cards:
        print("[3/3] Warning: could not parse flashcards; saving raw output.")
    return cards, raw


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Offline lecture notes")
    ap.add_argument("--audio", required=True, help="lecture audio file")
    ap.add_argument("--lang", default="auto", choices=["auto", "gu", "hi", "en"],
                    help="lecture language (auto = let Whisper detect)")
    ap.add_argument("--whisper-size", default="small",
                    help="tiny / base / small / medium")
    ap.add_argument("--llm-model", required=True,
                    help="path to a GGUF file (e.g. Llama 3.2 3B Instruct)")
    ap.add_argument("--cards", type=int, default=8, help="number of flashcards")
    ap.add_argument("--out", default="output", help="output folder")
    args = ap.parse_args()

    if not Path(args.audio).exists():
        sys.exit(f"Audio file not found: {args.audio}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report_engine()

    # 1. Transcribe
    transcript, detected = transcribe(args.audio, args.lang, args.whisper_size)
    (out / "transcript.txt").write_text(transcript, encoding="utf-8")
    if not transcript.strip():
        sys.exit("No speech detected in the audio.")

    lang_code = detected if args.lang == "auto" else args.lang
    lang_name = LANG_NAMES.get(lang_code, "the same language as the lecture")

    # 2. Summarize
    llm = load_llm(args.llm_model)
    summary = summarize(llm, transcript, lang_name)
    (out / "notes.md").write_text(
        "# Lecture Notes\n\n" + summary + "\n", encoding="utf-8")

    # 3. Flashcards
    cards, raw = make_flashcards(llm, summary, lang_name, args.cards)
    if cards:
        # utf-8-sig so Excel opens Gujarati/Hindi text correctly
        with open(out / "flashcards.csv", "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["question", "answer"])
            w.writerows(cards)
    else:
        (out / "flashcards_raw.txt").write_text(raw, encoding="utf-8")

    print(f"\nDone. Files saved in: {out.resolve()}")
    print("Reminder: AI output can contain mistakes. Review the notes and "
          "cards before studying from them.")


if __name__ == "__main__":
    main()
