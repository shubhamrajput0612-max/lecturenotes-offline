"""Speech-to-text using faster-whisper (runs fully offline once the model is downloaded)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_LANGS = {"gu": "Gujarati", "hi": "Hindi", "en": "English"}


@dataclass
class Segment:
    start: float
    end: float
    text: str


def transcribe_audio(
    audio_path: str | Path,
    model_size: str = "small",
    language: str | None = None,
    device: str = "auto",
    compute_type: str = "int8",
) -> tuple[list[Segment], str]:
    """Transcribe an audio/video file. Returns (segments, detected_language).

    `language` may be "gu", "hi", "en" or None for auto-detect.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    raw_segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,  # skips silence, helps a lot on long lectures
        beam_size=5,
    )
    segments = [Segment(s.start, s.end, s.text.strip()) for s in raw_segments if s.text.strip()]
    return segments, info.language


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def segments_to_text(segments: list[Segment], with_timestamps: bool = False) -> str:
    if with_timestamps:
        return "\n".join(f"[{format_timestamp(s.start)}] {s.text}" for s in segments)
    return " ".join(s.text for s in segments)
