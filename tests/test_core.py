from lecture_notes.chunking import chunk_text
from lecture_notes.llm import dedupe_cards, parse_cards
from lecture_notes.transcribe import Segment, format_timestamp, segments_to_text


def test_chunking_respects_limit():
    text = " ".join(["यह एक वाक्य है।"] * 500)
    chunks = chunk_text(text, max_chars=200)
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) > 1


def test_chunking_splits_unpunctuated_text():
    chunks = chunk_text("અ" * 1000, max_chars=300)
    assert all(len(c) <= 300 for c in chunks)
    assert sum(len(c) for c in chunks) == 1000


def test_parse_cards_handles_noise():
    raw = 'Sure! Here you go:\n[{"question": "Q1?", "answer": "A1"}, {"question": "", "answer": "x"}]'
    assert parse_cards(raw) == [{"question": "Q1?", "answer": "A1"}]


def test_parse_cards_bad_json():
    assert parse_cards("no json here") == []
    assert parse_cards("[not valid json]") == []


def test_dedupe_cards():
    cards = [{"question": "What?", "answer": "a"}, {"question": "what?", "answer": "b"}]
    assert len(dedupe_cards(cards)) == 1


def test_timestamps_and_text():
    segs = [Segment(0, 1, "hello"), Segment(3661, 3662, "world")]
    assert format_timestamp(3661) == "01:01:01"
    assert segments_to_text(segs) == "hello world"
    assert segments_to_text(segs, True).splitlines()[1].startswith("[01:01:01]")
