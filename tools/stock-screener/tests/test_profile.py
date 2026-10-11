from moonshot.profile import shorten
from moonshot.sector import ETF_NAMES, SIC_TO_ETF


def test_keeps_first_two_sentences():
    text = "Acme makes widgets. It sells them worldwide. It was founded in 1901."
    assert shorten(text) == "Acme makes widgets. It sells them worldwide."


def test_long_first_sentence_is_cut_at_a_word_boundary():
    out = shorten("word " * 100)
    assert out.endswith("…") and len(out) <= 281 and "wor…" not in out


def test_second_sentence_dropped_when_it_would_overflow():
    first = "A" * 200 + "."
    assert shorten(f"{first} {'B' * 200}.") == first


def test_empty_input():
    assert shorten("") == "" and shorten(None) == ""


def test_every_mapped_etf_has_a_display_name():
    assert set(SIC_TO_ETF.values()) <= set(ETF_NAMES)
