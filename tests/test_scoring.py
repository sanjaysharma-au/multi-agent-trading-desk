import json

import pandas as pd
import pytest

from moonshot.scoring import Checkpoint, format_fundamentals_table, html_to_text, parse_score, status_lines


def test_parse_score_plain_json():
    r = parse_score('{"score": 42, "rationale": "Strong growth."}')
    assert r == {"score": 42, "rationale": "Strong growth."}


def test_parse_score_strips_think_block_and_code_fence():
    content = (
        "<think>let me consider the revenue trend across quarters...</think>\n"
        '```json\n{"score": -75, "rationale": "Going concern language in the latest 10-Q."}\n```'
    )
    r = parse_score(content)
    assert r["score"] == -75
    assert "concern" in r["rationale"]


def test_parse_score_clamps_out_of_range():
    assert parse_score('{"score": 500, "rationale": "x"}')["score"] == 100
    assert parse_score('{"score": -500, "rationale": "x"}')["score"] == -100


def test_parse_score_missing_json_raises():
    with pytest.raises(ValueError):
        parse_score("I refuse to answer in JSON.")


def test_html_to_text_strips_tags_and_scripts():
    raw = "<html><head><style>.a{}</style></head><body><p>Revenue &amp; growth</p><script>evil()</script></body></html>"
    text = html_to_text(raw)
    assert "Revenue & growth" in text
    assert "evil" not in text
    assert "<p>" not in text


def test_format_fundamentals_table_has_no_price_fields():
    df = pd.DataFrame({
        "period_end": pd.to_datetime(["2024-03-31", "2024-06-30"]),
        "announced": pd.to_datetime(["2024-05-01", "2024-08-01"]),
        "revenue": [1_200_000_000.0, 1_500_000_000.0],
        "revenue_yoy_pct": [None, 25.0],
        "eps": [0.45, 0.52],
        "net_income": [-2_000_000.0, 10_000_000.0],
    })
    table = format_fundamentals_table(df)
    assert "$1.20B" in table and "$1.50B" in table
    assert "+25.0%" in table
    assert "-$2.0M" in table
    for forbidden in ("price", "multiple", "peak", "start_date"):
        assert forbidden not in table.lower()


def test_checkpoint_resume_skips_done_and_retries_failed(tmp_path):
    path = tmp_path / "cp.json"
    cp = Checkpoint(path)
    cp.mark_done("AAA", 80, "great", "model-x")
    cp.mark_failed("BBB", "timeout", "model-x")

    # A fresh Checkpoint instance loaded from disk sees the same state (simulates a resumed process).
    resumed = Checkpoint(path)
    assert resumed.status_of("AAA") == "done"
    assert resumed.status_of("BBB") == "failed"
    assert resumed.attempts_of("BBB") == 1

    resumed.mark_failed("BBB", "timeout again", "model-x")
    assert resumed.attempts_of("BBB") == 2

    n = resumed.reset_failed()
    assert n == 1
    assert resumed.status_of("BBB") == "pending"


def test_checkpoint_save_is_atomic_no_partial_file(tmp_path):
    path = tmp_path / "cp.json"
    cp = Checkpoint(path)
    cp.mark_done("AAA", 10, "x", "m")
    assert not path.with_suffix(".tmp").exists()
    assert json.loads(path.read_text())["AAA"]["score"] == 10


def test_status_lines_counts():
    cp = Checkpoint.__new__(Checkpoint)
    cp.data = {"AAA": {"status": "done"}, "BBB": {"status": "failed", "attempts": 2, "error": "boom"}}
    lines = status_lines([("AAA", None, None, None), ("BBB", None, None, None), ("CCC", None, None, None)], cp)
    assert lines[0] == "1/3 scored, 1 failed, 1 pending"
    assert any("BBB" in l and "boom" in l for l in lines[1:])
