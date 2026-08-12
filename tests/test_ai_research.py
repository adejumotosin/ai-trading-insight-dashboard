import json

from core.ai_research import parse_json_object, validate_research_payload


def valid_payload():
    return {
        "market_regime": "Neutral",
        "conviction": "Low",
        "time_horizon": "Swing",
        "thesis": "Test",
        "bullish_factors": [],
        "bearish_factors": [],
        "invalidation_conditions": [],
        "levels_to_watch": [],
        "news_context": "No news context available",
        "data_caveats": [],
        "bottom_line": "Test",
    }


def test_parse_json_object_plain_and_fenced():
    payload = valid_payload()
    raw = json.dumps(payload)
    assert parse_json_object(raw) == payload
    assert parse_json_object(f"```json\n{raw}\n```") == payload


def test_validate_research_payload_requires_schema():
    ok, error = validate_research_payload(valid_payload())
    assert ok is True
    assert error == ""
    broken = valid_payload()
    broken.pop("bottom_line")
    ok, error = validate_research_payload(broken)
    assert ok is False
    assert "bottom_line" in error
