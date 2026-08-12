from __future__ import annotations

import json
import re
from typing import Any

REQUIRED_KEYS = {
    "market_regime",
    "conviction",
    "time_horizon",
    "thesis",
    "bullish_factors",
    "bearish_factors",
    "invalidation_conditions",
    "levels_to_watch",
    "news_context",
    "data_caveats",
    "bottom_line",
}


def parse_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    for pattern in (r"```json\s*(\{.*\})\s*```", r"```\s*(\{.*\})\s*```", r"(\{.*\})"):
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            continue
        try:
            value = json.loads(match.group(1))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def validate_research_payload(data: dict[str, Any] | None) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "AI output was not a JSON object."
    missing = REQUIRED_KEYS - set(data)
    if missing:
        return False, f"AI output is missing required fields: {', '.join(sorted(missing))}."
    list_fields = (
        "bullish_factors",
        "bearish_factors",
        "invalidation_conditions",
        "levels_to_watch",
        "data_caveats",
    )
    for field in list_fields:
        if not isinstance(data.get(field), list):
            return False, f"AI field '{field}' must be a list."
    return True, ""


def build_research_prompt(
    symbol: str,
    company_name: str,
    snapshot: dict[str, Any],
    backtest_metrics: dict[str, Any] | None,
    news: list[dict[str, str]],
    data_context: dict[str, Any] | None = None,
) -> str:
    payload = {
        "symbol": symbol,
        "company_name": company_name,
        "data_context": data_context or {},
        "technical_snapshot": snapshot,
        "backtest": backtest_metrics,
        "news": news,
    }
    return f"""
You are a market research copilot. Analyze only the supplied payload. Do not invent fundamentals,
price targets, events, forecasts, or news that are not present. The deterministic trend model is
an input, not proof of future performance. Backtest results are historical simulations and must
not be presented as predictive guarantees. Distinguish clearly between observed data, historical
simulation, and interpretation. Do not issue a buy/sell command.

DATA:
{json.dumps(payload, indent=2, default=str)}

Return ONLY valid JSON with this exact schema:
{{
  "market_regime": "Bullish | Neutral | Bearish",
  "conviction": "Low | Medium | High",
  "time_horizon": "Short-term | Swing | Position",
  "thesis": "2-4 sentences grounded only in the supplied data",
  "bullish_factors": ["factor", "factor", "factor"],
  "bearish_factors": ["factor", "factor", "factor"],
  "invalidation_conditions": ["observable condition", "observable condition"],
  "levels_to_watch": ["level and why", "level and why"],
  "news_context": "1-3 sentences based only on supplied headlines, or 'No news context available'",
  "data_caveats": ["caveat", "caveat"],
  "bottom_line": "One concise research summary without a buy/sell instruction"
}}
""".strip()


def generate_research(client: Any, prompt: str, model: str = "llama-3.3-70b-versatile") -> tuple[dict[str, Any] | None, str | None, str | None]:
    kwargs = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model,
        "temperature": 0.1,
    }
    try:
        try:
            response = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
        except TypeError:
            response = client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content.strip()
        data = parse_json_object(raw)
        valid, error = validate_research_payload(data)
        if not valid:
            return None, raw, error
        return data, raw, None
    except Exception as exc:
        return None, None, f"AI request failed: {exc}"
