from __future__ import annotations

import json
import re
from typing import Any


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


def build_research_prompt(
    symbol: str,
    company_name: str,
    snapshot: dict[str, Any],
    backtest_metrics: dict[str, Any] | None,
    news: list[dict[str, str]],
) -> str:
    payload = {
        "symbol": symbol,
        "company_name": company_name,
        "technical_snapshot": snapshot,
        "backtest": backtest_metrics,
        "news": news,
    }
    return f"""
You are a market research copilot. Analyze only the supplied data. Do not invent fundamentals,
price targets, events, or news that are not in the payload. Treat the deterministic trend score
as an input, not as proof of future returns. Backtest results are historical and must not be
presented as predictive guarantees.

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


def generate_research(client: Any, prompt: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        return parse_json_object(raw), raw
    except TypeError:
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            return parse_json_object(raw), raw
        except Exception:
            return None, None
    except Exception:
        return None, None
