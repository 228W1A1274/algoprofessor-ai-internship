"""
response_parser.py
──────────────────
Parses raw LLM response strings and extracts a price as a float.

Parsing strategy (three layers):
  1. JSON parsing  — primary method, handles clean {"price": 749.99}
  2. Regex fallback — extracts the first valid number from plain text
  3. Returns None  — if all methods fail, logs a warning

Usage:
    from utils.response_parser import parse_price_response
    price = parse_price_response(raw_llm_string)
"""

import json
import re
import logging

logger = logging.getLogger(__name__)

# Reasonable price sanity bounds for consumer electronics (USD)
PRICE_MIN =     1.0
PRICE_MAX = 99999.0

# Key names an LLM might use in its JSON response
PRICE_KEYS = ["price", "Price", "PRICE", "cost", "value",
              "estimated_price", "retail_price", "usd"]


def parse_price_response(raw_response: str) -> float | None:
    """
    Parse an LLM's raw string response and extract the price as a float.

    Tries in order:
      1. Strip markdown fences, attempt json.loads()
      2. Regex number extraction from plain text
      3. Return None and log warning on complete failure

    Args:
        raw_response (str): Raw string returned by any LLM API.

    Returns:
        float | None: Parsed price rounded to 2 decimal places,
                      or None if parsing fails.

    Examples:
        >>> parse_price_response('{"price": 749.99}')
        749.99
        >>> parse_price_response('```json\\n{"price": 750}\\n```')
        750.0
        >>> parse_price_response('The price is approximately $750.00')
        750.0
        >>> parse_price_response('I cannot determine the price.')
        None
    """
    if not raw_response or not isinstance(raw_response, str):
        return None

    # ── Layer 1: Clean markdown code fences ───────────────────────
    cleaned = raw_response.strip()
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)   # remove ```json or ```
    cleaned = re.sub(r"```\s*", "",          cleaned)   # remove closing ```
    cleaned = cleaned.strip()

    # ── Layer 2: JSON parsing (primary) ───────────────────────────
    try:
        parsed = json.loads(cleaned)

        # Handle dict: {"price": 749.99} or {"Price": "749.99"} etc.
        if isinstance(parsed, dict):
            for key in PRICE_KEYS:
                if key in parsed:
                    raw_val = parsed[key]
                    # Strip currency symbols if value is a string
                    if isinstance(raw_val, str):
                        raw_val = re.sub(r"[^\d.]", "", raw_val)
                    price = float(raw_val)
                    if PRICE_MIN <= price <= PRICE_MAX:
                        return round(price, 2)

        # Handle bare number: 749.99
        elif isinstance(parsed, (int, float)):
            price = float(parsed)
            if PRICE_MIN <= price <= PRICE_MAX:
                return round(price, 2)

    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        pass  # fall through to regex

    # ── Layer 3: Regex fallback ────────────────────────────────────
    # Matches patterns like: $749.99  |  749.99  |  749
    numbers = re.findall(r"\$?\s*(\d{1,6}(?:\.\d{1,2})?)", cleaned)

    for num_str in numbers:
        try:
            price = float(num_str)
            if PRICE_MIN <= price <= PRICE_MAX:
                return round(price, 2)
        except ValueError:
            continue

    # ── Layer 4: Complete failure ──────────────────────────────────
    logger.warning("Could not parse price from response: %s", raw_response[:120])
    return None
