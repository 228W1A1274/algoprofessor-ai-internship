"""
prompt_builder.py
─────────────────
Builds structured prompts for LLM price prediction.

Each prompt instructs the model to return ONLY a JSON object
in the format: {"price": <float>}

Usage:
    from utils.prompt_builder import build_price_prompt
    prompt = build_price_prompt(product_info_dict)
"""


def build_price_prompt(product_info: dict) -> str:
    """
    Build a structured price prediction prompt for any LLM.

    The prompt enforces JSON-only output so the response parser
    can reliably extract the numeric price value.

    Args:
        product_info (dict): Must contain at least 'name'. Optional keys:
                             'category', 'brand', 'features', 'condition'

    Returns:
        str: Complete prompt string ready to send to any LLM API.

    Example:
        >>> product = {
        ...     "name": "Sony WH-1000XM5",
        ...     "category": "Headphones",
        ...     "brand": "Sony",
        ...     "features": "ANC, 30hr battery, Bluetooth 5.2"
        ... }
        >>> prompt = build_price_prompt(product)
    """
    name      = product_info.get("name",      "Unknown Product")
    category  = product_info.get("category",  "Electronics")
    brand     = product_info.get("brand",     "Unknown Brand")
    features  = product_info.get("features",  "Standard features")
    condition = product_info.get("condition", "New (retail box)")

    prompt = f"""You are a product pricing expert with deep knowledge of US consumer electronics retail markets.

Product Details:
- Name: {name}
- Category: {category}
- Brand: {brand}
- Key Features: {features}
- Condition: {condition}
- Market: United States

Task: Estimate the typical US retail price of this product based on your knowledge.

Rules:
1. Return ONLY a valid JSON object — no explanation, no markdown, no extra text.
2. Use exactly this format: {{"price": <number>}}
3. The price value must be a plain number (float or int). No dollar signs inside the JSON.
4. Example of a correct response: {{"price": 279.99}}"""

    return prompt
