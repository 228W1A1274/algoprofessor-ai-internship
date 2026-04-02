"""
utils/data_utils.py
===================
Data loading, feature engineering, prompt building, and price parsing.
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Category inference ───────────────────────────────────────

CATEGORY_MAP = {
    "electronics": ["phone", "laptop", "tablet", "camera", "headphone", "speaker",
                    "monitor", "keyboard", "mouse", "tv", "charger", "cable", "ipad"],
    "clothing":    ["shirt", "dress", "shoes", "pants", "jacket", "jeans", "boots", "sneaker"],
    "home":        ["furniture", "lamp", "chair", "table", "bed", "sofa", "rug", "cookware", "vacuum"],
    "beauty":      ["skincare", "makeup", "perfume", "lipstick", "serum", "cream", "foundation"],
    "sports":      ["gym", "yoga", "running", "bicycle", "fitness", "tennis", "basketball"],
    "books":       ["novel", "textbook", "biography", "cookbook", "guide", "hardcover"],
    "toys":        ["toy", "game", "puzzle", "lego", "doll", "action figure"],
}


def infer_category(title: str) -> str:
    t = title.lower()
    for cat, kws in CATEGORY_MAP.items():
        if any(k in t for k in kws):
            return cat
    return "other"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text).strip())
    text = re.sub(r"[^\w\s\-\.,()°%$/]", "", text)
    return text[:200]


# ── Synthetic data fallback ──────────────────────────────────

def generate_synthetic(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    TEMPLATES = [
        ("Apple iPhone {v} - {s}GB - {c}",                    "electronics",  799, 200),
        ("Samsung Galaxy S{v} Ultra Smartphone",               "electronics", 1099, 300),
        ("Sony WH-1000XM{v} Noise Cancelling Headphones",     "electronics",  349,  80),
        ("Logitech MX Master {v} Wireless Mouse",             "electronics",   79,  20),
        ("Dell XPS {v} 15-inch Laptop Intel Core i7",         "electronics", 1399, 400),
        ("Canon EOS R{v} Mirrorless Camera Body",             "electronics", 1299, 500),
        ("iPad Pro {v}-inch M2 Chip Wi-Fi",                   "electronics", 1099, 250),
        ("Nintendo Switch OLED {c} Edition",                  "electronics",  349,  50),
        ("Anker PowerCore {v}000mAh Portable Charger",        "electronics",   39,  15),
        ("Nike Air Max {v} Running Shoes Men",                "clothing",     129,  30),
        ("Levi's 501 Original Fit Jeans {c}",                 "clothing",      59,  15),
        ("KitchenAid Artisan Stand Mixer {c}",                "home",         449, 100),
        ("Dyson V{v} Cordless Vacuum Cleaner",                "home",         499, 150),
        ("Instant Pot Duo 7-in-1 Electric Pressure Cooker",   "home",          99,  25),
        ("IKEA KALLAX Shelf Unit {c}",                        "home",          89,  20),
        ("Atomic Habits by James Clear Hardcover",            "books",         17,   4),
        ("The Psychology of Money Morgan Housel",             "books",         15,   4),
        ("LEGO Technic {v} Building Set",                     "toys",          89,  40),
        ("Gaiam Premium Yoga Mat {c} 6mm Thick",              "sports",        35,  12),
        ("CeraVe Moisturizing Cream 19oz Daily Face Body",    "beauty",        18,   5),
    ]
    rows = []
    for _ in range(n):
        tmpl, cat, mu, sigma = TEMPLATES[rng.integers(0, len(TEMPLATES))]
        price = float(np.clip(rng.normal(mu, sigma), mu * 0.3, mu * 2.5))
        title = tmpl.format(
            v=rng.integers(5, 16),
            s=rng.choice([64, 128, 256, 512]),
            c=rng.choice(["Black", "White", "Silver", "Blue", "Red", "Green"]),
        )
        rows.append({"title": title, "price": round(price, 2), "category": cat})
    return pd.DataFrame(rows)


# ── Data loader ──────────────────────────────────────────────

def load_data(max_rows: int = 300) -> pd.DataFrame:
    """Load Amazon Electronics from HuggingFace, or fall back to synthetic data."""
    # Check local cache first
    cache = Path("data/products.csv")
    if cache.exists():
        print(f"Loading from cache: {cache}")
        df_raw = pd.read_csv(cache)
    else:
        try:
            from datasets import load_dataset as hf_load
            print("Loading Amazon Electronics from HuggingFace (streaming)...")
            ds = hf_load(
                "McAuley-Lab/Amazon-Reviews-2023",
                "raw_meta_Electronics",
                split="full",
                streaming=True,
                trust_remote_code=True,
            )
            rows = []
            for item in ds:
                if len(rows) >= max_rows:
                    break
                price = item.get("price")
                title = item.get("title", "")
                if not title or not price:
                    continue
                try:
                    pv = float(str(price).replace("$", "").replace(",", "").strip())
                except (ValueError, AttributeError):
                    continue
                if 0 < pv <= 5000:
                    rows.append({"title": title, "price": pv, "category": "Electronics"})
            if len(rows) < 50:
                raise ValueError("Too few rows from HuggingFace")
            df_raw = pd.DataFrame(rows)
            cache.parent.mkdir(exist_ok=True)
            df_raw.to_csv(cache, index=False)
            print(f"✅ Loaded {len(df_raw)} Amazon products")
        except Exception as e:
            print(f"HuggingFace failed ({e}). Using synthetic data.")
            df_raw = generate_synthetic(max_rows)
            cache.parent.mkdir(exist_ok=True)
            df_raw.to_csv(cache, index=False)
            print(f"✅ Generated {len(df_raw)} synthetic products")

    # Feature engineering
    df = df_raw.copy()
    df["title_clean"]       = df["title"].apply(clean_text)
    df["category_inferred"] = df["title_clean"].apply(infer_category)
    df["word_count"]        = df["title_clean"].str.split().str.len()
    df["log_price"]         = np.log1p(df["price"])
    df["price_bucket"]      = pd.cut(
        df["price"],
        bins=[0, 25, 75, 200, 500, 10000],
        labels=["budget", "low", "mid", "high", "premium"],
    )
    return df


# ── Prompt building ──────────────────────────────────────────

def build_prompt(title: str, category: Optional[str] = None) -> str:
    cat_hint = f" in the {category} category" if category else ""
    return (
        f"You are a product pricing expert. Based on the product title below"
        f"{cat_hint}, estimate its retail price in USD.\n\n"
        f"Product: {title}\n\n"
        f"Rules:\n"
        f"- Return ONLY a number (e.g., 29.99)\n"
        f"- No dollar sign, no text, no explanation\n"
        f"- Base your estimate on typical US retail prices\n\n"
        f"Price:"
    )


# ── Price parsing ────────────────────────────────────────────

def parse_price(response: str) -> Optional[float]:
    if not response:
        return None
    text  = str(response).strip().replace("$", "").replace(",", "").replace("USD", "")
    match = re.search(r"\b(\d{1,5}(?:\.\d{1,2})?)\b", text)
    if match:
        price = float(match.group(1))
        if 0.01 <= price <= 100_000:
            return price
    return None
