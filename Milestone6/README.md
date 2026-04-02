# 🔮 PriceOracle — Multi-LLM Product Price Benchmarking System

**Milestone 6 | LLM Evaluation Sprint | Days 41–45**

PriceOracle benchmarks state-of-the-art Large Language Models on their ability
to predict US retail prices for consumer electronics products.

---

## What This Project Does

- Evaluates **2–3 LLMs** on a dataset of 20 real products
- Uses **structured JSON prompting** to reliably extract numeric predictions
- Computes **MAE, MAPE, and RMSE** evaluation metrics per model
- Produces a **ranked leaderboard** and comparison visualizations

---

## Models Compared

| Model | Provider | API |
|-------|----------|-----|
| Gemini 2.5 Flash | Google | AI Studio |
| Llama 3.3 70B | Meta (via Groq) | Groq Console |
| GPT-4o-mini | OpenAI | Platform (optional) |

---

## Project Structure

```
Milestone_6_PriceOracle/
├── data/
│   └── products.csv          ← 20 real consumer electronics products
├── output/                   ← results saved here at runtime
├── utils/
│   ├── __init__.py
│   ├── prompt_builder.py     ← build_price_prompt()
│   ├── response_parser.py    ← parse_price_response()
│   └── api_callers.py        ← init_clients(), predict_with_*()
├── evaluator.py              ← compute_metrics(), build_leaderboard()
├── .gitignore
├── README.md
└── requirements.txt
```

The Colab notebook (`Milestone_6_PriceOracle.ipynb`) imports from
these modules and runs the full pipeline end-to-end.

---

## Setup

### 1. Get API Keys (all free tier)

| Key | Where to get it |
|-----|-----------------|
| `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey |
| `GROQ_API_KEY`   | https://console.groq.com/keys |
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys *(optional)* |

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run in Google Colab

1. Upload this entire folder to your Google Drive
2. Open `Milestone_6_PriceOracle.ipynb` in Colab
3. Enter your API keys in Cell 3
4. Run all cells in order (Runtime → Run all)

---

## Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **MAE** | mean(\|true − pred\|) | Average dollar error |
| **MAPE** | mean(\|true − pred\| / true) × 100 | Average % error — primary ranking metric |
| **RMSE** | √ mean((true − pred)²) | Error with extra penalty for large mistakes |
| **Coverage** | valid predictions / total | % of products where price was parsed |

**Lower values = better model.** Leaderboard is ranked by MAPE ascending.

---

## Dataset

`data/products.csv` — 20 consumer electronics products across 4 categories:

| Category | Count | Price Range |
|----------|-------|-------------|
| Television | 5 | $129 – $1,098 |
| Laptop | 5 | $329 – $1,600 |
| Smartphone | 5 | $199 – $799 |
| Headphones | 5 | $79 – $279 |

Ground truth prices reflect typical US retail market values (2024–2025).

---

## Output Files

After running the notebook, these files appear in `output/`:

| File | Description |
|------|-------------|
| `priceoralce_results.csv` | Full predictions table |
| `priceoralce_leaderboard.csv` | Metric summary per model |
| `priceoralce_benchmark_chart.png` | Bar chart visualization |

---

## Key Concepts

- **Structured output prompting** — enforcing JSON format across different LLM providers
- **Multi-layer response parsing** — JSON → regex → None fallback chain
- **Exponential backoff** — graceful handling of API rate limits
- **MAPE as primary metric** — scale-independent comparison across mixed price ranges
