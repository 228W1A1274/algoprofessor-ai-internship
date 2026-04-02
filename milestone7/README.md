# 🎯 DealHunter — Milestone 7
## AlgoProfessor AI R&D Internship 2026 | Phase 2 | Day 40

---

## What is DealHunter?
DealHunter is a **product price prediction system** that benchmarks multiple LLMs
on estimating retail prices from product titles. It produces a live Streamlit leaderboard.

---

## Models Compared
| Model | Provider | Cost |
|-------|----------|------|
| `llama-3.3-70b-versatile` | Groq | FREE |
| `gemini-2.0-flash` | Google | FREE |

---

## Metrics
| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error ($) |
| **MAPE** | Mean Absolute % Error — main metric, target < 20% |
| **RMSE** | Root Mean Squared Error |
| **±20% Acc** | % of predictions within 20% of true price |

---

## Project Structure
```
DealHunter/
├── app.py              # Streamlit leaderboard UI
├── evaluator.py        # CLI evaluation runner
├── requirements.txt
├── README.md
├── data/               # Cached product data (auto-generated)
│   └── products.csv
├── output/             # Evaluation results
│   ├── leaderboard.json
│   └── predictions_*.json
└── utils/
    ├── __init__.py
    ├── data_utils.py   # Data loading, prompts, price parsing
    ├── llm_clients.py  # Groq & Gemini client wrappers
    └── metrics.py      # MAE, MAPE, RMSE, accuracy_within
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API keys
```bash
export GROQ_API_KEY="your_groq_key"       # https://console.groq.com
export GOOGLE_API_KEY="your_gemini_key"   # https://aistudio.google.com/app/apikey
```

### 3. Run evaluation
```bash
python evaluator.py
```

### 4. Launch leaderboard UI
```bash
streamlit run app.py
```

---

## FREE API Keys
| Provider | URL | Daily Limit |
|----------|-----|-------------|
| Google Gemini | https://aistudio.google.com/app/apikey | 1M tokens/day |
| Groq (Llama) | https://console.groq.com | 14,400 req/day |

---

## Commit Message
```
[Day-40] Milestone 7: DealHunter — LLM price oracle
```
