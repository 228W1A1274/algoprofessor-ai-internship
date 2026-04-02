"""
DealHunter — evaluator.py
=========================
Core evaluation script. Run from command line:
    python evaluator.py

Loads data, builds prompts, calls configured LLM clients,
computes metrics, and saves results to output/.
"""

import os
import re
import json
import time
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.metrics import mae, mape, rmse, accuracy_within, compute_metrics
from utils.llm_clients import GroqClient, build_clients
from utils.data_utils import load_data, build_prompt, parse_price

# ── Config ──────────────────────────────────────────────────
EVAL_SAMPLES       = 50
SEED               = 42
USE_CATEGORY_HINT  = True
OUTPUT_DIR         = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_evaluation():
    print("=" * 60)
    print("  🎯 DealHunter — LLM Price Oracle Evaluation")
    print("=" * 60)

    # Load data
    df = load_data(max_rows=300)
    df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    split_idx   = int(len(df_shuffled) * 0.8)
    test_df     = df_shuffled.iloc[split_idx:].copy().reset_index(drop=True)
    test_eval   = test_df.head(EVAL_SAMPLES).reset_index(drop=True)

    print(f"\nEvaluating on {len(test_eval)} products")

    # Build clients
    clients = build_clients()
    if not clients:
        print("❌ No API clients available. Set GROQ_API_KEY or GOOGLE_API_KEY.")
        return

    all_results   = []
    all_preds_log = {}

    for client in clients:
        print(f"\n{'-'*55}")
        print(f"  Evaluating: {client.model_name}")
        print(f"{'-'*55}")

        y_true, y_pred, n_fail = [], [], 0
        pred_log = []

        for _, row in tqdm(test_eval.iterrows(), total=len(test_eval),
                           desc=client.model_name[:30]):
            cat    = row["category_inferred"] if USE_CATEGORY_HINT else None
            prompt = build_prompt(row["title_clean"], cat)
            raw    = client.predict(prompt)

            if not raw:
                n_fail += 1
                pred_log.append({
                    "title": row["title_clean"], "true": row["price"],
                    "predicted": None, "raw": raw, "error": "api_or_safety_fail"
                })
                continue

            price = parse_price(raw)
            if price is None:
                n_fail += 1
                pred_log.append({
                    "title": row["title_clean"], "true": row["price"],
                    "predicted": None, "raw": raw, "error": "parse_failure"
                })
                continue

            y_true.append(row["price"])
            y_pred.append(price)
            pred_log.append({
                "title": row["title_clean"], "true": row["price"],
                "predicted": price, "raw": raw
            })

        m = compute_metrics(y_true, y_pred, client.model_name, n_fail)
        m["avg_latency_ms"] = round(client.avg_latency_ms, 1)
        all_results.append(m)
        all_preds_log[client.model_name] = pred_log

        print(f"\n  Results — {client.model_name}:")
        print(f"    Evaluated : {m['n_evaluated']} / {m['n_evaluated']+m['n_failed']}")
        if m["mae"] is not None:
            print(f"    MAE       : ${m['mae']:.2f}")
            print(f"    MAPE      : {m['mape']:.2f}%")
            print(f"    RMSE      : ${m['rmse']:.2f}")
            print(f"    ±20% Acc  : {m['accuracy_within_20pct']:.1f}%")
        else:
            print("    ⚠️  All predictions failed — check API key")

    # Save results
    with open(OUTPUT_DIR / "leaderboard.json", "w") as f:
        json.dump(all_results, f, indent=2)

    for model_name, preds in all_preds_log.items():
        safe = model_name.replace("/", "_").replace(":", "_")
        with open(OUTPUT_DIR / f"predictions_{safe}.json", "w") as f:
            json.dump(preds, f, indent=2)

    print(f"\n💾 Results saved to {OUTPUT_DIR}/")
    print("\n✅ Evaluation complete!")

    # Summary
    valid = [r for r in all_results if r["mae"] is not None]
    if valid:
        best = min(valid, key=lambda r: r["mape"])
        print(f"\n🏆 Winner: {best['model']}")
        print(f"   MAPE={best['mape']}% | MAE=${best['mae']} | ±20%={best['accuracy_within_20pct']}%")


if __name__ == "__main__":
    run_evaluation()
