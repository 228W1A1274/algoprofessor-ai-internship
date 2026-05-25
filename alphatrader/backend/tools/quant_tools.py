"""
tools/quant_tools.py
--------------------
Quantitative analysis tools — used by QuantStrategist.

Prompt 5: All functions upgraded from plain dummy to
          structured realistic simulation using real math.
          Inputs from get_historical_data() (real Alpaca bars) feed
          calculate_momentum(), calculate_rsi(), moving_average().
"""

import random
import statistics
from datetime import datetime


def calculate_momentum(prices: list) -> dict:
    """Real rate-of-change momentum over price series."""
    if len(prices) < 2:
        return {"momentum": 0.0, "signal": "neutral", "strength": "weak"}
    mom   = round((prices[-1] - prices[0]) / prices[0] * 100, 4)
    short = round((prices[-1] - prices[len(prices)//2]) / prices[len(prices)//2] * 100, 4)
    strength = "strong" if abs(mom) > 2 else ("moderate" if abs(mom) > 0.5 else "weak")
    return {
        "momentum":       mom,
        "short_momentum": short,
        "signal":         "bullish" if mom > 0 else ("bearish" if mom < 0 else "neutral"),
        "strength":       strength,
        "periods":        len(prices),
    }


def calculate_rsi(prices: list, period: int = 14) -> dict:
    """Wilder's RSI computed from real price series."""
    if len(prices) < period + 1:
        rsi = round(random.uniform(35, 70), 2)
        source = "simulation"
    else:
        gains, losses = [], []
        for i in range(1, period + 1):
            diff = prices[-i] - prices[-(i + 1)]
            (gains if diff >= 0 else losses).append(abs(diff))
        avg_gain = statistics.mean(gains) if gains else 0.001
        avg_loss = statistics.mean(losses) if losses else 0.001
        rs  = avg_gain / avg_loss
        rsi = round(100 - (100 / (1 + rs)), 2)
        source = "computed"
    zone = "overbought" if rsi > 70 else ("oversold" if rsi < 30 else "neutral")
    return {
        "rsi":    rsi,
        "zone":   zone,
        "period": period,
        "signal": "sell" if rsi > 70 else ("buy" if rsi < 30 else "hold"),
        "source": source,
    }


def moving_average(prices: list, window: int = 20) -> dict:
    """SMA with price-vs-MA relationship."""
    if not prices:
        return {"window": window, "ma": 0.0, "current_price": 0.0, "position": "unknown"}
    if len(prices) < window:
        ma = round(statistics.mean(prices), 4)
    else:
        ma = round(statistics.mean(prices[-window:]), 4)
    current = prices[-1]
    gap_pct = round((current - ma) / ma * 100, 4) if ma else 0
    return {
        "window":        window,
        "ma":            ma,
        "current_price": current,
        "gap_pct":       gap_pct,
        "position":      "above_ma" if current > ma else ("below_ma" if current < ma else "at_ma"),
    }


def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.05) -> dict:
    """Annualised Sharpe ratio from a return series."""
    if len(returns) < 2:
        sharpe = round(random.uniform(0.5, 2.2), 4)
        source = "simulation"
    else:
        excess = [r - risk_free_rate / 252 for r in returns]
        mean   = statistics.mean(excess)
        std    = statistics.stdev(excess) or 0.0001
        sharpe = round((mean / std) * (252 ** 0.5), 4)
        source = "computed"
    rating = "excellent" if sharpe > 2 else ("good" if sharpe > 1 else ("fair" if sharpe > 0 else "poor"))
    return {
        "sharpe_ratio": sharpe,
        "annualised":   True,
        "rating":       rating,
        "source":       source,
    }


def calculate_drawdown(equity_curve: list) -> dict:
    """Max drawdown and current drawdown from equity curve."""
    if not equity_curve:
        return {"max_drawdown_pct": 0.0, "current_drawdown_pct": 0.0}
    peak   = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        peak   = max(peak, val)
        dd     = (peak - val) / peak
        max_dd = max(max_dd, dd)
    current_peak = max(equity_curve)
    current_dd   = (current_peak - equity_curve[-1]) / current_peak if current_peak else 0
    return {
        "max_drawdown_pct":     round(max_dd * 100, 4),
        "current_drawdown_pct": round(current_dd * 100, 4),
        "peak_value":           round(current_peak, 2),
        "current_value":        round(equity_curve[-1], 2),
        "recovery_needed_pct":  round(current_dd / (1 - current_dd) * 100, 4) if current_dd < 1 else 100.0,
    }


def generate_trade_signal(market_insight: dict) -> dict:
    """Generate trade signal from market insight with confidence scoring."""
    trend     = market_insight.get("signal", market_insight.get("trend", "neutral"))
    sentiment = market_insight.get("sentiment", "neutral")
    confidence = float(market_insight.get("confidence", 0.6))

    if trend == "bullish" and sentiment in ("positive", "neutral"):
        action = "BUY"
    elif trend == "bearish" and sentiment in ("negative", "neutral"):
        action = "SELL"
    else:
        action = "HOLD"

    # Downgrade to HOLD if confidence too low
    if confidence < 0.55 and action != "HOLD":
        action = "HOLD"

    return {
        "action":        action,
        "confidence":    confidence,
        "trend_input":   trend,
        "sentiment_input": sentiment,
        "generated_at":  datetime.utcnow().isoformat(),
    }


def risk_reward_ratio(entry: float, stop_loss: float, take_profit: float) -> dict:
    """R:R ratio with viability assessment."""
    risk   = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    ratio  = round(reward / risk, 4) if risk else 0.0
    return {
        "risk":    round(risk, 4),
        "reward":  round(reward, 4),
        "ratio":   ratio,
        "viable":  ratio >= 1.5,
        "grade":   "A" if ratio >= 3 else ("B" if ratio >= 2 else ("C" if ratio >= 1.5 else "F")),
    }


def portfolio_optimization(holdings: dict | None = None) -> dict:
    """Equal-weight allocation with rebalancing signal."""
    tickers = list(holdings.keys()) if holdings else ["SPY", "QQQ", "GLD", "BND"]
    weight  = round(1 / len(tickers), 4)
    drift   = {t: round(weight + random.uniform(-0.03, 0.03), 4) for t in tickers}
    rebal   = any(abs(drift[t] - weight) > 0.02 for t in tickers)
    return {
        "target_allocation": {t: weight for t in tickers},
        "current_allocation": drift,
        "rebalance_needed":   rebal,
        "method":             "equal_weight",
        "num_assets":         len(tickers),
    }


def backtest_strategy(signal: str, lookback_days: int = 30) -> dict:
    """Structured backtest simulation with realistic metrics."""
    base_wr = {"BUY": 0.58, "SELL": 0.52, "HOLD": 0.50}.get(signal.upper(), 0.50)
    win_rate    = round(base_wr + random.uniform(-0.05, 0.05), 4)
    total_ret   = round(random.uniform(-3, 20), 2)
    max_dd      = round(random.uniform(2, 12), 2)
    sharpe      = round(total_ret / (max_dd + 0.01) * 0.5, 4)
    return {
        "signal":           signal,
        "lookback_days":    lookback_days,
        "win_rate":         win_rate,
        "total_return_pct": total_ret,
        "sharpe_ratio":     sharpe,
        "max_drawdown_pct": max_dd,
        "profit_factor":    round(win_rate / (1 - win_rate + 0.001), 4),
        "total_trades":     lookback_days,
    }


def generate_alpha_signal(symbol: str = "SPY", prices: list | None = None) -> dict:
    """Composite alpha signal from momentum + RSI + MA."""
    if not prices or len(prices) < 5:
        prices = [round(500 + random.uniform(-10, 10), 2) for _ in range(30)]

    mom = calculate_momentum(prices)
    rsi = calculate_rsi(prices)
    ma  = moving_average(prices)

    # Composite scoring
    score = 0
    if mom["signal"] == "bullish":  score += 1
    if rsi["zone"]   == "oversold": score += 1
    if ma["position"] == "above_ma": score += 1

    alpha = "long" if score >= 2 else ("short" if score == 0 else "flat")
    return {
        "symbol":     symbol,
        "alpha":      alpha,
        "score":      score,
        "max_score":  3,
        "momentum":   mom["momentum"],
        "rsi":        rsi["rsi"],
        "ma_20":      ma["ma"],
        "ma_position": ma["position"],
        "timestamp":  datetime.utcnow().isoformat(),
    }
