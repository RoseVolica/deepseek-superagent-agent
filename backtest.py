"""Simple educational MA backtest; stock.csv contains synthetic prices."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SHORT_WINDOW = 10
LONG_WINDOW = 30
INITIAL_CASH = 100_000.0


def backtest(data):
    """Use yesterday's close-based signal at today's open; fractional shares."""
    frame = data.copy()
    frame["ma_short"] = frame["close"].rolling(SHORT_WINDOW).mean()
    frame["ma_long"] = frame["close"].rolling(LONG_WINDOW).mean()
    frame["signal"] = (frame["ma_short"] > frame["ma_long"]).astype(int)
    cash, shares = INITIAL_CASH, 0.0
    equities, positions = [], []
    for i in range(len(frame)):
        row = frame.iloc[i]
        target = int(frame.iloc[i - 1]["signal"]) if i else 0
        if target == 1 and shares == 0:
            shares = cash / row["open"]
            cash = 0.0
        elif target == 0 and shares > 0:
            cash = shares * row["open"]
            shares = 0.0
        equities.append(cash + shares * row["close"])
        positions.append(shares)
    frame["shares"] = positions
    frame["equity"] = equities
    frame["nav"] = frame["equity"] / INITIAL_CASH

    # Buy-and-hold benchmark: buy at first day's open, hold, daily close valuation
    first_open = frame.iloc[0]["open"]
    bh_shares = INITIAL_CASH / first_open
    frame["nav_bh"] = bh_shares * frame["close"] / INITIAL_CASH

    return frame


def compute_max_drawdown(nav_series):
    """Maximum drawdown as a positive percentage, with initial NAV 1 in the peak history."""
    peak = 1.0
    max_dd = 0.0
    for nav in nav_series:
        if nav > peak:
            peak = nav
        dd = (peak - nav) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE / "stock.csv")
    parser.add_argument("--output", type=Path, default=BASE / "output")
    args = parser.parse_args()
    data = pd.read_csv(args.input, parse_dates=["date"])
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(data.columns) or data.empty:
        raise ValueError("CSV must have date,open,high,low,close,volume and rows")
    data = data.sort_values("date").reset_index(drop=True)
    prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if data["date"].isna().any() or data["date"].duplicated().any():
        raise ValueError("Dates must be valid and unique")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Prices must be positive and finite")
    result = backtest(data)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "backtest_result.csv", index=False)
    total_return = result["nav"].iloc[-1] - 1
    max_dd = compute_max_drawdown(result["nav"])
    bh_return = result["nav_bh"].iloc[-1] - 1
    bh_max_dd = compute_max_drawdown(result["nav_bh"])
    summary_lines = [
        f"MA windows: {SHORT_WINDOW}/{LONG_WINDOW}",
        f"Strategy total return: {total_return:.4%}",
        f"Strategy max drawdown: {max_dd:.4%}",
        f"Buy-and-hold total return: {bh_return:.4%}",
        f"Buy-and-hold max drawdown: {bh_max_dd:.4%}",
    ]
    summary = "\n".join(summary_lines) + "\n"
    (args.output / "summary.txt").write_text(summary, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(result["date"], result["nav"], label="MA strategy", color="#2E6BAF")
    ax.plot(result["date"], result["nav_bh"], label="Buy & hold", color="#E0533D")
    ax.set(title="Synthetic prices - MA backtest", xlabel="Date", ylabel="NAV (initial capital = 1)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output / "equity_curve.png", dpi=160)
    plt.close(fig)
    print(summary, end="")
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()