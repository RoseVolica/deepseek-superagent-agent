"""Simple educational MA backtest; stock.csv contains synthetic prices."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SHORT_WINDOW = 5
LONG_WINDOW = 20
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
    return frame


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
    summary = f"MA windows: {SHORT_WINDOW}/{LONG_WINDOW}\nTotal return: {total_return:.4%}\n"
    (args.output / "summary.txt").write_text(summary, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(result["date"], result["nav"], label="MA strategy", color="#2E6BAF")
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
