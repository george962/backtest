from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt

from backtest_core import matched_benchmark_returns


def _equity_from_returns(returns):
    return (1 + returns.fillna(0)).cumprod()


def generate_report(
    output_dir,
    selected_strategy,
    strategy_returns,
    strategy_positions,
    fixed_df,
    summary_df,
    data,
    margin_rate_annual,
    walk_forward_df,
    oos_returns,
    oos_positions,
    oos_benchmark_returns,
    trade_stats_df,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_returns = strategy_returns[selected_strategy].fillna(0)
    selected_equity = _equity_from_returns(selected_returns)
    avg_exposure = strategy_positions[selected_strategy].mean()
    matched_bh_equity = _equity_from_returns(
        matched_benchmark_returns(data, avg_exposure, margin_rate_annual)
    )

    plt.figure(figsize=(11, 5))
    plt.plot(selected_equity.index, selected_equity.values, label=f"Static diagnostic: {selected_strategy}", linewidth=1.4)
    plt.plot(
        matched_bh_equity.index,
        matched_bh_equity.values,
        label=f"Matched-exposure benchmark ({avg_exposure:.2f}x)",
        linewidth=1.2,
        linestyle="--",
    )
    plt.yscale("log")
    plt.title("Static Diagnostic Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1, log scale")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "static_equity_curve.png", dpi=150)
    plt.close()

    oos_equity = _equity_from_returns(oos_returns)
    oos_benchmark_equity = _equity_from_returns(oos_benchmark_returns)
    plt.figure(figsize=(11, 5))
    plt.plot(oos_equity.index, oos_equity.values, label="Train-selected strategy", linewidth=1.5)
    plt.plot(oos_benchmark_equity.index, oos_benchmark_equity.values, label="Window-matched benchmark", linewidth=1.2, linestyle="--")
    plt.yscale("log")
    plt.title("Out-of-Sample Walk-Forward Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1, log scale")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "oos_equity_curve.png", dpi=150)
    plt.close()

    oos_drawdown = oos_equity / oos_equity.cummax() - 1
    plt.figure(figsize=(11, 3.5))
    plt.fill_between(oos_drawdown.index, oos_drawdown.values * 100, 0, color="firebrick", alpha=0.5)
    plt.title("Out-of-Sample Walk-Forward Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "oos_drawdown_chart.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    colors_bar = ["seagreen" if v > 0 else "firebrick" for v in walk_forward_df["Test True Edge"]]
    plt.bar(walk_forward_df["Window"], walk_forward_df["Test True Edge"] * 100, color=colors_bar)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Train-Selected True Edge by Test Window")
    plt.ylabel("Edge over matched benchmark (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "oos_edge_by_window.png", dpi=150)
    plt.close()

    summary_df.to_csv(output_dir / "static_summary_table.csv", index=False)
    fixed_df.to_csv(output_dir / "static_window_grid.csv", index=False)
    walk_forward_df.to_csv(output_dir / "walk_forward_selected.csv", index=False)
    trade_stats_df.to_csv(output_dir / "trade_stats.csv", index=False)

    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Backtest Run Summary\n\n")
        f.write("## Train-Then-Test Walk-Forward\n\n")
        f.write(walk_forward_df.to_markdown(index=False))
        f.write("\n\n## Static Grid Diagnostic: Top 20\n\n")
        f.write(summary_df.head(20).to_markdown(index=False))
