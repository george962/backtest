"""
report_generation.py

Drop this file in the same folder as backtest.py.
It generates charts and exports result tables without you needing to
paste any code into your main script — just import and call one function.
"""

import os
import matplotlib.pyplot as plt


def generate_report(
    selected_strategy,
    strategy_returns,
    strategy_positions,
    fixed_df,
    summary_df,
    df,
    margin_rate_annual,
    output_dir="outputs",
):
    """
    Generates and saves:
      - equity_curve.png   (strategy vs. matched-exposure benchmark, log scale)
      - drawdown_chart.png (drawdown over time for the selected strategy)
      - edge_by_window.png (bar chart of True Edge per walk-forward window)
      - summary_table.csv / summary_table.md (ranked strategy summary)

    Call this once, at the very end of backtest.py, after everything else
    has run — see the two-line snippet in README.md for exactly where.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- Equity curve: selected strategy vs. matched-exposure benchmark ----
    selected_returns = strategy_returns[selected_strategy].fillna(0)
    selected_equity = (1 + selected_returns).cumprod()

    avg_exposure = strategy_positions[selected_strategy].mean()
    matched_margin_exposure = max(avg_exposure - 1.0, 0)
    matched_bh_returns = (
        avg_exposure * df["Daily_Return"].fillna(0)
        - matched_margin_exposure * (margin_rate_annual / 252)
    )
    matched_bh_equity = (1 + matched_bh_returns).cumprod()

    plt.figure(figsize=(11, 5))
    plt.plot(selected_equity.index, selected_equity.values,
              label=f"Strategy: {selected_strategy}", linewidth=1.4)
    plt.plot(matched_bh_equity.index, matched_bh_equity.values,
              label=f"Matched-Exposure Benchmark ({avg_exposure:.2f}x)",
              linewidth=1.2, linestyle="--")
    plt.yscale("log")
    plt.title("Equity Curve — Strategy vs. Matched-Exposure Benchmark (log scale)")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1 (log scale)")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "equity_curve.png"), dpi=150)
    plt.close()

    # ---- Drawdown chart ----
    rolling_max = selected_equity.cummax()
    drawdown = selected_equity / rolling_max - 1

    plt.figure(figsize=(11, 3.5))
    plt.fill_between(drawdown.index, drawdown.values * 100, 0,
                      color="firebrick", alpha=0.5)
    plt.title(f"Drawdown — {selected_strategy}")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drawdown_chart.png"), dpi=150)
    plt.close()

    # ---- Walk-forward edge by window (bar chart) ----
    window_edges = fixed_df[fixed_df["Strategy"] == selected_strategy][["Window", "True Edge"]]

    plt.figure(figsize=(9, 4))
    colors_bar = ["seagreen" if v > 0 else "firebrick" for v in window_edges["True Edge"]]
    plt.bar(window_edges["Window"], window_edges["True Edge"] * 100, color=colors_bar)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"True Edge by Walk-Forward Window — {selected_strategy}")
    plt.ylabel("Edge over matched benchmark (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "edge_by_window.png"), dpi=150)
    plt.close()

    # ---- Export summary table as CSV and Markdown ----
    summary_df.to_csv(os.path.join(output_dir, "summary_table.csv"), index=False)

    with open(os.path.join(output_dir, "summary_table.md"), "w") as f:
        f.write("# Strategy Summary — Walk-Forward Results\n\n")
        f.write(summary_df.head(20).to_markdown(index=False))

    print()
    print(f"Saved charts and tables to ./{output_dir}/:")
    print(" - equity_curve.png")
    print(" - drawdown_chart.png")
    print(" - edge_by_window.png")
    print(" - summary_table.csv / summary_table.md")
