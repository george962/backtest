from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest_core import (
    BUY_HOLD_NAME,
    add_features,
    build_strategy_set,
    format_pct,
    load_config,
    load_price_data,
    performance_metrics,
    run_train_then_test_walk_forward,
    summarize_window_grid,
    trade_stats_for_strategy,
    write_run_config,
)


DEFAULT_CONFIG = "configs/spy_pullback.yaml"


def print_summary_table(title: str, results_df: pd.DataFrame, max_rows: int = 15) -> None:
    print()
    print(title)
    print()

    if results_df.empty:
        print("No strategies passed the filters.")
        return

    print(
        f"{'Rank':<5} "
        f"{'Strategy':<74} "
        f"{'AvgEdge':>9} "
        f"{'MedEdge':>9} "
        f"{'PosWin':>8} "
        f"{'WorstEdge':>11} "
        f"{'AvgAnn':>9} "
        f"{'WorstDD':>9} "
        f"{'AvgExp':>8}"
    )
    print("-" * 160)

    for rank, (_, row) in enumerate(results_df.head(max_rows).iterrows(), start=1):
        print(
            f"{rank:<5} "
            f"{row['Strategy']:<74} "
            f"{format_pct(row['Avg Test Edge']):>9} "
            f"{format_pct(row['Median Test Edge']):>9} "
            f"{int(row['Positive Windows']):>2}/{int(row['Total Windows']):<5} "
            f"{format_pct(row['Worst Test Edge']):>11} "
            f"{format_pct(row['Avg Annual Return']):>9} "
            f"{format_pct(row['Worst Max Drawdown']):>9} "
            f"{row['Avg Exposure']:>8.2f}"
        )


def print_walk_forward_table(walk_forward_df: pd.DataFrame) -> None:
    print()
    print("=== TRAIN-THEN-TEST WALK-FORWARD RESULTS ===")
    print("Each row selects parameters using only data available through TrainEnd.")
    print()
    print(
        f"{'Window':<11} "
        f"{'Selected Strategy':<68} "
        f"{'TrainEdge':>10} "
        f"{'TestEdge':>9} "
        f"{'TestAnn':>9} "
        f"{'TestDD':>9} "
        f"{'AvgExp':>8}"
    )
    print("-" * 138)

    for _, row in walk_forward_df.iterrows():
        print(
            f"{row['Window']:<11} "
            f"{row['Selected Strategy']:<68} "
            f"{format_pct(row['Train Edge']):>10} "
            f"{format_pct(row['Test True Edge']):>9} "
            f"{format_pct(row['Test Annual Return']):>9} "
            f"{format_pct(row['Test Max Drawdown']):>9} "
            f"{row['Test Avg Exposure']:>8.2f}"
        )


def print_trade_stats(
    data: pd.DataFrame,
    strategy_name: str,
    strategy_returns: dict,
    strategy_positions: dict,
    strategy_info: dict,
    config: dict,
) -> pd.DataFrame:
    periods = config.get(
        "trade_stat_periods",
        [
            ("1993-01-01", "2026-12-31"),
            ("2003-01-01", "2026-12-31"),
            ("2017-01-01", "2021-12-31"),
            ("2022-01-01", "2026-12-31"),
        ],
    )
    rows = [
        trade_stats_for_strategy(
            data,
            strategy_name,
            start,
            end,
            strategy_returns,
            strategy_positions,
            strategy_info,
            config,
        )
        for start, end in periods
    ]
    stats_df = pd.DataFrame(rows)

    print()
    print("=== BOOST TRADE STATS ===")
    print(f"Strategy: {strategy_name}")
    print("These stats measure the incremental overlay from boost periods, not the whole SPY position.")
    print()
    print(
        f"{'Period':<12} "
        f"{'Trades':>7} "
        f"{'WinRate':>9} "
        f"{'AvgTrade':>10} "
        f"{'MedTrade':>10} "
        f"{'Best':>10} "
        f"{'Worst':>10} "
        f"{'LoseStk':>8} "
        f"{'TotalOverlay':>13}"
    )
    print("-" * 105)

    for _, row in stats_df.iterrows():
        print(
            f"{row['Period']:<12} "
            f"{int(row['Trades']):>7} "
            f"{format_pct(row['Win Rate']):>9} "
            f"{format_pct(row['Avg Trade']):>10} "
            f"{format_pct(row['Median Trade']):>10} "
            f"{format_pct(row['Best Trade']):>10} "
            f"{format_pct(row['Worst Trade']):>10} "
            f"{int(row['Losing Streak']):>8} "
            f"{format_pct(row['Total Overlay Return']):>13}"
        )

    return stats_df


def run_backtest(args: argparse.Namespace) -> Path:
    config_path = Path(args.config)
    config = load_config(config_path)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{run_stamp}_{config['ticker'].lower()}_pullback"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(config_path, run_dir)

    data = add_features(load_price_data(config, refresh=args.refresh_data))
    strategy_returns, strategy_positions, strategy_info = build_strategy_set(data, config)

    fixed_df, summary_df, practical_df = summarize_window_grid(
        data,
        config,
        strategy_returns,
        strategy_positions,
    )
    walk_forward_df, oos_returns, oos_positions, oos_benchmark_returns = (
        run_train_then_test_walk_forward(
            data,
            config,
            strategy_returns,
            strategy_positions,
        )
    )

    oos_metrics = performance_metrics(
        data.loc[oos_returns.index],
        oos_returns,
        oos_positions,
        "Train-selected out-of-sample",
        config["costs"]["margin_rate_annual"],
    )

    selected_static = practical_df.iloc[0]["Strategy"] if not practical_df.empty else summary_df.iloc[0]["Strategy"]
    trade_stats_df = print_trade_stats(
        data,
        selected_static,
        strategy_returns,
        strategy_positions,
        strategy_info,
        config,
    )

    print()
    print("SPY PULLBACK OVERLAY - REPRODUCIBLE RESEARCH RUN")
    print(f"Ticker: {config['ticker']}")
    print(f"Trading cost: {config['costs']['trading_bps_per_side']} bps per side")
    print(f"Margin/financing cost: {config['costs']['margin_rate_annual'] * 100:.2f}% annual")
    print(f"Run directory: {run_dir}")

    print_walk_forward_table(walk_forward_df)
    print()
    print("Out-of-sample stitched result:")
    print(f"Annual return: {format_pct(oos_metrics['Annual Return'])}")
    print(f"Matched benchmark annual return: {format_pct(oos_metrics['Matched B&H Ann Ret'])}")
    print(f"True edge: {format_pct(oos_metrics['True Edge'])}")
    print(f"Sharpe: {oos_metrics['Sharpe']:.2f}")
    print(f"Max drawdown: {format_pct(oos_metrics['Max Drawdown'])}")
    print(f"Average exposure: {oos_metrics['Avg Exposure']:.2f}x")

    print_summary_table(
        "=== STATIC GRID DIAGNOSTIC: BEST TEST-WINDOW CANDIDATES ===",
        summary_df,
        max_rows=15,
    )
    print_summary_table(
        "=== STATIC GRID DIAGNOSTIC: PRACTICAL CANDIDATES ===",
        practical_df,
        max_rows=15,
    )

    from report_generation import generate_report

    generate_report(
        output_dir=run_dir,
        selected_strategy=selected_static,
        strategy_returns=strategy_returns,
        strategy_positions=strategy_positions,
        fixed_df=fixed_df,
        summary_df=summary_df,
        data=data,
        margin_rate_annual=config["costs"]["margin_rate_annual"],
        walk_forward_df=walk_forward_df,
        oos_returns=oos_returns,
        oos_positions=oos_positions,
        oos_benchmark_returns=oos_benchmark_returns,
        trade_stats_df=trade_stats_df,
    )

    print()
    print(f"Saved charts, tables, and config to {run_dir}")
    return run_dir


def signal_check(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    data = add_features(load_price_data(config, refresh=args.refresh_data))
    strategy_returns, strategy_positions, strategy_info = build_strategy_set(data, config)
    fixed_df, summary_df, practical_df = summarize_window_grid(
        data,
        config,
        strategy_returns,
        strategy_positions,
    )
    selected = practical_df.iloc[0]["Strategy"] if not practical_df.empty else summary_df.iloc[0]["Strategy"]
    selected_info = strategy_info[selected]
    latest_date = data.index[-1]
    latest = data.iloc[-1]

    print()
    print("=== CURRENT SIGNAL CHECK ===")
    print(f"Latest date: {latest_date.date()}")
    print(f"Close: {latest['Close']:.2f}")
    print(f"20DMA: {latest['SMA_20']:.2f}")
    print(f"200DMA: {latest['SMA_200']:.2f}")
    print(f"5D Return: {format_pct(latest['Return_5D'])}")
    print(f"Above 200DMA: {bool(latest['Above_200DMA'])}")
    print(f"Below 20DMA: {bool(latest['Below_20DMA'])}")
    print()
    print(f"Selected practical strategy: {selected}")
    print(f"Base exposure: {float(selected_info['Base Position'].iloc[-1]):.2f}")
    print(f"Active boost: {bool(selected_info['Active Signal'].iloc[-1])}")
    print(f"Boost days left: {int(selected_info['Days Remaining'].iloc[-1])}")
    print(f"Desired exposure: {float(strategy_positions[selected].iloc[-1]):.2f}")
    print(f"Effective exposure today: {float(strategy_positions[selected].shift(1).fillna(0).iloc[-1]):.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible SPY pullback-overlay backtest.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the full backtest and write artifacts.")
    run_parser.add_argument("--config", default=DEFAULT_CONFIG)
    run_parser.add_argument("--output-dir", default="runs")
    run_parser.add_argument("--refresh-data", action="store_true")
    run_parser.set_defaults(func=run_backtest)

    signal_parser = subparsers.add_parser("signal", help="Show the latest signal and exposure state.")
    signal_parser.add_argument("--config", default=DEFAULT_CONFIG)
    signal_parser.add_argument("--refresh-data", action="store_true")
    signal_parser.set_defaults(func=signal_check)

    parser.add_argument("--config", default=DEFAULT_CONFIG, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default="runs", help=argparse.SUPPRESS)
    parser.add_argument("--refresh-data", action="store_true", help=argparse.SUPPRESS)
    parser.set_defaults(func=run_backtest)
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    parsed_args.func(parsed_args)
