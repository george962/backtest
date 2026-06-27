from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


BUY_HOLD_NAME = "Buy and Hold SPY"
TRADING_DAYS = 252


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config.setdefault("costs", {})
    config["costs"].setdefault("trading_bps_per_side", 3.0)
    config["costs"].setdefault("margin_rate_annual", 0.06)
    config["costs"].setdefault("max_allowed_exposure", 2.0)

    config.setdefault("selection", {})
    config["selection"].setdefault("max_avg_exposure", 1.10)
    config["selection"].setdefault("max_drawdown_floor", -0.40)
    config["selection"].setdefault("min_train_edge", 0.0)

    return config


def write_run_config(config_path: str | Path, run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")


def load_price_data(config: dict, refresh: bool = False) -> pd.DataFrame:
    ticker = config["ticker"]
    start_date = config["start_date"]
    cache_path = Path(config.get("data_cache", f"data/{ticker}.parquet"))

    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    data = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check network access or the ticker.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data = data.dropna().copy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(cache_path)
    return data


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["Daily_Return"] = df["Close"].pct_change()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["Above_200DMA"] = df["Close"] > df["SMA_200"]
    df["Below_200DMA"] = df["Close"] < df["SMA_200"]
    df["Above_20DMA"] = df["Close"] > df["SMA_20"]
    df["Below_20DMA"] = df["Close"] < df["SMA_20"]
    df["Return_5D"] = df["Close"] / df["Close"].shift(5) - 1
    return df.dropna().copy()


def slice_period(data: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return data.loc[(data.index >= start) & (data.index <= end)].copy()


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:,.2f}%"


def max_drawdown(equity: pd.Series) -> float:
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    return float(drawdown.min())


def annual_return_from_returns(strategy_returns: pd.Series) -> float:
    strategy_returns = strategy_returns.dropna()
    if len(strategy_returns) == 0:
        return np.nan

    equity = (1 + strategy_returns).cumprod()
    total_return = equity.iloc[-1] - 1
    years = len(strategy_returns) / TRADING_DAYS
    if years <= 0:
        return np.nan

    return float((1 + total_return) ** (1 / years) - 1)


def sharpe_from_returns(strategy_returns: pd.Series) -> float:
    strategy_returns = strategy_returns.dropna()
    if len(strategy_returns) == 0 or strategy_returns.std() == 0:
        return np.nan
    return float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(TRADING_DAYS))


def build_active_signal(
    raw_signal: pd.Series,
    hold_days: int,
    exit_condition: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    raw = raw_signal.fillna(False).astype(bool).to_numpy()
    if exit_condition is None:
        exits = np.zeros(len(raw), dtype=bool)
    else:
        exits = exit_condition.fillna(False).astype(bool).to_numpy()

    active = np.zeros(len(raw), dtype=bool)
    days_remaining = np.zeros(len(raw), dtype=int)
    days_left = 0

    for i, current_signal in enumerate(raw):
        if days_left > 0:
            if exits[i]:
                days_left = 0
            else:
                active[i] = True
                days_remaining[i] = days_left
                days_left -= 1
        elif current_signal:
            active[i] = True
            days_remaining[i] = hold_days
            days_left = hold_days - 1

    return (
        pd.Series(active, index=raw_signal.index),
        pd.Series(days_remaining, index=raw_signal.index),
    )


def apply_costs(
    position: pd.Series,
    returns: pd.Series,
    cost_bps_per_side: float,
    margin_rate_annual: float,
    max_allowed_exposure: float = 2.0,
) -> pd.Series:
    desired_position = position.astype(float).clip(lower=0, upper=max_allowed_exposure)
    effective_position = desired_position.shift(1).fillna(0)

    turnover = effective_position.diff().abs().fillna(0)
    trading_cost = turnover * (cost_bps_per_side / 10000)

    margin_exposure = (effective_position - 1.0).clip(lower=0)
    margin_cost = margin_exposure * (margin_rate_annual / TRADING_DAYS)

    return effective_position * returns.fillna(0) - trading_cost - margin_cost


def count_exposure_changes(position: pd.Series) -> int:
    changes = position.diff().abs().fillna(0) > 0
    return int(changes.sum())


def matched_benchmark_returns(
    data: pd.DataFrame,
    avg_exposure: float,
    margin_rate_annual: float,
) -> pd.Series:
    matched_margin_exposure = max(avg_exposure - 1.0, 0)
    return (
        avg_exposure * data["Daily_Return"].fillna(0)
        - matched_margin_exposure * (margin_rate_annual / TRADING_DAYS)
    )


def performance_metrics(
    data: pd.DataFrame,
    strategy_returns: pd.Series,
    position: pd.Series,
    name: str,
    margin_rate_annual: float,
) -> dict:
    strategy_returns = strategy_returns.loc[data.index].fillna(0)
    position = position.loc[data.index].fillna(0)
    equity = (1 + strategy_returns).cumprod()

    annual_return = annual_return_from_returns(strategy_returns)
    sharpe = sharpe_from_returns(strategy_returns)
    mdd = max_drawdown(equity)
    avg_exposure = float(position.mean())
    max_exposure = float(position.max())

    matched_returns = matched_benchmark_returns(data, avg_exposure, margin_rate_annual)
    matched_return = annual_return_from_returns(matched_returns)
    true_edge = annual_return - matched_return

    return {
        "Strategy": name,
        "Annual Return": annual_return,
        "Matched B&H Ann Ret": matched_return,
        "True Edge": true_edge,
        "Sharpe": sharpe,
        "Max Drawdown": mdd,
        "Avg Exposure": avg_exposure,
        "Max Exposure": max_exposure,
        "Exposure Changes": count_exposure_changes(position),
    }


def build_strategy_set(data: pd.DataFrame, config: dict) -> tuple[dict, dict, dict]:
    strategy_returns = {}
    strategy_positions = {}
    strategy_info = {}

    returns = data["Daily_Return"].fillna(0)
    costs = config["costs"]
    grid = config["strategy_grid"]

    buyhold_position = pd.Series(1.0, index=data.index)
    strategy_positions[BUY_HOLD_NAME] = buyhold_position
    strategy_returns[BUY_HOLD_NAME] = apply_costs(
        buyhold_position,
        returns,
        cost_bps_per_side=0.0,
        margin_rate_annual=0.0,
        max_allowed_exposure=costs["max_allowed_exposure"],
    )
    strategy_info[BUY_HOLD_NAME] = {
        "Base Position": buyhold_position,
        "Active Signal": pd.Series(False, index=data.index),
        "Days Remaining": pd.Series(0, index=data.index),
        "Raw Signal": pd.Series(False, index=data.index),
    }

    for drop_threshold in grid["drop_thresholds"]:
        raw_signal = (
            data["Above_200DMA"]
            & data["Below_20DMA"]
            & (data["Return_5D"] < drop_threshold)
        )
        threshold_label = f"5D<{drop_threshold:.1%}"

        for hold_days in grid["hold_days"]:
            for exit_policy in grid["exit_policies"]:
                if exit_policy == "Fixed":
                    exit_condition = pd.Series(False, index=data.index)
                elif exit_policy == "StopBelow200":
                    exit_condition = data["Below_200DMA"]
                elif exit_policy == "Recover20":
                    exit_condition = data["Above_20DMA"]
                elif exit_policy == "StopBelow200+Recover20":
                    exit_condition = data["Below_200DMA"] | data["Above_20DMA"]
                else:
                    raise ValueError(f"Unknown exit policy: {exit_policy}")

                active_signal, days_remaining = build_active_signal(
                    raw_signal,
                    hold_days,
                    exit_condition=exit_condition,
                )

                for boost_exposure in grid["boost_exposures"]:
                    for mode in grid["modes"]:
                        if mode == "Core":
                            base_position = pd.Series(1.0, index=data.index)
                        elif mode == "Def":
                            base_position = pd.Series(0.5, index=data.index)
                            base_position.loc[data["Above_200DMA"]] = 1.0
                        else:
                            raise ValueError(f"Unknown mode: {mode}")

                        position = base_position.copy()
                        position.loc[active_signal] = boost_exposure
                        strategy_name = (
                            f"{mode}: Above200+Below20+{threshold_label}, "
                            f"{boost_exposure:.2f}x, {hold_days}d, {exit_policy}"
                        )

                        strategy_positions[strategy_name] = position
                        strategy_returns[strategy_name] = apply_costs(
                            position,
                            returns,
                            costs["trading_bps_per_side"],
                            costs["margin_rate_annual"],
                            costs["max_allowed_exposure"],
                        )
                        strategy_info[strategy_name] = {
                            "Base Position": base_position,
                            "Active Signal": active_signal,
                            "Days Remaining": days_remaining,
                            "Raw Signal": raw_signal,
                            "Drop Threshold": drop_threshold,
                            "Hold Days": hold_days,
                            "Boost Exposure": boost_exposure,
                            "Mode": mode,
                            "Exit Policy": exit_policy,
                        }

    return strategy_returns, strategy_positions, strategy_info


def evaluate_all_strategies(
    data: pd.DataFrame,
    strategy_returns: dict,
    strategy_positions: dict,
    start: str,
    end: str,
    margin_rate_annual: float,
) -> pd.DataFrame:
    period_df = slice_period(data, start, end)
    rows = []

    for strategy_name in strategy_returns:
        rows.append(
            performance_metrics(
                period_df,
                strategy_returns[strategy_name],
                strategy_positions[strategy_name],
                strategy_name,
                margin_rate_annual,
            )
        )

    return pd.DataFrame(rows)


def summarize_window_grid(
    data: pd.DataFrame,
    config: dict,
    strategy_returns: dict,
    strategy_positions: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed_rows = []
    margin_rate = config["costs"]["margin_rate_annual"]

    for window in config["walk_forward_windows"]:
        label = f"{window['TestStart'][:4]}-{window['TestEnd'][:4]}"
        results = evaluate_all_strategies(
            data,
            strategy_returns,
            strategy_positions,
            window["TestStart"],
            window["TestEnd"],
            margin_rate,
        )
        bh_return = results.loc[results["Strategy"] == BUY_HOLD_NAME, "Annual Return"].iloc[0]
        non_bh = results[results["Strategy"] != BUY_HOLD_NAME].copy()

        for _, row in non_bh.iterrows():
            fixed_rows.append(
                {
                    "Strategy": row["Strategy"],
                    "Window": label,
                    "Annual Return": row["Annual Return"],
                    "B&H Annual Return": bh_return,
                    "True Edge": row["True Edge"],
                    "Max Drawdown": row["Max Drawdown"],
                    "Sharpe": row["Sharpe"],
                    "Avg Exposure": row["Avg Exposure"],
                    "Max Exposure": row["Max Exposure"],
                }
            )

    fixed_df = pd.DataFrame(fixed_rows)
    summary_rows = []

    for strategy_name, group in fixed_df.groupby("Strategy"):
        summary_rows.append(
            {
                "Strategy": strategy_name,
                "Avg Test Edge": group["True Edge"].mean(),
                "Median Test Edge": group["True Edge"].median(),
                "Positive Windows": int((group["True Edge"] > 0).sum()),
                "Total Windows": len(group),
                "Worst Test Edge": group["True Edge"].min(),
                "Avg Annual Return": group["Annual Return"].mean(),
                "Worst Max Drawdown": group["Max Drawdown"].min(),
                "Avg Exposure": group["Avg Exposure"].mean(),
                "Max Exposure": group["Max Exposure"].max(),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["Positive Windows", "Avg Test Edge", "Worst Test Edge"],
        ascending=False,
    )

    selection = config["selection"]
    practical_df = summary_df[
        (summary_df["Worst Max Drawdown"] > selection["max_drawdown_floor"])
        & (summary_df["Avg Exposure"] <= selection["max_avg_exposure"])
    ].copy()
    practical_df = practical_df.sort_values(
        by=["Positive Windows", "Avg Test Edge", "Worst Test Edge"],
        ascending=False,
    )

    return fixed_df, summary_df, practical_df


def select_strategy(train_results: pd.DataFrame, config: dict) -> str:
    selection = config["selection"]
    candidates = train_results[train_results["Strategy"] != BUY_HOLD_NAME].copy()
    filtered = candidates[
        (candidates["True Edge"] >= selection["min_train_edge"])
        & (candidates["Max Drawdown"] > selection["max_drawdown_floor"])
        & (candidates["Avg Exposure"] <= selection["max_avg_exposure"])
    ].copy()

    if filtered.empty:
        filtered = candidates.copy()

    filtered = filtered.sort_values(
        by=["True Edge", "Sharpe", "Max Drawdown"],
        ascending=[False, False, False],
    )
    return str(filtered.iloc[0]["Strategy"])


def run_train_then_test_walk_forward(
    data: pd.DataFrame,
    config: dict,
    strategy_returns: dict,
    strategy_positions: dict,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    rows = []
    oos_returns = []
    oos_positions = []
    oos_benchmark_returns = []
    margin_rate = config["costs"]["margin_rate_annual"]

    for window in config["walk_forward_windows"]:
        label = f"{window['TestStart'][:4]}-{window['TestEnd'][:4]}"
        train_results = evaluate_all_strategies(
            data,
            strategy_returns,
            strategy_positions,
            config["start_date"],
            window["TrainEnd"],
            margin_rate,
        )
        selected = select_strategy(train_results, config)
        test_results = evaluate_all_strategies(
            data,
            strategy_returns,
            strategy_positions,
            window["TestStart"],
            window["TestEnd"],
            margin_rate,
        )
        train_row = train_results.loc[train_results["Strategy"] == selected].iloc[0]
        test_row = test_results.loc[test_results["Strategy"] == selected].iloc[0]
        test_df = slice_period(data, window["TestStart"], window["TestEnd"])
        test_position = strategy_positions[selected].loc[test_df.index]
        test_returns = strategy_returns[selected].loc[test_df.index]
        test_benchmark = matched_benchmark_returns(
            test_df,
            test_position.mean(),
            margin_rate,
        )

        oos_returns.append(test_returns)
        oos_positions.append(test_position)
        oos_benchmark_returns.append(test_benchmark)
        rows.append(
            {
                "Window": label,
                "Train End": window["TrainEnd"],
                "Test Start": window["TestStart"],
                "Test End": window["TestEnd"],
                "Selected Strategy": selected,
                "Train Edge": train_row["True Edge"],
                "Train Sharpe": train_row["Sharpe"],
                "Train Max Drawdown": train_row["Max Drawdown"],
                "Test Annual Return": test_row["Annual Return"],
                "Test Matched B&H Ann Ret": test_row["Matched B&H Ann Ret"],
                "Test True Edge": test_row["True Edge"],
                "Test Sharpe": test_row["Sharpe"],
                "Test Max Drawdown": test_row["Max Drawdown"],
                "Test Avg Exposure": test_row["Avg Exposure"],
                "Test Max Exposure": test_row["Max Exposure"],
            }
        )

    return (
        pd.DataFrame(rows),
        pd.concat(oos_returns).sort_index(),
        pd.concat(oos_positions).sort_index(),
        pd.concat(oos_benchmark_returns).sort_index(),
    )


def longest_losing_streak(trade_returns: pd.Series) -> int:
    streak = 0
    max_streak = 0

    for r in trade_returns:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return max_streak


def trade_stats_for_strategy(
    data: pd.DataFrame,
    strategy_name: str,
    start: str,
    end: str,
    strategy_returns: dict,
    strategy_positions: dict,
    strategy_info: dict,
    config: dict,
) -> dict:
    info = strategy_info[strategy_name]
    position = strategy_positions[strategy_name]
    base_position = info["Base Position"]
    costs = config["costs"]

    full_returns = strategy_returns[strategy_name]
    base_returns = apply_costs(
        base_position,
        data["Daily_Return"].fillna(0),
        costs["trading_bps_per_side"],
        costs["margin_rate_annual"],
        costs["max_allowed_exposure"],
    )
    overlay_returns = full_returns - base_returns
    extra_position = (position - base_position).clip(lower=0)
    effective_extra = extra_position.shift(1).fillna(0)
    period_mask = (data.index >= start) & (data.index <= end)

    overlay_returns = overlay_returns.loc[period_mask]
    effective_extra = effective_extra.loc[period_mask]

    in_trade = False
    start_idx = None
    trade_returns = []
    values = effective_extra.values
    rets = overlay_returns.values

    for i in range(len(values)):
        if not in_trade and values[i] > 0:
            in_trade = True
            start_idx = i

        if in_trade and (values[i] == 0 or i == len(values) - 1):
            end_idx = i if values[i] == 0 else i + 1
            trade_returns.append(np.prod(1 + rets[start_idx:end_idx]) - 1)
            in_trade = False
            start_idx = None

    trade_returns = pd.Series(trade_returns)
    if len(trade_returns) == 0:
        return {
            "Period": f"{start[:4]}-{end[:4]}",
            "Trades": 0,
            "Win Rate": np.nan,
            "Avg Trade": np.nan,
            "Median Trade": np.nan,
            "Best Trade": np.nan,
            "Worst Trade": np.nan,
            "Losing Streak": 0,
            "Total Overlay Return": np.nan,
        }

    return {
        "Period": f"{start[:4]}-{end[:4]}",
        "Trades": len(trade_returns),
        "Win Rate": (trade_returns > 0).mean(),
        "Avg Trade": trade_returns.mean(),
        "Median Trade": trade_returns.median(),
        "Best Trade": trade_returns.max(),
        "Worst Trade": trade_returns.min(),
        "Losing Streak": longest_losing_streak(trade_returns),
        "Total Overlay Return": np.prod(1 + trade_returns) - 1,
    }
