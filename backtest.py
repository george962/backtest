# SPY Practical Strategy Test - Optimized Fast Version
# Focus:
# Above200 + Below20 + 5D pullback
#
# Optimized fix:
# The old script repeatedly re-evaluated every strategy inside another strategy loop.
# This version evaluates all strategies once per walk-forward window and reuses the results.
#
# Install first if needed:
# pip install -r requirements.txt

import yfinance as yf
import pandas as pd
import numpy as np

# =============================
# 1. Settings
# =============================

TICKER = "SPY"
START_DATE = "1993-01-01"

BASE_COST_BPS_PER_SIDE = 3.0
BASE_MARGIN_RATE_ANNUAL = 0.06
MAX_ALLOWED_EXPOSURE = 2.00

DROP_THRESHOLDS = [-0.015, -0.020, -0.025, -0.030]
HOLD_DAYS_LIST = [3, 5, 7, 10, 15]
BOOST_EXPOSURES = [1.25, 1.50, 1.75, 2.00]
MODES = ["Core", "Def"]

EXIT_POLICIES = [
    "Fixed",
    "StopBelow200",
    "Recover20",
    "StopBelow200+Recover20"
]

WALK_FORWARD_WINDOWS = [
    {"TrainEnd": "2002-12-31", "TestStart": "2003-01-01", "TestEnd": "2005-12-31"},
    {"TrainEnd": "2005-12-31", "TestStart": "2006-01-01", "TestEnd": "2008-12-31"},
    {"TrainEnd": "2008-12-31", "TestStart": "2009-01-01", "TestEnd": "2011-12-31"},
    {"TrainEnd": "2011-12-31", "TestStart": "2012-01-01", "TestEnd": "2014-12-31"},
    {"TrainEnd": "2014-12-31", "TestStart": "2015-01-01", "TestEnd": "2017-12-31"},
    {"TrainEnd": "2017-12-31", "TestStart": "2018-01-01", "TestEnd": "2020-12-31"},
    {"TrainEnd": "2020-12-31", "TestStart": "2021-01-01", "TestEnd": "2023-12-31"},
    {"TrainEnd": "2023-12-31", "TestStart": "2024-01-01", "TestEnd": "2026-12-31"},
]

PRACTICAL_MAX_AVG_EXPOSURE = 1.10
PRACTICAL_MAX_DRAWDOWN = -0.40
PRACTICAL_MIN_POSITIVE_WINDOWS = 6

# =============================
# 2. Download data
# =============================

df = yf.download(TICKER, start=START_DATE, auto_adjust=True, progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

df = df.dropna().copy()

# =============================
# 3. Features
# =============================

df["Daily_Return"] = df["Close"].pct_change()

df["SMA_20"] = df["Close"].rolling(20).mean()
df["SMA_200"] = df["Close"].rolling(200).mean()

df["Above_200DMA"] = df["Close"] > df["SMA_200"]
df["Below_200DMA"] = df["Close"] < df["SMA_200"]

df["Above_20DMA"] = df["Close"] > df["SMA_20"]
df["Below_20DMA"] = df["Close"] < df["SMA_20"]

df["Return_5D"] = df["Close"] / df["Close"].shift(5) - 1

df = df.dropna().copy()

# =============================
# 4. Helper functions
# =============================

def slice_period(data, start, end):
    return data.loc[(data.index >= start) & (data.index <= end)].copy()


def format_pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:,.2f}%"


def max_drawdown(equity):
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    return drawdown.min()


def annual_return_from_returns(strategy_returns):
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) == 0:
        return np.nan

    equity = (1 + strategy_returns).cumprod()
    total_return = equity.iloc[-1] - 1
    years = len(strategy_returns) / 252

    if years <= 0:
        return np.nan

    return (1 + total_return) ** (1 / years) - 1


def sharpe_from_returns(strategy_returns):
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) == 0 or strategy_returns.std() == 0:
        return np.nan

    return strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)


def build_active_signal(raw_signal, hold_days, exit_condition=None):
    raw_signal = raw_signal.fillna(False).astype(bool)

    if exit_condition is None:
        exit_condition = pd.Series(False, index=raw_signal.index)
    else:
        exit_condition = exit_condition.fillna(False).astype(bool)

    active = pd.Series(False, index=raw_signal.index)
    days_remaining = pd.Series(0, index=raw_signal.index, dtype=int)

    days_left = 0

    for i in range(len(raw_signal)):
        current_signal = raw_signal.iloc[i]
        current_exit = exit_condition.iloc[i]

        if days_left > 0:
            if current_exit:
                active.iloc[i] = False
                days_remaining.iloc[i] = 0
                days_left = 0
            else:
                active.iloc[i] = True
                days_remaining.iloc[i] = days_left
                days_left -= 1

        elif current_signal:
            active.iloc[i] = True
            days_remaining.iloc[i] = hold_days
            days_left = hold_days - 1

    return active, days_remaining


def apply_costs(position, returns, cost_bps_per_side, margin_rate_annual):
    desired_position = position.astype(float).clip(lower=0, upper=MAX_ALLOWED_EXPOSURE)
    effective_position = desired_position.shift(1).fillna(0)

    turnover = effective_position.diff().abs().fillna(0)
    trading_cost = turnover * (cost_bps_per_side / 10000)

    margin_exposure = (effective_position - 1.0).clip(lower=0)
    margin_cost = margin_exposure * (margin_rate_annual / 252)

    strategy_returns = effective_position * returns - trading_cost - margin_cost

    return strategy_returns


def count_exposure_changes(position):
    changes = position.diff().abs().fillna(0) > 0
    return int(changes.sum())


def performance_metrics(data, strategy_returns, position, name, margin_rate_annual):
    strategy_returns = strategy_returns.loc[data.index].fillna(0)
    position = position.loc[data.index].fillna(0)

    equity = (1 + strategy_returns).cumprod()

    annual_return = annual_return_from_returns(strategy_returns)
    sharpe = sharpe_from_returns(strategy_returns)
    mdd = max_drawdown(equity)

    avg_exposure = position.mean()
    max_exposure = position.max()

    matched_margin_exposure = max(avg_exposure - 1.0, 0)
    matched_bh_returns = (
        avg_exposure * data["Daily_Return"].fillna(0)
        - matched_margin_exposure * (margin_rate_annual / 252)
    )

    matched_bh_annual_return = annual_return_from_returns(matched_bh_returns)
    true_edge = annual_return - matched_bh_annual_return

    exposure_changes = count_exposure_changes(position)

    return {
        "Strategy": name,
        "Annual Return": annual_return,
        "Matched B&H Ann Ret": matched_bh_annual_return,
        "True Edge": true_edge,
        "Sharpe": sharpe,
        "Max Drawdown": mdd,
        "Avg Exposure": avg_exposure,
        "Max Exposure": max_exposure,
        "Exposure Changes": exposure_changes
    }


def longest_losing_streak(trade_returns):
    streak = 0
    max_streak = 0

    for r in trade_returns:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return max_streak


def print_summary_table(title, results_df, max_rows=20):
    print()
    print(title)
    print()

    if len(results_df) == 0:
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


def print_detail_table(strategy_name, detail_df):
    print()
    print("=== DETAILED WALK-FORWARD WINDOWS ===")
    print(f"Strategy: {strategy_name}")
    print()

    print(
        f"{'Window':<13} "
        f"{'AnnRet':>9} "
        f"{'B&HRet':>9} "
        f"{'Edge':>9} "
        f"{'MaxDD':>9} "
        f"{'Sharpe':>8} "
        f"{'AvgExp':>8}"
    )

    print("-" * 82)

    for _, row in detail_df.iterrows():
        print(
            f"{row['Window']:<13} "
            f"{format_pct(row['Annual Return']):>9} "
            f"{format_pct(row['B&H Annual Return']):>9} "
            f"{format_pct(row['True Edge']):>9} "
            f"{format_pct(row['Max Drawdown']):>9} "
            f"{row['Sharpe']:>8.2f} "
            f"{row['Avg Exposure']:>8.2f}"
        )


# =============================
# 5. Build strategy set
# =============================

def build_strategy_set():
    strategy_returns_local = {}
    strategy_positions_local = {}
    strategy_info_local = {}

    returns = df["Daily_Return"].fillna(0)

    buyhold_position = pd.Series(1.0, index=df.index)

    strategy_positions_local["Buy and Hold SPY"] = buyhold_position
    strategy_returns_local["Buy and Hold SPY"] = apply_costs(
        buyhold_position,
        returns,
        cost_bps_per_side=0.0,
        margin_rate_annual=0.0
    )

    strategy_info_local["Buy and Hold SPY"] = {
        "Base Position": buyhold_position,
        "Active Signal": pd.Series(False, index=df.index),
        "Days Remaining": pd.Series(0, index=df.index),
        "Raw Signal": pd.Series(False, index=df.index),
    }

    for drop_threshold in DROP_THRESHOLDS:
        raw_signal = (
            df["Above_200DMA"] &
            df["Below_20DMA"] &
            (df["Return_5D"] < drop_threshold)
        )

        threshold_label = f"5D<{drop_threshold:.1%}"

        for hold_days in HOLD_DAYS_LIST:
            for exit_policy in EXIT_POLICIES:
                if exit_policy == "Fixed":
                    exit_condition = pd.Series(False, index=df.index)
                elif exit_policy == "StopBelow200":
                    exit_condition = df["Below_200DMA"]
                elif exit_policy == "Recover20":
                    exit_condition = df["Above_20DMA"]
                elif exit_policy == "StopBelow200+Recover20":
                    exit_condition = df["Below_200DMA"] | df["Above_20DMA"]
                else:
                    raise ValueError(f"Unknown exit policy: {exit_policy}")

                active_signal, days_remaining = build_active_signal(
                    raw_signal,
                    hold_days,
                    exit_condition=exit_condition
                )

                for boost_exposure in BOOST_EXPOSURES:
                    for mode in MODES:
                        if mode == "Core":
                            base_position = pd.Series(1.0, index=df.index)
                        elif mode == "Def":
                            base_position = pd.Series(0.5, index=df.index)
                            base_position.loc[df["Above_200DMA"]] = 1.0
                        else:
                            raise ValueError(f"Unknown mode: {mode}")

                        position = base_position.copy()
                        position.loc[active_signal] = boost_exposure

                        strategy_name = (
                            f"{mode}: Above200+Below20+{threshold_label}, "
                            f"{boost_exposure:.2f}x, {hold_days}d, {exit_policy}"
                        )

                        strategy_positions_local[strategy_name] = position
                        strategy_returns_local[strategy_name] = apply_costs(
                            position,
                            returns,
                            BASE_COST_BPS_PER_SIDE,
                            BASE_MARGIN_RATE_ANNUAL
                        )

                        strategy_info_local[strategy_name] = {
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

    return strategy_returns_local, strategy_positions_local, strategy_info_local


def evaluate_all_strategies(start, end):
    period_df = slice_period(df, start, end)
    rows = []

    for strategy_name in strategy_returns:
        metrics = performance_metrics(
            period_df,
            strategy_returns[strategy_name],
            strategy_positions[strategy_name],
            strategy_name,
            BASE_MARGIN_RATE_ANNUAL
        )
        rows.append(metrics)

    return pd.DataFrame(rows)


# =============================
# 6. Trade stats
# =============================

def trade_stats_for_strategy(strategy_name, start, end):
    info = strategy_info[strategy_name]

    position = strategy_positions[strategy_name]
    base_position = info["Base Position"]

    full_returns = strategy_returns[strategy_name]
    base_returns = apply_costs(
        base_position,
        df["Daily_Return"].fillna(0),
        BASE_COST_BPS_PER_SIDE,
        BASE_MARGIN_RATE_ANNUAL
    )

    overlay_returns = full_returns - base_returns

    extra_position = (position - base_position).clip(lower=0)
    effective_extra = extra_position.shift(1).fillna(0)

    period_mask = (df.index >= start) & (df.index <= end)

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
            trade_ret = np.prod(1 + rets[start_idx:end_idx]) - 1
            trade_returns.append(trade_ret)
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
            "Total Overlay Return": np.nan
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
        "Total Overlay Return": np.prod(1 + trade_returns) - 1
    }


def print_trade_stats(strategy_name):
    periods = [
        ("1993-01-01", "2026-12-31"),
        ("2003-01-01", "2026-12-31"),
        ("2017-01-01", "2021-12-31"),
        ("2022-01-01", "2026-12-31"),
    ]

    rows = []

    for start, end in periods:
        rows.append(trade_stats_for_strategy(strategy_name, start, end))

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


# =============================
# 7. Build all strategies once
# =============================

strategy_returns, strategy_positions, strategy_info = build_strategy_set()

# =============================
# 8. Fast walk-forward evaluation
# =============================

window_results = {}

for window in WALK_FORWARD_WINDOWS:
    label = f"{window['TestStart'][:4]}-{window['TestEnd'][:4]}"
    window_results[label] = evaluate_all_strategies(
        window["TestStart"],
        window["TestEnd"]
    )

fixed_rows = []

for label, results in window_results.items():
    bh_return = results.loc[
        results["Strategy"] == "Buy and Hold SPY",
        "Annual Return"
    ].iloc[0]

    non_bh = results[results["Strategy"] != "Buy and Hold SPY"].copy()

    for _, row in non_bh.iterrows():
        fixed_rows.append({
            "Strategy": row["Strategy"],
            "Window": label,
            "Annual Return": row["Annual Return"],
            "B&H Annual Return": bh_return,
            "True Edge": row["True Edge"],
            "Max Drawdown": row["Max Drawdown"],
            "Sharpe": row["Sharpe"],
            "Avg Exposure": row["Avg Exposure"],
            "Max Exposure": row["Max Exposure"],
        })

fixed_df = pd.DataFrame(fixed_rows)

summary_rows = []

for strategy_name, group in fixed_df.groupby("Strategy"):
    summary_rows.append({
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
    })

summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    by=["Positive Windows", "Avg Test Edge", "Worst Test Edge"],
    ascending=False
)

practical_df = summary_df[
    (summary_df["Positive Windows"] >= PRACTICAL_MIN_POSITIVE_WINDOWS) &
    (summary_df["Worst Max Drawdown"] > PRACTICAL_MAX_DRAWDOWN) &
    (summary_df["Avg Exposure"] <= PRACTICAL_MAX_AVG_EXPOSURE)
].copy()

practical_df = practical_df.sort_values(
    by=["Positive Windows", "Avg Test Edge", "Worst Test Edge"],
    ascending=False
)

# =============================
# 9. Output summary
# =============================

print()
print("PRACTICAL STRATEGY TEST - FAST VERSION")
print("Signal family: Above200 + Below20 + 5D pullback")
print(f"Trading cost: {BASE_COST_BPS_PER_SIDE} bps per side")
print(f"Margin/financing cost: {BASE_MARGIN_RATE_ANNUAL * 100:.2f}% annual")
print("New features tested: StopBelow200, Recover20, StopBelow200+Recover20")
print()

print_summary_table(
    title="=== BEST OVERALL WALK-FORWARD CANDIDATES ===",
    results_df=summary_df,
    max_rows=20
)

print_summary_table(
    title="=== PRACTICAL WALK-FORWARD CANDIDATES ===",
    results_df=practical_df,
    max_rows=20
)

if len(practical_df) > 0:
    selected_strategy = practical_df.iloc[0]["Strategy"]
else:
    selected_strategy = summary_df.iloc[0]["Strategy"]

selected_detail = fixed_df[fixed_df["Strategy"] == selected_strategy].copy()

print_detail_table(selected_strategy, selected_detail)

# =============================
# 10. Trade stats
# =============================

print_trade_stats(selected_strategy)

# =============================
# 11. Current signal checker
# =============================

print()
print("=== CURRENT SIGNAL CHECKER ===")

latest_date = df.index[-1]
latest = df.iloc[-1]

print(f"Latest date: {latest_date.date()}")
print(f"Close: {latest['Close']:.2f}")
print(f"20DMA: {latest['SMA_20']:.2f}")
print(f"200DMA: {latest['SMA_200']:.2f}")
print(f"5D Return: {format_pct(latest['Return_5D'])}")
print(f"Above 200DMA: {bool(latest['Above_200DMA'])}")
print(f"Below 20DMA: {bool(latest['Below_20DMA'])}")

raw_signal_2pct = (
    bool(latest["Above_200DMA"]) and
    bool(latest["Below_20DMA"]) and
    latest["Return_5D"] < -0.020
)

print(f"Raw signal for Above200 + Below20 + 5D<-2.0%: {raw_signal_2pct}")

print()
print("Selected strategy current status:")
print(f"Strategy: {selected_strategy}")

selected_info = strategy_info[selected_strategy]
active_boost = bool(selected_info["Active Signal"].iloc[-1])
days_left = int(selected_info["Days Remaining"].iloc[-1])
desired_exp = float(strategy_positions[selected_strategy].iloc[-1])
effective_exp_today = float(strategy_positions[selected_strategy].shift(1).fillna(0).iloc[-1])
base_exp = float(selected_info["Base Position"].iloc[-1])

print(f"Base exposure:            {base_exp:.2f}")
print(f"Active boost:             {active_boost}")
print(f"Boost days left:          {days_left}")
print(f"Desired exposure:         {desired_exp:.2f}")
print(f"Effective exposure today: {effective_exp_today:.2f}")

print()
print("Main 5D<-2.0% candidate statuses:")
print()

main_candidates = [
    "Def: Above200+Below20+5D<-2.0%, 1.25x, 7d, Fixed",
    "Def: Above200+Below20+5D<-2.0%, 1.50x, 7d, Fixed",
    "Def: Above200+Below20+5D<-2.0%, 1.75x, 7d, Fixed",
    "Def: Above200+Below20+5D<-2.0%, 2.00x, 7d, Fixed",
    "Def: Above200+Below20+5D<-2.0%, 1.50x, 7d, StopBelow200",
    "Def: Above200+Below20+5D<-2.0%, 1.50x, 7d, Recover20",
    "Def: Above200+Below20+5D<-2.0%, 1.50x, 7d, StopBelow200+Recover20",
    "Core: Above200+Below20+5D<-2.0%, 1.50x, 7d, Fixed",
    "Core: Above200+Below20+5D<-2.0%, 2.00x, 7d, Fixed",
]

print(
    f"{'Strategy':<78} "
    f"{'Active':>8} "
    f"{'Days':>6} "
    f"{'Base':>6} "
    f"{'Desired':>9} "
    f"{'EffToday':>9}"
)

print("-" * 125)

for candidate in main_candidates:
    if candidate not in strategy_positions:
        continue

    info = strategy_info[candidate]

    active = bool(info["Active Signal"].iloc[-1])
    days = int(info["Days Remaining"].iloc[-1])
    base = float(info["Base Position"].iloc[-1])
    desired = float(strategy_positions[candidate].iloc[-1])
    effective = float(strategy_positions[candidate].shift(1).fillna(0).iloc[-1])

    print(
        f"{candidate:<78} "
        f"{str(active):>8} "
        f"{days:>6} "
        f"{base:>6.2f} "
        f"{desired:>9.2f} "
        f"{effective:>9.2f}"
    )

# portfolio sizing
# What if only 25%, 50%, or 75% of my SPY allocation follows this strategy,
# while the rest stays buy-and-hold?

# =============================
# 12. Generate report (charts + tables)
# =============================

from report_generation import generate_report

generate_report(
    selected_strategy, strategy_returns, strategy_positions,
    fixed_df, summary_df, df, BASE_MARGIN_RATE_ANNUAL
)