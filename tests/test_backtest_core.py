import pandas as pd

from backtest_core import apply_costs, build_active_signal, max_drawdown, select_strategy


def test_apply_costs_shifts_position_and_charges_turnover_and_margin():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    position = pd.Series([1.0, 2.0, 2.0], index=dates)
    returns = pd.Series([0.01, 0.01, 0.01], index=dates)

    result = apply_costs(
        position,
        returns,
        cost_bps_per_side=10.0,
        margin_rate_annual=0.252,
        max_allowed_exposure=2.0,
    )

    expected = pd.Series([0.0, 0.009, 0.018], index=dates)
    pd.testing.assert_series_equal(result, expected)


def test_max_drawdown_uses_peak_to_trough_loss():
    equity = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert max_drawdown(equity) == -0.25


def test_build_active_signal_respects_exit_condition():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    raw_signal = pd.Series([True, False, False, False, False], index=dates)
    exit_condition = pd.Series([False, False, True, False, False], index=dates)

    active, days_remaining = build_active_signal(raw_signal, hold_days=4, exit_condition=exit_condition)

    assert active.tolist() == [True, True, False, False, False]
    assert days_remaining.tolist() == [4, 3, 0, 0, 0]


def test_select_strategy_uses_training_metrics_and_filters_risk():
    config = {
        "selection": {
            "max_avg_exposure": 1.10,
            "max_drawdown_floor": -0.40,
            "min_train_edge": 0.0,
        }
    }
    train_results = pd.DataFrame(
        [
            {
                "Strategy": "Buy and Hold SPY",
                "True Edge": 0.0,
                "Sharpe": 0.5,
                "Max Drawdown": -0.30,
                "Avg Exposure": 1.0,
            },
            {
                "Strategy": "Fast but too risky",
                "True Edge": 0.20,
                "Sharpe": 1.5,
                "Max Drawdown": -0.60,
                "Avg Exposure": 1.0,
            },
            {
                "Strategy": "Train winner",
                "True Edge": 0.05,
                "Sharpe": 1.1,
                "Max Drawdown": -0.25,
                "Avg Exposure": 1.05,
            },
            {
                "Strategy": "Lower train edge",
                "True Edge": 0.03,
                "Sharpe": 1.3,
                "Max Drawdown": -0.20,
                "Avg Exposure": 1.04,
            },
        ]
    )

    assert select_strategy(train_results, config) == "Train winner"
