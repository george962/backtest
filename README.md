# SPY Pullback Overlay Backtest

A reproducible research harness for testing a tactical SPY exposure overlay.

The strategy temporarily increases exposure during short-term pullbacks inside a
long-term uptrend, then measures whether the timing adds value beyond a
matched-exposure benchmark. The project is intentionally small, but it is built
to show the habits that matter in research infrastructure: explicit config,
cached data, walk-forward validation, cost modeling, tests, and repeatable run
artifacts.

## What the Strategy Tests

A raw pullback signal fires when:

- SPY is above its 200-day moving average
- SPY is below its 20-day moving average
- trailing 5-day return is below a configured threshold

When the signal fires, the strategy boosts exposure for a fixed holding period.
The grid tests multiple thresholds, hold periods, exposure levels, base exposure
modes, and exit policies.

## Why the Benchmark Is Matched

The strategy sometimes uses more than 1.0x exposure. Comparing that directly to
plain buy-and-hold would mostly reward leverage, not timing. This project
therefore compares each strategy to a buy-and-hold benchmark scaled to the same
average exposure and charged the same margin financing cost. The reported "True
Edge" is:

```text
strategy annual return - matched-exposure benchmark annual return
```

## Validation Approach

The main demo path uses a train-then-test walk-forward process:

1. Build the full strategy grid.
2. For each window, select the best strategy using only data through `TrainEnd`.
3. Test that selected strategy on the next unseen window.
4. Stitch the out-of-sample windows into one equity curve.

The run also writes a static grid diagnostic across test windows, but that is
kept separate from the train-selected out-of-sample result.

## How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full backtest:

```bash
python backtest.py run
```

You can also run the same default command with:

```bash
python backtest.py
```

Show the latest signal state:

```bash
python backtest.py signal
```

Force a fresh market data download:

```bash
python backtest.py run --refresh-data
```

The first run downloads SPY data and caches it at `data/SPY.parquet`. Later runs
reuse that cache unless `--refresh-data` is passed.

## Run Outputs

Each run creates a folder like:

```text
runs/20260627_163000_spy_pullback/
```

Inside it:

- `config.yaml`: exact config used for the run
- `walk_forward_selected.csv`: train-selected strategy for each test window
- `static_summary_table.csv`: static grid diagnostic summary
- `static_window_grid.csv`: per-window static grid results
- `trade_stats.csv`: overlay trade statistics
- `summary.md`: markdown summary tables
- `oos_equity_curve.png`: train-selected out-of-sample equity curve
- `oos_drawdown_chart.png`: train-selected out-of-sample drawdown
- `oos_edge_by_window.png`: out-of-sample edge by test window
- `static_equity_curve.png`: static diagnostic curve for the top practical strategy

## Run with Docker

Build the image:

```bash
docker build -t spy-pullback-backtest .
```

Run it and save artifacts back to your machine:

```bash
docker run --rm -v "$PWD/runs:/app/runs" -v "$PWD/data:/app/data" spy-pullback-backtest
```

## Run Tests

```bash
pytest
```

The tests focus on the parts that can quietly invalidate a backtest:

- position shifting so signals trade on the next bar
- turnover and margin cost math
- drawdown calculation
- train-period strategy selection under risk filters

## Interview Talking Point

This is not presented as a production trading system. It is a scoped research
infrastructure demo:

> I wanted to show how I would make financial experimentation reproducible and
> auditable: explicit configs, cached inputs, realistic costs, train-then-test
> validation, saved artifacts, tests around the financial math, and a Dockerized
> run path.
