# SPY Pullback Overlay — Walk-Forward Strategy Validation

A systematic test of a tactical exposure-boost strategy on SPY: increase exposure above 1x
during short-term pullbacks within a confirmed long-term uptrend, then compare the result
against a *fairly matched* benchmark — not just plain buy-and-hold — across eight historical
market regimes from 1993 to today.

## Why this exists

It's easy to convince yourself a trading idea works by testing it once, on one time period,
against a benchmark that isn't really comparable. This project is an attempt to make that
mistake hard to make by accident: every result here has to survive real trading costs, real
margin financing costs, multiple market regimes, and a benchmark that accounts for the fact
that taking on more exposure should be expected to produce more return on its own.

## Signal definition

A pullback signal fires when, simultaneously:
- Price is **above** its 200-day moving average (confirming a long-term uptrend)
- Price is **below** its 20-day moving average (confirming a short-term pullback)
- The trailing 5-day return is below a configurable threshold (e.g. -2.0%)

When the signal fires, exposure is temporarily boosted above the base allocation for a fixed
holding period, with several exit policy variants tested (fixed hold, exit on trend break,
exit on short-term recovery, or both).

## What makes the benchmark fair

Most "my strategy beats buy-and-hold" claims compare a leveraged strategy against a 1x
benchmark — which mostly just proves that more exposure produces more return, not that the
*timing* of that exposure added anything. This project instead constructs a **matched-exposure
benchmark**: a buy-and-hold position scaled to the same average exposure as the strategy,
carrying the same margin financing cost. The "True Edge" metric reported is the strategy's
return **minus** that matched benchmark — so a positive edge means the timing of *when*
exposure increased actually mattered, not just that exposure was higher on average.

## Cost model

- **Trading costs**: configurable basis points per side, applied on every change in position
- **Margin financing**: configurable annual rate, applied only to the exposure above 1x
- Both costs are applied directly inside the return series, not added as a footnote after
  the fact

## Validation method: walk-forward windows

Rather than one backtest over the full history, the strategy is evaluated independently
across eight non-overlapping windows spanning different regimes:

| Window | Regime context |
|---|---|
| 2003–2005 | Post dot-com recovery |
| 2006–2008 | Pre/post Global Financial Crisis |
| 2009–2011 | Post-GFC recovery |
| 2012–2014 | Steady bull market |
| 2015–2017 | Low-volatility bull market |
| 2018–2020 | Volatility spike + COVID crash |
| 2021–2023 | COVID recovery + 2022 rate-hike drawdown |
| 2024–2026 | Most recent regime |

A strategy only qualifies as "practical" if it clears a minimum number of positive-edge
windows, keeps worst-case drawdown within a defined limit, and keeps average exposure within
a realistic bound — not just if it looks good on average.

## How to run

```bash
pip install -r requirements.txt
python backtest.py
```

This will print:
1. The best overall walk-forward candidates (ranked by consistency of edge, not just average)
2. The subset that passes practical risk/exposure filters
3. A detailed window-by-window breakdown for the top candidate
4. Trade-level statistics (win rate, average trade, longest losing streak) for the boost
   overlay specifically
5. A live signal check against the most recent market data

## Results

*(Fill in after your most recent run — pull these directly from the script's console output)*

- Selected strategy: `[paste selected_strategy here]`
- Positive windows: `[X / 8]`
- Average annual edge over matched benchmark: `[X%]`
- Worst-case window edge: `[X%]`
- Worst-case max drawdown: `[X%]`

## What I'd build next

- Save chart output (equity curve, drawdown curve) automatically per run instead of console
  tables only
- Extend to partial-allocation sizing (running this strategy on only 25/50/75% of a SPY
  position while the rest stays buy-and-hold)
- Add a small test suite around the cost/margin calculation functions, since a silent error
  there would quietly invalidate every result above it
