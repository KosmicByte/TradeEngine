# stochax-market: Workflow Guide

Step-by-step reference for running the V3 (`stochax_market`) pipeline
end-to-end, plus daily-refresh instructions for keeping a live forecast
scored against newly-realised market data.

This guide complements:

- [`README.md`](../README.md) — high-level Quick Start.
- [`docs/diagnostics.md`](diagnostics.md) — per-finding reference for
  `stochax-diagnose`.
- [`docs/api.md`](api.md) — function-level signatures.

The workflow below uses concrete dates throughout (anchored on
**Monday, 11 May 2026**) so each step has a real reference point. Swap
in your own dates where appropriate.

---

## Contents

1. [Conceptual model — what is "the workflow"?](#conceptual-model)
2. [Prerequisites](#prerequisites)
3. [The forecast lifecycle](#the-forecast-lifecycle)
4. [Workflow A — First-time run (live forward forecast)](#workflow-a)
5. [Workflow B — Daily refresh (filling in actuals)](#workflow-b)
6. [Workflow C — Backtest from a historical anchor](#workflow-c)
7. [Workflow D — Forecast rotation (issuing a new forecast)](#workflow-d)
8. [Output files at a glance](#output-files)
9. [Troubleshooting common symptoms](#troubleshooting)

---

<a id="conceptual-model"></a>
## 1. Conceptual model — what is "the workflow"?

The `stochax-market` pipeline produces **a single forecast that's
incrementally evaluated against reality.** Three time windows are
involved:

| Window         | What it is                                                 | Refreshes?              |
|----------------|------------------------------------------------------------|-------------------------|
| Training       | All historical data up to and including the anchor day.    | Only when you re-fit.   |
| Forecast       | N business days starting the day after the anchor.         | Fixed once issued.      |
| Realisation    | The subset of the forecast window where actuals now exist. | Grows by one row per trading day. |

A "forecast lifecycle" is the period from issuing a forecast (Workflow
A) until its horizon is fully realised (after ~N trading days for an
N-step forecast). During that lifecycle, **you do not re-fit or
re-simulate** — the model output is fixed, and you simply re-run merge
(and optionally diagnose / visualize) each day to grow the realised
window (Workflow B).

When the horizon is exhausted, you start a new lifecycle by re-fitting
and re-simulating from the new anchor (Workflow D).

If you want to test "what would the model have predicted starting on
some past date?" — that's Workflow C, the backtest. It uses the same
machinery but anchors at a historical date rather than today.

---

<a id="prerequisites"></a>
## 2. Prerequisites

Before running any of the workflows below, confirm:

**Package installed in editable mode**

```bash
cd ~/github/TradeEngine
uv pip install -e .
```

This registers all seven CLI commands (`stochax-fit`, `stochax-simulate`,
`stochax-predict`, `stochax-merge`, `stochax-diagnose`,
`stochax-visualize`, `stochax-export`).

**Historical data file present**

Place the V1.1.0-fetcher output at `./data/{SYMBOL}.csv`:

```
~/github/TradeEngine/
├── data/
│   └── RELIANCE.csv          ← required schema (capitalised cols)
├── src/
└── ...
```

Required columns: `Date, Symbol, Series, Prev Close, Open, High, Low,
Close, Volume, VWAP, Turnover, Trades, Deliverable Volume, %Deliverble`.
Dates can be tz-aware (e.g. `2026-05-08 00:00:00+05:30`) or tz-naive ISO
— v0.4 of the loader normalises both.

For the running example, the file should contain RELIANCE daily bars
through Friday 8 May 2026 (last row date: `2026-05-08`).

**Quick sanity check on the data:**

```bash
python3 -c "
from stochax_market.data.loader import load_stock
df = load_stock('RELIANCE')
print(f'Rows: {len(df)}')
print(f'Date range: {df[\"Date\"].min()} → {df[\"Date\"].max()}')
print(f'Date dtype: {df[\"Date\"].dtype}')
"
```

Expected output (paraphrased):
```
Rows: 3055
Date range: 2014-01-01 00:00:00 → 2026-05-08 00:00:00
Date dtype: datetime64[ns]
```

If `dtype` shows `datetime64[ns, Asia/Kolkata]` or similar, the loader
hasn't stripped the timezone — see [Troubleshooting](#troubleshooting).

---

<a id="the-forecast-lifecycle"></a>
## 3. The forecast lifecycle

A concrete timeline for a 30-step RELIANCE forecast issued on Monday
11 May 2026:

```
Mon May 11   issue forecast    ── Workflow A
             ├── data: through Fri May 8 (last close: ₹1435.20)
             ├── forecast horizon: 30 BDays
             ├── forecast anchor: Mon May 11 (step 0)
             ├── forecast end:    ≈ Fri Jun 19 (step 29)
             └── actuals filled:  0 / 30

Tue May 12   daily refresh     ── Workflow B
             └── actuals filled:  1 / 30  (May 11 Close now known)

Wed May 13   daily refresh
             └── actuals filled:  2 / 30

…
Fri Jun 19   horizon reached
             └── actuals filled:  30 / 30
                 → final accuracy report

Mon Jun 22   issue new forecast ── Workflow D
             ├── re-fit on updated data
             ├── re-simulate 30 BDays
             └── lifecycle restarts
```

Throughout the lifecycle, `RELIANCE_sim.csv` and `params.pkl` are
**fixed** — only the `actual_price` column of the sim CSV grows as days
pass and merge picks up new realised closes.

---

<a id="workflow-a"></a>
## 4. Workflow A — First-time run (live forward forecast)

The full one-time sequence to issue a new 30-step forecast for RELIANCE,
anchored at the next business day after history. Run from
`~/github/TradeEngine/`.

```bash
cd ~/github/TradeEngine

# (1) one-time: data/RELIANCE.csv must be in place (see Prerequisites)

# (2) fit the SPDE + GARCH parameters on the full history
stochax-fit --symbol RELIANCE --n-steps 500 --output params.pkl

# (3) simulate 30 trading days forward
#     → writes RELIANCE_sim.csv with (step, predicted_price)
stochax-simulate --symbol RELIANCE --steps 30

# (4) predict (optional — terminal-only ensemble view with CI bands)
#     → prints day-by-day mean / 5% / 95% to the terminal
#     → does NOT feed into the merge → diagnose chain
stochax-predict --symbol RELIANCE --horizon 21 --params params.pkl

# (5) merge predictions with realised actuals
#     → augments RELIANCE_sim.csv with (actual_price, date) columns
#     → all actuals are NaN on first issue; that's expected
stochax-merge --sim-csv RELIANCE_sim.csv --symbol RELIANCE

# (6) diagnose forecast quality
#     → reads merged CSV + params; writes RELIANCE_diag.md
#     → realised-data sections (Accuracy, Directional, Variance,
#       Residual) will report "fewer than 2 realised points" warnings
#       on the first run — that's expected. Model-health sections
#       (GARCH Health, Drift, Calibration Convergence) still produce
#       meaningful output.
stochax-diagnose --symbol RELIANCE \
                 --sim-csv RELIANCE_sim.csv \
                 --params  params.pkl \
                 --out     RELIANCE_diag.md

# (7) visualize — generates 7 Plotly PNGs in plots/
#     → run_all internally calls predict() to render the CI band plot,
#       so step (4) is redundant for plotting purposes
stochax-visualize --symbol RELIANCE \
                  --sim-csv RELIANCE_sim.csv \
                  --params  params.pkl \
                  --horizon 21 \
                  --recent-n 60 \
                  --out-dir plots/
```

### What you should see at each step

**Step (2) — fit.** Final terminal line should be a "✓ Calibration
complete" with a final loss in the 1e-3 to 1e-4 range for a well-fit
model. Hitting the 1000-step BFGS cap is flagged by diagnostics in
step (6) — not a hard error.

**Step (3) — simulate.** A Rich table with `Steps simulated: 30`,
`Trajectory shape: (30, 128)`, `Output file: RELIANCE_sim.csv`. Debug
lines show `last_price` matching the last `Close` in your data
(₹1435.20 in this example).

**Step (4) — predict (optional).** A Rich table with one row per
forecast day showing Mean / 5% CI / 95% CI prices. Useful for quickly
seeing the forecast band without opening a plot.

**Step (5) — merge.** A Rich table reporting:

| Metric          | Value             |
|-----------------|-------------------|
| Output file     | `RELIANCE_sim.csv` |
| Total steps     | `30`              |
| Date range      | `11/05/26 → 19/06/26` |
| Actuals filled  | `0`               |
| Forecast-only   | `30`              |

`Actuals filled: 0` is expected on day one — markets haven't traded yet
on May 11. The Date range tells you the realisation window you'll be
tracking.

**Step (6) — diagnose.** A Rich summary table with overall verdict
(`Healthy` / `Issues detected` / `Significant problems`), followed by a
per-section pass/warn/fail count grid. With zero realised data, several
sections will report `warn` ("fewer than 2 realised points") — this is
informational, not a failure of the model.

**Step (7) — visualize.** Seven PNGs written to `plots/`:

```
plots/
├── RELIANCE_historical.png
├── RELIANCE_log_returns.png
├── RELIANCE_volatility.png
├── RELIANCE_sim_vs_actual.png
├── RELIANCE_actual_vs_predicted.png    (auto-included once merged)
├── RELIANCE_prediction_21d.png         (CI band, calls predict internally)
└── RELIANCE_garch_fit.png
```

After step 7, you have a complete first-pass workflow: a forecast issued,
plots generated, and a diagnostic report stating the initial model
health.

---

<a id="workflow-b"></a>
## 5. Workflow B — Daily refresh (filling in actuals)

Once a forecast has been issued (Workflow A), the daily routine is to
refresh the data, re-merge, and re-diagnose. **Do not re-fit or
re-simulate during this phase** — the forecast you're scoring is the
one issued at the start of the lifecycle.

### Example daily run — Tuesday 12 May 2026

```bash
cd ~/github/TradeEngine

# (1) refresh data/RELIANCE.csv from V1.1.0
#     → this is project-external; use your upstox-historical fetcher
#       e.g. (from the upstox-historical repo):
#       cd ~/upstox_historical
#       upstox-fetcher update --symbol RELIANCE --nse-enrich
#       cp ./data/RELIANCE.csv ~/github/TradeEngine/data/
#     → after this, data/RELIANCE.csv should include the 11/05/26 row

cd ~/github/TradeEngine

# (2) re-merge — picks up the new May 11 actual
stochax-merge --sim-csv RELIANCE_sim.csv --symbol RELIANCE

# (3) re-diagnose — recomputes accuracy now that 1 realised point exists
stochax-diagnose --symbol RELIANCE \
                 --sim-csv RELIANCE_sim.csv \
                 --params  params.pkl \
                 --out     RELIANCE_diag.md

# (4) re-visualize — refreshes the actual-vs-predicted plot
stochax-visualize --symbol RELIANCE \
                  --sim-csv RELIANCE_sim.csv \
                  --params  params.pkl \
                  --out-dir plots/
```

### What changes day by day

After **Tue 12 May**:
- `RELIANCE_sim.csv` row for `step=0, date=11/05/26` now has an
  `actual_price` value.
- Merge table shows `Actuals filled: 1`.
- Diagnostics' Forecast Accuracy section is still mostly warnings
  (1 point isn't enough for MAPE, RMSE, etc.) but the section header
  changes to "1 realised."

After **Fri 16 May** (5 trading days in):
- `Actuals filled: 5`.
- Diagnostics Accuracy section now produces real numbers (MAPE, MAE,
  RMSE).
- Directional Performance section becomes meaningful (4 consecutive
  changes available).

After **Fri 13 Jun** (≈25 trading days in):
- `Actuals filled: 25`.
- All seven diagnostic sections produce reliable findings.
- `actual_vs_predicted.png` shows 25 dashed actuals against the full
  30-day predicted line, with `MAPE on realised window` in the
  subtitle.

After **Fri 19 Jun** (full horizon):
- `Actuals filled: 30`.
- The forecast is fully scored. Time to issue a new one (Workflow D).

### Why merge is idempotent

`stochax-merge` rebuilds the `actual_price` and `date` columns from
scratch each call. There's no state to manage, no risk of stale NaNs.
You can run it as often as you want — once daily after market close is
typical, but running it more often (or skipping a day) has no
side-effects.

### Cadence guidance

| Frequency  | Activities                                          |
|------------|-----------------------------------------------------|
| **Daily**  | Refresh data → re-merge → optionally re-visualize.  |
| **Weekly** | Add re-diagnose. Review the report; flag any new `warn`/`fail` findings. |
| **End of lifecycle** | Workflow D: re-fit, re-simulate, restart cycle. |

Re-fitting daily is technically valid but **not recommended** during a
forecast lifecycle — the parameters drift slightly day-to-day and
obscure the cleaner question of "did this forecast hold up?". Save
re-fits for lifecycle transitions or major regime shifts.

---

<a id="workflow-c"></a>
## 6. Workflow C — Backtest from a historical anchor

Use this workflow to ask: *"What would the model have predicted starting
on some past date, and how did it actually score against subsequent
realised data?"*

### Example — backtest from Tuesday 21 April 2026

You want to evaluate the model as if it had been run on 21 April 2026,
with 30 BDays of forecast horizon. Today's date (11 May 2026) means
you have roughly 14 trading days of realised data available for that
backtest window.

```bash
cd ~/github/TradeEngine

# (1) data/RELIANCE.csv already in place (covers 2014–today)

# (2) fit and simulate — same as Workflow A
#     ⚠ caveat below
stochax-fit --symbol RELIANCE --n-steps 500 --output params.pkl
stochax-simulate --symbol RELIANCE --steps 30

# (3) merge with EXPLICIT historical anchor
stochax-merge --sim-csv RELIANCE_sim.csv \
              --symbol  RELIANCE \
              --start-date 21/04/26
#     → forecast step 0 = Tue 21 April 2026
#     → forecast step 29 ≈ Fri 5 June 2026
#     → actuals filled: ~14 (21 April to 8 May, our data's end)

# (4) diagnose — now with substantial realised data
stochax-diagnose --symbol RELIANCE \
                 --sim-csv RELIANCE_sim.csv \
                 --params  params.pkl \
                 --out     RELIANCE_backtest_21apr.md

# (5) visualize
stochax-visualize --symbol RELIANCE \
                  --sim-csv RELIANCE_sim.csv \
                  --params  params.pkl \
                  --out-dir plots_backtest_21apr/
```

### The look-ahead caveat

⚠ **`stochax-fit` and `stochax-simulate` use the entire historical CSV.**
This means in step (2) above, the model is calibrated on data through
8 May 2026 — *including* the post-21-April-2026 prices that the
backtest is supposed to compare against. The fit has implicitly "seen"
some of the future when constructing parameters.

For a casual sanity check ("does the model produce sensible-looking
forecasts on past data?") this is acceptable. For rigorous walk-forward
validation, you'd need to:

1. Truncate `data/RELIANCE.csv` to rows on or before 18 April 2026.
2. Run fit + simulate against the truncated CSV.
3. Restore the full CSV before running merge (so actuals are available
   for comparison).

The current pipeline doesn't automate this truncation, so it's a
manual step. Workflow C as written above is the quick version with
look-ahead bias understood.

---

<a id="workflow-d"></a>
## 7. Workflow D — Forecast rotation (issuing a new forecast)

When the active forecast's horizon is fully realised (Workflow B
completes with `Actuals filled: 30 / 30`), or when a major regime shift
suggests parameter staleness, rotate to a new forecast.

### Example — rotation on Monday 22 June 2026

```bash
cd ~/github/TradeEngine

# (1) archive the completed forecast (optional but recommended)
mkdir -p archive/2026-05-11_to_2026-06-19
mv RELIANCE_sim.csv     archive/2026-05-11_to_2026-06-19/
mv RELIANCE_diag.md     archive/2026-05-11_to_2026-06-19/
mv params.pkl           archive/2026-05-11_to_2026-06-19/
mv plots                archive/2026-05-11_to_2026-06-19/

# (2) refresh data through last close (Fri 19 Jun 2026)
#     → run V1.1.0 fetcher, copy data/RELIANCE.csv

# (3) re-run Workflow A end-to-end with the fresh data
stochax-fit --symbol RELIANCE --n-steps 500 --output params.pkl
stochax-simulate --symbol RELIANCE --steps 30
stochax-merge --sim-csv RELIANCE_sim.csv --symbol RELIANCE
stochax-diagnose --symbol RELIANCE \
                 --sim-csv RELIANCE_sim.csv \
                 --params  params.pkl \
                 --out     RELIANCE_diag.md
stochax-visualize --symbol RELIANCE \
                  --sim-csv RELIANCE_sim.csv \
                  --params  params.pkl \
                  --out-dir plots/
```

After rotation: new forecast window is Mon 22 June 2026 to ≈ Fri 31
July 2026. Workflow B resumes daily against this new forecast.

### How often should you rotate?

The natural rhythm is **rotate when the horizon is exhausted** — every
~30 trading days for a 30-step forecast. Rotating mid-horizon is valid
but discards the partially-realised accuracy data, so most users
complete each lifecycle before rotating.

Two cases where mid-horizon rotation makes sense:

- **Diagnostic-flagged regime shift.** If the diagnose report's "Drift
  Calibration" section shows recent-60d drift diverging strongly from
  the full-history mean (status: `fail`), the parameters are tracking
  a regime that no longer reflects current dynamics. Rotate to refit.
- **Substantial volatility breakout.** If `GARCH Health` flags
  unconditional/realised variance ratio outside the OK band, the model
  is mis-scaled for current vol. Rotate.

---

<a id="output-files"></a>
## 8. Output files at a glance

| Path                          | Produced by         | Refreshes? | Lifecycle of value         |
|-------------------------------|---------------------|------------|----------------------------|
| `data/RELIANCE.csv`           | external fetcher    | daily      | growing                    |
| `params.pkl`                  | `stochax-fit`       | per lifecycle | fixed during lifecycle  |
| `RELIANCE_sim.csv`            | `stochax-simulate`  | per lifecycle | predicted_price column is fixed |
| `RELIANCE_sim.csv` (post-merge) | `stochax-merge`   | daily      | actual_price column grows  |
| `RELIANCE_diag.md`            | `stochax-diagnose`  | daily      | latest snapshot            |
| `plots/*.png` (7 files)       | `stochax-visualize` | daily      | latest snapshot            |
| Terminal table                | `stochax-predict`   | ad-hoc     | inspection only            |

### Which files go to git?

| File                          | Track in git? | Why                              |
|-------------------------------|---------------|----------------------------------|
| `data/*.csv`                  | No            | Large, external, derived         |
| `params.pkl`                  | No            | Binary, regenerable, large       |
| `RELIANCE_sim.csv`            | No            | Regenerable from params + data   |
| `RELIANCE_diag.md`            | Optional      | Useful for cross-run comparisons |
| `plots/*.png`                 | No            | Regenerable from CSVs            |

The `.gitignore` in the repo already covers `data/`, `*.csv`, `*.pkl`,
and `plots/`. Diagnostics reports are not currently ignored — track
them if you want a paper trail of model evolution.

---

<a id="troubleshooting"></a>
## 9. Troubleshooting common symptoms

### "Actuals filled: 0" after merge

**Most likely cause: timezone mismatch (pre-v0.4 bug).**

Your CSV has tz-aware dates (e.g. `2026-05-08 00:00:00+05:30`) but
merge constructs tz-naive forecast grids. Lookups silently return NaN
for every date.

**Fix:** Upgrade to v0.4 (loader.py now strips timezones on read).
Verify with:

```bash
python3 -c "
from stochax_market.data.loader import load_stock
df = load_stock('RELIANCE')
print(df['Date'].dtype)   # should be: datetime64[ns]
"
```

If `dtype` ends in `[ns, Asia/Kolkata]` or similar, the fix isn't
applied. Re-check that `src/stochax_market/data/loader.py` has the v0.4
content and reinstall with `uv pip install -e .`.

**Other possible causes:**

- The `--start-date` is outside the data's coverage. Check
  `df['Date'].min() / max()`.
- The CSV is missing rows for the realised window (data fetcher
  failed mid-update). Re-run the fetcher.

### "FileNotFoundError: load_stock('RELIANCE')"

The loader couldn't find `data/RELIANCE.csv`. The error message
includes the absolute path it checked. Possible fixes:

- You're running from the wrong directory. CLI commands resolve `./data/`
  relative to the current working directory. Run from
  `~/github/TradeEngine/`.
- The symbol is misspelled (`RELIANCE` not `Reliance`).
- The file is in a different directory. Pass `data_dir=` explicitly
  if using the loader from Python directly.

### "BFGS hit 1000-step cap" warning in diagnose

Calibration may not have fully converged. Two things to try:

- Rerun fit with more steps: `stochax-fit ... --n-steps 2000`.
- Check whether the diagnose report's `Calibration Convergence` section
  also reports a high `Final calibration loss` (> 0.01). If so, the
  initialisation might be poor — see project memory for the `raw_omega
  ≈ -11.0` anchor convention for daily-frequency data.

### Predicted curve looks completely flat across the horizon

Run diagnose. If `Variance Diagnostics` reports the
`Predicted/actual std ratio` as `fail` (< 0.3), this is the v0.2
bug-regression signature. The constant-drift or frozen-volatility patch
to `predict.py` may have been undone — re-check the `_single_forecast`
function against the v0.2 fix.

### After re-fit, parameters look very different from previous lifecycle

Possible regime shift. The diagnostics' `Drift Calibration` section
will tell you whether recent and historical drifts diverged. If they
have, the new parameters are tracking the new regime and the prior
forecast lifecycle's accuracy isn't a fair benchmark for the next one.
This is informational rather than a bug — just be aware that
cross-lifecycle metric comparisons need that context.

### Diagnostics section shows "fewer than 2 realised points"

Normal for the first 1–2 days of a new forecast. Wait for the
realisation window to grow.

---

## Appendix: One-line command cheat sheet

```bash
# First-time forecast (Workflow A)
stochax-fit --symbol RELIANCE --n-steps 500 --output params.pkl && \
stochax-simulate --symbol RELIANCE --steps 30 && \
stochax-merge --sim-csv RELIANCE_sim.csv --symbol RELIANCE && \
stochax-diagnose --symbol RELIANCE --sim-csv RELIANCE_sim.csv --params params.pkl --out RELIANCE_diag.md && \
stochax-visualize --symbol RELIANCE --sim-csv RELIANCE_sim.csv --params params.pkl --out-dir plots/

# Daily refresh (Workflow B) — assumes data/RELIANCE.csv is up to date
stochax-merge --sim-csv RELIANCE_sim.csv --symbol RELIANCE && \
stochax-diagnose --symbol RELIANCE --sim-csv RELIANCE_sim.csv --params params.pkl --out RELIANCE_diag.md && \
stochax-visualize --symbol RELIANCE --sim-csv RELIANCE_sim.csv --params params.pkl --out-dir plots/

# Backtest (Workflow C) — same as A but with --start-date on merge
stochax-merge --sim-csv RELIANCE_sim.csv --symbol RELIANCE --start-date 21/04/26
```
