# Diagnostics Module — `stochax_market.diagnostics`

Post-hoc analysis of V3 simulation and forecast results. Reads the merged
sim CSV produced by `stochax-merge` (and optionally the fitted `params.pkl`
from `stochax-fit`), runs a battery of statistical and model-health checks,
and emits a structured Markdown report describing what's working, what
isn't, and what to investigate.

The module exists for two reasons that are easy to conflate but distinct:

1. **Quality assessment** — "How good are these results?" Concrete numbers
   (MAPE, RMSE, directional accuracy) you can compare across runs.
2. **Failure detection** — "What's going wrong?" Targeted checks designed
   to catch the *specific* failure modes of an SPDE-GARCH model, including
   the constant-drift and frozen-volatility bugs that were patched in v0.2.

A passing report is therefore not just "low error" but "low error *and*
none of the known failure-mode signatures are present."

---

## Pipeline placement

```
fit  →  simulate  →  merge  →  diagnose  →  visualize
                              ↑
                              you are here
```

`diagnose` runs after `merge` because it needs the 4-column merged schema
(predicted alongside actuals on a business-day grid). It runs before or
alongside `visualize` — they consume the same input but answer different
questions: `visualize` produces plots for human inspection, `diagnose`
produces structured findings that machines can compare across runs.

---

## CLI usage

```bash
stochax-diagnose --symbol RELIANCE \
                 --sim-csv RELIANCE_sim.csv \
                 --params  params.pkl \
                 --out     RELIANCE_diagnostics.md
```

| Flag         | Required | Purpose                                                              |
|--------------|:--------:|----------------------------------------------------------------------|
| `--symbol`   | ✓        | Stock ticker; used to load V1.1.0 historical data via `load_stock`.  |
| `--sim-csv`  | ✓        | Merged sim CSV from `stochax-merge` (4 cols).                        |
| `--params`   |          | Fitted `params.pkl`. If omitted, GARCH and Calibration sections are skipped. |
| `--out`      |          | Output path. If omitted, the full Markdown report streams to stdout. |

The CLI also prints two Rich-styled summary tables before the markdown:
the headline (overall verdict + MAPE + mean predicted/actual) and a
per-section pass/warn/fail count grid.

---

## Library usage

```python
from pathlib import Path
from stochax_market.diagnostics import analyze

report = analyze(
    symbol      = "RELIANCE",
    sim_csv     = Path("RELIANCE_sim.csv"),
    params_path = Path("params.pkl"),
    out_md      = Path("RELIANCE_diagnostics.md"),  # optional
)

print(report.overall_status())     # "ok" | "warn" | "fail"
print(report.summary)              # {"MAPE (realised)": "1.23%", ...}
for section, findings in report.sections.items():
    for f in findings:
        print(f.metric, f.status, f.value, f.diagnosis)
```

The `DiagnosticsReport` dataclass exposes the same data the markdown
renderer consumes, so you can plug it into custom pipelines (e.g. CI
gates that fail on `overall_status() == "fail"`, or aggregation across
many runs to track trends over time).

---

## Report structure

Every report has the same skeleton:

1. **Header** — symbol, overall status (worst across all findings), step
   counts, and headline summary metrics (MAPE, mean predicted, mean actual).
2. **Sections** — seven of them, each rendered as a table of findings.
3. **Recommendations** — auto-extracted from any non-`ok` findings.

A `Finding` is `(metric, value, status, diagnosis)`. The diagnosis is empty
for `ok` rows and carries a one-line cause/remedy for `warn` and `fail`.
The auto-recommendations section at the end aggregates these so you can
skim straight to a TODO list without reading every section.

---

## The seven sections

Each section below documents:

- **What it measures** — the question the section answers.
- **Findings** — individual checks within the section, with formulas,
  status thresholds, and what each threshold actually means.

### Section 1 — Forecast Accuracy

**What it measures.** Standard regression-style error of the predicted
prices against realised actual prices, plus a baseline-comparison metric
(Theil's U) that asks whether the SPDE forecast is actually better than
predicting "today's price = yesterday's price."

#### MAPE

Mean absolute percentage error on the realised window:

$$\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left|\frac{\hat{p}_i - p_i}{p_i}\right|$$

| Threshold | Status |
|-----------|--------|
| < 2%      | ok     |
| 2–5%      | warn   |
| > 5%      | fail   |

For daily equity prediction, anything above 2% is borderline; above 5% is
poor. The 2% threshold is a deliberately strict ceiling — daily NIFTY
moves average roughly 1–1.5%, so a MAPE consistently above the typical
daily move means the model isn't reliably tracking even the average day.

#### RMSE / MAE

Reported in INR for absolute interpretability. No status thresholds —
they're informational, used to compare across runs at the same scale.
Use these for run-to-run regression detection: if MAE jumps 30% between
two otherwise comparable runs, something changed.

#### Bias (mean error)

$$\text{Bias} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - p_i)$$

Status is on |bias / mean(actual)|:

| Threshold | Status | Diagnosis                                          |
|-----------|--------|----------------------------------------------------|
| < 1%      | ok     |                                                    |
| 1–3%      | warn   | systematic over/under-prediction                   |
| > 3%      | fail   | check μ_scale calibration and drift handling       |

Persistent positive or negative bias means the model has a systematic
directional offset. Most often this is a μ_scale or drift-baseline
miscalibration — the SPDE's deterministic μ component is pulling the field
in the wrong direction.

#### Theil's U vs random walk

Theil's U is the ratio of the model's RMSE to the RMSE of the random-walk
baseline ($\hat{p}_i = p_{i-1}$):

$$U = \frac{\sqrt{\frac{1}{n-1}\sum (\hat{p}_i - p_i)^2}}{\sqrt{\frac{1}{n-1}\sum (p_{i-1} - p_i)^2}}$$

| Threshold | Status | Diagnosis                                                |
|-----------|--------|----------------------------------------------------------|
| < 0.95    | ok     | model beats naive baseline                                |
| 0.95–1.10 | warn   | roughly tied with naive RW baseline                       |
| > 1.10    | fail   | WORSE than naive — SPDE not adding signal vs predict-yesterday |

This is the most important single number in the report. A complex
multi-parameter SPDE-GARCH model has to clear a high bar to be worth
running over the trivial random-walk baseline. If U > 1, the model is
losing to "predict tomorrow's price = today's price."

### Section 2 — Directional Performance

**What it measures.** Whether the model gets the sign of day-over-day
moves right, independent of how close it gets in magnitude.

#### Directional accuracy (consecutive)

Fraction of consecutive predicted moves whose sign matches the actual
move's sign:

$$\text{DA} = \frac{1}{n-1} \sum_{i=2}^{n} \mathbf{1}\!\left[\text{sign}(\hat{p}_i - \hat{p}_{i-1}) = \text{sign}(p_i - p_{i-1})\right]$$

| Threshold | Status | Diagnosis                                                |
|-----------|--------|----------------------------------------------------------|
| ≥ 55%     | ok     | better than chance                                        |
| 45–55%    | warn   | indistinguishable from coin flip                          |
| < 45%     | fail   | model can't tell up-days from down-days                   |

Random guessing is 50%. Below 45% is statistically suspect — close to
guessing, with the slight skew possibly indicating actively wrong sign
attribution. In equities, a *consistently* well-calibrated model should
clear 55% on directional accuracy alone.

#### Return correlation (ρ)

Pearson correlation between predicted and actual one-day returns:

| Threshold | Status |
|-----------|--------|
| > 0.20    | ok     |
| -0.05–0.20| warn   |
| < -0.05   | fail   |

Near-zero correlation with positive directional accuracy is unusual; it
typically means the model gets the *sign* right by accident through a
slight directional bias, but isn't actually tracking the underlying
return generating process.

### Section 3 — Variance Diagnostics

**What it measures.** Whether the predicted price path has the right
amount of *movement* to be plausible — neither a flat line nor wild
swings. This is the section that explicitly catches regressions of the
constant-drift and frozen-volatility bugs fixed in v0.2.

#### Predicted/actual std ratio

$$\text{ratio} = \frac{\sigma(\hat{p})}{\sigma(p)}$$

Computed over the realised window only, where both series have data.

| Threshold     | Status | Diagnosis                                                              |
|---------------|--------|------------------------------------------------------------------------|
| 0.5–2.0       | ok     |                                                                        |
| 0.3–0.5       | warn   | predicted variance much smaller than actual                            |
| < 0.3 or > 3.0| fail   | classic signature of frozen-vol or constant-drift bugs in `predict.py` |

Why this catches the bug regressions: the constant-drift bug
(`drift = jnp.full(horizon, last_drift)`) and the frozen-vol bug
(`sigma_fields = jnp.full(horizon, last_sigma)`) both produce predicted
trajectories with collapsed variance. The drift becomes a deterministic
slope and the noise term shrinks toward whatever single sample sigma was
frozen at. The result is a near-flat predicted curve with std much smaller
than the actual price path — which manifests as ratio < 0.3.

#### Predicted relative range (full horizon)

$$\text{rel range} = \frac{\max \hat{p} - \min \hat{p}}{\overline{\hat{p}}}$$

Computed over the *full* forecast horizon, not just the realised window.

| Threshold | Status | Diagnosis                                |
|-----------|--------|------------------------------------------|
| > 2%      | ok     |                                          |
| 0.5–2%    | warn   | forecast is fairly flat                  |
| < 0.5%    | fail   | forecast is essentially a flat line      |

This is a stricter, full-horizon version of the std ratio check. A
forecast that varies by less than 0.5% across 20+ trading days is a clear
sign that either the noise term is suppressed or the deterministic drift
is dominating to the point of erasing stochastic movement entirely.

### Section 4 — Residual Analysis

**What it measures.** Properties of the prediction errors. A
well-calibrated model should produce residuals that look like noise: no
autocorrelation, no extreme skew or kurtosis. Structure in residuals
means the model is leaving exploitable signal on the table.

#### Lag-1 autocorrelation

$$\rho_1 = \text{corr}(r_t, r_{t-1}), \quad r_t = \hat{p}_t - p_t$$

| Threshold      | Status | Diagnosis                                                |
|----------------|--------|----------------------------------------------------------|
| \|ρ\| < 0.3    | ok     |                                                          |
| 0.3 ≤ \|ρ\| < 0.5 | warn   | leftover structure                                    |
| \|ρ\| ≥ 0.5    | fail   | strong residual autocorrelation                          |

If today's residual predicts tomorrow's residual, the model has a
systematic miss that an AR component or longer drift window would
capture. For an SPDE that already includes diffusive smoothing, strong
residual autocorrelation often points at drift baseline issues — the
deterministic μ component is consistently mis-aligned with the regime.

#### Residual skewness and excess kurtosis

Standardised third and fourth moments of residuals (zero-mean, unit-std):

$$\text{skew} = \mathbb{E}[z^3], \quad \text{kurt}_{ex} = \mathbb{E}[z^4] - 3$$

| Metric    | Threshold | Status | Diagnosis                                  |
|-----------|-----------|--------|--------------------------------------------|
| \|skew\|  | < 1       | ok     |                                            |
| \|skew\|  | ≥ 1       | warn   | asymmetric error distribution              |
| \|kurt\|  | < 3       | ok     |                                            |
| \|kurt\|  | ≥ 3       | warn   | fat-tailed residuals                       |

Asymmetric residuals usually indicate an unmodeled regime shift in the
realised window. Fat tails mean the model under-estimates the probability
of large moves — common in equity models that don't fully capture
volatility clustering, even with GARCH.

### Section 5 — GARCH Health

**What it measures.** Whether the fitted GARCH(1,1) volatility model is
internally consistent and tracking realised volatility dynamics. Skipped
if `--params` is not provided.

#### Persistence (α + β)

| Threshold | Status | Diagnosis                                                          |
|-----------|--------|--------------------------------------------------------------------|
| < 0.99    | ok     |                                                                    |
| 0.99–1.00 | warn   | near-unit-root persistence; vol shocks decay extremely slowly      |
| ≥ 1.00    | fail   | non-stationary GARCH — refit with bounded β                        |

α + β is the persistence parameter of the GARCH(1,1) recursion. Equity
GARCH typically lands in 0.95–0.99, so values in this range are normal.
Above 0.99, shock decay becomes pathologically slow (a vol spike persists
for hundreds of days). At or above 1.00, the model is mathematically
non-stationary — the unconditional variance ω/(1-α-β) is undefined and
the long-horizon forecast diverges.

The half-life (`-ln(2)/ln(α+β)`) is reported as a sanity check; persistence
0.97 → ~23 days, 0.99 → ~69 days, 1.00 → infinity.

#### Unconditional / realised variance ratio

$$\text{ratio} = \frac{\omega/(1-\alpha-\beta)}{\text{Var}(\log\text{ returns})}$$

| Threshold    | Status | Diagnosis                                                                          |
|--------------|--------|------------------------------------------------------------------------------------|
| 0.7–1.5      | ok     |                                                                                    |
| 0.4–2.5      | warn   | mild mismatch between long-run GARCH variance and realised variance                |
| < 0.4 or > 2.5 | fail | inflated/deflated GARCH long-run variance — check ω initialisation (~−11.0 raw)    |

A well-calibrated GARCH should have its long-run unconditional variance
match the realised variance of log-returns. Large mismatches almost always
trace back to ω initialisation; raw_omega ≈ −11.0 is the right anchor for
daily NIFTY data, while raw_omega ≈ −3.0 (a common naive default) inflates
unconditional variance by orders of magnitude.

#### GARCH vs 30d realised vol correlation

Pearson correlation between fitted GARCH conditional volatility σ_t and a
30-day rolling realised volatility computed from the same log-returns.

| Threshold | Status | Diagnosis                                                                |
|-----------|--------|--------------------------------------------------------------------------|
| > 0.50    | ok     |                                                                          |
| 0.20–0.50 | warn   | weak tracking                                                            |
| < 0.20    | fail   | GARCH essentially decoupled from observed dynamics                       |

The GARCH model is supposed to track realised volatility dynamics. If the
correlation between fitted σ and rolling realised σ is below 0.5, the
GARCH calibration is not actually picking up the volatility clustering it
exists to model — typically a sign the calibration loss is mis-weighted
or the optimiser hit a flat-vol local minimum.

### Section 6 — Drift Calibration

**What it measures.** Whether the historical drift used as the
deterministic μ component is sane in absolute terms and stable across the
fit window.

#### Mean historical drift (annualised)

$$\bar{\mu}_{annual} = 252 \cdot \frac{1}{T} \sum_{t=1}^{T} \log\frac{p_t}{p_{t-1}}$$

| Threshold       | Status | Diagnosis                                                              |
|-----------------|--------|------------------------------------------------------------------------|
| \|μ\| < 40%     | ok     |                                                                        |
| 40% ≤ \|μ\| < 80% | warn | annualised drift outside ±40% — possible single-regime fit             |
| \|μ\| ≥ 80%     | fail   | unrealistic drift — likely a window-of-fit anomaly                     |

NIFTY equities historically show ~10–15% annualised drift across long
windows. ±40% is a very generous band that allows for high-growth or
high-decline periods; outside it, the fit window is almost certainly
dominated by a single trending regime that won't continue indefinitely.

#### Recent-60d drift divergence

The same metric computed over the most recent 60 trading days, compared
against the full-history mean:

| |Δ μ_annual| | Status | Diagnosis                                          |
|--------------|--------|----------------------------------------------------|
| < 30%        | ok     |                                                    |
| 30–60%       | warn   | regime shift                                       |
| ≥ 60%        | fail   | major regime shift — predictions will lag          |

If recent drift diverges sharply from the historical mean, the
deterministic μ component (anchored to historical mean by `predict.py`'s
v0.2 fix) will lag the current regime. This isn't a bug — it's a
deliberate choice to favour stability over reactivity — but the user
should be aware that in fast-moving regimes, the predicted curve will
under-shoot the realised trend until enough new data flows into the next
fit.

### Section 7 — Calibration Convergence

**What it measures.** Whether the calibration step (`stochax-fit`)
produced a healthy parameter set or hit edge cases like step-cap
non-convergence. Skipped if `--params` is not provided or the pickled
file lacks `loss_info`.

#### Final calibration loss

| Threshold | Status | Diagnosis                                                |
|-----------|--------|----------------------------------------------------------|
| < 0.01    | ok     |                                                          |
| 0.01–0.10 | warn   | borderline calibration                                   |
| ≥ 0.10    | fail   | under-calibrated; rerun with more BFGS steps             |

The calibration loss combines four components: teacher-forced MSE
(weight 1000), directional penalty (0.1), GARCH variance-ratio anchor
(0.5), and rollout-based shift-variance penalty (0.3). See
[`calibration.md`](calibration.md) for the per-component rationale.

The final-loss thresholds above are heuristic — they pre-date the v0.5
four-term loss and assume the older unit-weighted formulation. With the
weights above, a healthy v0.5 fit lands around 0.3–0.5 final loss
(MSE-weighted ≈ 0.15, var-ratio ≈ 0; shift-var ≈ 0.05–0.10; directional
≈ 0.05). Treat the threshold here as an outlier detector rather than a
quality indicator — for actual quality, look at the rollout diff-std
ratio in `result_info["rollout_diff_std_ratio"]` and the per-component
shares in the fit's postamble. The CHANGELOG documents representative
loss values across versions.

#### BFGS steps taken

If `info` records the step count and it equals the documented 1000-step
cap, this is flagged as `warn`: the optimiser may not have fully
converged. Refit with `--n-steps` higher or examine the loss landscape
for plateaus (typically caused by overly aggressive penalty weights in
the loss).

#### Price scale L

Reported informationally. For RELIANCE-like NIFTY large-caps, L typically
lands in the ₹1000–2000 range; values outside this are worth a sanity
check against `features["L"]` from the loader.

---

## Common failure patterns

Some failure modes show up as a *combination* of findings. The patterns
below are worth memorising.

### Constant-drift / frozen-volatility bug regression

| Section | Finding                              | Status |
|---------|--------------------------------------|--------|
| Variance| Predicted/actual std ratio < 0.3     | fail   |
| Variance| Predicted relative range < 0.5%      | fail   |
| Direct. | Directional accuracy ~ 50%           | warn   |
| Accuracy| Theil's U > 1                        | fail   |

If you see this pattern in v0.5+, the multiplicative-form regression
is unlikely (the SPDE itself is now advective), but `predict.py` could
still have `last_sigma` / `last_drift` regressions in the forward
rollout. Re-check `_single_forecast` against the v0.2 fix pattern: σ
must be rolled forward via `lax.scan` of the GARCH recursion, and
drift must use a (windowed) mean rather than `last_drift`.

### Drift-baseline regime mismatch

| Section | Finding                                  | Status |
|---------|------------------------------------------|--------|
| Drift   | Recent-60d drift divergence ≥ 30%        | warn   |
| Accuracy| Bias > 1% (and consistent sign)          | warn   |
| Residual| Lag-1 autocorrelation ≥ 0.3              | warn   |

The model is anchored to a historical mean drift that no longer reflects
current dynamics. Mitigations: shorter fit window for `stochax-fit`, or a
rolling-window drift instead of full-history mean inside `predict.py`.

### Inflated GARCH long-run variance

| Section | Finding                                    | Status |
|---------|--------------------------------------------|--------|
| GARCH   | Unconditional / realised variance > 2.5    | fail   |
| Variance| Predicted/actual std ratio > 3.0           | fail   |
| Calib.  | Final loss > 0.10                          | warn   |

Almost always traces back to `raw_omega` initialisation. The default
since v0.5 is `raw_omega = −11.0` (correct for daily-scale data); if
you see this pattern, verify that the active `GARCHVolatility` is using
the v0.5 default. The persistence-cap (α+β ≤ 0.97, since v0.4) prevents
the unit-root corner that earlier versions could fall into.

### Calibrated against wrong frequency

| Section | Finding                                     | Status |
|---------|---------------------------------------------|--------|
| Drift   | Mean historical drift > 80%                 | fail   |
| GARCH   | Unconditional / realised variance < 0.4     | fail   |
| Residual| Excess kurtosis ≥ 3                         | warn   |

This is the 15-minute-intraday-data-treated-as-daily pitfall documented in
the project memory. If you see this combination, verify the loader is
reading daily bars, not intraday.

---

## Architecture and extension

The module is organised as one section function per check group:

```python
_accuracy_findings(realised)              -> list[Finding]
_directional_findings(realised)           -> list[Finding]
_variance_findings(sim, realised)         -> list[Finding]
_residual_findings(realised)              -> list[Finding]
_garch_findings(garch, df)                -> list[Finding]
_drift_findings(df)                       -> list[Finding]
_calibration_findings(saved)              -> list[Finding]
```

The orchestrator `analyze()` calls each in turn, packs the results into a
`DiagnosticsReport`, optionally writes the markdown, and returns the
report.

### Adding a new finding to an existing section

Append a `Finding(metric, value, status, diagnosis)` to the list returned
by the appropriate `_*_findings` function. The `_fmt` and `_pct` helpers
handle display formatting and NaN/inf safely.

### Adding a new section

1. Add a new `_X_findings(...)` function returning `list[Finding]`.
2. Call it from `analyze()` and assign to `sections["Section Name"]`.

The markdown renderer walks `sections.items()` in insertion order, so the
section appears in the report at whatever position you call it. The
recommendations block automatically picks up any non-`ok` findings from
the new section without further changes.

### Custom thresholds

Thresholds are inline in each `_*_findings` function for readability. If
you need to tune them per-symbol or per-asset-class (e.g. small-caps need
looser MAPE thresholds than large-caps), the cleanest pattern is to add a
`thresholds: dict | None = None` parameter to `analyze()` and thread it
through, with the inline values as fallbacks.

---

## Output schema reference

### `Finding`

| Field      | Type     | Notes                                                      |
|------------|----------|------------------------------------------------------------|
| `metric`   | `str`    | Short name shown in the report (e.g. "MAPE").              |
| `value`    | `str`    | Pre-formatted display string. Use `_fmt` / `_pct` helpers. |
| `status`   | `str`    | `"ok"` / `"warn"` / `"fail"`.                              |
| `diagnosis`| `str`    | One-line cause/remedy. Empty for `ok` rows.                |

### `DiagnosticsReport`

| Field        | Type                       | Notes                                  |
|--------------|----------------------------|----------------------------------------|
| `symbol`     | `str`                      |                                        |
| `n_total`    | `int`                      | Total forecast steps in the sim CSV.   |
| `n_realised` | `int`                      | Steps with non-NaN `actual_price`.     |
| `sections`   | `dict[str, list[Finding]]` | Section name → findings, insertion-ordered. |
| `summary`    | `dict[str, str]`           | Headline metrics for the report header.|

Methods:

- `overall_status() -> str` — worst status across all findings.
- `recommendations() -> list[str]` — flattened non-ok findings as
  actionable items.
- `to_markdown() -> str` — full report as a markdown string.

### Markdown report layout

```
# {symbol} — V3 Diagnostics Report

**Overall:** {glyph} {verdict}

- Forecast steps: ... · realised: ... · forecast-only: ...
- {summary metrics}

## Forecast Accuracy
| Status | Metric | Value | Diagnosis |
| ... |

## Directional Performance
...

(seven sections total)

## Recommendations
- **[Section]** Metric: diagnosis
- ...
```

Section ordering is fixed, finding ordering within each section is fixed.
Consumers can therefore parse reports across runs and diff them for
regression detection.
