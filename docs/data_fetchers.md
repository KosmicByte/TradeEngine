# Data Fetching — Documentation

---

## What the script does 

It fetches **FII/DII flows**, **India VIX**, **NIFTY spot**, and the **nearest-expiry NIFTY option chain**; computes **near-ATM PCR (±3%)** and a coarse **market regime** label; then **saves four artifacts** per day under `results/YYYY-MM-DD/` and prints a clean console summary. It’s hardened with **cookie warm-up, HTTP retries, jitter**, and **targeted re-fetch** if any piece comes back missing

**Outputs (per day):**

* `results/YYYY-MM-DD/fii_dii.csv`
* `results/YYYY-MM-DD/nifty_option_chain_nearest_expiry.csv`
* `results/YYYY-MM-DD/snapshot.json`
* `results/YYYY-MM-DD/snapshot.txt`&#x20;

---

## Data sources (NSE public endpoints)

* `GET /api/allIndices` — primary for **INDIA VIX** and fallback for **NIFTY spot**.
* `GET /api/fiidiiTradeReact` — **FII/DII** flows (script tolerates dict or list payload).
* `GET /api/option-chain-indices?symbol=NIFTY` — **option chain** (nearest expiry filtered with fallbacks).
* `GET /api/option-chain-indices?symbol=INDIAVIX` — fallback for **VIX** via `underlyingValue`

The script “**warms up**” the session by visiting the homepage and a cheap API to obtain cookies before hitting JSON endpoints. This avoids intermittent `401/403` and empty payloads

---

## Where files are saved

* The script computes repo root from `__file__`, then writes under:

  ```
  <repo-root>/results/<YYYY-MM-DD>/
  ```

  This is done via `REPO_ROOT`, `RESULTS_DIR`, `DATE_DIR` and `Path(...).mkdir(..., exist_ok=True)`

---

## Prerequisites

* Python 3.10+ recommended
* `pandas`, `requests`, `urllib3` (installed already in your venv)

Optional but recommended:

* Run as a **single instance** at a time to reduce rate-limit risk.

---

## How it works — workflow

**A. Setup**

1. Build a `requests.Session` with **HTTPAdapter retries** (connect/read/status).
2. Configure short **per-request timeouts** and a **global alarm** (`RUN_TIMEOUT_SECS`) to guarantee a deterministic finish.
3. Define hard **STATUS\_FORCELIST** (401/403/429/5xx) for retry.
4. Resolve **results/YYYY-MM-DD/** and create directories

**B. Each attempt** (`_attempt_snapshot`)

1. **Warm-up** (homepage → `allIndices`) to set cookies.
2. Fetch **FII/DII**, **NIFTY OC**, **NIFTY spot**, **India VIX** (with small **jitter** between calls).
3. Compute **PCR** near ATM (±3%) and **regime** from VIX.
4. Mark which components are “OK” (OC/spot/VIX present)

**C. Healing pass** (`_heal_missing`)
If anything is missing:

* Re-warm and **re-fetch OC** (then recompute PCR if spot is valid).
* If **spot** is missing, try **indices-first** path explicitly, then OC fallback.
* If **VIX** is missing, re-fetch via indices (tolerant name match) then OC fallback

**D. Fallback and save**

* Up to **3 passes**: attempt → heal → (maybe next attempt).
* Even if still imperfect, the script **always writes** all four files, then prints a summary.
* The alarm is cleared before exit

---

## Calculations & labeling

* **PCR near ATM (±3%)**: sum OI for strikes within ±3% of spot; fallback to entire chain if window empty.
* **Sentiment** from PCR: `<0.8` **Bearish**, `0.8–1.2` **Neutral**, `>1.2` **Bullish**.
* **Market Regime** from VIX (baseline 12):

  * `<max(10, 0.9×baseline)` → **Calm**
  * `>max(16, 1.2×baseline)` → **Stressed**
  * else **Normal**
  * `nan/≤0` → **Unknown**

---

## Console summary (what you’ll see)

* FII/DII head (or `<no data>`)
* OC head (or `<no data>`)
* NIFTY Spot, India VIX, PCR, Sentiment, Regime
* “Saved:” list of the four absolute file paths under the date folder

---

## Running it

From anywhere inside the repo (activate your venv first):

```bash
python src/tradesystem/data_fetchers/fetch_v3.py
```

**Pro tip:** Add a tiny random delay in your scheduler (cron/systemd) to avoid hitting NSE at the same second every day (the script already jitters between internal calls)

---

## Cron / systemd examples

**Cron (runs at 09:10 IST on trading days):**

```cron
# m h dom mon dow  command
40 5 * * 1-5  /path/to/venv/bin/python /path/to/repo/src/tradesystem/data_fetchers/fetch_v3.py >> /path/to/repo/results/runner.log 2>&1
```

**systemd unit** (reliable restarts, logs to journal):

```
[Unit]
Description=NSE Snapshot Fetch

[Service]
Type=oneshot
WorkingDirectory=/path/to/repo
ExecStart=/path/to/venv/bin/python src/tradesystem/data_fetchers/fetch_v3.py
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
```

Timer unit to schedule it daily.

---

## File formats (schemas)

**`fii_dii.csv`** (tidy):

```
category,date,buyValue,sellValue,netValue
DII **,2025-09-09,10422.84,10339.76,83.08
FII/FPI *,2025-09-09,11896.67,9846.21,2050.46
...
```

The script normalizes column names and dates, and coerces numerics safely. Empty results still produce a CSV with headers

**`nifty_option_chain_nearest_expiry.csv`** (nearest expiry; duplicates collapsed by strike):

```
strike,ce_oi,ce_chg_oi,ce_vol,ce_iv,pe_oi,pe_chg_oi,pe_vol,pe_iv
...
```

Nearest expiry selection is robust: nearest ≥ today, else closest past; fallback to expiry with **most rows**, then **all rows** if needed

**`snapshot.json`**:

```json
{
  "timestamp": "2025-09-10T10:30:13",
  "nifty_spot": 20012.55,
  "india_vix": 10.69,
  "pcr_near_atm": 0.774,
  "market_sentiment": "Bearish",
  "market_regime": "Calm",
  "files": {
    "fii_dii_csv": "results/2025-09-10/fii_dii.csv",
    "option_chain_csv": "results/2025-09-10/nifty_option_chain_nearest_expiry.csv"
  }
}
```

**`snapshot.txt`** mirrors the console summary

---

## Troubleshooting

**Issue: VIX/spot/OC shows `nan` or `<no data>` intermittently**

* That’s the NSE wall. The script already mitigates via warm-up, retries, jitter, and healing passes. If it persists:

  * Bump `attempts` from `3` → `4–5`.
  * Increase `RUN_TIMEOUT_SECS` from `60` → `90`.
  * Ensure only one instance runs at a time

**Issue: Script “hangs”**

* Protected by the global alarm (`signal.alarm`). If you see a `[FATAL] Global run timeout exceeded`, raise the timeout a bit or reduce your network backoff

**Issue: `403`/`429` in logs**

* You’re rate-limited. The session warm-up and backoff help. Consider shifting the schedule a few minutes or randomizing start time

---

## Extending it (clear seams)

* **Add Bank Nifty**: replicate OC fetch for `BANKNIFTY`; save parallel CSV & PCR.
* **Historical snapshots**: append daily JSON/CSV into a Parquet dataset for fast analysis.
* **Max Pain**: compute at run-time from the chain and include in `snapshot.json`.
* **Alerting**: if VIX spikes, or PCR crosses thresholds, emit a Slack/Telegram message.
* **Unit tests**: inject sample payloads into fetchers; verify parsing & fallbacks deterministically

---

## Guardrails & design choices (why this way)

* **Cookie warm-up** is necessary; hitting JSON APIs cold often yields empty bodies or 403s.
* **Two-layer retries** (HTTPAdapter + manual) with **jitter** spreads load and avoids synchronized retry storms.
* **Nearest-expiry with fallbacks** prevents “empty chain” when date formats drift.
* **Always write files** ensures downstream steps (like your analyzer) don’t break on missing artifacts.
* **Global timeout** keeps CI/cron predictable

---

## Daily workflow 

1. **Fetch** (this script) → drops four files into `results/YYYY-MM-DD/`.
2. **Analyze** (`analyze_option_chain.py`) → reads the day’s CSV/JSON in the same folder and writes plots to `results/YYYY-MM-DD/analysis/`.
3. **Review** the text snapshot and plots; push results or ship alerts if thresholds are hit 


---

# Analyzing the Option Chain — Documentation

**What**

* Loads `nifty_option_chain_nearest_expiry.csv` (nearest expiry only, already de-duplicated by strike).
* Safely coerces numerics; fills missing analytical columns (`*_iv`, `*_chg_oi`, `*_vol`) with `0.0`.
* Optionally loads `snapshot.json` for **spot** and **VIX**.
* Adds derived columns:

  * `total_oi = ce_oi + pe_oi`
  * `oi_diff = pe_oi − ce_oi` (support minus resistance)
  * `chg_oi_diff = pe_chg_oi − ce_chg_oi`
  * `distance_from_spot_pct` if spot is known.&#x20;

**Why**

* Ensures robust plots and metrics even if some fields are missing or malformed.
* `oi_diff`/`chg_oi_diff` give at-a-glance tilt toward support/resistance and where fresh positions are concentrating.&#x20;

**How**

* `safe_float()` guards parsing; empty window falls back to whole chain where applicable.
* Sorting by `strike` makes all plots orderly along the x-axis.&#x20;

---

# Chart 1 — OI Support vs Resistance (`oi_support_resistance.png`)

**What**

* Side-by-side bars per strike: **CE OI** (resistance) vs **PE OI** (support).&#x20;

**Why**

* **PE OI highs** ≈ levels where put writers defend (support).
* **CE OI highs** ≈ levels where call writers cap moves (resistance).
* Quickest visual to spot “walls” in the market.&#x20;

**How**

* Uses bar offsets based on median strike step so bars don’t overlap.
* Grid on Y aids reading relative magnitudes.&#x20;

**Interpretation tips**

* Strongest **support** = top **PE OI** clusters.
* Strongest **resistance** = top **CE OI** clusters.
* Tight clusters around spot → range-bound expectations; wide separation → skewed risk.&#x20;

---

# Chart 2 — OI Build-up (Change in OI) (`oi_change_build_up.png`)

**What**

* Bars of **ΔOI** per strike: `ce_chg_oi` and `pe_chg_oi`.&#x20;

**Why**

* Absolute OI shows stockpiles; **ΔOI** shows **today’s positioning**.
* Rising **PE ΔOI** near/under spot → bullish defense emerging.
* Rising **CE ΔOI** near/over spot → bearish capping intensifying.&#x20;

**How**

* Same side-by-side approach as Chart 1; signed values (can be negative).&#x20;

**Interpretation tips**

* Look for **confluence**: a strike with high OI **and** high ΔOI is a live level that traders are defending/attacking now.&#x20;

---

# Chart 3 — IV Skew (`iv_skew.png`)

**What**

* Line plot of **CE IV** and **PE IV** across strikes.&#x20;

**Why**

* Implied Volatility = priced-in uncertainty.
* **Skew** (puts pricier than calls or vice versa) reveals asymmetry in tail risk.
* Local IV spikes flag strikes where traders expect jumpy behavior.&#x20;

**How**

* Simple `plot()` against strikes; no smoothing—keeps true surface shape.&#x20;

**Interpretation tips**

* **PE IV > CE IV** commonly (downside protection demand).
* A sudden hump near a strike suggests event risk or localized positioning.&#x20;

---

# Chart 4 — PCR by Window (`pcr_by_window.png`)

**What**

* PCR computed near ATM over multiple windows: **±2%**, **±3%**, **±5%** around spot.&#x20;

**Why**

* PCR is sensitive to the strikes included. Seeing PCR across windows checks **robustness**:

  * If PCR \~ stable across windows → sentiment is broad-based.
  * If PCR flips wildly → sentiment is **localized** to a narrow band.&#x20;

**How**

* `compute_pcr_window(df, spot, pct_window)` sums **PE OI** and **CE OI** only within the window; if empty, falls back to the entire chain; returns `nan` if **spot** not available.&#x20;

**Interpretation tips**

* Heuristic:

  * `< 0.8` Bearish, `0.8–1.2` Neutral, `> 1.2` Bullish (your fetcher uses the same thresholds).
* Check if small-window PCR disagrees with large-window PCR → likely conflicting positioning on either side of spot.&#x20;

---

# Chart 5 — Max Pain (`max_pain.png`)

**What**

* **Total pain curve** vs hypothetical settlement `S`, with the **minimum** marked = **Max Pain strike**.
* Pain proxy:

  $$
  \text{pain}(S) = \sum_K \Big( \text{CE_OI}(K)\,\max(0, S-K) \;+\; \text{PE_OI}(K)\,\max(0, K-S) \Big)
  $$

  where `K` = strike, `CE_OI(K)` = Call OI at strike `K`, `PE_OI(K)` = Put OI at strike `K`.

**Why**

* Max Pain is the strike where option writers (collectively) lose the **least**; markets often **gravitate** toward it near expiry (not a law, just a tendency).
* It complements PCR/OI views with an **expiry magnet** hypothesis.&#x20;

**How**

* Evaluates `pain(S)` on a **grid** consisting of all strikes + midpoints to smooth the curve; marks the argmin.
* Outputs the full curve to `max_pain_curve.csv` as well.&#x20;

**Interpretation tips**

* If Max Pain is far from spot early in the series, it’s less actionable. Convergence into expiry is the useful signal.
* Always balance with **ΔOI trends** (Chart 2) to see if positioning is fighting or aligning with that magnet.&#x20;

---

# CSV Outputs (what to read quickly)

1. **`summary_per_strike.csv`**

   * All base + derived columns (`total_oi`, `oi_diff`, `chg_oi_diff`, `distance_from_spot_pct`).
   * Use it for programmatic filters (e.g., strikes within ±2% with top `oi_diff`).&#x20;

2. **`top_support_resistance.csv`**

   * Top-N **PE OI** (support) and **CE OI** (resistance), concatenated for quick review.
   * Directly answers: “What are the biggest walls right now?”&#x20;

3. **`top_build_up.csv`**

   * Top-N **PE ΔOI** and **CE ΔOI** (fresh build-ups).
   * This is your **today** signal; combine with (2) for **where** traders are reinforcing defenses.&#x20;

4. **`max_pain_curve.csv`** (when available)

   * `S`, `total_pain`—for backtesting convergence or custom visualizations.&#x20;

---

# Assumptions & Caveats (read this)

* All analytics are on the **nearest expiry only**. Wider expiries can tell a different story.
* **OI ≠ direction** by itself. Writers dominate many markets; use ΔOI and price context.
* **IV** is noisy intraday; comparing across data snapshots can be misleading if collection times differ.
* **PCR thresholds** are rule-of-thumb. You can tune them or analyze PCR **change** over time instead.
* **Max Pain** is a heuristic, not a target. It’s more valuable as **expiry approaches** and when OI is concentrated.&#x20;

---

# A practical reading workflow

1. Open **OI Support/Resistance** → mark the top 2–3 support and resistance shelves.
2. Check **ΔOI Build-up** → are those shelves being **reinforced** today?
3. Glance at **IV Skew** → any lopsided fear pockets?
4. Look at **PCR by window** → is the sentiment robust around spot?
5. Note **Max Pain** → is it drifting toward spot as expiry nears?
6. Cross-check with **spot trend** and **VIX** (from `snapshot.json`) before acting.&#x20;

---
 