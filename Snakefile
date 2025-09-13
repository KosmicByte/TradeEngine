# Snakefile
# Pipeline: fetch market snapshot -> analyze nearest-expiry option chain
# Assumes this Snakefile sits at the REPO ROOT.
# Script paths below follow your code's expected location.

import datetime as dt
from pathlib import Path

# --- Config (override via `--config window=0.03 top=5`) ---
window = float(config.get("window", 0.03))   # PCR window for analysis
top    = int(config.get("top", 5))           # Top-N strikes lists

# --- Date & directories (must match scripts' behavior) ---
TODAY = dt.date.today().isoformat()
RESULTS_DIR = Path("results") / TODAY
ANALYSIS_DIR = RESULTS_DIR / "analysis"

# --- Script locations (as implied by parents[3] in your files) ---
FETCH_SCRIPT   = "src/tradesystem/data_fetchers/fetch_v3.py"
ANALYZE_SCRIPT = "src/tradesystem/data_fetchers/analyze_option_chain.py"

# --- Files produced by fetch_v3.py ---
FII_DII_CSV   = RESULTS_DIR / "fii_dii.csv"
OC_CSV        = RESULTS_DIR / "nifty_option_chain_nearest_expiry.csv"
SNAPSHOT_JSON = RESULTS_DIR / "snapshot.json"
SNAPSHOT_TXT  = RESULTS_DIR / "snapshot.txt"

# --- Files produced by analyze_option_chain.py (documented in the script) ---
AN_PNG_OI        = ANALYSIS_DIR / "oi_support_resistance.png"
AN_PNG_OI_CHG    = ANALYSIS_DIR / "oi_change_build_up.png"
AN_PNG_IV_SKEW   = ANALYSIS_DIR / "iv_skew.png"
AN_PNG_PCR       = ANALYSIS_DIR / "pcr_by_window.png"
AN_PNG_MAXPAIN   = ANALYSIS_DIR / "max_pain.png"

AN_CSV_SUMMARY   = ANALYSIS_DIR / "summary_per_strike.csv"
AN_CSV_TOP_OI    = ANALYSIS_DIR / "top_support_resistance.csv"
AN_CSV_TOP_BUILD = ANALYSIS_DIR / "top_build_up.csv"
AN_CSV_MAXPAIN   = ANALYSIS_DIR / "max_pain_curve.csv"  # script writes only if available

# Some runs may skip max_pain_curve.csv if pain_df is empty; list it as "temp"
# so Snakemake won’t fail if it’s missing.
temp_optional = [AN_CSV_MAXPAIN]

rule all:
    input:
        # Fetch outputs
        FII_DII_CSV,
        OC_CSV,
        SNAPSHOT_JSON,
        SNAPSHOT_TXT,
        # Analysis PNGs
        AN_PNG_OI,
        AN_PNG_OI_CHG,
        AN_PNG_IV_SKEW,
        AN_PNG_PCR,
        AN_PNG_MAXPAIN,
        # Analysis CSVs
        AN_CSV_SUMMARY,
        AN_CSV_TOP_OI,
        AN_CSV_TOP_BUILD,
        # Optional curve (don’t require strictly)
        *temp_optional

rule fetch_market_snapshot:
    """
    Pull FII/DII, NIFTY spot, India VIX, and nearest-expiry option chain.
    Writes into results/YYYY-MM-DD/.
    """
    output:
        fii=FII_DII_CSV,
        oc=OC_CSV,
        snap_json=SNAPSHOT_JSON,
        snap_txt=SNAPSHOT_TXT
    # Optional: enable conda/pip env here if you want
    shell:
        "python {FETCH_SCRIPT}"

rule analyze_option_chain:
    """
    Analyze the fetched chain & snapshot; produce plots and summary CSVs.
    """
    input:
        oc=OC_CSV,
        snap=SNAPSHOT_JSON
    output:
        png_oi=AN_PNG_OI,
        png_oi_chg=AN_PNG_OI_CHG,
        png_iv=AN_PNG_IV_SKEW,
        png_pcr=AN_PNG_PCR,
        png_maxpain=AN_PNG_MAXPAIN,
        csv_summary=AN_CSV_SUMMARY,
        csv_top_oi=AN_CSV_TOP_OI,
        csv_top_build=AN_CSV_TOP_BUILD,
        csv_maxpain=temp(AN_CSV_MAXPAIN)
    params:
        window=window,
        top=top,
        outdir=str(ANALYSIS_DIR)
    shell:
        (
            "python {ANALYZE_SCRIPT} "
            "--csv {input.oc} "
            "--snapshot {input.snap} "
            "--window {params.window} "
            "--top {params.top} "
            "--outdir {params.outdir}"
        )