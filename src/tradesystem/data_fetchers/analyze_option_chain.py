"""
Analyze NIFTY nearest-expiry option chain and plot meaningful insights.

Inputs
------
- CSV (required): tidy chain with columns:
  strike, ce_oi, ce_chg_oi, ce_vol, ce_iv, pe_oi, pe_chg_oi, pe_vol, pe_iv
- snapshot.json (optional): for spot and VIX (keys: nifty_spot, india_vix)

Outputs (in project_root/results/yyyy-mm-dd/analysis/)
------------------------
- PNGs:
  1. oi_support_resistance.png      (PE/CE OI by strike)
  2. oi_change_build_up.png         (PE/CE ΔOI by strike)
  3. iv_skew.png                    (PE/CE IV by strike)
  4. pcr_by_window.png              (PCR across ±window% windows)
  5. max_pain.png                   (Total pain vs strike; min marked)
- CSVs:
  - summary_per_strike.csv          (computed metrics per strike)
  - top_support_resistance.csv      (top-N PE/CE OI)
  - top_build_up.csv                (top-N PE/CE ΔOI)
- Console summary with key levels

Usage
-----
python analyze_option_chain.py --csv nifty_option_chain_nearest_expiry.csv --snapshot snapshot.json \
    --window 0.03 --top 5 --outdir analysis
"""

from __future__ import annotations
import argparse, json, os
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TODAY = dt.date.today().isoformat()
REPO_ROOT = Path(__file__).resolve().parents[3]  # go from src/tradesystem/data_fetchers → repo root
DATE_DIR = REPO_ROOT / "results" / TODAY
DATE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------

def ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)

def load_snapshot(path: str | None) -> tuple[float | None, float | None]:
    if not path or not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            js = json.load(f)
        spot = js.get("nifty_spot", None)
        vix  = js.get("india_vix", None)
        return (float(spot) if spot is not None else None,
                float(vix) if vix is not None else None)
    except Exception:
        return None, None

def safe_float(x):
    try:
        v = float(x)
        return 0.0 if np.isnan(v) else v
    except Exception:
        return 0.0

def compute_pcr_window(df: pd.DataFrame, spot: float, pct_window: float = 0.03) -> float:
    if not np.isfinite(spot) or spot <= 0 or df.empty:
        return np.nan
    lo, hi = spot * (1 - pct_window), spot * (1 + pct_window)
    sub = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
    if sub.empty:
        sub = df
    ce = sub["ce_oi"].sum()
    pe = sub["pe_oi"].sum()
    return (pe / ce) if ce > 0 else np.nan

def compute_max_pain(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """
    Approximate max pain:
    pain(S) = sum_K [ CE_OI(K) * max(0, S - K) + PE_OI(K) * max(0, K - S) ]
    """
    if df.empty:
        return np.nan, pd.DataFrame()
    strikes = df["strike"].values
    ce_oi = df["ce_oi"].values
    pe_oi = df["pe_oi"].values

    # Evaluate at each strike (plus optional midpoints for smoother curve)
    grid = np.unique(np.concatenate([strikes, (strikes[:-1] + strikes[1:]) / 2.0]))
    pain = []
    for S in grid:
        pain_S = np.sum(ce_oi * np.maximum(0.0, S - strikes)) + np.sum(pe_oi * np.maximum(0.0, strikes - S))
        pain.append(pain_S)
    pain = np.array(pain)
    idx = int(np.argmin(pain))
    return float(grid[idx]), pd.DataFrame({"S": grid, "total_pain": pain})

# ---------- plotting (matplotlib only; no styles/colors forced) ----------

def plot_oi(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(12,6))
    # offset so bars are visible side-by-side
    plt.bar(df["strike"]-0.4*(df["strike"].diff().median() or 50), df["ce_oi"], width=(df["strike"].diff().median() or 50)*0.8, label="CE OI", alpha=0.6)
    plt.bar(df["strike"]+0.4*(df["strike"].diff().median() or 50), df["pe_oi"], width=(df["strike"].diff().median() or 50)*0.8, label="PE OI", alpha=0.6)
    plt.title("Open Interest by Strike (Support vs Resistance)")
    plt.xlabel("Strike")
    plt.ylabel("Open Interest")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "oi_support_resistance.png"), dpi=160)
    plt.close()

def plot_oi_change(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(12,6))
    plt.bar(df["strike"]-0.4*(df["strike"].diff().median() or 50), df["ce_chg_oi"], width=(df["strike"].diff().median() or 50)*0.8, label="CE ΔOI", alpha=0.7)
    plt.bar(df["strike"]+0.4*(df["strike"].diff().median() or 50), df["pe_chg_oi"], width=(df["strike"].diff().median() or 50)*0.8, label="PE ΔOI", alpha=0.7)
    plt.title("Change in Open Interest (Build-up)")
    plt.xlabel("Strike")
    plt.ylabel("Change in OI")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "oi_change_build_up.png"), dpi=160)
    plt.close()

def plot_iv_skew(df: pd.DataFrame, outdir: str):
    plt.figure(figsize=(12,6))
    plt.plot(df["strike"], df["ce_iv"], marker="o", label="CE IV")
    plt.plot(df["strike"], df["pe_iv"], marker="o", label="PE IV")
    plt.title("IV Skew (Nearest Expiry)")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "iv_skew.png"), dpi=160)
    plt.close()

def plot_pcr_curve(df: pd.DataFrame, spot: float | None, outdir: str, windows=(0.02, 0.03, 0.05)):
    plt.figure(figsize=(10,5))
    if spot and np.isfinite(spot):
        vals = []
        for w in windows:
            p = compute_pcr_window(df, spot, w)
            vals.append((w, p))
            plt.plot([w*100], [p], marker="o", label=f"±{int(w*100)}% → PCR={p:.3f}" if np.isfinite(p) else f"±{int(w*100)}% → PCR=nan")
        plt.title(f"PCR near ATM @ Spot={spot:.2f}")
        plt.xlabel("Window (±%)")
        plt.ylabel("PCR")
        plt.grid(True, alpha=0.5)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Spot not available → PCR curve skipped", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pcr_by_window.png"), dpi=160)
    plt.close()

def plot_max_pain(pain_df: pd.DataFrame, max_pain_strike: float, outdir: str):
    if pain_df.empty:
        return
    plt.figure(figsize=(12,5))
    plt.plot(pain_df["S"], pain_df["total_pain"])
    plt.axvline(max_pain_strike, linestyle="--")
    plt.title(f"Max Pain (minimum total loss) @ {max_pain_strike:.2f}")
    plt.xlabel("Settlement Price (S)")
    plt.ylabel("Total Pain (proxy)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "max_pain.png"), dpi=160)
    plt.close()

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Analyze NIFTY nearest-expiry option chain.")
    ap.add_argument(
        "--csv",
        default=DATE_DIR / "nifty_option_chain_nearest_expiry.csv",
        help="Path to option chain CSV (default: results/YYYY-MM-DD/nifty_option_chain_nearest_expiry.csv)",
    )
    ap.add_argument(
        "--snapshot",
        default=DATE_DIR / "snapshot.json",
        help="Optional snapshot.json for spot/VIX "
             "(default: results/YYYY-MM-DD/snapshot.json)",
    )
    ap.add_argument("--window", type=float, default=0.03,
                    help="PCR window (± fraction around spot). Default 0.03.")
    ap.add_argument("--top", type=int, default=5,
                    help="Top-N strikes to list for support/resistance/build-up.")
    ap.add_argument(
        "--outdir",
        default=DATE_DIR / "analysis",
        help="Output directory for plots/CSVs (default: results/YYYY-MM-DD/analysis).",
    )

    args = ap.parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Load chain
    df = pd.read_csv(args.csv)
    # Sanity: fill missing columns if absent
    for col in ["ce_iv","pe_iv","ce_chg_oi","pe_chg_oi","ce_vol","pe_vol"]:
        if col not in df.columns:
            df[col] = 0.0
    # Clean types
    for col in ["strike","ce_oi","pe_oi","ce_chg_oi","pe_chg_oi","ce_iv","pe_iv","ce_vol","pe_vol"]:
        if col in df.columns:
            df[col] = df[col].map(safe_float)
    df = df.sort_values("strike").reset_index(drop=True)

    # Load snapshot (spot, VIX) if available
    spot, vix = load_snapshot(args.snapshot)

    # Per-strike summary (add totals & ratios)
    out = df.copy()
    out["total_oi"] = out["ce_oi"] + out["pe_oi"]
    out["oi_diff"]  = out["pe_oi"] - out["ce_oi"]      # +ve → more support
    out["chg_oi_diff"] = out["pe_chg_oi"] - out["ce_chg_oi"]
    if spot and np.isfinite(spot):
        out["distance_from_spot_pct"] = (out["strike"] - spot) / spot * 100.0
    else:
        out["distance_from_spot_pct"] = np.nan

    # PCR (single window)
    pcr = compute_pcr_window(out, spot, args.window) if (spot and np.isfinite(spot)) else np.nan

    # Max pain
    mp_strike, pain_df = compute_max_pain(out)

    # Top lists
    topN = args.top
    top_support = out.sort_values("pe_oi", ascending=False).head(topN).assign(level="Support (PE OI)")
    top_resist  = out.sort_values("ce_oi", ascending=False).head(topN).assign(level="Resistance (CE OI)")
    top_build_pe = out.sort_values("pe_chg_oi", ascending=False).head(topN).assign(level="PE Build-up (ΔOI)")
    top_build_ce = out.sort_values("ce_chg_oi", ascending=False).head(topN).assign(level="CE Build-up (ΔOI)")

    # Save CSVs
    out.to_csv(os.path.join(outdir, "summary_per_strike.csv"), index=False)
    pd.concat([top_support, top_resist], ignore_index=True)\
        .to_csv(os.path.join(outdir, "top_support_resistance.csv"), index=False)
    pd.concat([top_build_pe, top_build_ce], ignore_index=True)\
        .to_csv(os.path.join(outdir, "top_build_up.csv"), index=False)
    if not pain_df.empty:
        pain_df.to_csv(os.path.join(outdir, "max_pain_curve.csv"), index=False)

    # Plots
    plot_oi(out, outdir)
    plot_oi_change(out, outdir)
    plot_iv_skew(out, outdir)
    plot_pcr_curve(out, spot, outdir, windows=(0.02, 0.03, 0.05))
    plot_max_pain(pain_df, mp_strike, outdir)

    # Console summary
    print("\n=== Snapshot ===")
    if spot and np.isfinite(spot):
        print(f"Spot: {spot:.2f}")
    else:
        print("Spot: NA")
    if vix and np.isfinite(vix):
        print(f"India VIX: {vix:.2f}")
    else:
        print("India VIX: NA")

    if np.isfinite(pcr):
        print(f"PCR (±{int(args.window*100)}%): {pcr:.3f}")
    else:
        print(f"PCR (±{int(args.window*100)}%): NA")

    print("\nTop Support (PE OI):")
    print(top_support[["strike","pe_oi"]].to_string(index=False))

    print("\nTop Resistance (CE OI):")
    print(top_resist[["strike","ce_oi"]].to_string(index=False))

    print("\nTop Build-up (ΔOI):")
    print("PE:")
    print(top_build_pe[["strike","pe_chg_oi"]].to_string(index=False))
    print("\nCE:")
    print(top_build_ce[["strike","ce_chg_oi"]].to_string(index=False))

    if np.isfinite(mp_strike):
        print(f"\nMax Pain Strike: {mp_strike:.2f}")
    else:
        print("\nMax Pain Strike: NA")

    print(f"\nSaved plots & tables → {os.path.abspath(outdir)}\n")

if __name__ == "__main__":
    main()