
import pandas as pd

def analyze_option_flow(df):
    df = df.copy()
    df["ce_oi_ratio"] = df["ce_chg_oi"] / (df["ce_oi"] + 1)
    df["pe_oi_ratio"] = df["pe_chg_oi"] / (df["pe_oi"] + 1)
    df["ce_vol_ratio"] = df["ce_vol"] / (df["ce_oi"] + 1)
    df["pe_vol_ratio"] = df["pe_vol"] / (df["pe_oi"] + 1)

    # Overall Positioning
    total_ce_oi = df["ce_oi"].sum()
    total_pe_oi = df["pe_oi"].sum()
    pcr = total_pe_oi / (total_ce_oi + 1)

    # Directional Score
    bullish_strikes = df[df["pe_chg_oi"] > df["ce_chg_oi"]]
    bearish_strikes = df[df["ce_chg_oi"] > df["pe_chg_oi"]]

    if pcr > 1.1 and len(bullish_strikes) > len(bearish_strikes):
        bias = "Bullish"
    elif pcr < 0.9 and len(bearish_strikes) > len(bullish_strikes):
        bias = "Bearish"
    else:
        bias = "Neutral"

    summary = {
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
        "pcr": round(pcr, 2),
        "bias": bias
    }
    return summary, df[["strike", "ce_oi", "ce_chg_oi", "pe_oi", "pe_chg_oi", "ce_vol", "pe_vol"]]

# Example:
# from nse_option_chain_fetcher import fetch_nse_option_chain
# df = fetch_nse_option_chain()
# summary, flow_df = analyze_option_flow(df)
# print(summary)
