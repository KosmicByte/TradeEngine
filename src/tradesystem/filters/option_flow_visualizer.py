
from src.tradesystem.data_fetchers.nse_option_chain_fetcher import fetch_nse_option_chain
from option_flow_analyzer import analyze_option_flow


def display_option_flow():
    df = fetch_nse_option_chain()
    summary, flow_df = analyze_option_flow(df)

    print("\n--- Option Flow Summary ---")
    print(f"Total CE OI: {summary['total_ce_oi']}")
    print(f"Total PE OI: {summary['total_pe_oi']}")
    print(f"PCR: {summary['pcr']}")
    print(f"Bias: {summary['bias']}")

    print("\n--- Key Strike Data (Top 10 Sorted by CE OI) ---")
    display_df = flow_df.sort_values("ce_oi", ascending=False).head(10)
    print(display_df.to_string(index=False))

if __name__ == "__main__":
    display_option_flow()
