#!/usr/bin/env python3
import argparse
from datetime import datetime
import sys
import pandas as pd
from Derivatives import Sensibull

def parse_args():
    p = argparse.ArgumentParser(description="Fetch option-chain with Greeks via Sensibull (no try/except).")
    p.add_argument("--symbol", required=True, help="Underlying (e.g., 'NIFTY 50', 'BANKNIFTY', 'RELIANCE').")
    p.add_argument("--expiry", required=True, help="Expiry YYYY-MM-DD.")
    p.add_argument("--num-lookups", type=int, default=3, help="# of strikes on each side of ATM (default: 3).")
    p.add_argument("--out", default="greeks.csv", help="Output CSV file.")
    p.add_argument("--auto-pick", type=int, default=None, help="Auto-pick Nth candidate (1-based) if multiple.")
    return p.parse_args()

def fetch_all_instruments(sb: Sensibull):
    resp = sb.hit_and_get_data(f"{sb._base_url}/cache/underlying_instruments")
    if not isinstance(resp, dict) or "data" not in resp:
        print("ERROR: Could not fetch instrument list from Sensibull.", file=sys.stderr)
        sys.exit(1)
    return resp["data"]

def fuzzy_candidates(instruments, query: str):
    q = query.strip().lower()
    seen = set()
    out = []
    for it in instruments:
        ts = str(it.get("tradingsymbol", "")).lower()
        sy = str(it.get("symbol", "")).lower()
        nm = str(it.get("name", "")).lower()
        if q in ts or q in sy or q in nm:
            tok = it.get("instrument_token")
            if tok not in seen:
                seen.add(tok)
                out.append(it)
    return out

def pick_candidate(cands, auto_pick: int | None):
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    if auto_pick is not None:
        idx = auto_pick - 1
        if 0 <= idx < len(cands):
            return cands[idx]
        print(f"ERROR: --auto-pick {auto_pick} out of range (1..{len(cands)}).", file=sys.stderr)
        sys.exit(2)
    print("\nMultiple matches found. Pick one:\n")
    for i, it in enumerate(cands, start=1):
        print(f"[{i}] {it.get('tradingsymbol','?'):20}  "
              f"symbol={it.get('symbol','?'):15}  token={it.get('instrument_token','?')}")
    choice = input(f"\nEnter choice [1..{len(cands)}]: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(cands)):
        print("ERROR: Invalid choice.", file=sys.stderr)
        sys.exit(2)
    return cands[int(choice)-1]

def get_per_expiry(sb: Sensibull, token: int):
    resp = sb.hit_and_get_data(f"{sb._base_url}/cache/live_derivative_prices/{token}")
    if not isinstance(resp, dict):
        print("ERROR: Unexpected response fetching live_derivative_prices.", file=sys.stderr)
        sys.exit(1)
    data = resp.get("data", {})
    per = data.get("per_expiry_data", {})
    if not isinstance(per, dict) or not per:
        print("ERROR: No per_expiry_data available for this instrument.", file=sys.stderr)
        sys.exit(1)
    return per

def main():
    args = parse_args()

    # Parse expiry
    expiry_dt = datetime.strptime(args.expiry, "%Y-%m-%d")
    expiry_key = expiry_dt.strftime("%Y-%m-%d")

    sb = Sensibull()

    # Resolve instrument
    exact = sb.search_token(args.symbol)
    if exact:
        ticker_data = exact
    else:
        instruments = fetch_all_instruments(sb)
        cands = fuzzy_candidates(instruments, args.symbol)
        if not cands:
            print(f"ERROR: No instruments matching '{args.symbol}'. "
                  f"Try 'NIFTY 50', 'NIFTY INDEX', 'BANKNIFTY', 'RELIANCE'.", file=sys.stderr)
            sys.exit(2)
        ticker_data = pick_candidate(cands, args.auto_pick)
        if not ticker_data:
            print("ERROR: Could not resolve instrument.", file=sys.stderr)
            sys.exit(2)

    token = ticker_data.get("instrument_token")
    tradingsymbol = ticker_data.get("tradingsymbol")
    if token is None or tradingsymbol is None:
        print("ERROR: Resolved instrument missing token/tradingsymbol.", file=sys.stderr)
        sys.exit(2)

    # Validate expiry BEFORE calling Greeks
    per_exp = get_per_expiry(sb, token)
    available = sorted(per_exp.keys())
    if expiry_key not in per_exp:
        print(f"ERROR: Expiry {expiry_key} not available for {tradingsymbol}.\n"
              f"Available expiries: {', '.join(available)}", file=sys.stderr)
        sys.exit(3)

    # Fetch Greeks (now guaranteed to have this expiry)
    df, atm_strike = sb.get_options_data_with_greeks(
        ticker_data=ticker_data,
        num_look_ups_from_atm=args.num_lookups,
        expiry_date=expiry_dt,
    )

    if df is None or df.empty:
        print("ERROR: Empty DataFrame returned. Check market hours/symbol/expiry.", file=sys.stderr)
        sys.exit(4)

    keep = [c for c in df.columns if c in ("future_price", "strike") or c.startswith("CE.") or c.startswith("PE.")]
    print(f"\nResolved: {tradingsymbol} (token={token})")
    print(f"ATM strike: {atm_strike} | rows: {len(df)}\n")
    print(df[keep].head(10).to_string(index=False))

    df.to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out}")

if __name__ == "__main__":
    main()
