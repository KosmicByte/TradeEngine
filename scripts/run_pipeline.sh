#!/usr/bin/env bash
# scripts/run_pipeline.sh
# Usage: bash scripts/run_pipeline.sh
# Runs simulate → fit → visualize for every symbol in data/

set -euo pipefail

DATA_DIR="data"
PLOTS_DIR="plots"
STEPS_SIM=252
STEPS_FIT=2000
HORIZON=21
RECENT_N=60

mkdir -p "$PLOTS_DIR"

for csv_file in "$DATA_DIR"/*.csv; do
    # Extract symbol name from filename (e.g. data/RELIANCE.csv → RELIANCE)
    symbol=$(basename "$csv_file" .csv)

    echo ""
    echo "════════════════════════════════════════"
    echo "  Processing: $symbol"
    echo "════════════════════════════════════════"

    sim_csv="${symbol}_sim.csv"
    params_pkl="${symbol}_params.pkl"

    # 1. Simulate
    echo "[$symbol] Step 1/3 — Simulating $STEPS_SIM steps..."
    stochax-simulate \
        --symbol "$symbol" \
        --steps "$STEPS_SIM" \
        --output "$sim_csv" \
        --params "$params_pkl" 2>/dev/null \
        || stochax-simulate \
            --symbol "$symbol" \
            --steps "$STEPS_SIM" \
            --output "$sim_csv"

    # 2. Fit
    echo "[$symbol] Step 2/3 — Fitting $STEPS_FIT optimisation steps..."
    stochax-fit \
        --symbol "$symbol" \
        --n-steps "$STEPS_FIT" \
        --output "$params_pkl"

    # 3. Visualize
    echo "[$symbol] Step 3/3 — Generating plots..."
    stochax-visualize \
        --symbol "$symbol" \
        --sim-csv "$sim_csv" \
        --params "$params_pkl" \
        --horizon "$HORIZON" \
        --recent-n "$RECENT_N" \
        --out-dir "$PLOTS_DIR/$symbol"

    echo "[$symbol] ✓ Done → plots saved to $PLOTS_DIR/$symbol/"
done

echo ""
echo "════════════════════════════════════════"
echo "  All symbols complete."
echo "════════════════════════════════════════"