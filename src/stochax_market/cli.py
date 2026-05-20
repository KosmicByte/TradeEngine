"""Typer CLI for stochax-market: simulate, fit, predict, merge, diagnose, visualize, and export commands."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()

# --- simulate CLI ---

simulate_app = typer.Typer(name="stochax-simulate", add_completion=False)


@simulate_app.callback(invoke_without_command=True)
def simulate_main(
    symbol: str = typer.Option("RELIANCE", "--symbol", "-s", help="Stock symbol"),
    steps: int = typer.Option(252, "--steps", "-n", help="Number of timesteps"),
    params:  Path = typer.Option(None,         help="Path to fitted params.pkl"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output CSV path"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    drift_window: int | None = typer.Option(
        None, "--drift-window", "-d",
        help="Use only the last N historical timesteps for the forward-drift "
             "mean. Useful when diagnostics flags a regime shift (e.g. recent "
             "drift diverges from full-history mean). Try 126 (~6 months) or "
             "252 (~1 year). Default: full history.",
    ),
):
    """Run SPDE simulation for a NIFTY50 stock."""
    from stochax_market.simulate import simulate

    console.print(f"[bold blue]Simulating {symbol} for {steps} steps...[/bold blue]")
    result = simulate(
        symbol,
        n_steps      = steps,
        output_path  = output,
        seed         = seed,
        params_path  = params,
        drift_window = drift_window,
    )

    table = Table(title=f"Simulation Results: {symbol}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Steps simulated", str(result["n_steps"]))
    table.add_row("Trajectory shape", str(result["trajectory_shape"]))
    table.add_row("Output file", result["output_path"])
    console.print(table)
    console.print("[bold green]✓ Simulation complete[/bold green]")


# --- fit CLI ---

fit_app = typer.Typer(name="stochax-fit", add_completion=False)


@fit_app.callback(invoke_without_command=True)
def fit_main(
    symbol: str = typer.Option("RELIANCE", "--symbol", "-s", help="Stock symbol"),
    n_steps: int = typer.Option(
        1000, "--n-steps", "-n",
        help="BFGS optimisation steps (the cap is enforced by Optimistix; "
             "fit may terminate earlier on convergence)",
    ),
    training_window: int | None = typer.Option(
        None, "--training-window", "-w",
        help="Restrict fitting to the last N timesteps of history. "
             "Default: full history. The old 50-timestep silent cap is gone; "
             "use this if you want short-window experiments.",
    ),
    output: str = typer.Option("params.pkl", "--output", "-o", help="Output params path"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Suppress the pre-fit / post-fit diagnostic blocks.",
    ),
):
    """Calibrate SPDE + GARCH parameters for a NIFTY50 stock."""
    import jax

    from stochax_market.calibration.fit import fit
    from stochax_market.data.features import encode_features
    from stochax_market.data.loader import load_stock
    from stochax_market.model.spde import SPDEStepper
    from stochax_market.model.volatility import GARCHVolatility

    console.print(f"[bold blue]Fitting {symbol} ({n_steps} BFGS steps)...[/bold blue]")

    df = load_stock(symbol)
    features = encode_features(df)

    spde = SPDEStepper()
    garch = GARCHVolatility()
    key = jax.random.key(seed)

    fitted_spde, fitted_garch, info = fit(
        spde, garch, features,
        n_steps         = n_steps,
        training_window = training_window,
        key             = key,
        verbose         = not quiet,
    )

    L = float(features["L"])
    with open(output, "wb") as f:
        pickle.dump(
            {
                "spde":      fitted_spde,
                "garch":     fitted_garch,
                "L":         L,
                "loss_info": info,
            },
            f,
        )

    table = Table(title=f"Calibration Results: {symbol}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in info.items():
        # Compact float formatting for headline numbers
        if isinstance(v, float):
            value_str = f"{v:.6g}"
        else:
            value_str = str(v)
        table.add_row(str(k), value_str)
    table.add_row("Output file", output)
    console.print(table)
    if "final_loss" in info:
        console.print(f"[bold yellow]Final loss: {info['final_loss']:.6f}[/bold yellow]")
    console.print("[bold green]✓ Calibration complete[/bold green]")


# --- predict CLI ---

predict_app = typer.Typer(name="stochax-predict", add_completion=False)


@predict_app.callback(invoke_without_command=True)
def predict_main(
    symbol: str = typer.Option("RELIANCE", "--symbol", "-s", help="Stock symbol"),
    horizon: int = typer.Option(5, "--horizon", "-h", help="Forecast horizon (days)"),
    params: str = typer.Option("params.pkl", "--params", "-p", help="Fitted params path"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    samples: int = typer.Option(100, "--samples", "-k", help="Monte Carlo samples"),
    drift_window: int | None = typer.Option(
        None, "--drift-window", "-d",
        help="Use only the last N historical timesteps for the mean-drift "
             "baseline. Useful for regime-shifted markets (e.g. when "
             "diagnostics flags 'recent drift diverges from historical "
             "mean'). Try 126 (~6 months) or 252 (~1 year). "
             "Default: full history.",
    ),
):
    """Predict future prices for a NIFTY50 stock."""
    from stochax_market.predict import predict

    console.print(
        f"[bold blue]Predicting {symbol} {horizon} steps ahead "
        f"({samples} MC samples)...[/bold blue]"
    )

    result = predict(
        symbol,
        horizon      = horizon,
        params_path  = params,
        seed         = seed,
        n_samples    = samples,
        drift_window = drift_window,
    )

    table = Table(title=f"Predictions: {symbol} ({horizon}-day horizon)")
    table.add_column("Day", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("5% CI", style="yellow")
    table.add_column("95% CI", style="yellow")

    for i in range(horizon):
        table.add_row(
            str(i + 1),
            f"{float(result['mean_prediction'][i]):.2f}",
            f"{float(result['lower_ci'][i]):.2f}",
            f"{float(result['upper_ci'][i]):.2f}",
        )

    console.print(table)
    console.print("[bold green]✓ Prediction complete[/bold green]")


# --- merge CLI ---

merge_app = typer.Typer(name="stochax-merge", add_completion=False)


@merge_app.callback(invoke_without_command=True)
def merge_main(
    sim_csv:    Path        = typer.Option(..., "--sim-csv",    help="Path to simulate output CSV (cols: step, predicted_price)"),
    symbol:     str         = typer.Option(..., "--symbol", "-s", help="Stock symbol e.g. RELIANCE"),
    start_date: str | None  = typer.Option(None, "--start-date", help="Date for forecast step 0 (YYYY-MM-DD or DD/MM/YY); defaults to next BDay after last historical date"),
    out:        Path | None = typer.Option(None, "--out", "-o", help="Output CSV path; defaults to overwriting --sim-csv in place"),
):
    """Merge simulate predictions with realised actuals from V1.1.0 historical data."""
    import pandas as pd

    from stochax_market.merge import merge_predicted_with_actuals

    console.print(f"[bold blue]Merging {sim_csv.name} with {symbol} actuals...[/bold blue]")

    out_path = merge_predicted_with_actuals(
        sim_csv    = sim_csv,
        symbol     = symbol,
        start_date = start_date,
        out        = out,
    )

    merged    = pd.read_csv(out_path)
    n_total   = len(merged)
    n_filled  = int(merged["actual_price"].notna().sum())
    n_pending = n_total - n_filled

    table = Table(title=f"Merge Results: {symbol}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="green")
    table.add_row("Output file",      str(out_path))
    table.add_row("Total steps",      str(n_total))
    table.add_row("Date range",       f"{merged['date'].iloc[0]} → {merged['date'].iloc[-1]}")
    table.add_row("Actuals filled",   str(n_filled))
    table.add_row("Forecast-only",    str(n_pending))
    console.print(table)
    console.print("[bold green]✓ Merge complete[/bold green]")


# --- diagnose CLI ---

diagnose_app = typer.Typer(name="stochax-diagnose", add_completion=False)


@diagnose_app.callback(invoke_without_command=True)
def diagnose_main(
    symbol:  str         = typer.Option(..., "--symbol", "-s", help="Stock symbol e.g. RELIANCE"),
    sim_csv: Path        = typer.Option(..., "--sim-csv",      help="Merged sim CSV from stochax-merge (4 cols)"),
    params:  Path | None = typer.Option(None, "--params", "-p", help="Optional fitted params.pkl for GARCH + calibration sections"),
    out:     Path | None = typer.Option(None, "--out", "-o",    help="Output Markdown report path; if omitted, prints to stdout"),
):
    """Run V3 diagnostics and emit a Markdown report on forecast quality."""
    from stochax_market.diagnostics import analyze

    console.print(f"[bold blue]Analysing {symbol} forecast results...[/bold blue]")

    report = analyze(
        symbol      = symbol,
        sim_csv     = sim_csv,
        params_path = params,
        out_md      = out,
    )

    overall       = report.overall_status()
    overall_glyph = {"ok": "[green]✓[/green]", "warn": "[yellow]⚠[/yellow]", "fail": "[red]✗[/red]"}[overall]
    overall_word  = {"ok": "Healthy", "warn": "Issues detected", "fail": "Significant problems"}[overall]

    table = Table(title=f"Diagnostics Summary: {symbol}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="green")
    table.add_row("Overall",       f"{overall_glyph} {overall_word}")
    table.add_row("Total steps",   str(report.n_total))
    table.add_row("Realised",      str(report.n_realised))
    table.add_row("Forecast-only", str(report.n_total - report.n_realised))
    for k, v in report.summary.items():
        table.add_row(k, v)
    console.print(table)

    counts = Table(title="Findings by Section", show_lines=False)
    counts.add_column("Section",  style="bold")
    counts.add_column("✓ ok",     justify="right", style="green")
    counts.add_column("⚠ warn",   justify="right", style="yellow")
    counts.add_column("✗ fail",   justify="right", style="red")
    for section, findings in report.sections.items():
        n_ok   = sum(1 for f in findings if f.status == "ok")
        n_warn = sum(1 for f in findings if f.status == "warn")
        n_fail = sum(1 for f in findings if f.status == "fail")
        counts.add_row(section, str(n_ok), str(n_warn), str(n_fail))
    console.print(counts)

    if out is None:
        # No output path → dump full markdown to stdout for piping / quick read.
        console.print()
        console.print(report.to_markdown())
    else:
        console.print(f"[bold green]✓[/bold green] Report written to [cyan]{out}[/cyan]")


# --- visualize CLI ---

visualize_app = typer.Typer()

@visualize_app.command()

def visualize(
    symbol:      str            = typer.Option(...,        help="Stock symbol e.g. RELIANCE"),
    sim_csv:     Path           = typer.Option(None,       help="Path to simulation CSV"),
    params:      Path           = typer.Option(None,       help="Path to params.pkl"),
    horizon:     int            = typer.Option(21,         help="Prediction horizon (trading days)"),
    recent_n:    int            = typer.Option(60,         help="Recent days shown in prediction chart"),
    out_dir:     Path           = typer.Option("plots",    help="Output directory for PNGs"),
    seed:        int            = typer.Option(42,         help="Random seed for MC sampling"),
):
    """Generate all diagnostic and results plots for a stock symbol."""
    from stochax_market.visualize import run_all

    console.print(f"[bold cyan]Generating plots for {symbol}...[/bold cyan]")

    saved = run_all(
        symbol      = symbol,
        sim_csv     = sim_csv,
        params_path = params,
        horizon     = horizon,
        recent_n    = recent_n,
        out_dir     = out_dir,
        seed        = seed,
    )

    table = Table(title=f"Visualisation Output: {symbol}", show_lines=True)
    table.add_column("Plot",        style="bold")
    table.add_column("Saved To",    style="green")

    for name, path in saved.items():
        table.add_row(name.replace("_", " ").title(), str(path))

    console.print(table)
    console.print("[bold green]✓ All plots saved[/bold green]")


# --- export CLI ---

export_app = typer.Typer(name="stochax-export", add_completion=False)


@export_app.command()
def latex(
    symbol: str = typer.Option(..., help="Stock symbol"),
    params: Path = typer.Option("params.pkl", help="Path to fitted params"),
    output: Path = typer.Option(None, help="Output PDF path"),
):
    """Export fitted model equations to LaTeX PDF."""
    from stochax_market.export.latex import export_model_pdf

    pdf_path = export_model_pdf(symbol, params, output)
    console.print(f"[green]✓[/green] Model exported to {pdf_path}")
