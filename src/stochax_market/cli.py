"""Typer CLI for stochax-market: simulate, fit, and predict commands."""

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
):
    """Run SPDE simulation for a NIFTY50 stock."""
    from stochax_market.simulate import simulate

    console.print(f"[bold blue]Simulating {symbol} for {steps} steps...[/bold blue]")
    result = simulate(symbol, n_steps=steps, output_path=output, seed=seed, params_path=params)

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
    n_steps: int = typer.Option(100, "--n-steps", "-n", help="Optimization steps"),
    output: str = typer.Option("params.pkl", "--output", "-o", help="Output params path"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    """Calibrate SPDE + GARCH parameters for a NIFTY50 stock."""
    import jax

    from stochax_market.calibration.fit import fit
    from stochax_market.data.features import encode_features
    from stochax_market.data.loader import load_stock
    from stochax_market.model.spde import SPDEStepper
    from stochax_market.model.volatility import GARCHVolatility

    console.print(f"[bold blue]Fitting {symbol} ({n_steps} steps)...[/bold blue]")

    df = load_stock(symbol)
    features = encode_features(df)

    spde = SPDEStepper()
    garch = GARCHVolatility()
    key = jax.random.key(seed)

    fitted_spde, fitted_garch, info = fit(spde, garch, features, n_steps=n_steps, key=key)

    with open(output, "wb") as f:
        pickle.dump((fitted_spde, fitted_garch, features["L"]), f)

    table = Table(title=f"Calibration Results: {symbol}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in info.items():
        table.add_row(str(k), str(v))
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
):
    """Predict future prices for a NIFTY50 stock."""
    from stochax_market.predict import predict

    console.print(
        f"[bold blue]Predicting {symbol} {horizon} steps ahead "
        f"({samples} MC samples)...[/bold blue]"
    )

    result = predict(
        symbol, horizon=horizon, params_path=params, seed=seed, n_samples=samples
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