#!/usr/bin/env python3
"""
scripts/compile_report.py

Compile all per-symbol visualisation PNGs into a single HTML report.

Usage:
    python scripts/compile_report.py
    python scripts/compile_report.py --plots-dir plots --output report.html
    python scripts/compile_report.py --plots-dir plots --output report.html --title "NIFTY50 SPDE Report"
"""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path

# ── Plot display names and order within each symbol section ──────────────────
PLOT_ORDER = [
    ("historical",       "Historical Close & VWAP"),
    ("prediction",       "Prediction with 90% CI"),
    ("sim_vs_actual",    "SPDE Simulation vs Actual"),
    ("garch_fit",        "GARCH(1,1) vs Realised Volatility"),
    ("volatility",       "Volatility Clustering"),
    ("log_returns",      "Log-Return Distribution"),
]

# ── HTML template ─────────────────────────────────────────────────────────────
_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg:       #0f1117;
    --surface:  #1a1d27;
    --border:   #2a2d3a;
    --accent:   #5b8def;
    --text:     #e2e8f0;
    --muted:    #8892a4;
    --green:    #22c55e;
    --orange:   #f97316;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 15px;
    line-height: 1.6;
  }}

  /* ── Header ── */
  .site-header {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 24px 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }}
  .site-header h1 {{
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: -0.3px;
  }}
  .site-header .meta {{
    color: var(--muted);
    font-size: 0.85rem;
  }}

  /* ── Nav sidebar ── */
  .layout {{ display: flex; min-height: 100vh; }}
  .sidebar {{
    width: 200px;
    flex-shrink: 0;
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 24px 0;
    position: sticky;
    top: 73px;
    height: calc(100vh - 73px);
    overflow-y: auto;
  }}
  .sidebar h3 {{
    padding: 0 20px 12px;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
  }}
  .sidebar a {{
    display: block;
    padding: 8px 20px;
    color: var(--text);
    text-decoration: none;
    font-size: 0.9rem;
    border-left: 3px solid transparent;
    transition: all 0.15s;
  }}
  .sidebar a:hover {{
    background: rgba(91,141,239,0.08);
    border-left-color: var(--accent);
    color: var(--accent);
  }}

  /* ── Main content ── */
  .main {{ flex: 1; padding: 40px; max-width: 1400px; }}

  /* ── Symbol section ── */
  .symbol-section {{
    margin-bottom: 64px;
    scroll-margin-top: 90px;
  }}
  .symbol-header {{
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 24px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }}
  .symbol-header h2 {{
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
  }}
  .symbol-header .badge {{
    background: rgba(91,141,239,0.15);
    color: var(--accent);
    border: 1px solid rgba(91,141,239,0.3);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
  }}

  /* ── Plot grid ── */
  .plot-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(620px, 1fr));
    gap: 24px;
  }}
  .plot-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    transition: border-color 0.2s;
  }}
  .plot-card:hover {{ border-color: var(--accent); }}
  .plot-card.full-width {{
    grid-column: 1 / -1;
  }}
  .plot-card-header {{
    padding: 14px 18px 10px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .plot-card-header .dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
  }}
  .plot-card-header span {{
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text);
  }}
  .plot-card img {{
    width: 100%;
    height: auto;
    display: block;
  }}
  .plot-missing {{
    padding: 48px;
    text-align: center;
    color: var(--muted);
    font-size: 0.85rem;
  }}

  /* ── Summary table ── */
  .summary-section {{ margin-bottom: 48px; }}
  .summary-section h2 {{
    font-size: 1.2rem;
    margin-bottom: 16px;
    color: var(--text);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
  }}
  th {{
    background: var(--surface);
    color: var(--muted);
    font-weight: 600;
    text-align: left;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  td {{
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }}
  tr:hover td {{ background: rgba(255,255,255,0.02); }}
  .status-ok   {{ color: var(--green);  font-weight: 600; }}
  .status-miss {{ color: var(--orange); font-weight: 600; }}

  /* ── Footer ── */
  footer {{
    text-align: center;
    padding: 32px;
    color: var(--muted);
    font-size: 0.82rem;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>
"""

_HTML_TAIL = """
<footer>
  Generated by stochax_market &nbsp;·&nbsp; {generated_at}
</footer>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _img_to_b64(path: Path) -> str | None:
    """Read a PNG and return a base64 data-URI string, or None if missing."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _load_meta(png_path: Path) -> dict:
    """Load sidecar .meta.json if present, else return empty dict."""
    meta_path = png_path.with_suffix(".png.meta.json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            pass
    return {}


def _find_plot(symbol_dir: Path, key: str) -> Path | None:
    """
    Find a plot PNG for a given key inside symbol_dir.
    Matches files containing the key substring (e.g. 'prediction' matches
    'RELIANCE_prediction_21d.png').
    """
    for p in sorted(symbol_dir.glob("*.png")):
        if key in p.stem:
            return p
    return None


def _build_summary_table(symbols: list[tuple[str, dict[str, Path | None]]]) -> str:
    """Build an HTML summary table showing which plots exist per symbol."""
    plot_keys = [k for k, _ in PLOT_ORDER]
    header_cells = "".join(f"<th>{label}</th>" for _, label in PLOT_ORDER)
    rows = ""
    for symbol, plots in symbols:
        cells = ""
        for key in plot_keys:
            if plots.get(key):
                cells += '<td class="status-ok">✓</td>'
            else:
                cells += '<td class="status-miss">–</td>'
        rows += f"<tr><td><strong>{symbol}</strong></td>{cells}</tr>"
    return f"""
    <div class="summary-section">
      <h2>Coverage Summary</h2>
      <table>
        <thead><tr><th>Symbol</th>{header_cells}</tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# Main builder
# ══════════════════════════════════════════════════════════════════════════════

def build_report(
    plots_dir: Path,
    output_path: Path,
    title: str = "SPDE Stock Model — Diagnostic Report",
    embed_images: bool = True,
) -> None:
    """
    Scan plots_dir for per-symbol subdirectories, collect PNGs, and
    write a self-contained HTML report.

    Args:
        plots_dir    : Root directory containing per-symbol subdirs
                       (e.g. plots/RELIANCE/, plots/TCS/).
        output_path  : Destination HTML file.
        title        : Page title shown in header.
        embed_images : If True, embed PNGs as base64 (fully self-contained).
                       If False, use relative src paths (smaller file).
    """
    plots_dir   = Path(plots_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover symbols — each subdirectory is one symbol
    symbol_dirs = sorted(
        [d for d in plots_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    # If no subdirs, try flat layout (all PNGs directly in plots_dir)
    if not symbol_dirs:
        symbol_dirs = [plots_dir]

    # Collect plot paths per symbol
    symbol_data: list[tuple[str, dict[str, Path | None]]] = []
    for sym_dir in symbol_dirs:
        symbol = sym_dir.name if sym_dir != plots_dir else "ALL"
        plots: dict[str, Path | None] = {}
        for key, _ in PLOT_ORDER:
            plots[key] = _find_plot(sym_dir, key)
        symbol_data.append((symbol, plots))

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Build HTML ─────────────────────────────────────────────────────────────
    html_parts = [
        _HTML_HEAD.format(title=title),
        f"""
        <div class="site-header">
          <h1>📈 {title}</h1>
          <span class="meta">Generated {generated_at} &nbsp;·&nbsp; {len(symbol_data)} symbol(s)</span>
        </div>
        <div class="layout">
        """,
    ]

    # Sidebar navigation
    nav_links = "".join(
        f'<a href="#{sym}">{sym}</a>'
        for sym, _ in symbol_data
    )
    html_parts.append(f"""
        <nav class="sidebar">
          <h3>Symbols</h3>
          {nav_links}
        </nav>
        <main class="main">
    """)

    # Summary table
    html_parts.append(_build_summary_table(symbol_data))

    # Per-symbol sections
    for symbol, plots in symbol_data:
        n_plots = sum(1 for v in plots.values() if v is not None)
        html_parts.append(f"""
        <section class="symbol-section" id="{symbol}">
          <div class="symbol-header">
            <h2>{symbol}</h2>
            <span class="badge">{n_plots}/{len(PLOT_ORDER)} plots</span>
          </div>
          <div class="plot-grid">
        """)

        for key, label in PLOT_ORDER:
            png_path = plots.get(key)

            # Historical and prediction span full width
            full = "full-width" if key in ("historical", "prediction") else ""

            html_parts.append(f'<div class="plot-card {full}">')
            html_parts.append(f"""
              <div class="plot-card-header">
                <div class="dot"></div>
                <span>{label}</span>
              </div>
            """)

            if png_path and png_path.exists():
                meta = _load_meta(png_path)
                caption = meta.get("caption", "")
                desc    = meta.get("description", "")

                if embed_images:
                    src = _img_to_b64(png_path)
                else:
                    src = str(png_path.relative_to(output_path.parent))

                alt = desc or caption or f"{symbol} {label}"
                html_parts.append(
                    f'<img src="{src}" alt="{alt}" loading="lazy">'
                )
                if caption:
                    html_parts.append(
                        f'<div style="padding:8px 18px;font-size:0.78rem;'
                        f'color:var(--muted);">{caption}</div>'
                    )
            else:
                html_parts.append(
                    f'<div class="plot-missing">Plot not available<br>'
                    f'<small>{symbol}_{key}.png not found</small></div>'
                )

            html_parts.append("</div>")  # plot-card

        html_parts.append("</div></section>")  # plot-grid, symbol-section

    html_parts.append("</main></div>")  # main, layout
    html_parts.append(_HTML_TAIL.format(generated_at=generated_at))

    output_path.write_text("".join(html_parts), encoding="utf-8")
    print(f"✓ Report written to {output_path}  ({output_path.stat().st_size // 1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile per-symbol SPDE plots into a single HTML report."
    )
    parser.add_argument(
        "--plots-dir", default="plots",
        help="Root directory of per-symbol plot subdirectories (default: plots/)",
    )
    parser.add_argument(
        "--output", default="report.html",
        help="Output HTML file path (default: report.html)",
    )
    parser.add_argument(
        "--title", default="SPDE Stock Model — Diagnostic Report",
        help="Report title shown in the page header",
    )
    parser.add_argument(
        "--no-embed", action="store_true",
        help="Use relative image paths instead of base64 embedding "
             "(smaller HTML, requires keeping plots/ directory alongside)",
    )
    args = parser.parse_args()

    build_report(
        plots_dir    = Path(args.plots_dir),
        output_path  = Path(args.output),
        title        = args.title,
        embed_images = not args.no_embed,
    )


if __name__ == "__main__":
    main()