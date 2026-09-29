"""
bootstrap.py
------------
First-time setup script. Run this once after cloning the repo.
Downloads Statcast data, builds features, and trains the models.

Usage:
    python bootstrap.py                        # current + last season
    python bootstrap.py --seasons 2024 2025    # specific seasons
    python bootstrap.py --quick                # 2025 only (faster)
    python bootstrap.py --check                # verify setup only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path


PYTHON = sys.executable


def run(cmd: list, label: str, required: bool = True) -> bool:
    print(f"\n  Running: {label}...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  ERROR: {label} failed (exit code {result.returncode})")
        if required:
            print("  Fix the error above and re-run bootstrap.py")
            sys.exit(1)
        return False
    print(f"  Done: {label}")
    return True


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_env() -> bool:
    """Verify .env file and required keys exist."""
    env_path = Path(".env")
    if not env_path.exists():
        print("\n  ERROR: .env file not found.")
        print("  Copy .env.example to .env and add your ODDS_API_KEY.")
        print("  Get a free key at: https://the-odds-api.com/")
        return False

    with open(env_path) as f:
        content = f.read()

    if "ODDS_API_KEY" not in content or "your_odds_api_key_here" in content:
        print("\n  WARNING: ODDS_API_KEY not set in .env")
        print("  Sportsbook odds features will not work.")
        print("  Get a free key at: https://the-odds-api.com/")
    else:
        print("  .env file found with ODDS_API_KEY")

    return True


def check_dependencies() -> bool:
    """Verify required packages are installed."""
    required = ["xgboost", "lightgbm", "sklearn", "pybaseball",
                "pandas", "numpy", "requests", "loguru", "dotenv"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False

    print("  All dependencies installed")
    return True


def get_season_dates(season: int) -> tuple[str, str]:
    """Return start and end dates for a season."""
    return f"{season}-03-20", f"{season}-11-01"


def main():
    parser = argparse.ArgumentParser(
        description="First-time setup for MLB Analytics Engine"
    )
    parser.add_argument("--seasons", nargs="+", type=int,
                        help="Seasons to download (e.g. --seasons 2024 2025)")
    parser.add_argument("--quick", action="store_true",
                        help="Download current season only (faster)")
    parser.add_argument("--check", action="store_true",
                        help="Verify setup without downloading data")
    args = parser.parse_args()

    current_year = datetime.now().year

    if args.quick:
        seasons = [current_year]
    elif args.seasons:
        seasons = args.seasons
    else:
        seasons = [current_year - 1, current_year]

    print(f"\n{'='*60}")
    print(f"  MLB Analytics Engine — First-Time Setup")
    print(f"{'='*60}")
    print(f"  Seasons: {seasons}")
    print(f"  Python:  {sys.version.split()[0]}")

    # -------------------------------------------------------
    # STEP 1 — Check environment
    # -------------------------------------------------------
    banner("STEP 1/5 — Checking environment")

    if not check_dependencies():
        sys.exit(1)

    if not check_env():
        sys.exit(1)

    if args.check:
        print("\n  Environment looks good! Run bootstrap.py to download data.")
        return

    # -------------------------------------------------------
    # STEP 2 — Download Statcast data
    # -------------------------------------------------------
    banner(f"STEP 2/5 — Downloading Statcast data ({seasons})")
    print("  This is the longest step — 15-45 min per season depending on connection.")
    print("  Statcast data is pitch-level (~700K pitches per season).")

    for season in seasons:
        start, end = get_season_dates(season)
        run([
            PYTHON, "-m", "mlb_analytics.ingestion.statcast_ingest",
            "--start", start, "--end", end
        ], f"Statcast {season}", required=True)

    # -------------------------------------------------------
    # STEP 3 — Download Baseball Savant season stats
    # -------------------------------------------------------
    banner("STEP 3/5 — Downloading Baseball Savant season stats")

    season_args = [str(s) for s in seasons]
    run([
        PYTHON, "-m", "mlb_analytics.ingestion.savant_season_stats",
        "--seasons", *season_args
    ], "Savant season stats", required=True)

    # -------------------------------------------------------
    # STEP 4 — Build rolling features
    # -------------------------------------------------------
    banner("STEP 4/5 — Building rolling features")
    print("  Computing 5-game rolling batter and pitcher features...")

    run([
        PYTHON, "-m", "mlb_analytics.run_pipeline",
        "--stages", "features",
        "--seasons", *season_args
    ], "Rolling features", required=True)

    # -------------------------------------------------------
    # STEP 5 — Train models
    # -------------------------------------------------------
    banner("STEP 5/5 — Training prediction models")
    print("  Training XGBoost + LightGBM + Random Forest + Logistic Regression ensemble...")

    run([
        PYTHON, "train_props.py",
        "--seasons", *season_args
    ], "Props model training", required=True)

    # -------------------------------------------------------
    # Done
    # -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Setup complete!")
    print(f"{'='*60}")
    print(f"""
  You're ready to go. Run this every day after lineups post (~2-3 PM ET):

      python morning.py

  Other useful commands:
      python morning.py --quick          # Skip retraining (faster)
      python morning.py --picks-only     # Just show today's picks
      python backtest.py --start {seasons[0]}-04-01 --end {seasons[-1]}-09-30
      python best_odds.py --ks           # Pitcher K props
      python track_results.py --summary  # Full season record
""")


if __name__ == "__main__":
    main()
