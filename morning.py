"""
morning.py
----------
Run this every morning before making picks.
Does everything in the right order automatically.

Usage:
    python morning.py                  # full update + show picks
    python morning.py --skip-training  # faster, skip retraining
    python morning.py --picks-only     # just show today's picks (no updates)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PYTHON = ".venv\\Scripts\\python" if sys.platform == "win32" \
         else ".venv/bin/python"


def run(cmd: str, label: str = "", check: bool = True) -> bool:
    """Run a command and return True if successful."""
    print(f"\n  ▶  {label or cmd}")
    result = subprocess.run(cmd, shell=True, check=False)
    if result.returncode != 0:
        logger.warning(f"  ⚠️  {label} returned code {result.returncode}")
        return False
    return True


def step_banner(n: int, total: int, label: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  [{n}/{total}]  {label}")
    print(f"{'─'*60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLB morning update — run before making picks"
    )
    parser.add_argument("--skip-training",  action="store_true",
                        help="Skip model retraining (faster)")
    parser.add_argument("--skip-statcast",  action="store_true",
                        help="Skip pulling yesterday's Statcast data")
    parser.add_argument("--picks-only",     action="store_true",
                        help="Skip all updates, just show today's picks")
    args = parser.parse_args()

    today     = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    print(f"\n{'='*60}")
    print(f"  ⚾  MLB MORNING UPDATE  —  {today}")
    print(f"{'='*60}")

    if args.picks_only:
        print("\n  Skipping updates — showing today's picks only\n")
        show_picks()
        return

    total = 7 if not args.skip_training else 6

    # 1 — Pull yesterday's Statcast
    if not args.skip_statcast:
        step_banner(1, total, "Pulling yesterday's Statcast data")
        run(
            f"{PYTHON} -m mlb_analytics.ingestion.statcast_ingest "
            f"--start {yesterday} --end {yesterday}",
            "Statcast ingest"
        )
    else:
        logger.info("Skipping Statcast pull")

    # 2 — Fetch today's lineups
    step_banner(2, total, "Fetching today's lineups & starters")
    run(f"{PYTHON} fetch_lineups.py", "Lineup fetch")

    # 3 — Update bullpen availability
    step_banner(3, total, "Updating bullpen rest days")
    run(f"{PYTHON} -c \""
        f"from data_fetchers.bullpen_and_umpires import build_bullpen_availability; "
        f"print('Bullpen updated')\"",
        "Bullpen update", check=False
    )

    # 4 — Fetch weather
    step_banner(4, total, "Fetching game-time weather")
    Path("cache").mkdir(exist_ok=True)
    run(f"{PYTHON} data_fetchers/weather.py", "Weather fetch", check=False)

    # 5 — Refresh rolling features + matchups
    step_banner(5, total, "Refreshing rolling features & matchups")
    run(
        f"{PYTHON} -m mlb_analytics.run_pipeline --stages features",
        "Rolling features"
    )
    run(
        f"{PYTHON} -m mlb_analytics.features.build_matchups",
        "Matchup features"
    )

    # 6 — Retrain models (optional)
    if not args.skip_training:
        step_banner(6, total, "Retraining prediction models")
        run(
            f"{PYTHON} train_props.py --seasons 2023 2024 2025",
            "Props model training"
        )

    # 7 — Show picks
    step_banner(total, total, "Today's picks")
    show_picks()

    print(f"\n{'='*60}")
    print(f"  ✅  Morning update complete — {today}")
    print(f"{'='*60}")
    print("""
  Commands available:
    python morning.py --picks-only      ← just see today's picks again
    python morning.py --skip-training   ← faster daily update
    python matchup_predictor.py         ← interactive game selector
    python parlay_builder.py            ← best parlays
""")


def show_picks() -> None:
    """Show parlay recommendations and prompt for matchup analysis."""
    print("\n" + "="*60)
    print("  🏆  TODAY'S PARLAY RECOMMENDATIONS")
    print("="*60)

    result = subprocess.run(
        f"{PYTHON} parlay_builder.py",
        shell=True, capture_output=False
    )

    print("\n" + "="*60)
    print("  📊  MATCHUP PREDICTOR")
    print("="*60)
    print("\n  Run this for detailed player prop predictions:")
    print("  python matchup_predictor.py")
    print("\n  Or analyze a specific game:")
    print("  python matchup_predictor.py --pitcher 'Skenes' "
          "--pitcher-team PIT --batting-team NYM\n")


if __name__ == "__main__":
    main()
