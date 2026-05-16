"""
daily_update.py
---------------
Master daily pipeline. Run every morning before making picks.

What it does:
  1. Pulls yesterday's Statcast data
  2. Fetches today's lineups + starters (MLB API)
  3. Updates bullpen availability + rest days
  4. Fetches weather for all outdoor parks
  5. Fetches umpire assignments
  6. Checks injury report
  7. Refreshes rolling features
  8. Rebuilds matchup features
  9. Retrains models on latest data

Usage:
    python daily_update.py
    python daily_update.py --skip-training   (faster, use existing models)
    python daily_update.py --date 2026-03-26 (specific date)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

if sys.platform == "win32":
    PYTHON = ".venv\\Scripts\\python"
else:
    PYTHON = ".venv/bin/python"


def run(cmd: str, check: bool = True) -> None:
    print(f"\n  >>> {cmd}")
    subprocess.run(cmd, shell=True, check=check)


# ---------------------------------------------------------------------------
# Individual update steps
# ---------------------------------------------------------------------------

def step_statcast(yesterday: str) -> None:
    logger.info(f"[1/8] Pulling Statcast for {yesterday}...")
    run(f"{PYTHON} -m mlb_analytics.ingestion.statcast_ingest "
        f"--start {yesterday} --end {yesterday}")


def step_lineups(today: str) -> dict:
    """Fetch lineups and return parsed game data."""
    logger.info(f"[2/8] Fetching lineups for {today}...")
    from fetch_lineups import get_todays_games, parse_game, write_daily_lineup

    games_raw = get_todays_games(today)
    games     = [parse_game(g) for g in games_raw if parse_game(g)]
    write_daily_lineup(games, today)
    return {g["home"]: g for g in games if g}


def step_bullpen(games: dict) -> None:
    """Update bullpen availability for all teams playing today."""
    logger.info("[3/8] Updating bullpen availability...")
    from data_fetchers.bullpen_and_umpires import build_bullpen_availability

    try:
        from daily_lineup import BULLPEN_AVAILABILITY as existing
    except ImportError:
        existing = {}

    updated_bullpen = {}
    teams = set()
    for game in games.values():
        teams.add(game.get("home", ""))
        teams.add(game.get("away", ""))

    for team in teams:
        if not team:
            continue
        existing_arms = existing.get(team, [])
        updated_bullpen[team] = build_bullpen_availability(team, existing_arms)
        logger.info(f"  {team}: {len(updated_bullpen[team])} bullpen arms tracked")

    # Patch bullpen into daily_lineup.py
    _patch_bullpen(updated_bullpen)


def _patch_bullpen(bullpen_data: dict) -> None:
    """Update just the BULLPEN_AVAILABILITY section in daily_lineup.py."""
    try:
        with open("daily_lineup.py", "r", encoding="utf-8") as f:
            content = f.read()

        new_section = "BULLPEN_AVAILABILITY = {\n"
        for team, arms in sorted(bullpen_data.items()):
            new_section += f'    "{team}": [\n'
            for arm in arms:
                new_section += (
                    f'        {{"name": "{arm["name"]}", '
                    f'"hand": "{arm["hand"]}", '
                    f'"rest_days": {arm["rest_days"]}, '
                    f'"available": {arm["available"]}}},'
                    f'\n'
                )
            new_section += "    ],\n"
        new_section += "}\n"

        import re
        content = re.sub(
            r"BULLPEN_AVAILABILITY\s*=\s*\{.*?\}",
            new_section,
            content,
            flags=re.DOTALL,
        )
        with open("daily_lineup.py", "w", encoding="utf-8") as f:
            f.write(content)
        logger.success("Bullpen availability updated in daily_lineup.py")
    except Exception as exc:
        logger.warning(f"Could not patch bullpen: {exc}")


def step_weather(games: dict, today: str) -> dict:
    """Fetch weather for all outdoor home parks."""
    logger.info("[4/8] Fetching weather forecasts...")
    from data_fetchers.weather import fetch_weather, print_weather_report

    weather_data = {}
    home_teams   = list(games.keys())

    for team in home_teams:
        w = fetch_weather(team, today)
        weather_data[team] = w
        if w:
            print(f"\n  {team}:")
            print_weather_report(w)

    # Save to JSON for use by other scripts
    with open("cache/weather_today.json", "w") as f:
        json.dump(weather_data, f, indent=2)
    logger.success(f"Weather saved for {len(weather_data)} parks")
    return weather_data


def step_umpires(today: str) -> dict:
    """Fetch HP umpire assignments."""
    logger.info("[5/8] Fetching umpire assignments...")
    from data_fetchers.bullpen_and_umpires import (
        get_todays_umpires, print_umpire_report
    )

    umpires = get_todays_umpires(today)

    print(f"\n  Today's HP Umpires:")
    for gk, u in umpires.items():
        k_label = "🔴 Tight" if u["k_factor"] > 1.04 else \
                  "🟢 Wide"  if u["k_factor"] < 0.96 else "🟡 Avg"
        print(f"  {u['away_team']:>4} @ {u['home_team']:<4} — "
              f"{u['hp_umpire']:<25} {k_label} zone")

    with open("cache/umpires_today.json", "w") as f:
        json.dump(umpires, f, indent=2)
    return umpires


def step_injuries() -> None:
    """Print injury report for all teams."""
    logger.info("[6/8] Checking injury report...")
    try:
        from daily_lineup import TODAYS_LINEUPS
        from data_fetchers.injuries import check_lineup_for_injuries

        for team, lineup in TODAYS_LINEUPS.items():
            flagged = check_lineup_for_injuries(lineup, team)
            if flagged:
                print(f"\n  ⚠️  INJURY FLAGS — {team}:")
                for p in flagged:
                    print(f"     ❌ {p['name']}: {p['description']}")
            else:
                print(f"  ✅ {team} lineup clear")
    except Exception as exc:
        logger.warning(f"Injury check skipped: {exc}")


def step_features() -> None:
    logger.info("[7/8] Refreshing rolling features...")
    run(f"{PYTHON} -m mlb_analytics.run_pipeline --stages features")
    run(f"{PYTHON} -m mlb_analytics.features.build_matchups")


def step_train() -> None:
    logger.info("[8/8] Retraining models...")
    run(f"{PYTHON} -m mlb_analytics.run_pipeline --stages train")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MLB daily update pipeline")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model retraining (faster)")
    parser.add_argument("--skip-statcast", action="store_true",
                        help="Skip Statcast pull")
    parser.add_argument("--date", type=str, default=None,
                        help="Override today's date YYYY-MM-DD")
    args = parser.parse_args()

    today     = args.date or date.today().isoformat()
    yesterday = (date.fromisoformat(today) - timedelta(days=1)).isoformat()

    # Ensure cache folder exists
    Path("cache").mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ⚾  MLB DAILY UPDATE — {today}")
    print(f"{'='*60}")

    # 1. Statcast
    if not args.skip_statcast:
        try:
            step_statcast(yesterday)
        except Exception as exc:
            logger.warning(f"Statcast pull failed: {exc} — continuing")

    # 2. Lineups
    try:
        games = step_lineups(today)
    except Exception as exc:
        logger.warning(f"Lineup fetch failed: {exc}")
        games = {}

    # 3. Bullpen
    if games:
        try:
            step_bullpen(games)
        except Exception as exc:
            logger.warning(f"Bullpen update failed: {exc}")

    # 4. Weather
    if games:
        try:
            step_weather(games, today)
        except Exception as exc:
            logger.warning(f"Weather fetch failed: {exc}")

    # 5. Umpires
    try:
        step_umpires(today)
    except Exception as exc:
        logger.warning(f"Umpire fetch failed: {exc}")

    # 6. Injuries
    try:
        step_injuries()
    except Exception as exc:
        logger.warning(f"Injury check failed: {exc}")

    # 7. Features
    try:
        step_features()
    except Exception as exc:
        logger.warning(f"Feature refresh failed: {exc}")

    # 8. Train
    if not args.skip_training:
        try:
            step_train()
        except Exception as exc:
            logger.warning(f"Model training failed: {exc}")

    # Done
    print(f"\n{'='*60}")
    print(f"  ✅  DAILY UPDATE COMPLETE — {today}")
    print(f"{'='*60}")
    print("""
  Now run:
    python parlay_builder.py          ← best parlays
    python matchup_predictor.py \\
      --pitcher "Webb" \\
      --pitcher-team SF \\
      --batting-team NYY              ← matchup breakdown
""")


if __name__ == "__main__":
    main()