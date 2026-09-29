"""
fetch_historical_lineups.py
---------------------------
Fetches confirmed lineups for past dates from the MLB Stats API
and saves them to cache/lineups_YYYY-MM-DD.json

This allows the backtest to use real historical lineups instead
of today's lineups for past games.

Usage:
    python fetch_historical_lineups.py --days 30
    python fetch_historical_lineups.py --start 2026-05-01 --end 2026-06-01
"""

from __future__ import annotations
import argparse
import json
import requests
from datetime import date, timedelta
from pathlib import Path
from loguru import logger

Path("cache").mkdir(exist_ok=True)

MLB_API  = "https://statsapi.mlb.com/api/v1"
HEADERS  = {"User-Agent": "Mozilla/5.0"}
HAND_MAP = {"L": "L", "R": "R", "S": "S", "B": "S"}


def fetch_lineups_for_date(game_date: str) -> dict:
    """Fetch confirmed lineups for a specific date."""
    try:
        r = requests.get(f"{MLB_API}/schedule", params={
            "sportId": 1,
            "date":    game_date,
            "hydrate": "lineups,probablePitcher",
        }, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.debug(f"API error for {game_date}: {e}")
        return {}

    lineups  = {}
    starters = {}

    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            home = game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation")
            away = game.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation")

            # Fix team abbreviations
            ABBREV_FIX = {"ATH": "OAK", "ARI": "AZ"}
            home = ABBREV_FIX.get(home, home)
            away = ABBREV_FIX.get(away, away)

            # Starting pitchers
            home_sp = game.get("teams", {}).get("home", {}).get("probablePitcher", {})
            away_sp = game.get("teams", {}).get("away", {}).get("probablePitcher", {})
            if home_sp:
                starters[home] = {
                    "name": home_sp.get("fullName", "TBD"),
                    "hand": home_sp.get("pitchHand", {}).get("code", "R"),
                    "role": "starter",
                }
            if away_sp:
                starters[away] = {
                    "name": away_sp.get("fullName", "TBD"),
                    "hand": away_sp.get("pitchHand", {}).get("code", "R"),
                    "role": "starter",
                }

            # Lineups
            lineups_data  = game.get("lineups", {})
            home_players  = lineups_data.get("homePlayers", [])
            away_players  = lineups_data.get("awayPlayers", [])

            def parse_players(players):
                return [
                    {
                        "order": i + 1,
                        "name":  p.get("fullName", "Unknown"),
                        "bats":  HAND_MAP.get(p.get("batSide", {}).get("code", "R"), "R"),
                        "pos":   p.get("primaryPosition", {}).get("abbreviation", ""),
                    }
                    for i, p in enumerate(players)
                ]

            if len(home_players) >= 8:
                lineups[home] = parse_players(home_players)
            if len(away_players) >= 8:
                lineups[away] = parse_players(away_players)

    return {"lineups": lineups, "starters": starters}


def fetch_range(start: str, end: str, skip_existing: bool = True) -> None:
    start_dt = date.fromisoformat(start)
    end_dt   = date.fromisoformat(end)
    current  = start_dt
    saved    = 0
    skipped  = 0

    total_days = (end_dt - start_dt).days + 1
    logger.info(f"Fetching lineups for {total_days} days ({start} to {end})")

    while current <= end_dt:
        ds         = current.isoformat()
        cache_path = f"cache/lineups_{ds}.json"

        if skip_existing and Path(cache_path).exists():
            skipped += 1
            current += timedelta(days=1)
            continue

        data = fetch_lineups_for_date(ds)

        if data.get("lineups"):
            # Save lineups in same format as fetch_lineups.py
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data["lineups"], f, indent=2)

            # Save starters separately
            starters_path = f"cache/starters_{ds}.json"
            with open(starters_path, "w", encoding="utf-8") as f:
                json.dump(data["starters"], f, indent=2)

            n_teams = len(data["lineups"])
            saved  += 1
            logger.success(f"  {ds}: saved {n_teams} team lineups")
        else:
            logger.debug(f"  {ds}: no lineups found")

        current += timedelta(days=1)

    print(f"\nDone! Saved {saved} dates, skipped {skipped} existing.")
    print(f"Lineup cache files: cache/lineups_YYYY-MM-DD.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",  type=int, default=30,
                        help="Fetch last N days (default 30)")
    parser.add_argument("--start", type=str)
    parser.add_argument("--end",   type=str)
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if cache exists")
    args = parser.parse_args()

    if args.start and args.end:
        start, end = args.start, args.end
    else:
        end   = (date.today() - timedelta(days=1)).isoformat()
        start = (date.today() - timedelta(days=args.days)).isoformat()

    fetch_range(start, end, skip_existing=not args.force)


if __name__ == "__main__":
    main()
