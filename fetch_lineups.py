# -*- coding: utf-8 -*-
"""
fetch_lineups.py
----------------
Fetches today's lineups and starting pitchers from the MLB Stats API.

Rules:
  - Only uses lineups confirmed TODAY by the MLB API (8+ players)
  - Saves confirmed lineups to cache/lineups_YYYY-MM-DD.json
  - Next day, loads yesterday's cache for teams without confirmed lineups
  - Falls back to hardcoded 2026 rosters if no cache exists
  - Pitchers always pulled fresh from today's probable pitcher data

Usage:
    python fetch_lineups.py
    python fetch_lineups.py --date 2026-04-13
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import requests
from loguru import logger

Path("cache").mkdir(exist_ok=True)

MLB_API_BASE     = "https://statsapi.mlb.com/api/v1"
MLB_SCHEDULE_URL = f"{MLB_API_BASE}/schedule"
HEADERS          = {"User-Agent": "Mozilla/5.0"}
HAND_MAP         = {"L": "L", "R": "R", "S": "S", "B": "S"}

# ---------------------------------------------------------------------------
# 2026 fallback rosters — last resort only
# ---------------------------------------------------------------------------
FALLBACK_ROSTERS = {
    "NYY": [
        {"order": 1, "name": "Aaron Judge",       "bats": "R", "pos": "RF"},
        {"order": 2, "name": "Juan Soto",          "bats": "L", "pos": "LF"},
        {"order": 3, "name": "Jazz Chisholm Jr.",  "bats": "L", "pos": "2B"},
        {"order": 4, "name": "Giancarlo Stanton",  "bats": "R", "pos": "DH"},
        {"order": 5, "name": "Cody Bellinger",     "bats": "L", "pos": "1B"},
        {"order": 6, "name": "Austin Wells",       "bats": "L", "pos": "C"},
        {"order": 7, "name": "Trent Grisham",      "bats": "L", "pos": "CF"},
        {"order": 8, "name": "Ben Rice",           "bats": "L", "pos": "3B"},
        {"order": 9, "name": "Jose Caballero",     "bats": "R", "pos": "SS"},
    ],
    "LAD": [
        {"order": 1, "name": "Mookie Betts",       "bats": "R", "pos": "SS"},
        {"order": 2, "name": "Shohei Ohtani",      "bats": "L", "pos": "DH"},
        {"order": 3, "name": "Freddie Freeman",    "bats": "L", "pos": "1B"},
        {"order": 4, "name": "Teoscar Hernandez",  "bats": "R", "pos": "RF"},
        {"order": 5, "name": "Will Smith",         "bats": "R", "pos": "C"},
        {"order": 6, "name": "Max Muncy",          "bats": "L", "pos": "3B"},
        {"order": 7, "name": "Andy Pages",         "bats": "R", "pos": "CF"},
        {"order": 8, "name": "Gavin Lux",          "bats": "L", "pos": "2B"},
        {"order": 9, "name": "Miguel Rojas",       "bats": "R", "pos": "SS"},
    ],
    "SF": [
        {"order": 1, "name": "Wilmer Flores",      "bats": "R", "pos": "1B"},
        {"order": 2, "name": "Matt Chapman",       "bats": "R", "pos": "3B"},
        {"order": 3, "name": "Rafael Devers",      "bats": "L", "pos": "DH"},
        {"order": 4, "name": "Patrick Bailey",     "bats": "S", "pos": "C"},
        {"order": 5, "name": "Tyler Fitzgerald",   "bats": "R", "pos": "SS"},
        {"order": 6, "name": "Grant McCray",       "bats": "L", "pos": "CF"},
        {"order": 7, "name": "Heliot Ramos",       "bats": "R", "pos": "RF"},
        {"order": 8, "name": "Mike Yastrzemski",   "bats": "L", "pos": "LF"},
        {"order": 9, "name": "Brett Wisely",       "bats": "R", "pos": "2B"},
    ],
    "ATL": [
        {"order": 1, "name": "Ronald Acuna Jr.",   "bats": "R", "pos": "RF"},
        {"order": 2, "name": "Ozzie Albies",       "bats": "S", "pos": "2B"},
        {"order": 3, "name": "Matt Olson",         "bats": "L", "pos": "1B"},
        {"order": 4, "name": "Austin Riley",       "bats": "R", "pos": "3B"},
        {"order": 5, "name": "Jurickson Profar",   "bats": "L", "pos": "DH"},
        {"order": 6, "name": "Michael Harris II",  "bats": "L", "pos": "CF"},
        {"order": 7, "name": "Sean Murphy",        "bats": "R", "pos": "C"},
        {"order": 8, "name": "Jarred Kelenic",     "bats": "L", "pos": "LF"},
        {"order": 9, "name": "Orlando Arcia",      "bats": "R", "pos": "SS"},
    ],
    "HOU": [
        {"order": 1, "name": "Jose Altuve",        "bats": "R", "pos": "2B"},
        {"order": 2, "name": "Alex Bregman",       "bats": "R", "pos": "3B"},
        {"order": 3, "name": "Yordan Alvarez",     "bats": "L", "pos": "DH"},
        {"order": 4, "name": "Christian Walker",   "bats": "R", "pos": "1B"},
        {"order": 5, "name": "Yainer Diaz",        "bats": "R", "pos": "C"},
        {"order": 6, "name": "Jake Meyers",        "bats": "R", "pos": "CF"},
        {"order": 7, "name": "Mauricio Dubon",     "bats": "R", "pos": "RF"},
        {"order": 8, "name": "Jon Singleton",      "bats": "L", "pos": "1B"},
        {"order": 9, "name": "Jeremy Pena",        "bats": "R", "pos": "SS"},
    ],
    "NYM": [
        {"order": 1, "name": "Francisco Lindor",   "bats": "S", "pos": "SS"},
        {"order": 2, "name": "Juan Soto",          "bats": "L", "pos": "LF"},
        {"order": 3, "name": "Pete Alonso",        "bats": "R", "pos": "1B"},
        {"order": 4, "name": "Francisco Alvarez",  "bats": "R", "pos": "C"},
        {"order": 5, "name": "Mark Vientos",       "bats": "R", "pos": "3B"},
        {"order": 6, "name": "Brandon Nimmo",      "bats": "L", "pos": "CF"},
        {"order": 7, "name": "Tyrone Taylor",      "bats": "R", "pos": "RF"},
        {"order": 8, "name": "Jeff McNeil",        "bats": "L", "pos": "2B"},
        {"order": 9, "name": "Jose Iglesias",      "bats": "R", "pos": "SS"},
    ],
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def save_lineup_cache(lineups: dict, game_date: str) -> None:
    """Save confirmed lineups to cache for use next day."""
    cache_path = f"cache/lineups_{game_date}.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(lineups, f, indent=2)
    logger.info(f"Saved {len(lineups)} confirmed lineups to {cache_path}")


def load_lineup_cache(game_date: str) -> dict:
    """Load confirmed lineups from a specific date's cache."""
    cache_path = f"cache/lineups_{game_date}.json"
    try:
        with open(cache_path, encoding="utf-8") as f:
            lineups = json.load(f)
        logger.info(f"Loaded {len(lineups)} lineups from cache {cache_path}")
        return lineups
    except FileNotFoundError:
        return {}


def load_yesterday_cache(today: str) -> dict:
    """
    Load yesterday's confirmed lineups from cache.
    Tries yesterday first, then 2 days ago, then 3 days ago.
    Only loads from 2026 season (March 25+).
    """
    today_dt = date.fromisoformat(today)
    season_start = date(2026, 3, 25)

    for days_back in [1, 2, 3]:
        check_date = today_dt - timedelta(days=days_back)
        if check_date < season_start:
            break
        lineups = load_lineup_cache(check_date.isoformat())
        if lineups:
            logger.info(f"Using lineups from {check_date.isoformat()} "
                        f"as carryover for {today}")
            return lineups

    logger.info("No recent lineup cache found -- will use fallback rosters")
    return {}


# ---------------------------------------------------------------------------
# Fetch schedule
# ---------------------------------------------------------------------------

def get_todays_games(game_date: Optional[str] = None) -> list[dict]:
    game_date = game_date or date.today().isoformat()
    params = {
        "sportId": 1,
        "date":    game_date,
        "hydrate": "team,probablePitcher(note),lineups",
    }
    try:
        resp = requests.get(MLB_SCHEDULE_URL, params=params,
                            headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error(f"MLB API schedule fetch failed: {exc}")
        return []

    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            games.append(game)

    logger.info(f"Found {len(games)} games for {game_date}")
    return games


# ---------------------------------------------------------------------------
# Parse a single game
# ---------------------------------------------------------------------------

def parse_game(game: dict) -> Optional[dict]:
    game_pk   = game.get("gamePk")
    game_time = game.get("gameDate", "")
    home_data = game.get("teams", {}).get("home", {})
    away_data = game.get("teams", {}).get("away", {})

    home_abbrev = home_data.get("team", {}).get("abbreviation", "")
    away_abbrev = away_data.get("team", {}).get("abbreviation", "")

    # Starting pitchers
    home_pitcher = home_data.get("probablePitcher", {})
    away_pitcher = away_data.get("probablePitcher", {})

    starters = {}
    if home_pitcher:
        starters[home_abbrev] = {
            "name": home_pitcher.get("fullName", "TBD"),
            "hand": home_pitcher.get("pitchHand", {}).get("code", "R"),
            "role": "starter",
        }
    if away_pitcher:
        starters[away_abbrev] = {
            "name": away_pitcher.get("fullName", "TBD"),
            "hand": away_pitcher.get("pitchHand", {}).get("code", "R"),
            "role": "starter",
        }

    # Only accept lineups confirmed today with 8+ players
    lineups_data = game.get("lineups", {})
    home_players = lineups_data.get("homePlayers", [])
    away_players = lineups_data.get("awayPlayers", [])

    def parse_players(players: list) -> list:
        lineup = []
        for i, p in enumerate(players, 1):
            lineup.append({
                "order": i,
                "name":  p.get("fullName", "Unknown"),
                "bats":  HAND_MAP.get(
                    p.get("batSide", {}).get("code", "R"), "R"
                ),
                "pos":   p.get("primaryPosition", {}).get("abbreviation", ""),
            })
        return lineup

    lineups = {}
    if len(home_players) >= 8:
        lineups[home_abbrev] = parse_players(home_players)
    if len(away_players) >= 8:
        lineups[away_abbrev] = parse_players(away_players)

    return {
        "game_pk":   game_pk,
        "home":      home_abbrev,
        "away":      away_abbrev,
        "starters":  starters,
        "lineups":   lineups,
        "game_time": game_time,
    }


# ---------------------------------------------------------------------------
# Write daily_lineup.py
# ---------------------------------------------------------------------------

def write_daily_lineup(games: list[dict], game_date: str) -> None:
    # Carry over bullpen data
    existing_bullpen = {}
    try:
        from daily_lineup import BULLPEN_AVAILABILITY
        existing_bullpen = BULLPEN_AVAILABILITY
        for team, arms in existing_bullpen.items():
            for arm in arms:
                arm["rest_days"] = min(arm["rest_days"] + 1, 5)
                if arm["rest_days"] >= 1:
                    arm["available"] = True
        logger.info("Carried over bullpen data (+1 rest day)")
    except (ImportError, Exception):
        pass

    # Collect today's confirmed data
    all_starters      = {}
    all_game_times    = {}
    confirmed_lineups = {}   # team -> lineup, confirmed by API today
    no_pitcher_teams  = set()

    for game in games:
        all_starters.update(game.get("starters", {}))
        home = game.get("home", "")
        away = game.get("away", "")

        if home:
            all_game_times[home] = game.get("game_time", "")

        # Only confirmed lineups from today's API
        for team, lineup in game.get("lineups", {}).items():
            confirmed_lineups[team] = lineup

        # Track teams with no pitcher posted
        for team in [home, away]:
            if team and team not in game.get("starters", {}):
                no_pitcher_teams.add(team)

    # Save today's confirmed lineups to cache
    if confirmed_lineups:
        save_lineup_cache(confirmed_lineups, game_date)

    # Build final lineups — confirmed first, then yesterday's cache, then fallback
    yesterday_cache = load_yesterday_cache(game_date)
    all_lineups     = {}
    carried_teams   = set()
    fallback_teams  = set()

    # All teams playing today
    all_teams = set()
    for game in games:
        all_teams.add(game.get("home", ""))
        all_teams.add(game.get("away", ""))
    all_teams.discard("")

    for team in all_teams:
        if team in confirmed_lineups:
            # Best case — confirmed today
            all_lineups[team] = confirmed_lineups[team]
        elif team in yesterday_cache:
            # Good case — use yesterday's actual confirmed lineup
            all_lineups[team] = yesterday_cache[team]
            carried_teams.add(team)
        elif team in FALLBACK_ROSTERS:
            # Last resort — hardcoded 2026 roster
            all_lineups[team] = FALLBACK_ROSTERS[team]
            fallback_teams.add(team)

    # Write daily_lineup.py
    lines = [
        "# -*- coding: utf-8 -*-",
        '"""',
        "daily_lineup.py",
        "----------------",
        f"Auto-generated by fetch_lineups.py on {game_date}",
        "Source: MLB Stats API + lineup cache",
        '"""',
        "",
        "from datetime import date",
        "",
        f'TODAY = "{game_date}"',
        "",
        "# Teams with no pitcher posted today",
        "NO_PITCHER_TEAMS = " + repr(sorted(no_pitcher_teams)),
        "",
        "# Teams using yesterday's confirmed lineup",
        "CARRIED_LINEUP_TEAMS = " + repr(sorted(carried_teams)),
        "",
        "# Teams using fallback hardcoded roster",
        "FALLBACK_LINEUP_TEAMS = " + repr(sorted(fallback_teams)),
        "",
        "GAME_TIMES_UTC = {",
    ]

    for team, gt in sorted(all_game_times.items()):
        lines.append(f'    "{team}": "{gt}",')
    lines += ["}", ""]

    lines += ["TODAYS_STARTERS = {"]
    for team, pitcher in sorted(all_starters.items()):
        lines.append(
            f'    "{team}": {{"name": "{pitcher["name"]}", '
            f'"hand": "{pitcher["hand"]}", "role": "starter"}},'
        )
    if not all_starters:
        lines.append("    # No starters confirmed yet")
    lines += ["}", ""]

    lines += ["BULLPEN_AVAILABILITY = {"]
    if existing_bullpen:
        for team, arms in sorted(existing_bullpen.items()):
            lines.append(f'    "{team}": [')
            for arm in arms:
                lines.append(
                    f'        {{"name": "{arm["name"]}", '
                    f'"hand": "{arm["hand"]}", '
                    f'"rest_days": {arm["rest_days"]}, '
                    f'"available": {arm["available"]}}},'
                )
            lines.append("    ],")
    lines += ["}", ""]

    lines += ["TODAYS_LINEUPS = {"]
    for team, lineup in sorted(all_lineups.items()):
        lines.append(f'    "{team}": [')
        for p in lineup:
            lines.append(
                f'        {{"order": {p["order"]}, '
                f'"name": "{p["name"]}", '
                f'"bats": "{p["bats"]}", '
                f'"pos": "{p["pos"]}"}}, '
            )
        lines.append("    ],")
    if not all_lineups:
        lines.append(
            "    # No lineups available yet"
        )
    lines += ["}", "", "SCRATCHES = []", ""]
    lines += ["GAME_TIMES = GAME_TIMES_UTC", ""]

    with open("daily_lineup.py", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Print summary
    print(f"\n{'='*60}")
    print(f"  LINEUP UPDATE -- {game_date}")
    print(f"{'='*60}")

    if all_starters:
        print(f"\n  Starting Pitchers ({len(all_starters)}):")
        for team, p in sorted(all_starters.items()):
            gt = all_game_times.get(team, "")
            print(f"  [OK] {team}: {p['name']} ({p['hand']}HP)  {gt}")

    if no_pitcher_teams:
        print(f"\n  No pitcher posted:")
        for t in sorted(no_pitcher_teams):
            print(f"  [--] {t}")

    print(f"\n  Lineups:")
    if confirmed_lineups:
        print(f"  Confirmed today:    {', '.join(sorted(confirmed_lineups.keys()))}")
    if carried_teams:
        print(f"  From yesterday:     {', '.join(sorted(carried_teams))}")
    if fallback_teams:
        print(f"  Hardcoded fallback: {', '.join(sorted(fallback_teams))}")

    total = len(confirmed_lineups) + len(carried_teams) + len(fallback_teams)
    print(f"  Total: {total}/{len(all_teams)} teams have lineups")

    print(f"\n  Next steps:")
    print(f"    python save_predictions.py")
    print(f"    python parlay_builder.py")
    print(f"    python matchup_predictor.py")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch today's MLB lineups"
    )
    parser.add_argument("--date", type=str, default=None,
                        help="Date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    game_date = args.date or date.today().isoformat()
    logger.info(f"Fetching MLB lineups for {game_date}...")

    games_raw = get_todays_games(game_date)
    if not games_raw:
        logger.warning("No games found")
        return

    games = [parse_game(g) for g in games_raw if parse_game(g)]
    write_daily_lineup(games, game_date)


if __name__ == "__main__":
    main()