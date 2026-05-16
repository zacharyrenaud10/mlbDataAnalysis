"""
data_fetchers/bullpen_and_umpires.py
-------------------------------------
Fetches from the official MLB Stats API (free, no key):
  1. Bullpen usage — who pitched yesterday, rest days
  2. Home plate umpire assignments — affects K/BB rates
"""

from __future__ import annotations
from datetime import date, timedelta
from typing import Optional
import requests
from loguru import logger

MLB_API = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# ---------------------------------------------------------------------------
# Umpire tendencies (2025 data — update as 2026 accumulates)
# Strikeout tendency: >1.0 = more Ks than average, <1.0 = fewer
# ---------------------------------------------------------------------------
UMPIRE_TENDENCIES = {
    "Angel Hernandez":   {"k_factor": 0.94, "bb_factor": 1.08, "notes": "Wide zone"},
    "CB Bucknor":        {"k_factor": 0.91, "bb_factor": 1.12, "notes": "Very wide zone"},
    "Joe West":          {"k_factor": 0.96, "bb_factor": 1.05, "notes": "Slightly wide"},
    "Doug Eddings":      {"k_factor": 1.05, "bb_factor": 0.97, "notes": "Tight zone"},
    "Laz Diaz":          {"k_factor": 1.08, "bb_factor": 0.94, "notes": "Very tight zone"},
    "Lance Barksdale":   {"k_factor": 0.97, "bb_factor": 1.03, "notes": "Slightly wide"},
    "Hunter Wendelstedt":{"k_factor": 1.03, "bb_factor": 0.98, "notes": "Average"},
    "Jim Reynolds":      {"k_factor": 1.06, "bb_factor": 0.95, "notes": "Tight zone"},
    "Mark Carlson":      {"k_factor": 1.02, "bb_factor": 0.99, "notes": "Average"},
    "Dan Iassogna":      {"k_factor": 1.04, "bb_factor": 0.97, "notes": "Slightly tight"},
}

DEFAULT_UMPIRE = {"k_factor": 1.0, "bb_factor": 1.0, "notes": "No data"}


# ---------------------------------------------------------------------------
# Bullpen usage
# ---------------------------------------------------------------------------

def get_yesterdays_pitchers(team_abbrev: str) -> list[dict]:
    """
    Returns list of pitchers who appeared in yesterday's game
    for a given team — so we know who needs rest today.
    """
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    # Get yesterday's schedule
    params = {
        "sportId": 1,
        "date":    yesterday,
        "hydrate": "team",
    }
    try:
        resp = requests.get(
            f"{MLB_API}/schedule", params=params,
            headers=HEADERS, timeout=10
        )
        resp.raise_for_status()
        schedule = resp.json()
    except Exception as exc:
        logger.error(f"Schedule fetch failed: {exc}")
        return []

    # Find game involving this team
    game_pk = None
    for date_entry in schedule.get("dates", []):
        for game in date_entry.get("games", []):
            home = game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
            away = game.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")
            if team_abbrev in (home, away):
                game_pk = game.get("gamePk")
                break

    if not game_pk:
        logger.info(f"{team_abbrev} did not play yesterday")
        return []

    # Get boxscore for that game
    try:
        resp = requests.get(
            f"{MLB_API}/game/{game_pk}/boxscore",
            headers=HEADERS, timeout=10
        )
        resp.raise_for_status()
        box = resp.json()
    except Exception as exc:
        logger.error(f"Boxscore fetch failed: {exc}")
        return []

    # Find pitchers for our team
    teams_data = box.get("teams", {})
    pitchers   = []

    for side in ["home", "away"]:
        team_data = teams_data.get(side, {})
        abbrev    = team_data.get("team", {}).get("abbreviation", "")
        if abbrev != team_abbrev:
            continue

        players = team_data.get("players", {})
        for player_key, player in players.items():
            stats = player.get("stats", {}).get("pitching", {})
            if stats.get("inningsPitched", "0.0") != "0.0":
                info   = player.get("person", {})
                detail = player.get("gameStatus", {})
                pitchers.append({
                    "name":           info.get("fullName", "Unknown"),
                    "id":             info.get("id"),
                    "innings_pitched":stats.get("inningsPitched", "0.0"),
                    "pitches":        stats.get("numberOfPitches", 0),
                    "is_starter":     detail.get("isCurrentPitcher", False),
                    "rest_days":      0,  # pitched yesterday
                    "available":      False,
                })

    logger.info(f"Found {len(pitchers)} pitchers from {team_abbrev} yesterday")
    return pitchers


def build_bullpen_availability(
    team_abbrev: str,
    existing_bullpen: Optional[list] = None,
) -> list[dict]:
    """
    Builds updated bullpen availability by:
    1. Starting from existing data (adds +1 rest day to everyone)
    2. Marking anyone who pitched yesterday as unavailable
    """
    # Get yesterday's pitchers
    yesterdays_pitchers = get_yesterdays_pitchers(team_abbrev)
    yesterday_names     = {p["name"] for p in yesterdays_pitchers}

    if existing_bullpen:
        # Update existing roster
        updated = []
        for arm in existing_bullpen:
            pitched_yesterday = arm["name"] in yesterday_names
            updated.append({
                "name":      arm["name"],
                "hand":      arm["hand"],
                "rest_days": 0 if pitched_yesterday else min(arm["rest_days"] + 1, 5),
                "available": not pitched_yesterday,
            })
        # Add any new pitchers we haven't seen before
        known_names = {a["name"] for a in existing_bullpen}
        for p in yesterdays_pitchers:
            if p["name"] not in known_names:
                updated.append({
                    "name":      p["name"],
                    "hand":      "R",  # default — update manually
                    "rest_days": 0,
                    "available": False,
                })
        return updated
    else:
        # Fresh start — just use yesterday's pitchers
        return [
            {
                "name":      p["name"],
                "hand":      "R",  # will need manual update for handedness
                "rest_days": p["rest_days"],
                "available": p["available"],
            }
            for p in yesterdays_pitchers
        ]


# ---------------------------------------------------------------------------
# Umpire assignments
# ---------------------------------------------------------------------------

def get_todays_umpires(game_date: Optional[str] = None) -> dict[str, dict]:
    """
    Returns dict of {game_pk: umpire_info} for today's games.
    MLB API doesn't always expose HP umpire pre-game,
    so we fall back to the officials endpoint.
    """
    game_date = game_date or date.today().isoformat()

    params = {
        "sportId": 1,
        "date":    game_date,
        "hydrate": "officials",
    }
    try:
        resp = requests.get(
            f"{MLB_API}/schedule", params=params,
            headers=HEADERS, timeout=10
        )
        resp.raise_for_status()
        schedule = resp.json()
    except Exception as exc:
        logger.error(f"Umpire fetch failed: {exc}")
        return {}

    results = {}
    for date_entry in schedule.get("dates", []):
        for game in date_entry.get("games", []):
            game_pk   = game.get("gamePk")
            officials = game.get("officials", [])
            home_team = game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
            away_team = game.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")

            hp_ump = None
            for official in officials:
                if official.get("officialType") == "Home Plate":
                    hp_ump = official.get("official", {}).get("fullName", "Unknown")
                    break

            if not hp_ump and officials:
                # Sometimes listed as first official
                hp_ump = officials[0].get("official", {}).get("fullName", "Unknown")

            tendencies = UMPIRE_TENDENCIES.get(hp_ump, DEFAULT_UMPIRE)

            results[game_pk] = {
                "home_team":   home_team,
                "away_team":   away_team,
                "hp_umpire":   hp_ump or "TBD",
                "k_factor":    tendencies["k_factor"],
                "bb_factor":   tendencies["bb_factor"],
                "notes":       tendencies["notes"],
            }

    logger.info(f"Found umpires for {len(results)} games")
    return results


def get_umpire_for_teams(
    home_team: str,
    away_team: str,
    game_date: Optional[str] = None,
) -> dict:
    """Get umpire info for a specific matchup."""
    all_umpires = get_todays_umpires(game_date)
    for game_pk, info in all_umpires.items():
        if info["home_team"] == home_team or info["away_team"] == away_team:
            return info
    return {"hp_umpire": "TBD", "k_factor": 1.0, "bb_factor": 1.0, "notes": "Not found"}


def print_umpire_report(ump: dict) -> None:
    if not ump or ump.get("hp_umpire") == "TBD":
        print("  ⚖️  HP Umpire: TBD")
        return

    k   = ump["k_factor"]
    bb  = ump["bb_factor"]

    k_label  = ("🔴 Tight zone (+Ks)" if k > 1.04
                else "🟢 Wide zone (-Ks)" if k < 0.96
                else "🟡 Average zone")
    bb_label = ("🟢 More walks" if bb > 1.04
                else "🔴 Fewer walks" if bb < 0.96
                else "🟡 Average walks")

    print(f"  ⚖️  HP Umpire: {ump['hp_umpire']}")
    print(f"     K Factor:  {k:.2f}x  — {k_label}")
    print(f"     BB Factor: {bb:.2f}x  — {bb_label}")
    print(f"     Notes:     {ump['notes']}")


if __name__ == "__main__":
    print("Testing bullpen fetch for NYY...")
    arms = get_yesterdays_pitchers("NYY")
    for a in arms:
        print(f"  {a['name']} — {a['innings_pitched']} IP")

    print("\nTesting umpire fetch...")
    umps = get_todays_umpires()
    for gk, u in umps.items():
        print(f"  {u['away_team']} @ {u['home_team']}: {u['hp_umpire']} "
              f"(K:{u['k_factor']:.2f}x)")