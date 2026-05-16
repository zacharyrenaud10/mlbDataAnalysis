# -*- coding: utf-8 -*-
"""
data_fetchers/injuries.py
-------------------------
Pulls today's MLB injury list and flags games where
a starting pitcher or key batter is on the IL or questionable.

Source: MLB Stats API (free, no key needed)

Usage:
    python data_fetchers/injuries.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import requests
from loguru import logger

Path("cache").mkdir(exist_ok=True)

MLB_API  = "https://statsapi.mlb.com/api/v1"
HEADERS  = {"User-Agent": "Mozilla/5.0"}

# Teams playing today
def get_todays_teams() -> dict:
    """Get home/away teams for today's games."""
    try:
        from daily_lineup import TODAYS_STARTERS, GAME_TIMES
        teams = {}
        for team in TODAYS_STARTERS:
            teams[team] = TODAYS_STARTERS[team].get("name", "")
        return teams
    except ImportError:
        return {}


def fetch_team_injuries(team_id: int, team_abb: str) -> list[dict]:
    """Fetch IL and day-to-day players for a team."""
    try:
        resp = requests.get(
            f"{MLB_API}/teams/{team_id}/roster",
            params={"rosterType": "injuries"},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data    = resp.json()
        players = data.get("roster", [])

        injured = []
        for p in players:
            name   = p.get("person", {}).get("fullName", "")
            status = p.get("status", {}).get("description", "")
            pos    = p.get("position", {}).get("abbreviation", "")
            injured.append({
                "name":   name,
                "status": status,
                "pos":    pos,
                "team":   team_abb,
            })
        return injured
    except Exception:
        return []


def fetch_all_injuries() -> dict:
    """Fetch injuries for all 30 MLB teams."""
    # Team ID mapping
    TEAM_IDS = {
        "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
        "CWS": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
        "HOU": 117, "KC":  118, "LAA": 108, "LAD": 119, "MIA": 146,
        "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
        "ATH": 133, "PHI": 143, "PIT": 134, "SD":  135, "SF":  137,
        "SEA": 136, "STL": 138, "TB":  139, "TEX": 140, "TOR": 141,
        "WSH": 120,
    }

    all_injuries = {}
    todays_teams = get_todays_teams()

    # Only fetch for teams playing today
    teams_to_check = list(todays_teams.keys()) if todays_teams else list(TEAM_IDS.keys())

    for team_abb in teams_to_check:
        team_id = TEAM_IDS.get(team_abb)
        if not team_id:
            continue
        injured = fetch_team_injuries(team_id, team_abb)
        if injured:
            all_injuries[team_abb] = injured

    return all_injuries


def check_starter_injuries(injuries: dict) -> list[dict]:
    """Flag games where today's starting pitcher is injured or questionable."""
    alerts = []
    try:
        from daily_lineup import TODAYS_STARTERS
    except ImportError:
        return alerts

    for team, sp in TODAYS_STARTERS.items():
        sp_name   = sp.get("name", "")
        team_il   = injuries.get(team, [])

        for player in team_il:
            # Check if starter name matches IL player
            if any(part.lower() in player["name"].lower()
                   for part in sp_name.split() if len(part) > 3):
                alerts.append({
                    "team":    team,
                    "pitcher": sp_name,
                    "status":  player["status"],
                    "alert":   f"⚠️  {sp_name} ({team}) is on IL: {player['status']} -- SKIP this game!",
                })

    return alerts


def flag_key_batters(injuries: dict) -> list[dict]:
    """Flag games where key batters (top 5 in lineup) are injured."""
    alerts = []
    try:
        from daily_lineup import TODAYS_LINEUPS
    except ImportError:
        return alerts

    # Consider top 5 in lineup as "key batters"
    for team, lineup in TODAYS_LINEUPS.items():
        key_batters = [p["name"] for p in lineup if p.get("order", 9) <= 5]
        team_il     = injuries.get(team, [])

        for player in team_il:
            for batter in key_batters:
                if any(part.lower() in player["name"].lower()
                       for part in batter.split() if len(part) > 3):
                    alerts.append({
                        "team":   team,
                        "batter": batter,
                        "status": player["status"],
                        "alert":  f"⚠️  {batter} ({team}) is on IL: {player['status']}",
                    })

    return alerts


def save_injury_cache(injuries: dict, alerts: list) -> None:
    cache = {
        "date":     date.today().isoformat(),
        "injuries": injuries,
        "alerts":   alerts,
    }
    with open("cache/injuries_today.json", "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def main() -> None:
    print(f"\n{'='*60}")
    print(f"  MLB INJURY REPORT -- {date.today()}")
    print(f"{'='*60}")

    print("\n  Fetching injury data...")
    injuries = fetch_all_injuries()

    # Check starters
    starter_alerts = check_starter_injuries(injuries)
    batter_alerts  = flag_key_batters(injuries)
    all_alerts     = starter_alerts + batter_alerts

    if starter_alerts:
        print(f"\n  🚨 PITCHER ALERTS (DO NOT BET THESE GAMES):")
        for a in starter_alerts:
            print(f"    {a['alert']}")
    else:
        print(f"\n  ✅ No starting pitcher injuries flagged")

    if batter_alerts:
        print(f"\n  ⚠️  KEY BATTER ALERTS:")
        for a in batter_alerts[:5]:  # show top 5
            print(f"    {a['alert']}")

    # Show full IL by team
    print(f"\n  Full IL list for today's teams:")
    for team, players in sorted(injuries.items()):
        if players:
            names = [f"{p['name']} ({p['status'][:15]})" for p in players[:3]]
            print(f"    {team}: {', '.join(names)}"
                  + (" +" + str(len(players)-3) + " more" if len(players) > 3 else ""))

    save_injury_cache(injuries, all_alerts)
    print(f"\n  Saved to cache/injuries_today.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
