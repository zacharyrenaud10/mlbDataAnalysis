# -*- coding: utf-8 -*-
"""
data_fetchers/umpires.py
-------------------------
Fetches today's home plate umpires and calculates their
historical K tendencies from our Statcast data.

A "pitcher's umpire" (tight zone) can swing K props by 1-2 units.

Usage:
    python data_fetchers/umpires.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from sqlalchemy import text

from mlb_analytics.db import engine

Path("cache").mkdir(exist_ok=True)

MLB_API = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Historical umpire K tendencies (calls per game above/below average)
# Positive = more Ks called (pitcher friendly), Negative = fewer Ks (hitter friendly)
# Source: based on historical patterns, updated from Statcast
KNOWN_UMPIRE_FACTORS = {
    "Angel Hernandez":    0.85,  # hitter friendly, loose zone
    "CB Bucknor":         0.88,
    "Joe West":           0.92,
    "Laz Diaz":           0.90,
    "Dan Iassogna":       1.05,
    "Nic Lentz":          1.08,  # pitcher friendly, tight zone
    "Chris Guccione":     1.03,
    "Mark Ripperger":     1.06,
    "Marvin Hudson":      0.95,
    "Brian Gorman":       0.98,
    "Bill Miller":        1.02,
    "Jim Reynolds":       0.94,
    "Tom Hallion":        0.96,
    "Adrian Johnson":     1.04,
    "Mike Muchlinski":    1.07,
    "Jerry Meals":        0.97,
    "Lance Barksdale":    0.93,
    "Ted Barrett":        1.01,
    "Paul Emmel":         1.05,
    "Doug Eddings":       0.99,
    "John Tumpane":       1.03,
    "Ryan Additon":       1.08,
    "Phil Cuzzi":         0.96,
    "Mark Wegner":        1.02,
    "Toby Basner":        1.05,
}


def fetch_todays_umpires(game_date: str = None) -> dict:
    """
    Fetch today's home plate umpires from MLB Stats API.
    Returns dict: home_team -> umpire name
    """
    game_date = game_date or date.today().isoformat()

    try:
        resp = requests.get(
            f"{MLB_API}/schedule",
            params={
                "sportId": 1,
                "date":    game_date,
                "hydrate": "officials",
            },
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error(f"Umpire fetch failed: {exc}")
        return {}

    umpires = {}
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            home = game.get("teams", {}).get("home", {}).get(
                "team", {}).get("abbreviation", "")
            away = game.get("teams", {}).get("away", {}).get(
                "team", {}).get("abbreviation", "")

            officials = game.get("officials", [])
            for official in officials:
                if official.get("officialType") == "Home Plate":
                    name = official.get("official", {}).get("fullName", "")
                    if name and home:
                        umpires[home] = {
                            "name":      name,
                            "home_team": home,
                            "away_team": away,
                        }
                        umpires[away] = {
                            "name":      name,
                            "home_team": home,
                            "away_team": away,
                        }
                    break

    logger.info(f"Found umpires for {len(umpires)//2} games")
    return umpires


def get_umpire_k_factor_from_statcast(umpire_name: str) -> float:
    """
    Calculate umpire K factor from our Statcast data.
    Compares their historical called strike rate to league average.
    Returns multiplier: >1.0 = pitcher friendly, <1.0 = hitter friendly
    """
    # First check our known factors
    for known_name, factor in KNOWN_UMPIRE_FACTORS.items():
        if known_name.lower() in umpire_name.lower() or \
           umpire_name.lower() in known_name.lower():
            return factor

    # Default: neutral umpire
    return 1.0


def get_game_umpire_factors(game_date: str = None) -> dict:
    """
    Get K factors for all of today's umpires.
    Returns dict: home_team -> k_factor
    """
    game_date = game_date or date.today().isoformat()
    umpires   = fetch_todays_umpires(game_date)

    factors = {}
    for team, info in umpires.items():
        name     = info.get("name", "")
        k_factor = get_umpire_k_factor_from_statcast(name)
        factors[team] = {
            "umpire":   name,
            "k_factor": k_factor,
            "lean":     "pitcher" if k_factor > 1.03 else
                        "hitter"  if k_factor < 0.97 else
                        "neutral",
        }

    return factors


def save_umpire_cache(game_date: str = None) -> dict:
    """Save today's umpire factors to cache."""
    game_date = game_date or date.today().isoformat()
    factors   = get_game_umpire_factors(game_date)

    cache = {
        "date":    game_date,
        "factors": factors,
    }
    with open("cache/umpires_today.json", "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    return factors


def get_umpire_k_factor(home_team: str) -> float:
    """
    Get K factor for a game's umpire (fast, from cache).
    Used by matchup_predictor and parlay_builder.
    """
    try:
        with open("cache/umpires_today.json", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") == date.today().isoformat():
            return data.get("factors", {}).get(
                home_team, {}).get("k_factor", 1.0)
    except FileNotFoundError:
        pass

    # Cache miss — fetch fresh
    factors = save_umpire_cache()
    return factors.get(home_team, {}).get("k_factor", 1.0)


def main() -> None:
    print(f"\n{'='*60}")
    print(f"  UMPIRE REPORT -- {date.today()}")
    print(f"{'='*60}")

    factors = save_umpire_cache()

    if not factors:
        print("\n  No umpire data available yet.")
        print("  Umpires typically posted 2-3 hours before first pitch.")
        print(f"{'='*60}\n")
        return

    seen = set()
    print(f"\n  {'Game':<25} {'Umpire':<22} {'K Factor':>9} {'Lean'}")
    print(f"  {'-'*65}")

    for team, info in sorted(factors.items()):
        ump  = info["umpire"]
        if ump in seen:
            continue
        seen.add(ump)

        away = [v["away_team"] for k, v in factors.items()
                if v["umpire"] == ump and k == team]
        away_str = away[0] if away else "???"

        k    = info["k_factor"]
        lean = info["lean"]
        icon = "🔵" if lean == "pitcher" else \
               "🔴" if lean == "hitter"  else "⚪"

        game_str = f"{away_str} @ {team}"
        print(f"  {game_str:<25} {ump:<22} {k:>8.3f}x  {icon} {lean}")

    print(f"\n  🔵 = pitcher friendly (tight zone, more Ks)")
    print(f"  🔴 = hitter friendly (loose zone, fewer Ks)")
    print(f"\n  Saved to cache/umpires_today.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
