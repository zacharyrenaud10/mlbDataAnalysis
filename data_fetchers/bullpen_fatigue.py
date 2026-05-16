# -*- coding: utf-8 -*-
"""
data_fetchers/bullpen_fatigue.py
---------------------------------
Calculates bullpen fatigue for each team based on recent
pitcher usage from our Statcast DB.

Logic:
  - Relievers = pitchers with avg < 60 pitches per appearance
  - FATIGUED arm = 30+ pitches in last 2 days OR pitched 2 consecutive days
  - Team fatigue score = % of key relievers who are fatigued
  - Penalty applied to team's effective pitcher score in parlay_builder

Usage:
    python data_fetchers/bullpen_fatigue.py
    from data_fetchers.bullpen_fatigue import get_team_bullpen_fatigue
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import text

from mlb_analytics.db import engine

Path("cache").mkdir(exist_ok=True)


def _build_pitcher_team_map() -> dict:
    """
    Build a pitcher_id -> team mapping using lineup cache files.
    Since 2026 Statcast has NULL team columns, we use the lineup
    cache to map players to teams.
    """
    pitcher_team = {}
    try:
        from daily_lineup import TODAYS_STARTERS, TODAYS_LINEUPS
        # Map starters
        for team, sp in TODAYS_STARTERS.items():
            # We don't have pitcher IDs in starters dict, skip
            pass
        # Map batters from lineups to teams (for batter_id lookup)
        # This helps us cross-reference game_pk participants
    except ImportError:
        pass

    # Load from lineup cache files for last 3 days
    today = date.today()
    for days_back in range(1, 4):
        check_date = (today - timedelta(days=days_back)).isoformat()
        cache_path = Path(f"cache/lineups_{check_date}.json")
        if not cache_path.exists():
            continue
        try:
            import json
            with open(cache_path) as f:
                lineups = json.load(f)
            for team, players in lineups.items():
                for p in players:
                    name = p.get("name", "")
                    pitcher_team[name.lower()] = team
        except Exception:
            continue

    return pitcher_team


def get_bullpen_fatigue(days_back: int = 3) -> dict:
    """
    Calculate bullpen fatigue for all teams.
    Returns dict: team_abbrev -> fatigue info

    Since 2026 Statcast has NULL team columns, we identify relievers
    by pitch count per appearance and use game_pk + pitcher stats
    to determine which team they pitched for.
    """
    today     = date.today()
    cutoff    = (today - timedelta(days=days_back)).isoformat()
    today_str = today.isoformat()

    try:
        # Get pitcher appearances — use pitch count to identify relievers vs starters
        query = text("""
            SELECT
                sp.pitcher_id,
                sp.game_date,
                sp.game_pk,
                p.full_name,
                COUNT(*) as pitches
            FROM statcast_pitches sp
            LEFT JOIN players p ON p.player_id = sp.pitcher_id
            WHERE sp.game_date >= :cutoff
              AND sp.game_date < :today
              AND sp.season = 2026
            GROUP BY sp.pitcher_id, sp.game_date, sp.game_pk, p.full_name
            ORDER BY sp.pitcher_id, sp.game_date
        """)

        with engine.connect() as conn:
            df = pd.read_sql(query, conn,
                             params={"cutoff": cutoff, "today": today_str})

        if df.empty:
            logger.warning("No recent Statcast data for bullpen fatigue")
            return {}

        # Get rolling Savant pitcher data to identify starters vs relievers
        # Starters have batters_faced >= 60 in rolling window
        # Relievers have batters_faced < 60
        starter_query = text("""
            SELECT DISTINCT player_id
            FROM pitcher_rolling_features
            WHERE batters_faced >= 60
              AND window_games = 5
        """)
        with engine.connect() as conn:
            starters_df = pd.read_sql(starter_query, conn)
        starter_ids = set(starters_df["player_id"].tolist())

        # Identify relievers: not a starter AND avg < 55 pitches per game
        pitcher_avg = df.groupby("pitcher_id")["pitches"].mean()
        relievers   = set(
            pid for pid, avg in pitcher_avg.items()
            if pid not in starter_ids and avg < 55
        )

        # Filter to relievers only
        rel_df = df[df["pitcher_id"].isin(relievers)].copy()
        if rel_df.empty:
            logger.info("No relievers found in recent data")
            return {}

        # Check fatigue conditions per reliever
        fatigued = set()
        for pid, group in rel_df.groupby("pitcher_id"):
            group  = group.sort_values("game_date")
            dates  = sorted(group["game_date"].tolist())
            pitches = group.groupby("game_date")["pitches"].sum().to_dict()

            # Condition 1: 30+ pitches in last 2 days
            recent = [d for d in dates
                      if d >= (today - timedelta(days=2)).isoformat()]
            if sum(pitches.get(d, 0) for d in recent) >= 30:
                fatigued.add(pid)
                continue

            # Condition 2: pitched on 2 consecutive days
            for i in range(len(dates) - 1):
                d1 = date.fromisoformat(dates[i])
                d2 = date.fromisoformat(dates[i+1])
                if (d2 - d1).days == 1:
                    fatigued.add(pid)
                    break

        # Map to teams using lineup cache + TODAYS_STARTERS
        # Build name -> team mapping
        name_team = _build_pitcher_team_map()

        # For each game_pk, use the batter_id to cross-ref which team the pitcher is on
        # by looking at what team had that pitcher in their recent lineups
        team_relievers = {}
        team_fatigued  = {}

        # Use pitcher names from players table
        name_df = rel_df.dropna(subset=["full_name"])
        for _, row in name_df.iterrows():
            name = (row.get("full_name") or "").lower()
            team = name_team.get(name)
            if not team:
                # Try partial match
                for key, val in name_team.items():
                    if key.split()[-1] in name or name.split()[-1] in key:
                        team = val
                        break
            if not team:
                continue

            pid = row["pitcher_id"]
            if team not in team_relievers:
                team_relievers[team] = set()
                team_fatigued[team]  = set()
            team_relievers[team].add(pid)
            if pid in fatigued:
                team_fatigued[team].add(pid)

        # If we still have very few teams, fall back to a simpler estimate
        if len(team_relievers) < 3:
            # Just return fatigue counts without team mapping
            logger.info(f"Found {len(fatigued)} fatigued relievers "
                        f"(team mapping limited)")

        result = {}
        for team in team_relievers:
            total   = len(team_relievers[team])
            tired   = len(team_fatigued[team])
            fat_pct = tired / total if total > 0 else 0
            result[team] = {
                "total_relievers": total,
                "fatigued_count":  tired,
                "fatigue_pct":     round(fat_pct, 3),
                "penalty":         round(fat_pct * 0.15, 3),
                "status":          "FATIGUED" if fat_pct >= 0.4 else
                                   "TIRED"    if fat_pct >= 0.2 else
                                   "FRESH",
            }

        logger.success(f"Bullpen fatigue computed for {len(result)} teams")
        return result

        if df.empty:
            logger.warning("No recent Statcast data for bullpen fatigue")
            return {}

        # Identify relievers: avg < 55 pitches per appearance
        pitcher_avg = df.groupby("pitcher_id")["pitches"].mean()
        relievers   = set(pitcher_avg[pitcher_avg < 55].index)

        # Filter to relievers only
        rel_df = df[df["pitcher_id"].isin(relievers)].copy()

        if rel_df.empty:
            return {}

        # For each reliever check fatigue conditions
        fatigued = set()
        for pid, group in rel_df.groupby("pitcher_id"):
            group     = group.sort_values("game_date")
            dates     = sorted(group["game_date"].tolist())
            pitches   = group.set_index("game_date")["pitches"].to_dict()

            # Condition 1: 30+ pitches in last 2 days
            recent = [d for d in dates
                      if d >= (today - timedelta(days=2)).isoformat()]
            recent_pitches = sum(pitches.get(d, 0) for d in recent)
            if recent_pitches >= 30:
                fatigued.add(pid)
                continue

            # Condition 2: pitched on 2 consecutive days in last 3 days
            if len(dates) >= 2:
                for i in range(len(dates) - 1):
                    d1 = date.fromisoformat(dates[i])
                    d2 = date.fromisoformat(dates[i+1])
                    if (d2 - d1).days == 1:
                        fatigued.add(pid)
                        break

        # Map fatigued pitchers to teams
        team_relievers  = {}
        team_fatigued   = {}

        for pid, group in rel_df.groupby("pitcher_id"):
            teams = [t for t in group["pitcher_team"].dropna().unique() if t]
            if not teams:
                continue
            team = teams[0]

            if team not in team_relievers:
                team_relievers[team] = set()
                team_fatigued[team]  = set()

            team_relievers[team].add(pid)
            if pid in fatigued:
                team_fatigued[team].add(pid)

        # Build result
        result = {}
        for team in team_relievers:
            total    = len(team_relievers[team])
            tired    = len(team_fatigued[team])
            fatigue_pct = tired / total if total > 0 else 0

            result[team] = {
                "total_relievers":   total,
                "fatigued_count":    tired,
                "fatigue_pct":       round(fatigue_pct, 3),
                "penalty":           round(fatigue_pct * 0.15, 3),
                "status":            "FATIGUED" if fatigue_pct >= 0.4 else
                                     "TIRED"    if fatigue_pct >= 0.2 else
                                     "FRESH",
            }

        logger.success(f"Bullpen fatigue computed for {len(result)} teams")
        return result

    except Exception as exc:
        logger.error(f"Bullpen fatigue failed: {exc}")
        return {}


def get_team_bullpen_fatigue(team: str) -> dict:
    """Get fatigue info for a specific team."""
    all_fatigue = get_bullpen_fatigue()
    return all_fatigue.get(team, {
        "total_relievers": 0,
        "fatigued_count":  0,
        "fatigue_pct":     0.0,
        "penalty":         0.0,
        "status":          "UNKNOWN",
    })


def save_fatigue_cache() -> dict:
    """Save today's bullpen fatigue to cache."""
    fatigue = get_bullpen_fatigue()
    cache   = {
        "date":    date.today().isoformat(),
        "fatigue": fatigue,
    }
    with open("cache/bullpen_fatigue.json", "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    return fatigue


def load_fatigue_cache() -> dict:
    """Load cached bullpen fatigue (fast, no DB query)."""
    try:
        with open("cache/bullpen_fatigue.json", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") == date.today().isoformat():
            return data.get("fatigue", {})
    except FileNotFoundError:
        pass
    # Cache miss — compute fresh
    return save_fatigue_cache()


def main() -> None:
    print(f"\n{'='*60}")
    print(f"  BULLPEN FATIGUE REPORT -- {date.today()}")
    print(f"{'='*60}")

    fatigue = save_fatigue_cache()

    if not fatigue:
        print("\n  No recent Statcast data available.")
        print("  Run: python -m mlb_analytics.ingestion.statcast_ingest")
        print(f"{'='*60}\n")
        return

    # Sort by fatigue level
    sorted_teams = sorted(fatigue.items(),
                          key=lambda x: x[1]["fatigue_pct"], reverse=True)

    print(f"\n  {'Team':<6} {'Status':<10} {'Fatigued':>9} {'Total':>6} {'Penalty':>8}")
    print(f"  {'-'*45}")

    for team, info in sorted_teams:
        if info["total_relievers"] == 0:
            continue
        status = info["status"]
        icon   = "🔴" if status == "FATIGUED" else \
                 "🟡" if status == "TIRED"    else "🟢"
        print(f"  {team:<6} {icon} {status:<8} "
              f"{info['fatigued_count']:>6}/{info['total_relievers']:<3} "
              f"{info['penalty']*100:>6.1f}%")

    fatigued_teams = [t for t, i in fatigue.items() if i["status"] == "FATIGUED"]
    if fatigued_teams:
        print(f"\n  🔴 FATIGUED bullpens (avoid backing these teams late):")
        for t in fatigued_teams:
            print(f"     {t} — {fatigue[t]['fatigued_count']} tired arms, "
                  f"{fatigue[t]['penalty']*100:.0f}% penalty applied")

    print(f"\n  Saved to cache/bullpen_fatigue.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()