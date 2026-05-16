"""
mlb_analytics/ingestion/savant_season_stats.py
-----------------------------------------------
Pulls full season Statcast stats from Baseball Savant
for all batters and pitchers.

No API key needed — uses the public CSV export endpoint.
Matches players by MLBAM player_id so no name matching issues.

Stores results in:
  - savant_batter_season  (new table)
  - savant_pitcher_season (new table)

Usage:
    python -m mlb_analytics.ingestion.savant_season_stats
    python -m mlb_analytics.ingestion.savant_season_stats --seasons 2023 2024 2025
"""

from __future__ import annotations

import argparse
import io
import time
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from mlb_analytics.db import engine

SAVANT_URL = "https://baseballsavant.mlb.com/leaderboard/custom"
HEADERS    = {"User-Agent": "Mozilla/5.0"}

# ---------------------------------------------------------------------------
# Batter stat selections
# ---------------------------------------------------------------------------
BATTER_SELECTIONS = ",".join([
    "xba", "xslg", "xwoba", "xobp",
    "exit_velocity_avg", "hard_hit_percent",
    "barrel_batted_rate", "k_percent", "bb_percent",
    "sprint_speed", "avg_hyper_speed",
])

# ---------------------------------------------------------------------------
# Pitcher stat selections
# ---------------------------------------------------------------------------
PITCHER_SELECTIONS = ",".join([
    "xera", "xba", "xslg", "xwoba",
    "exit_velocity_avg", "hard_hit_percent",
    "barrel_batted_rate", "k_percent", "bb_percent",
    "whiff_percent", "put_away",
    "p_era", "p_formatted_ip",
])


def fetch_savant_csv(
    year: int,
    player_type: str,
    selections: str,
    min_pa: int = 25,
) -> Optional[pd.DataFrame]:
    """
    Pull a Savant leaderboard as a CSV and return as DataFrame.
    player_type: 'batter' or 'pitcher'
    """
    # Use lower min PA for current season
    import datetime
    current_year = datetime.date.today().year
    if year == current_year:
        min_pa = max(10, min_pa // 3)  # lower threshold for in-progress season

    params = {
        "year":       str(year),
        "type":       player_type,
        "filter":     "",
        "sort":       "xwoba",
        "sortDir":    "desc",
        "min":        str(min_pa),
        "selections": selections,
        "csv":        "true",
    }

    try:
        resp = requests.get(
            SAVANT_URL, params=params,
            headers=HEADERS, timeout=30
        )
        resp.raise_for_status()

        # Strip BOM if present
        text = resp.text.lstrip("\ufeff")
        df   = pd.read_csv(io.StringIO(text))

        # Clean column names
        df.columns = [c.strip().strip('"').strip() for c in df.columns]

        logger.success(f"  Savant {player_type} {year}: {len(df)} players")
        return df

    except Exception as exc:
        logger.error(f"Savant fetch failed ({player_type} {year}): {exc}")
        return None


def process_batter_stats(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Clean and standardize batter stat columns."""
    rename = {
        "last_name, first_name": "name_raw",
        "player_id":             "player_id",
        "xba":                   "xba",
        "xslg":                  "xslg",
        "xwoba":                 "xwoba",
        "xobp":                  "xobp",
        "exit_velocity_avg":     "avg_exit_velo",
        "hard_hit_percent":      "hard_hit_pct",
        "barrel_batted_rate":    "barrel_pct",
        "k_percent":             "k_pct",
        "bb_percent":            "bb_pct",
        "sprint_speed":          "sprint_speed",
    }

    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["season"] = year

    # Convert pct columns from 0-100 to 0-1
    for col in ["hard_hit_pct", "barrel_pct", "k_pct", "bb_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    # Convert string stats to float
    for col in ["xba", "xslg", "xwoba", "xobp", "avg_exit_velo"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = ["player_id", "season", "xba", "xslg", "xwoba", "xobp",
            "avg_exit_velo", "hard_hit_pct", "barrel_pct",
            "k_pct", "bb_pct", "sprint_speed"]
    return df[[c for c in keep if c in df.columns]]


def process_pitcher_stats(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Clean and standardize pitcher stat columns."""
    rename = {
        "last_name, first_name": "name_raw",
        "player_id":             "player_id",
        "xera":                  "xera",
        "xba":                   "xba",
        "xslg":                  "xslg",
        "xwoba":                 "xwoba",
        "exit_velocity_avg":     "avg_exit_velo_allowed",
        "hard_hit_percent":      "hard_hit_pct_allowed",
        "barrel_batted_rate":    "barrel_pct_allowed",
        "k_percent":             "k_pct",
        "bb_percent":            "bb_pct",
        "whiff_percent":         "whiff_pct",
        "put_away":              "put_away_pct",
    }

    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["season"] = year

    for col in ["hard_hit_pct_allowed", "barrel_pct_allowed",
                "k_pct", "bb_pct", "whiff_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    for col in ["xera", "xba", "xslg", "xwoba", "avg_exit_velo_allowed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = ["player_id", "season", "xera", "xba", "xslg", "xwoba",
            "avg_exit_velo_allowed", "hard_hit_pct_allowed",
            "barrel_pct_allowed", "k_pct", "bb_pct",
            "whiff_pct", "put_away_pct"]
    return df[[c for c in keep if c in df.columns]]


def save_batter_stats(df: pd.DataFrame) -> None:
    if df.empty:
        return
    # Use replace to handle re-runs and updates mid-season
    # Use named columns to avoid column count mismatch issues
    from sqlalchemy import text as _text
    with engine.begin() as conn:
        for _, row in df.iterrows():
            d = row.to_dict()
            conn.execute(_text("""
                INSERT OR REPLACE INTO savant_batter_season
                    (player_id, season, xba, xslg, xwoba, xobp,
                     avg_exit_velo, hard_hit_pct, barrel_pct,
                     k_pct, bb_pct, sprint_speed)
                VALUES
                    (:player_id, :season, :xba, :xslg, :xwoba, :xobp,
                     :avg_exit_velo, :hard_hit_pct, :barrel_pct,
                     :k_pct, :bb_pct, :sprint_speed)
            """), d)
    logger.success(f"  Saved {len(df)} batter season rows")


def save_pitcher_stats(df: pd.DataFrame) -> None:
    if df.empty:
        return
    from sqlalchemy import text as _text
    with engine.begin() as conn:
        for _, row in df.iterrows():
            d = row.to_dict()
            # Normalize column names — Savant uses _allowed suffix for pitcher stats
            normalized = {
                "player_id":    d.get("player_id"),
                "season":       d.get("season"),
                "xera":         d.get("xera"),
                "xba":          d.get("xba"),
                "xslg":         d.get("xslg"),
                "xwoba":        d.get("xwoba"),
                "avg_exit_velo":d.get("avg_exit_velo_allowed") or d.get("avg_exit_velo"),
                "hard_hit_pct": d.get("hard_hit_pct_allowed") or d.get("hard_hit_pct"),
                "barrel_pct":   d.get("barrel_pct_allowed") or d.get("barrel_pct"),
                "k_pct":        d.get("k_pct"),
                "bb_pct":       d.get("bb_pct"),
                "whiff_pct":    d.get("whiff_pct"),
                "put_away":     d.get("put_away_pct") or d.get("put_away"),
            }
            conn.execute(_text("""
                INSERT OR REPLACE INTO savant_pitcher_season
                    (player_id, season, xera, xba, xslg, xwoba,
                     avg_exit_velo_allowed, hard_hit_pct_allowed, barrel_pct_allowed,
                     k_pct, bb_pct, whiff_pct, put_away_pct)
                VALUES
                    (:player_id, :season, :xera, :xba, :xslg, :xwoba,
                     :avg_exit_velo, :hard_hit_pct, :barrel_pct,
                     :k_pct, :bb_pct, :whiff_pct, :put_away)
            """), normalized)
    logger.success(f"  Saved {len(df)} pitcher season rows")


def ingest_season(year: int) -> None:
    """Pull and store both batter and pitcher stats for one season."""
    logger.info(f"Ingesting Savant season stats for {year}...")

    # Batters
    bat_raw = fetch_savant_csv(year, "batter", BATTER_SELECTIONS)
    if bat_raw is not None:
        bat_clean = process_batter_stats(bat_raw, year)
        save_batter_stats(bat_clean)

    time.sleep(1)

    # Pitchers
    pit_raw = fetch_savant_csv(year, "pitcher", PITCHER_SELECTIONS)
    if pit_raw is not None:
        pit_clean = process_pitcher_stats(pit_raw, year)
        save_pitcher_stats(pit_clean)


def get_player_season_stats(
    player_id: int,
    season: int,
    player_type: str = "batter",
) -> dict:
    """
    Quick lookup of a player's season stats from DB.
    Returns dict of stats or empty dict if not found.
    """
    from sqlalchemy import text
    table = "savant_batter_season" if player_type == "batter" \
            else "savant_pitcher_season"
    try:
        with engine.connect() as conn:
            row = pd.read_sql(
                text(f"SELECT * FROM {table} "
                     f"WHERE player_id = :pid AND season = :s"),
                conn, params={"pid": player_id, "s": season}
            )
        return row.iloc[0].to_dict() if not row.empty else {}
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull Savant season stats for all players"
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int,
        default=[2023, 2024, 2025, 2026],
    )
    args = parser.parse_args()

    # Create tables if needed
    with engine.begin() as conn:
        conn.execute(__import__("sqlalchemy").text("""
            CREATE TABLE IF NOT EXISTS savant_batter_season (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id    INTEGER,
                season       INTEGER,
                xba          REAL,
                xslg         REAL,
                xwoba        REAL,
                xobp         REAL,
                avg_exit_velo REAL,
                hard_hit_pct REAL,
                barrel_pct   REAL,
                k_pct        REAL,
                bb_pct       REAL,
                sprint_speed REAL,
                UNIQUE(player_id, season)
            )
        """))
        conn.execute(__import__("sqlalchemy").text("""
            CREATE TABLE IF NOT EXISTS savant_pitcher_season (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id             INTEGER,
                season                INTEGER,
                xera                  REAL,
                xba                   REAL,
                xslg                  REAL,
                xwoba                 REAL,
                avg_exit_velo_allowed REAL,
                hard_hit_pct_allowed  REAL,
                barrel_pct_allowed    REAL,
                k_pct                 REAL,
                bb_pct                REAL,
                whiff_pct             REAL,
                put_away_pct          REAL,
                UNIQUE(player_id, season)
            )
        """))

    for season in args.seasons:
        ingest_season(season)
        time.sleep(2)

    logger.success("Savant season stats ingestion complete!")
    logger.info("Now retrain: python train_props.py --seasons 2023 2024 2025")


if __name__ == "__main__":
    main()