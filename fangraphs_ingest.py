"""
mlb_analytics/ingestion/fangraphs_ingest.py
-------------------------------------------
Pulls FanGraphs seasonal batting and pitching leaderboard data
via pybaseball's pitching_stats() / batting_stats() wrappers.

Usage (CLI):
    python -m mlb_analytics.ingestion.fangraphs_ingest --seasons 2023 2024 2025
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
from loguru import logger
from pybaseball import batting_stats, pitching_stats

from mlb_analytics.db import engine

# ---------------------------------------------------------------------------
# Column rename maps — FanGraphs → our schema
# ---------------------------------------------------------------------------
BATTER_RENAME = {
    "IDfg":    "fangraphs_id",
    "Season":  "season",
    "Team":    "team",
    "G":       "g",
    "PA":      "pa",
    "AB":      "ab",
    "H":       "h",
    "2B":      "doubles",
    "3B":      "triples",
    "HR":      "hr",
    "RBI":     "rbi",
    "BB":      "bb",
    "HBP":     "hbp",
    "SO":      "so",
    "SB":      "sb",
    "AVG":     "avg",
    "OBP":     "obp",
    "SLG":     "slg",
    "OPS":     "ops",
    "wOBA":    "woba",
    "wRC+":    "wrc_plus",
    "EV":      "ev",
    "maxEV":   "max_ev",
    "LA":      "la",
    "Hard%":   "hard_hit_pct",
    "Barrel%": "barrel_pct",
    "K%":      "k_pct",
    "BB%":     "bb_pct",
    "Sprint Speed": "sprint_speed",
    "WAR":     "war",
    "Name":    "full_name",
    "MLBAMID": "player_id",
}

PITCHER_RENAME = {
    "IDfg":    "fangraphs_id",
    "Season":  "season",
    "Team":    "team",
    "G":       "g",
    "GS":      "gs",
    "IP":      "ip",
    "W":       "w",
    "L":       "l",
    "SV":      "sv",
    "ERA":     "era",
    "FIP":     "fip",
    "xFIP":    "xfip",
    "SIERA":   "siera",
    "K/9":     "k_9",
    "BB/9":    "bb_9",
    "HR/9":    "hr_9",
    "K%":      "k_pct",
    "BB%":     "bb_pct",
    "K-BB%":   "k_bb",
    "FB%":     "fb_pct",
    "SL%":     "sl_pct",
    "CT%":     "ct_pct",
    "CB%":     "cb_pct",
    "CH%":     "ch_pct",
    "vFB":     "avg_fastball_velo",
    "Whiff%":  "whiff_pct",
    "Zone%":   "zone_pct",
    "O-Swing%":"chase_pct",
    "CSW%":    "csw_pct",
    "WAR":     "war",
    "Name":    "full_name",
    "MLBAMID": "player_id",
}


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_batting(season: int) -> pd.DataFrame:
    """
    Pull FanGraphs batting leaderboard for a given season.
    qual=1 keeps all plate appearances (including part-time players).
    """
    logger.info(f"Fetching FanGraphs batting — {season}")
    try:
        df = batting_stats(season, season, qual=1)
    except Exception as exc:
        logger.error(f"batting_stats({season}) failed: {exc}")
        return pd.DataFrame()

    if df is None or df.empty:
        logger.warning(f"Empty batting response for {season}")
        return pd.DataFrame()

    df = df.rename(columns={k: v for k, v in BATTER_RENAME.items() if k in df.columns})
    df["season"] = season

    # Convert pct columns stored as 0-1 fractions → keep as-is for consistency
    for col in ["k_pct", "bb_pct", "hard_hit_pct", "barrel_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.success(f"  ✓ {len(df)} batters for {season}")
    return df


def fetch_pitching(season: int) -> pd.DataFrame:
    """
    Pull FanGraphs pitching leaderboard for a given season.
    """
    logger.info(f"Fetching FanGraphs pitching — {season}")
    try:
        df = pitching_stats(season, season, qual=1)
    except Exception as exc:
        logger.error(f"pitching_stats({season}) failed: {exc}")
        return pd.DataFrame()

    if df is None or df.empty:
        logger.warning(f"Empty pitching response for {season}")
        return pd.DataFrame()

    df = df.rename(columns={k: v for k, v in PITCHER_RENAME.items() if k in df.columns})
    df["season"] = season

    logger.success(f"  ✓ {len(df)} pitchers for {season}")
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_batting(df: pd.DataFrame, if_exists: str = "append") -> None:
    if df.empty:
        return
    # Only keep columns that match our schema
    schema_cols = [
        "player_id", "fangraphs_id", "season", "team", "g", "pa", "ab", "h",
        "doubles", "triples", "hr", "rbi", "bb", "hbp", "so", "sb",
        "avg", "obp", "slg", "ops", "woba", "wrc_plus",
        "ev", "max_ev", "la", "hard_hit_pct", "barrel_pct",
        "k_pct", "bb_pct", "sprint_speed", "war",
    ]
    df_out = df[[c for c in schema_cols if c in df.columns]]
    df_out.to_sql("fg_batter_season", con=engine, if_exists=if_exists,
                  index=False, method="multi", chunksize=1_000)
    logger.success(f"Saved {len(df_out)} batting rows to fg_batter_season")


def save_pitching(df: pd.DataFrame, if_exists: str = "append") -> None:
    if df.empty:
        return
    schema_cols = [
        "player_id", "fangraphs_id", "season", "team", "g", "gs", "ip",
        "w", "l", "sv", "era", "fip", "xfip", "siera",
        "k_9", "bb_9", "hr_9", "k_pct", "bb_pct", "k_bb",
        "fb_pct", "sl_pct", "ct_pct", "cb_pct", "ch_pct",
        "avg_fastball_velo", "whiff_pct", "zone_pct", "chase_pct", "csw_pct",
        "war",
    ]
    df_out = df[[c for c in schema_cols if c in df.columns]]
    df_out.to_sql("fg_pitcher_season", con=engine, if_exists=if_exists,
                  index=False, method="multi", chunksize=1_000)
    logger.success(f"Saved {len(df_out)} pitching rows to fg_pitcher_season")


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def ingest_seasons(seasons: list[int]) -> dict[str, pd.DataFrame]:
    """
    Fetch and store batting + pitching stats for multiple seasons.
    Returns dict with combined DataFrames (useful for feature engineering).
    """
    all_batting:  list[pd.DataFrame] = []
    all_pitching: list[pd.DataFrame] = []

    for season in seasons:
        bat = fetch_batting(season)
        pit = fetch_pitching(season)

        save_batting(bat)
        save_pitching(pit)

        all_batting.append(bat)
        all_pitching.append(pit)
        time.sleep(1)  # throttle

    return {
        "batting":  pd.concat(all_batting,  ignore_index=True) if all_batting  else pd.DataFrame(),
        "pitching": pd.concat(all_pitching, ignore_index=True) if all_pitching else pd.DataFrame(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest FanGraphs seasonal stats.")
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=[2023, 2024, 2025],
        help="Seasons to pull (default: 2023 2024 2025)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logger.info(f"Ingesting FanGraphs data for seasons: {args.seasons}")
    ingest_seasons(args.seasons)
    logger.success("FanGraphs ingestion complete.")


if __name__ == "__main__":
    main()
