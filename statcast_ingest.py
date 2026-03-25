"""
mlb_analytics/ingestion/statcast_ingest.py
------------------------------------------
Pulls Statcast pitch-level data from Baseball Savant via pybaseball,
cleans it, computes derived flags, and persists to the database.

Usage (CLI):
    python -m mlb_analytics.ingestion.statcast_ingest --season 2025
    python -m mlb_analytics.ingestion.statcast_ingest --season 2025 --team NYY
    python -m mlb_analytics.ingestion.statcast_ingest --start 2025-04-01 --end 2025-04-30
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# pybaseball
import pybaseball
from pybaseball import statcast, statcast_pitcher, statcast_batter, cache

# Internal
from mlb_analytics.db import engine

# ---------------------------------------------------------------------------
# Enable pybaseball's built-in disk cache so repeated pulls are instant.
# Comment out if you prefer to always hit the live endpoint.
# ---------------------------------------------------------------------------
cache.enable()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEASON_DATES: dict[int, tuple[str, str]] = {
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
}

# Columns we actually keep (avoids storing Statcast's ~90 columns verbatim).
KEEP_COLS = [
    "game_pk", "game_date", "batter", "pitcher",
    "pitcher_team", "batter_team",  # added by us after merge
    "pitch_type",
    "release_speed", "release_spin_rate", "release_extension",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "zone",
    "description", "type", "events", "bb_type",
    "launch_speed", "launch_angle", "hit_distance_sc",
    "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
    "woba_value",
    "balls", "strikes", "outs_when_up", "inning", "inning_topbot",
    "on_1b", "on_2b", "on_3b",
    "stand", "p_throws",
]

# ---------------------------------------------------------------------------
# Derived flag computation
# ---------------------------------------------------------------------------
SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "foul_bunt", "missed_bunt",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
ZONE_NUMS = set(range(1, 10))  # Statcast zones 1-9 are the strike zone


def _add_derived_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean swing / whiff / zone flags to a pitch dataframe."""
    df = df.copy()
    df["is_swing"]   = df["description"].isin(SWING_DESCRIPTIONS)
    df["is_whiff"]   = df["description"].isin(WHIFF_DESCRIPTIONS)
    df["is_in_zone"] = df["zone"].isin(ZONE_NUMS)
    df["is_hard_hit"] = df["launch_speed"].ge(95)
    return df


def _clean_statcast(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column types and names coming out of pybaseball.
    Returns a tidy DataFrame ready for DB insertion.
    """
    if raw.empty:
        return raw

    # Rename MLBAM id columns to match our schema
    raw = raw.rename(columns={"batter": "batter_id", "pitcher": "pitcher_id"})

    # Parse game_date
    raw["game_date"] = pd.to_datetime(raw["game_date"]).dt.date
    raw["season"]    = pd.to_datetime(raw["game_date"]).apply(lambda d: d.year)

    # Normalise floats
    float_cols = [
        "release_speed", "release_spin_rate", "release_extension",
        "pfx_x", "pfx_z", "plate_x", "plate_z",
        "launch_speed", "launch_angle", "hit_distance_sc",
        "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
        "woba_value",
    ]
    for col in float_cols:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").astype("float32")

    # Int columns
    int_cols = ["balls", "strikes", "outs_when_up", "inning", "zone",
                "on_1b", "on_2b", "on_3b"]
    for col in int_cols:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").astype("Int64")

    # Derived flags
    raw = _add_derived_flags(raw)

    # Select only available keep cols
    available = [c for c in KEEP_COLS + ["batter_id", "pitcher_id", "season",
                                          "is_swing", "is_whiff", "is_in_zone",
                                          "is_hard_hit"]
                 if c in raw.columns]
    return raw[available].drop_duplicates()


# ---------------------------------------------------------------------------
# Fetching helpers
# ---------------------------------------------------------------------------

def fetch_statcast_range(
    start_dt: str,
    end_dt: str,
    chunk_days: int = 7,
    retry_wait: int = 10,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Pull Statcast data for a date range in weekly chunks to avoid timeouts.
    Baseball Savant throttles large single requests.

    Parameters
    ----------
    start_dt    : 'YYYY-MM-DD'
    end_dt      : 'YYYY-MM-DD'
    chunk_days  : size of each pull window (default 7 days)
    retry_wait  : seconds to wait between retries
    max_retries : retries per chunk on failure
    """
    start = date.fromisoformat(start_dt)
    end   = date.fromisoformat(end_dt)
    frames: list[pd.DataFrame] = []

    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        s = cursor.isoformat()
        e = chunk_end.isoformat()

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"  Pulling Statcast {s} → {e}  (attempt {attempt})")
                chunk = statcast(start_dt=s, end_dt=e, verbose=False)
                if chunk is not None and not chunk.empty:
                    frames.append(_clean_statcast(chunk))
                    logger.success(f"    ✓ {len(chunk):,} pitches")
                else:
                    logger.warning(f"    ⚠ Empty response for {s}→{e}")
                break
            except Exception as exc:
                logger.error(f"    ✗ Error: {exc}")
                if attempt < max_retries:
                    logger.info(f"    Retrying in {retry_wait}s …")
                    time.sleep(retry_wait)
                else:
                    logger.error(f"    Giving up on {s}→{e}")

        cursor = chunk_end + timedelta(days=1)
        time.sleep(2)  # be polite to Baseball Savant

    if not frames:
        logger.warning("No data fetched for the requested range.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total pitches fetched: {len(combined):,}")
    return combined


def fetch_statcast_season(season: int, **kwargs) -> pd.DataFrame:
    """Convenience wrapper: fetch a full season by year."""
    if season not in SEASON_DATES:
        raise ValueError(f"Season {season} not in SEASON_DATES. Add it manually.")
    start, end = SEASON_DATES[season]
    logger.info(f"=== Fetching Statcast for {season} season ({start} – {end}) ===")
    return fetch_statcast_range(start, end, **kwargs)


# ---------------------------------------------------------------------------
# Single-player pulls (useful for targeted refreshes)
# ---------------------------------------------------------------------------

def fetch_pitcher_statcast(
    pitcher_id: int,
    start_dt: str,
    end_dt: str,
) -> pd.DataFrame:
    """Pull pitch data for a single pitcher MLBAM ID."""
    logger.info(f"Fetching pitcher {pitcher_id}: {start_dt} → {end_dt}")
    raw = statcast_pitcher(start_dt=start_dt, end_dt=end_dt, player_id=pitcher_id)
    return _clean_statcast(raw) if raw is not None and not raw.empty else pd.DataFrame()


def fetch_batter_statcast(
    batter_id: int,
    start_dt: str,
    end_dt: str,
) -> pd.DataFrame:
    """Pull pitch data for a single batter MLBAM ID."""
    logger.info(f"Fetching batter {batter_id}: {start_dt} → {end_dt}")
    raw = statcast_batter(start_dt=start_dt, end_dt=end_dt, player_id=batter_id)
    return _clean_statcast(raw) if raw is not None and not raw.empty else pd.DataFrame()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def upsert_statcast_to_db(df: pd.DataFrame, if_exists: str = "append") -> int:
    """
    Write cleaned Statcast data to `statcast_pitches` table.
    Uses pandas to_sql for simplicity; for production consider
    a true UPSERT via psycopg2 + ON CONFLICT DO NOTHING.

    Returns number of rows written.
    """
    if df.empty:
        return 0

    # Rename to match DB column names
    df = df.rename(columns={
        "batter_id": "batter_id",
        "pitcher_id": "pitcher_id",
    })

    rows_before = _count_rows("statcast_pitches")

    df.to_sql(
        name="statcast_pitches",
        con=engine,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=5_000,
    )

    rows_after = _count_rows("statcast_pitches")
    written = rows_after - rows_before
    logger.success(f"Wrote {written:,} rows to statcast_pitches (total: {rows_after:,})")
    return written


def _count_rows(table: str) -> int:
    from sqlalchemy import text as sqla_text
    with engine.connect() as conn:
        result = conn.execute(sqla_text(f"SELECT COUNT(*) FROM {table}"))
        return result.scalar() or 0


# ---------------------------------------------------------------------------
# Parquet cache (fast local access without DB round-trips)
# ---------------------------------------------------------------------------

def save_to_parquet(df: pd.DataFrame, path: str) -> None:
    """Cache a DataFrame as a compressed Parquet file."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    logger.info(f"Saved {len(df):,} rows → {path}")


def load_from_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest Statcast pitch-level data into the MLB analytics DB."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--season", type=int, help="Full season year (2023-2025)")
    group.add_argument("--start",  type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   type=str, help="End date YYYY-MM-DD (used with --start)")
    parser.add_argument("--team",  type=str, help="Filter by team abbreviation (not yet implemented)")
    parser.add_argument("--chunk", type=int, default=7, help="Chunk size in days (default 7)")
    parser.add_argument("--parquet", type=str, default=None,
                        help="Also save raw data to this Parquet path")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip DB write (useful with --parquet for inspection)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.season:
        df = fetch_statcast_season(args.season, chunk_days=args.chunk)
    elif args.start:
        end = args.end or date.today().isoformat()
        df = fetch_statcast_range(args.start, end, chunk_days=args.chunk)
    else:
        # Default: pull the entire 2025 season
        logger.info("No arguments supplied — defaulting to full 2025 season pull.")
        df = fetch_statcast_season(2025)

    if df.empty:
        logger.warning("Nothing to write.")
        return

    if args.parquet:
        save_to_parquet(df, args.parquet)

    if not args.no_db:
        upsert_statcast_to_db(df)


if __name__ == "__main__":
    main()
