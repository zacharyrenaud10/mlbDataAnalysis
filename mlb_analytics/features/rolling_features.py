"""
mlb_analytics/features/rolling_features.py
-------------------------------------------
Computes N-game rolling averages for batters and pitchers from
raw Statcast pitch data stored in the database (or a DataFrame).

Key outputs
-----------
  batter_rolling_features  — exit velo, launch angle, K%, BB%, wOBA, …
  pitcher_rolling_features — fastball velo, whiff%, zone%, K%, BB%, …

All windows are "last N *games*", not N calendar days, matching
how analysts typically construct recency features.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

from mlb_analytics.db import engine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_WINDOW = 5   # games
HARD_HIT_THRESHOLD = 95.0  # mph


# ---------------------------------------------------------------------------
# Batter rolling features
# ---------------------------------------------------------------------------

def _batter_game_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up pitch-level data to per-game, per-batter aggregates.
    Expects columns: batter_id, game_date, game_pk, launch_speed,
    launch_angle, is_whiff, is_swing, description, events, woba_value,
    balls, strikes, is_hard_hit.
    """
    df = df.copy()

    # Flag PA-ending events
    hit_events = {"single", "double", "triple", "home_run"}
    pa_end_events = hit_events | {
        "strikeout", "strikeout_double_play", "walk", "hit_by_pitch",
        "field_out", "force_out", "grounded_into_double_play",
        "double_play", "fielders_choice", "fielders_choice_out",
        "sac_fly", "sac_bunt",
    }
    df["is_hit"]         = df["events"].isin(hit_events)
    df["is_pa_end"]      = df["events"].isin(pa_end_events)
    df["is_k"]           = df["events"].isin({"strikeout", "strikeout_double_play"})
    df["is_bb"]          = df["events"].isin({"walk"})
    df["is_in_play"]     = df["type"] == "X"

    # Per-game aggregates
    g = df.groupby(["batter_id", "game_pk", "game_date"]).agg(
        pa             = ("is_pa_end",    "sum"),
        hits           = ("is_hit",       "sum"),
        k              = ("is_k",         "sum"),
        bb             = ("is_bb",        "sum"),
        avg_ev         = ("launch_speed",  lambda x: x[x.notna()].mean()),
        avg_la         = ("launch_angle",  lambda x: x[x.notna()].mean()),
        hard_hit_cnt   = ("is_hard_hit",  "sum"),
        balls_in_play  = ("is_in_play",   "sum"),
        swings         = ("is_swing",     "sum"),
        whiffs         = ("is_whiff",     "sum"),
        woba_sum       = ("woba_value",   "sum"),
        woba_cnt       = ("woba_value",   "count"),
    ).reset_index()

    g["k_pct"]       = g["k"]  / g["pa"].replace(0, np.nan)
    g["bb_pct"]      = g["bb"] / g["pa"].replace(0, np.nan)
    g["hard_hit_pct"]= g["hard_hit_cnt"] / g["balls_in_play"].replace(0, np.nan)
    g["whiff_pct"]   = g["whiffs"] / g["swings"].replace(0, np.nan)
    g["ba"]          = g["hits"] / g["pa"].replace(0, np.nan)
    g["woba"]        = g["woba_sum"] / g["woba_cnt"].replace(0, np.nan)

    return g.sort_values(["batter_id", "game_date"]).reset_index(drop=True)


def compute_batter_rolling(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Given pitch-level data, produce a DataFrame of rolling batter features.
    Each row represents a batter's rolling stats *entering* a game
    (i.e. the window is computed on the *preceding* N games).

    Returns DataFrame with columns:
        player_id, as_of_date, window_games,
        avg_exit_velo, avg_launch_angle, hard_hit_pct, k_pct, bb_pct,
        ba, woba, pa_in_window, games_in_window
    """
    logger.info(f"Computing {window}-game batter rolling features …")

    game_aggs = _batter_game_aggregates(df)

    records = []

    for batter_id, grp in game_aggs.groupby("batter_id"):
        grp = grp.sort_values("game_date").reset_index(drop=True)

        # Rolling window = last N rows (games)
        rolling = grp.rolling(window=window, min_periods=min_periods)

        # Weighted means using PA as weight where possible
        for i in range(1, len(grp) + 1):
            window_df = grp.iloc[max(0, i - window): i]
            total_pa  = window_df["pa"].sum()

            if total_pa == 0:
                continue

            records.append({
                "player_id":       batter_id,
                "as_of_date":      grp.iloc[i - 1]["game_date"],
                "window_games":    window,
                "avg_exit_velo":   window_df["avg_ev"].mean(),
                "avg_launch_angle":window_df["avg_la"].mean(),
                "hard_hit_pct":    window_df["hard_hit_cnt"].sum() /
                                   max(window_df["balls_in_play"].sum(), 1),
                "k_pct":           window_df["k"].sum() / max(total_pa, 1),
                "bb_pct":          window_df["bb"].sum() / max(total_pa, 1),
                "ba":              window_df["hits"].sum() / max(total_pa, 1),
                "woba":            (window_df["woba_sum"].sum() /
                                    max(window_df["woba_cnt"].sum(), 1)),
                "pa_in_window":    int(total_pa),
                "games_in_window": len(window_df),
            })

    result = pd.DataFrame(records)
    logger.success(f"  ✓ {len(result):,} batter rolling rows computed")
    return result


# ---------------------------------------------------------------------------
# Pitcher rolling features
# ---------------------------------------------------------------------------

def _pitcher_game_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up pitch-level Statcast data to per-game pitcher aggregates.
    """
    df = df.copy()

    k_events = {"strikeout", "strikeout_double_play"}
    bb_events = {"walk"}
    hr_events = {"home_run"}
    pa_end    = k_events | bb_events | hr_events | {
        "field_out", "force_out", "grounded_into_double_play",
        "double_play", "fielders_choice", "fielders_choice_out",
        "single", "double", "triple", "hit_by_pitch", "sac_fly", "sac_bunt",
    }

    df["is_k"]      = df["events"].isin(k_events)
    df["is_bb"]     = df["events"].isin(bb_events)
    df["is_hr"]     = df["events"].isin(hr_events)
    df["is_pa_end"] = df["events"].isin(pa_end)
    df["is_fb"]     = df["pitch_type"].isin({"FF", "SI", "FC"})

    g = df.groupby(["pitcher_id", "game_pk", "game_date"]).agg(
        batters_faced  = ("is_pa_end",    "sum"),
        k              = ("is_k",         "sum"),
        bb             = ("is_bb",        "sum"),
        hr             = ("is_hr",        "sum"),
        pitches        = ("pitch_type",   "count"),
        swings         = ("is_swing",     "sum"),
        whiffs         = ("is_whiff",     "sum"),
        zones          = ("is_in_zone",   "sum"),
        fb_velo_sum    = ("release_speed", lambda x: x[df.loc[x.index, "is_fb"]].sum()),
        fb_velo_cnt    = ("release_speed", lambda x: x[df.loc[x.index, "is_fb"]].count()),
    ).reset_index()

    g["k_pct"]           = g["k"]      / g["batters_faced"].replace(0, np.nan)
    g["bb_pct"]          = g["bb"]     / g["batters_faced"].replace(0, np.nan)
    g["whiff_pct"]       = g["whiffs"] / g["swings"].replace(0, np.nan)
    g["zone_pct"]        = g["zones"]  / g["pitches"].replace(0, np.nan)
    g["avg_fb_velo"]     = g["fb_velo_sum"] / g["fb_velo_cnt"].replace(0, np.nan)

    return g.sort_values(["pitcher_id", "game_date"]).reset_index(drop=True)


def compute_pitcher_rolling(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Produce a DataFrame of N-game rolling pitcher features.

    Returns DataFrame with columns:
        player_id, as_of_date, window_games,
        avg_fastball_velo, whiff_pct, zone_pct, k_pct, bb_pct,
        hr_per_9, batters_faced, games_in_window
    """
    logger.info(f"Computing {window}-game pitcher rolling features …")

    game_aggs = _pitcher_game_aggregates(df)
    records   = []

    for pitcher_id, grp in game_aggs.groupby("pitcher_id"):
        grp = grp.sort_values("game_date").reset_index(drop=True)

        for i in range(1, len(grp) + 1):
            window_df    = grp.iloc[max(0, i - window): i]
            total_bf     = window_df["batters_faced"].sum()
            total_swings = window_df["swings"].sum()
            total_pitches= window_df["pitches"].sum()
            total_k      = window_df["k"].sum()
            total_bb     = window_df["bb"].sum()
            total_hr     = window_df["hr"].sum()

            if total_bf == 0:
                continue

            # Estimated IP (3 outs per inning, ~3.3 BF/IP rule of thumb)
            est_ip = total_bf / 3.3

            records.append({
                "player_id":          pitcher_id,
                "as_of_date":         grp.iloc[i - 1]["game_date"],
                "window_games":       window,
                "avg_fastball_velo":  window_df["avg_fb_velo"].mean(),
                "whiff_pct":          (window_df["whiffs"].sum() /
                                       max(total_swings, 1)),
                "zone_pct":           (window_df["zones"].sum() /
                                       max(total_pitches, 1)),
                "k_pct":              total_k  / max(total_bf, 1),
                "bb_pct":             total_bb / max(total_bf, 1),
                "hr_per_9":           (total_hr / max(est_ip, 0.001)) * 9,
                "batters_faced":      int(total_bf),
                "games_in_window":    len(window_df),
            })

    result = pd.DataFrame(records)
    logger.success(f"  ✓ {len(result):,} pitcher rolling rows computed")
    return result


# ---------------------------------------------------------------------------
# Convenience: load from DB, compute, write back
# ---------------------------------------------------------------------------

def refresh_rolling_features(
    season: int,
    window: int = DEFAULT_WINDOW,
) -> None:
    """
    Full refresh pipeline:
    1. Load Statcast data for `season` from DB
    2. Compute batter + pitcher rolling features
    3. Write results to the rolling feature tables
    """
    logger.info(f"Refreshing {window}-game rolling features for {season} …")

    query = text(
        "SELECT * FROM statcast_pitches WHERE season = :season"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"season": season})

    logger.info(f"  Loaded {len(df):,} pitches from DB")

    # --- Batters ---
    batter_features = compute_batter_rolling(df, window=window)
    if not batter_features.empty:
        batter_features.to_sql(
            "batter_rolling_features", con=engine,
            if_exists="replace", index=False,
            method="multi", chunksize=200,
        )
        logger.success(f"  Saved {len(batter_features):,} batter feature rows")

    # --- Pitchers ---
    pitcher_features = compute_pitcher_rolling(df, window=window)
    if not pitcher_features.empty:
        pitcher_features.to_sql(
            "pitcher_rolling_features", con=engine,
            if_exists="replace", index=False,
            method="multi", chunksize=200,
        )
        logger.success(f"  Saved {len(pitcher_features):,} pitcher feature rows")


def get_batter_features_as_of(
    player_id: int,
    as_of_date: str,
    window: int = DEFAULT_WINDOW,
) -> Optional[pd.Series]:
    """
    Retrieve the most recent rolling feature row for a batter
    at or before `as_of_date`.
    """
    query = text("""
        SELECT * FROM batter_rolling_features
        WHERE  player_id   = :pid
          AND  as_of_date  <= :dt
          AND  window_games = :win
        ORDER  BY as_of_date DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = pd.read_sql(query, conn,
                          params={"pid": player_id, "dt": as_of_date, "win": window})
    return row.iloc[0] if not row.empty else None


def get_pitcher_features_as_of(
    player_id: int,
    as_of_date: str,
    window: int = DEFAULT_WINDOW,
) -> Optional[pd.Series]:
    """
    Retrieve the most recent rolling feature row for a pitcher
    at or before `as_of_date`.
    """
    query = text("""
        SELECT * FROM pitcher_rolling_features
        WHERE  player_id   = :pid
          AND  as_of_date  <= :dt
          AND  window_games = :win
        ORDER  BY as_of_date DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = pd.read_sql(query, conn,
                          params={"pid": player_id, "dt": as_of_date, "win": window})
    return row.iloc[0] if not row.empty else None
