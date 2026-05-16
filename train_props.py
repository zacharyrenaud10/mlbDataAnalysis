"""
train_props.py
--------------
Builds per-game stat lines from Statcast data and trains
all player prop models.

Extracts from statcast_pitches:
  Per batter per game: hits, doubles, triples, hr, bb, k, sb, runs, rbi, total_bases
  Per pitcher per game: k, ip (estimated), er (estimated)

Then trains XGBoost classifiers for every prop target.

Usage:
    python train_props.py
    python train_props.py --seasons 2024 2025
    python train_props.py --quick   (2025 only, faster)
"""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

from mlb_analytics.db import engine
from mlb_analytics.models.player_props_model import PlayerPropsModel


# ---------------------------------------------------------------------------
# Step 1 — Build per-game batter stat lines from Statcast
# ---------------------------------------------------------------------------

def build_batter_game_stats(seasons: list[int]) -> pd.DataFrame:
    """
    Aggregate pitch-level Statcast data into per-game batter stat lines.
    Returns one row per (batter_id, game_pk) with hit, HR, K, BB, etc.
    """
    logger.info(f"Building batter game stats for seasons {seasons}...")

    season_str = ",".join(str(s) for s in seasons)

    query = text(f"""
        SELECT
            batter_id,
            game_pk,
            game_date,
            pitcher_id,
            stand          AS batter_hand,
            p_throws       AS pitcher_hand,

            -- Hits
            SUM(CASE WHEN events IN ('single','double','triple','home_run')
                THEN 1 ELSE 0 END) AS hits,

            -- Extra base hits
            SUM(CASE WHEN events = 'double'   THEN 1 ELSE 0 END) AS doubles,
            SUM(CASE WHEN events = 'triple'   THEN 1 ELSE 0 END) AS triples,
            SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hr,

            -- Walks
            SUM(CASE WHEN events = 'walk'     THEN 1 ELSE 0 END) AS bb,

            -- Strikeouts
            SUM(CASE WHEN events IN ('strikeout','strikeout_double_play')
                THEN 1 ELSE 0 END) AS k,

            -- Plate appearances (approximation)
            COUNT(DISTINCT CASE WHEN events IS NOT NULL
                THEN events END) AS pa_proxy,

            -- Batted ball metrics
            AVG(CASE WHEN launch_speed IS NOT NULL
                THEN launch_speed END) AS avg_ev,
            AVG(CASE WHEN launch_angle IS NOT NULL
                THEN launch_angle END) AS avg_la,
            SUM(CASE WHEN launch_speed >= 95
                THEN 1 ELSE 0 END) AS hard_hit_cnt,
            COUNT(CASE WHEN launch_speed IS NOT NULL
                THEN 1 END) AS balls_in_play

        FROM statcast_pitches
        WHERE season IN ({season_str})
          AND batter_id IS NOT NULL
          AND game_pk IS NOT NULL
        GROUP BY batter_id, game_pk, game_date, pitcher_id,
                 stand, p_throws
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    logger.info(f"  Raw batter-game rows: {len(df):,}")

    # Calculate total bases
    df["total_bases"] = (
        df["hits"] +
        df["doubles"] +
        df["triples"] * 2 +
        df["hr"] * 3
    )

    # Hard hit %
    df["hard_hit_pct"] = df["hard_hit_cnt"] / df["balls_in_play"].replace(0, np.nan)

    logger.success(f"  Built {len(df):,} batter-game rows")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Build per-game pitcher stat lines
# ---------------------------------------------------------------------------

def build_pitcher_game_stats(seasons: list[int]) -> pd.DataFrame:
    """
    Aggregate pitch-level data into per-game pitcher stat lines.
    """
    logger.info(f"Building pitcher game stats for seasons {seasons}...")

    season_str = ",".join(str(s) for s in seasons)

    query = text(f"""
        SELECT
            pitcher_id,
            game_pk,
            game_date,
            p_throws AS pitcher_hand,

            -- Strikeouts
            SUM(CASE WHEN events IN ('strikeout','strikeout_double_play')
                THEN 1 ELSE 0 END) AS k,

            -- Walks
            SUM(CASE WHEN events = 'walk'
                THEN 1 ELSE 0 END) AS bb,

            -- Home runs allowed
            SUM(CASE WHEN events = 'home_run'
                THEN 1 ELSE 0 END) AS hr,

            -- Total batters faced (approx)
            COUNT(DISTINCT CASE WHEN events IS NOT NULL
                THEN rowid END) AS bf,

            -- Pitch metrics
            AVG(release_speed)   AS avg_velo,
            AVG(CASE WHEN is_whiff = 1 THEN 1.0
                     WHEN is_swing = 1 THEN 0.0
                     ELSE NULL END) AS whiff_pct,
            AVG(CASE WHEN is_in_zone = 1 THEN 1.0
                     ELSE 0.0 END) AS zone_pct

        FROM statcast_pitches
        WHERE season IN ({season_str})
          AND pitcher_id IS NOT NULL
          AND game_pk IS NOT NULL
        GROUP BY pitcher_id, game_pk, game_date, p_throws
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    logger.info(f"  Raw pitcher-game rows: {len(df):,}")

    # Estimated IP from batters faced (rough: ~3.3 BF per IP)
    df["est_ip"] = df["bf"] / 3.3

    # Quality start flag: 6+ estimated IP and 3 or fewer HR allowed (proxy)
    df["qs"] = ((df["est_ip"] >= 6.0) & (df["hr"] <= 1)).astype(int)

    logger.success(f"  Built {len(df):,} pitcher-game rows")
    return df


# ---------------------------------------------------------------------------
# Step 3 — Join with rolling features to build feature matrix
# ---------------------------------------------------------------------------

def build_batter_feature_matrix(
    game_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Join batter game stats with rolling features to create
    (X_features, y_targets) for model training.
    """
    logger.info("Building batter feature matrix...")

    # Load rolling features
    with engine.connect() as conn:
        brf = pd.read_sql(
            "SELECT * FROM batter_rolling_features WHERE window_games = 5",
            conn
        )
        prf = pd.read_sql(
            "SELECT * FROM pitcher_rolling_features WHERE window_games = 5",
            conn
        )

    logger.info(f"  Batter rolling rows: {len(brf):,}")
    logger.info(f"  Pitcher rolling rows: {len(prf):,}")

    if brf.empty or prf.empty:
        logger.error("Rolling features not found — run: "
                     "python -m mlb_analytics.run_pipeline --stages features")
        return pd.DataFrame(), pd.DataFrame()

    # Convert dates
    game_stats["game_date"] = pd.to_datetime(game_stats["game_date"])
    brf["as_of_date"]       = pd.to_datetime(brf["as_of_date"])
    prf["as_of_date"]       = pd.to_datetime(prf["as_of_date"])

    # Rename rolling feature columns
    brf = brf.rename(columns={
        "player_id":        "batter_id",
        "as_of_date":       "batter_as_of",
        "avg_exit_velo":    "b_avg_ev",
        "avg_launch_angle": "b_avg_la",
        "hard_hit_pct":     "b_hard_hit_pct",
        "k_pct":            "b_k_pct",
        "bb_pct":           "b_bb_pct",
        "woba":             "b_woba_rolling",
        "ba":               "b_ba",
    })

    prf = prf.rename(columns={
        "player_id":          "pitcher_id",
        "as_of_date":         "pitcher_as_of",
        "avg_fastball_velo":  "p_avg_velo",
        "whiff_pct":          "p_whiff_pct",
        "zone_pct":           "p_zone_pct",
        "k_pct":              "p_k_pct",
        "bb_pct":             "p_bb_pct",
    })

    brf_cols = ["batter_id", "batter_as_of", "b_avg_ev", "b_avg_la",
                "b_hard_hit_pct", "b_k_pct", "b_bb_pct",
                "b_woba_rolling", "b_ba"]
    prf_cols = ["pitcher_id", "pitcher_as_of", "p_avg_velo", "p_whiff_pct",
                "p_zone_pct", "p_k_pct", "p_bb_pct"]

    # Merge batter rolling — get most recent features before game date
    df = game_stats.merge(brf[brf_cols], on="batter_id", how="left")
    df = df[df["batter_as_of"] < df["game_date"]]
    df = (df.sort_values("batter_as_of")
            .groupby(["batter_id", "game_pk", "pitcher_id"])
            .last()
            .reset_index())

    # Merge pitcher rolling
    df = df.merge(prf[prf_cols], on="pitcher_id", how="left")
    df = df[df["pitcher_as_of"] < df["game_date"]]
    df = (df.sort_values("pitcher_as_of")
            .groupby(["batter_id", "game_pk", "pitcher_id"])
            .last()
            .reset_index())

    # Add context features
    df["park_factor"]    = 100.0  # placeholder
    df["batter_hand_R"]  = (df["batter_hand"] == "R").astype(int)
    df["batter_hand_S"]  = (df["batter_hand"] == "S").astype(int)
    df["pitcher_hand_R"] = (df["pitcher_hand"] == "R").astype(int)
    df["h2h_pa"]         = 0
    df["h2h_ba"]         = 0.250
    df["b_season_woba"]  = df["b_woba_rolling"].fillna(0.315)
    df["b_season_wrc_plus"] = 100.0
    df["p_season_era"]   = 4.0
    df["p_season_k_pct"] = df["p_k_pct"].fillna(0.22)
    df["batting_order"]  = 5
    df["is_home"]        = 0
    df["barrel_pct"]     = 0.08

    # Rename for model
    df = df.rename(columns={"b_avg_ev": "b_avg_ev",
                              "b_hard_hit_pct": "b_hard_hit_pct"})
    df["b_barrel_pct"] = df.get("barrel_pct", 0.08)

    from mlb_analytics.models.player_props_model import BATTER_FEATURES
    feature_cols = [c for c in BATTER_FEATURES if c in df.columns]
    target_cols  = ["hits", "doubles", "triples", "hr", "bb", "k",
                    "total_bases", "hard_hit_pct"]

    X = df[feature_cols].fillna(0.0)
    y = df[target_cols].fillna(0)

    logger.success(f"  Feature matrix: {X.shape}")
    return X, y


def build_pitcher_feature_matrix(
    game_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build pitcher feature matrix from game stats + rolling features."""
    logger.info("Building pitcher feature matrix...")

    with engine.connect() as conn:
        prf = pd.read_sql(
            "SELECT * FROM pitcher_rolling_features WHERE window_games = 5",
            conn
        )

    if prf.empty:
        return pd.DataFrame(), pd.DataFrame()

    game_stats["game_date"] = pd.to_datetime(game_stats["game_date"])
    prf["as_of_date"]       = pd.to_datetime(prf["as_of_date"])

    prf = prf.rename(columns={
        "player_id":         "pitcher_id",
        "as_of_date":        "pitcher_as_of",
        "avg_fastball_velo": "p_avg_velo",
        "whiff_pct":         "p_whiff_pct",
        "zone_pct":          "p_zone_pct",
        "k_pct":             "p_k_pct",
        "bb_pct":            "p_bb_pct",
        "hr_per_9":          "p_hr_per_9",
    })

    prf_cols = ["pitcher_id", "pitcher_as_of", "p_avg_velo", "p_whiff_pct",
                "p_zone_pct", "p_k_pct", "p_bb_pct", "p_hr_per_9"]

    df = game_stats.merge(prf[prf_cols], on="pitcher_id", how="left")
    df = df[df["pitcher_as_of"] < df["game_date"]]
    df = (df.sort_values("pitcher_as_of")
            .groupby(["pitcher_id", "game_pk"])
            .last()
            .reset_index())

    df["park_factor"]        = 100.0
    df["is_home"]            = 1
    df["days_rest"]          = 4
    df["opp_k_pct"]          = 0.22
    df["p_season_era"]       = 4.0
    df["p_season_fip"]       = 4.0
    df["p_season_k_9"]       = 8.0

    from mlb_analytics.models.player_props_model import PITCHER_FEATURES
    feature_cols = [c for c in PITCHER_FEATURES if c in df.columns]
    target_cols  = ["k", "qs"]

    X = df[feature_cols].fillna(0.0)
    y = df[target_cols].fillna(0)

    logger.success(f"  Pitcher feature matrix: {X.shape}")
    return X, y


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train(seasons: list[int]) -> None:
    logger.info(f"Training props models on seasons: {seasons}")

    # Build stat lines
    batter_stats  = build_batter_game_stats(seasons)
    pitcher_stats = build_pitcher_game_stats(seasons)

    # Build feature matrices
    X_bat, y_bat = build_batter_feature_matrix(batter_stats)
    X_pit, y_pit = build_pitcher_feature_matrix(pitcher_stats)

    if X_bat.empty:
        logger.error("Batter feature matrix is empty — check rolling features")
        return

    # Train
    model = PlayerPropsModel()

    logger.info("Training batter models...")
    model.fit_batter_models(X_bat, y_bat)

    if not X_pit.empty:
        logger.info("Training pitcher models...")
        model.fit_pitcher_models(X_pit, y_pit)

    # Save
    path = model.save()
    logger.success(f"Props model saved → {path}")
    logger.success(f"Trained {len(model.models)} prop classifiers")
    logger.info("Now run: python matchup_predictor.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train player prop prediction models from Statcast data"
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int,
        default=[2023, 2024, 2025],
        help="Seasons to train on (default: 2023 2024 2025)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 2025 only"
    )
    return parser.parse_args()


def main() -> None:
    args   = _parse_args()
    seasons = [2025] if args.quick else args.seasons
    train(seasons)


if __name__ == "__main__":
    main()
