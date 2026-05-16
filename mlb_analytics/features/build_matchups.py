"""
Builds the matchup_features table by joining
batter and pitcher rolling features to statcast pitch data.
"""
import pandas as pd
from sqlalchemy import text
from loguru import logger
from mlb_analytics.db import engine

def build_matchup_features():
    logger.info("Loading statcast pitch data...")
    with engine.connect() as conn:
        # Get one row per batter-pitcher-game combination
        df = pd.read_sql(text("""
            SELECT DISTINCT
                game_pk,
                game_date,
                batter_id,
                pitcher_id,
                stand as batter_hand,
                p_throws as pitcher_hand
            FROM statcast_pitches
            WHERE game_pk IS NOT NULL
              AND batter_id IS NOT NULL
              AND pitcher_id IS NOT NULL
        """), conn)

    logger.info(f"Found {len(df):,} unique matchups")

    with engine.connect() as conn:
        brf = pd.read_sql("SELECT * FROM batter_rolling_features", conn)
        prf = pd.read_sql("SELECT * FROM pitcher_rolling_features", conn)

    logger.info("Joining rolling features to matchups...")

    # Join batter features
    brf = brf.rename(columns={
        "player_id": "batter_id",
        "as_of_date": "batter_as_of",
        "avg_exit_velo": "b_avg_ev",
        "avg_launch_angle": "b_avg_la",
        "hard_hit_pct": "b_hard_hit_pct",
        "k_pct": "b_k_pct",
        "bb_pct": "b_bb_pct",
        "woba": "b_woba_rolling",
    })

    prf = prf.rename(columns={
        "player_id": "pitcher_id",
        "as_of_date": "pitcher_as_of",
        "avg_fastball_velo": "p_avg_velo",
        "whiff_pct": "p_whiff_pct",
        "zone_pct": "p_zone_pct",
        "k_pct": "p_k_pct",
        "bb_pct": "p_bb_pct",
    })

    # Keep only window=5 rows
    brf = brf[brf["window_games"] == 5]
    prf = prf[prf["window_games"] == 5]

    # Convert dates
    df["game_date"]      = pd.to_datetime(df["game_date"])
    brf["batter_as_of"]  = pd.to_datetime(brf["batter_as_of"])
    prf["pitcher_as_of"] = pd.to_datetime(prf["pitcher_as_of"])

    # Merge batter features — get most recent row before game date
    brf_cols = ["batter_id", "batter_as_of", "b_avg_ev", "b_avg_la",
                "b_hard_hit_pct", "b_k_pct", "b_bb_pct", "b_woba_rolling"]
    prf_cols = ["pitcher_id", "pitcher_as_of", "p_avg_velo", "p_whiff_pct",
                "p_zone_pct", "p_k_pct", "p_bb_pct"]

    df = df.merge(brf[brf_cols], on="batter_id", how="left")
    df = df[df["batter_as_of"] <= df["game_date"]]
    df = df.sort_values("batter_as_of").groupby(
        ["game_pk", "batter_id", "pitcher_id"]
    ).last().reset_index()

    df = df.merge(prf[prf_cols], on="pitcher_id", how="left")
    df = df[df["pitcher_as_of"] <= df["game_date"]]
    df = df.sort_values("pitcher_as_of").groupby(
        ["game_pk", "batter_id", "pitcher_id"]
    ).last().reset_index()

    # Add placeholder columns the model expects
    df["park_factor"]     = 100.0
    df["h2h_pa"]          = 0
    df["h2h_hits"]        = 0
    df["h2h_ba"]          = 0.250
    df["b_season_woba"]   = 0.320
    df["b_season_wrc_plus"] = 100.0
    df["p_season_era"]    = 4.0
    df["p_season_k_pct"]  = 0.22
    df["batter_hand_R"]   = (df["batter_hand"] == "R").astype(int)
    df["batter_hand_S"]   = (df["batter_hand"] == "S").astype(int)
    df["pitcher_hand_R"]  = (df["pitcher_hand"] == "R").astype(int)

    logger.info(f"Built {len(df):,} matchup rows")

    # Save to DB
    df.to_sql(
        "matchup_features", con=engine,
        if_exists="replace", index=False,
        method="multi", chunksize=200,
    )
    logger.success(f"Saved {len(df):,} rows to matchup_features")

if __name__ == "__main__":
    build_matchup_features()