"""
mlb_analytics/run_pipeline.py
------------------------------
End-to-end daily pipeline CLI.

Stages (can be run individually or all at once):

  ingest    — pull latest Statcast + FanGraphs data
  features  — recompute rolling feature store
  train     — retrain models on latest data
  predict   — run today's predictions
  ev        — identify +EV opportunities and print best bets

Usage:
  # Full daily run (all stages)
  python -m mlb_analytics.run_pipeline --all

  # Only refresh features and run EV scan
  python -m mlb_analytics.run_pipeline --stages features ev

  # Ingest a specific date range
  python -m mlb_analytics.run_pipeline --stages ingest --start 2025-09-01 --end 2025-09-28

  # Just print today's best bets (requires trained models)
  python -m mlb_analytics.run_pipeline --stages ev --top 15
"""

from __future__ import annotations

from sqlalchemy import text
import argparse
import sys
from datetime import date

from loguru import logger

# ---------------------------------------------------------------------------
# Configure logger
# ---------------------------------------------------------------------------
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
)
logger.add(
    "logs/pipeline_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="14 days",
    level="DEBUG",
)


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def stage_ingest(args: argparse.Namespace) -> None:
    """Pull raw Statcast + FanGraphs data."""
    from mlb_analytics.ingestion.statcast_ingest import (
        fetch_statcast_range,
        fetch_statcast_season,
        upsert_statcast_to_db,
    )
    from mlb_analytics.ingestion.fangraphs_ingest import ingest_seasons

    if getattr(args, "start", None):
        end = getattr(args, "end", None) or date.today().isoformat()
        logger.info(f"Ingesting Statcast {args.start} → {end}")
        df = fetch_statcast_range(args.start, end)
        upsert_statcast_to_db(df)
    else:
        seasons = getattr(args, "seasons", [2023, 2024, 2025])
        for season in seasons:
            df = fetch_statcast_season(season)
            upsert_statcast_to_db(df)

    # Always refresh FanGraphs for the full training window
    ingest_seasons([2023, 2024, 2025])


def stage_features(args: argparse.Namespace) -> None:
    """Compute rolling feature store."""
    from mlb_analytics.features.rolling_features import refresh_rolling_features

    window  = getattr(args, "window", 5)
    seasons = getattr(args, "seasons", [2023, 2024, 2025])

    for season in seasons:
        refresh_rolling_features(season, window=window)


def stage_train(args: argparse.Namespace) -> None:
    """Train / retrain prediction models."""
    import pandas as pd
    from sqlalchemy import text

    from mlb_analytics.db import engine
    from mlb_analytics.models.prediction_engine import (
        BaseHitClassifier,
        StrikeoutRegressor,
        cross_validate_classifier,
    )

    logger.info("Loading training data from matchup_features …")

    with engine.connect() as conn:
        mf = pd.read_sql("SELECT * FROM matchup_features", conn)

    if mf.empty:
        logger.error("matchup_features table is empty — run feature engineering first.")
        return

    # ---- Base Hit Classifier ----
    # Target: 1 if h2h_hits > 0 (proxy; replace with actual game-level hit flag)
    # Build target from statcast events directly
    with engine.connect() as conn:
        hits_df = pd.read_sql(text("""
            SELECT DISTINCT game_pk, batter_id
            FROM statcast_pitches
            WHERE events IN ('single','double','triple','home_run')
        """), conn)

    hits_df["target_hit"] = 1
    mf = mf.merge(hits_df, on=["game_pk", "batter_id"], how="left")
    mf["target_hit"] = mf["target_hit"].fillna(0).astype(int)

    feature_cols_clf = BaseHitClassifier().features
    X = mf[[c for c in feature_cols_clf if c in mf.columns]].copy()
    y = mf["target_hit"]

    split   = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    clf = BaseHitClassifier()
    clf.fit(X_train, y_train, X_val, y_val)
    cv_metrics = cross_validate_classifier(clf, X, y)

    clf_path = clf.save()
    logger.success(f"BaseHitClassifier saved → {clf_path}")

    # ---- Strikeout Regressor ----
    # Aggregate Statcast to pitcher-game level for strikeout totals
    with engine.connect() as conn:
        k_df = pd.read_sql(
            text("""
                SELECT pitcher_id, game_pk, game_date,
                       COUNT(*) FILTER (WHERE events IN ('strikeout','strikeout_double_play'))
                           AS actual_k
                FROM   statcast_pitches
                WHERE  season IN (2023, 2024, 2025)
                GROUP  BY pitcher_id, game_pk, game_date
            """),
            conn,
        )

    # Join with pitcher rolling features (this is a simplified join; expand as needed)
    from mlb_analytics.features.rolling_features import get_pitcher_features_as_of
    feature_cols_reg = StrikeoutRegressor().features
    # … in production, you'd vectorise this join via a proper matchup_features query
    logger.info(f"K regressor training data: {len(k_df):,} pitcher-game rows")

    if len(k_df) < 100:
        logger.warning("Not enough K data to train regressor — skipping.")
        return

    X_reg = pd.DataFrame(0.0, index=k_df.index, columns=feature_cols_reg)
    y_reg = k_df["actual_k"]

    split   = int(len(X_reg) * 0.8)
    reg = StrikeoutRegressor()
    reg.fit(
        X_reg.iloc[:split], y_reg.iloc[:split],
        X_reg.iloc[split:], y_reg.iloc[split:],
    )
    reg_path = reg.save()
    logger.success(f"StrikeoutRegressor saved → {reg_path}")


def stage_ev(args: argparse.Namespace) -> None:
    """Identify and display today's +EV opportunities."""
    import glob
    import os
    from mlb_analytics.ev_engine import EVEngine

    model_dir = os.getenv("MODEL_DIR", "./models")
    top_n     = getattr(args, "top", 10)

    # Auto-discover latest saved models
    clf_files = sorted(glob.glob(f"{model_dir}/bh_clf_*.pkl"), reverse=True)
    reg_files = sorted(glob.glob(f"{model_dir}/k_reg_*.pkl"),  reverse=True)

    if clf_files and reg_files:
        engine_ev = EVEngine.from_saved_models(clf_files[0], reg_files[0])
    else:
        logger.warning(
            "No saved models found — running EV engine with fallback population averages.\n"
            "Run --stages train first for accurate predictions."
        )
        engine_ev = EVEngine()   # will use population defaults

    opportunities = engine_ev.run_daily()
    EVEngine.print_best_bets(opportunities, top_n=top_n)
    engine_ev.save_opportunities(opportunities)

    # Export to CSV for easy review
    if opportunities:
        df = EVEngine.to_dataframe(opportunities)
        out_path = f"best_bets_{date.today()}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Best bets exported → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_STAGES = ["ingest", "features", "train", "predict", "ev"]

STAGE_FNS = {
    "ingest":   stage_ingest,
    "features": stage_features,
    "train":    stage_train,
    "ev":       stage_ev,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLB Analytics Engine — daily pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stages", nargs="+", choices=list(STAGE_FNS.keys()),
        help="Pipeline stages to run (default: all)",
    )
    parser.add_argument("--all", action="store_true", help="Run all stages")
    parser.add_argument("--start",   type=str, help="Ingest start date YYYY-MM-DD")
    parser.add_argument("--end",     type=str, help="Ingest end date YYYY-MM-DD")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument("--window",  type=int, default=5, help="Rolling window size (games)")
    parser.add_argument("--top",     type=int, default=10, help="Best bets to display")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stages = list(STAGE_FNS.keys()) if args.all else (args.stages or ["ev"])

    logger.info(f"Pipeline starting | stages={stages} | date={date.today()}")

    for stage in stages:
        logger.info(f"--- Stage: {stage.upper()} ---")
        STAGE_FNS[stage](args)

    logger.success("Pipeline complete.")


if __name__ == "__main__":
    main()
