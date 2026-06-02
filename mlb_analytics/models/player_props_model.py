"""
mlb_analytics/models/player_props_model.py
-------------------------------------------
Trains and predicts full per-batter and per-pitcher stat lines
for player prop betting.

Batter models (per game):
  - P(1+ hits)
  - P(1+ total bases), P(2+ TB), P(3+ TB), P(4+ TB)
  - P(1+ runs)
  - P(1+ RBI)
  - P(1+ walks)
  - P(1+ strikeouts)
  - P(1+ home runs)
  - P(stolen base attempt)
  - Expected total bases (regression)

Pitcher models (per start):
  - Expected strikeouts (regression)
  - P(6+ K), P(7+ K), P(8+ K)
  - Expected innings pitched
  - P(win)
  - P(quality start — 6+ IP, 3 or fewer ER)
"""

from __future__ import annotations

import os
import pickle
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
BATTER_FEATURES = [
    "b_avg_ev", "b_avg_la", "b_hard_hit_pct", "b_barrel_pct",
    "b_k_pct", "b_bb_pct", "b_woba_rolling", "b_ba",
    "p_avg_velo", "p_whiff_pct", "p_zone_pct", "p_k_pct", "p_bb_pct",
    "park_factor", "batter_hand_R", "batter_hand_S", "pitcher_hand_R",
    "h2h_pa", "h2h_ba",
    "b_season_woba", "b_season_wrc_plus",
    "p_season_era", "p_season_k_pct",
    "batting_order",
    "is_home",
]

PITCHER_FEATURES = [
    "p_avg_velo", "p_whiff_pct", "p_zone_pct",
    "p_k_pct", "p_bb_pct", "p_hr_per_9",
    "p_season_era", "p_season_fip", "p_season_k_9",
    "opp_k_pct",
    "park_factor", "is_home",
    "days_rest",
]

# ---------------------------------------------------------------------------
# Batter prop targets
# ---------------------------------------------------------------------------
BATTER_PROP_TARGETS = {
    "hit_1plus":     {"desc": "1+ Hits",          "market": "batter_hits",         "line": 0.5},
    "total_bases_1": {"desc": "1+ Total Bases",   "market": "batter_total_bases",  "line": 0.5},
    "total_bases_2": {"desc": "2+ Total Bases",   "market": "batter_total_bases",  "line": 1.5},
    "total_bases_3": {"desc": "3+ Total Bases",   "market": "batter_total_bases",  "line": 2.5},
    "run_1plus":     {"desc": "1+ Runs",           "market": "batter_runs",         "line": 0.5},
    "rbi_1plus":     {"desc": "1+ RBI",            "market": "batter_rbis",         "line": 0.5},
    "walk_1plus":    {"desc": "1+ Walks",          "market": "batter_walks",        "line": 0.5},
    "k_1plus":       {"desc": "1+ Strikeouts",     "market": "batter_strikeouts",   "line": 0.5},
    "hr_1plus":      {"desc": "1+ Home Runs",      "market": "batter_home_runs",    "line": 0.5},
    "sb_1plus":      {"desc": "1+ Stolen Bases",   "market": "batter_stolen_bases", "line": 0.5},
}

PITCHER_PROP_TARGETS = {
    "k_4plus":  {"desc": "4+ Strikeouts",  "market": "pitcher_strikeouts", "line": 3.5},
    "k_5plus":  {"desc": "5+ Strikeouts",  "market": "pitcher_strikeouts", "line": 4.5},
    "k_6plus":  {"desc": "6+ Strikeouts",  "market": "pitcher_strikeouts", "line": 5.5},
    "k_7plus":  {"desc": "7+ Strikeouts",  "market": "pitcher_strikeouts", "line": 6.5},
    "k_8plus":  {"desc": "8+ Strikeouts",  "market": "pitcher_strikeouts", "line": 7.5},
    "qs":       {"desc": "Quality Start",  "market": "pitcher_qs",         "line": 0.5},
    "win":      {"desc": "Win",            "market": "pitcher_win",        "line": 0.5},
}


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class PlayerPropsModel:
    """
    Unified model for all player prop predictions.
    Trains one XGBoost classifier per prop target.
    """

    def __init__(self, version: Optional[str] = None):
        self.version     = version or f"props_{date.today():%Y%m%d}"
        self.models:     dict = {}
        self.regressors: dict = {}
        self.features_b  = BATTER_FEATURES
        self.features_p  = PITCHER_FEATURES

    # ------------------------------------------------------------------ fit

    def fit_batter_models(
        self,
        X: pd.DataFrame,
        game_stats: pd.DataFrame,
    ) -> None:
        """Train all batter prop classifiers."""
        logger.info("Training batter prop models...")
        X = self._align(X, self.features_b)

        for target, info in BATTER_PROP_TARGETS.items():

            # Build binary target — skip if column not available
            y = self._build_batter_target(target, game_stats)
            if y is None:
                logger.info(f"  Skipping {target} — column not in data")
                continue

            hit_rate = y.mean()
            if hit_rate < 0.01 or hit_rate > 0.99:
                logger.warning(
                    f"  Skipping {target} — degenerate rate ({hit_rate:.3f})"
                )
                continue

            logger.info(
                f"  Training {target} | n={len(y):,} | rate={hit_rate:.3f}"
            )

            split = int(len(X) * 0.8)
            Xt, Xv = X.iloc[:split], X.iloc[split:]
            yt, yv = y.iloc[:split], y.iloc[split:]

            # --- XGBoost ---
            xgb = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=(1 - hit_rate) / max(hit_rate, 0.001),
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
            xgb.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)

            # --- Random Forest ---
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(Xt, yt)

            # --- Logistic Regression ---
            lr = LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=3000,
                random_state=42,
            )
            lr.fit(Xt, yt)

            # --- LightGBM (if available) ---
            estimators = [("xgb", xgb), ("rf", rf), ("lr", lr)]
            if HAS_LGBM:
                lgbm = LGBMClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
                lgbm.fit(Xt, yt)
                estimators.append(("lgbm", lgbm))

            # --- Voting Ensemble ---
            ensemble = VotingClassifier(
                estimators=estimators,
                voting="soft",
                weights=[3, 2, 1] + ([2] if HAS_LGBM else []),
            )
            ensemble.fit(Xt, yt)

            # Calibrate the ensemble
            cal = CalibratedClassifierCV(ensemble, method="isotonic", cv="prefit")
            cal.fit(Xv, yv)

            proba = cal.predict_proba(Xv)[:, 1]
            auc   = roc_auc_score(yv, proba)

            # Cross-validation AUC
            tscv = TimeSeriesSplit(n_splits=3)
            cv_aucs = []
            for train_idx, val_idx in tscv.split(X):
                Xtr, Xval = X.iloc[train_idx], X.iloc[val_idx]
                ytr, yval = y.iloc[train_idx], y.iloc[val_idx]
                if len(yval.unique()) < 2:
                    continue
                ensemble_cv = VotingClassifier(
                    estimators=[
                        ("xgb", XGBClassifier(n_estimators=100, max_depth=4,
                            learning_rate=0.05, use_label_encoder=False,
                            eval_metric="logloss", random_state=42, n_jobs=-1)),
                        ("rf",  RandomForestClassifier(n_estimators=100,
                            max_depth=6, random_state=42, n_jobs=-1)),
                        ("lr",  LogisticRegression(C=0.1, max_iter=1000,
                            random_state=42)),
                    ],
                    voting="soft",
                )
                ensemble_cv.fit(Xtr, ytr)
                cv_aucs.append(roc_auc_score(yval, ensemble_cv.predict_proba(Xval)[:, 1]))

            cv_mean = sum(cv_aucs) / len(cv_aucs) if cv_aucs else auc
            logger.info(f"    AUC={auc:.4f}  CV-AUC={cv_mean:.4f}")

            self.models[target] = cal

        # Total bases regression
        if "total_bases" in game_stats.columns:
            logger.info("  Training total_bases regressor...")
            y_tb = game_stats["total_bases"].fillna(0)
            reg  = XGBRegressor(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, random_state=42, n_jobs=-1
            )
            reg.fit(X, y_tb, verbose=False)
            self.regressors["total_bases"] = reg
            logger.success("  Total bases regressor trained")

        logger.success(f"Trained {len(self.models)} batter prop models")

    def _build_batter_target(
        self,
        target: str,
        game_stats: pd.DataFrame,
    ) -> Optional[pd.Series]:
        """
        Build a binary target series for a given prop.
        Returns None if required column is missing.
        """
        col_map = {
            "hit_1plus":     ("hits",        1),
            "total_bases_1": ("total_bases", 1),
            "total_bases_2": ("total_bases", 2),
            "total_bases_3": ("total_bases", 3),
            "run_1plus":     ("runs",        1),
            "rbi_1plus":     ("rbi",         1),
            "walk_1plus":    ("bb",          1),
            "k_1plus":       ("k",           1),
            "hr_1plus":      ("hr",          1),
            "sb_1plus":      ("sb",          1),
        }

        if target not in col_map:
            return None

        col, threshold = col_map[target]

        if col not in game_stats.columns:
            return None

        return (game_stats[col] >= threshold).astype(int)

    def fit_pitcher_models(
        self,
        X: pd.DataFrame,
        game_stats: pd.DataFrame,
    ) -> None:
        """Train all pitcher prop classifiers."""
        logger.info("Training pitcher prop models...")
        X = self._align(X, self.features_p)

        col_map = {
            "k_4plus": ("k", 4),
            "k_5plus": ("k", 5),
            "k_6plus": ("k", 6),
            "k_7plus": ("k", 7),
            "k_8plus": ("k", 8),
            "qs":      ("qs", 1),
            "win":     ("win", 1),
        }

        for target in PITCHER_PROP_TARGETS:
            if target not in col_map:
                continue

            col, threshold = col_map[target]

            if col not in game_stats.columns:
                logger.info(f"  Skipping {target} — column '{col}' not in data")
                continue

            y        = (game_stats[col] >= threshold).astype(int)
            hit_rate = y.mean()

            if hit_rate < 0.01 or hit_rate > 0.99:
                continue

            logger.info(f"  Training {target} | rate={hit_rate:.3f}")

            split = int(len(X) * 0.8)
            Xt, Xv = X.iloc[:split], X.iloc[split:]
            yt, yv = y.iloc[:split], y.iloc[split:]

            base = XGBClassifier(
                n_estimators=300, max_depth=4,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=(1 - hit_rate) / max(hit_rate, 0.001),
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42, n_jobs=-1,
            )
            base.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)

            cal = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
            cal.fit(Xv, yv)
            self.models[target] = cal

        # K regression
        if "k" in game_stats.columns:
            logger.info("  Training pitcher K regressor...")
            reg = XGBRegressor(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, random_state=42, n_jobs=-1
            )
            reg.fit(X, game_stats["k"].fillna(0), verbose=False)
            self.regressors["pitcher_k"] = reg
            logger.success("  Pitcher K regressor trained")

        logger.success(f"Trained {len(self.models)} pitcher prop models")

    # -------------------------------------------------------------- predict

    def predict_batter(
        self,
        batter_features: dict,
        pitcher_features: dict,
        context: dict,
    ) -> dict[str, float]:
        """Predict all prop probabilities for a batter in a matchup."""
        row = self._build_batter_row(batter_features, pitcher_features, context)
        X   = pd.DataFrame([row])
        X   = self._align(X, self.features_b)

        results = {}

        for target, info in BATTER_PROP_TARGETS.items():
            if target in self.models:
                prob = float(self.models[target].predict_proba(X)[0, 1])
                results[target] = round(prob, 4)
            else:
                results[target] = self._pop_avg(target)

        # Total bases regression
        if "total_bases" in self.regressors:
            results["expected_tb"] = round(
                float(self.regressors["total_bases"].predict(X)[0]), 3
            )
        else:
            results["expected_tb"] = self._estimate_tb(results)

        return results

    def predict_pitcher(
        self,
        pitcher_features: dict,
        context: dict,
    ) -> dict[str, float]:
        """Predict all prop probabilities for a pitcher start."""
        row = self._build_pitcher_row(pitcher_features, context)
        X   = pd.DataFrame([row])
        X   = self._align(X, self.features_p)

        results = {}
        for target in PITCHER_PROP_TARGETS:
            if target in self.models:
                prob = float(self.models[target].predict_proba(X)[0, 1])
                results[target] = round(prob, 4)
            else:
                results[target] = self._pop_avg_pitcher(target)

        if "pitcher_k" in self.regressors:
            raw_k = float(self.regressors["pitcher_k"].predict(X)[0])
            results["expected_k"] = round(min(raw_k, 12.0), 2
            )
        else:
            results["expected_k"] = round(
                pitcher_features.get("k_pct", 0.22) * 27, 1
            )

        return results

    # --------------------------------------------------------- row builders

    def _build_batter_row(self, bf: dict, pf: dict, ctx: dict) -> dict:
        hand = ctx.get("batter_hand", "R")
        return {
            "b_avg_ev":          bf.get("avg_exit_velo",    88.5),
            "b_avg_la":          bf.get("avg_launch_angle", 12.0),
            "b_hard_hit_pct":    bf.get("hard_hit_pct",     0.38),
            "b_barrel_pct":      bf.get("barrel_pct",       0.08),
            "b_k_pct":           bf.get("k_pct",            0.22),
            "b_bb_pct":          bf.get("bb_pct",           0.085),
            "b_woba_rolling":    bf.get("woba",             0.315),
            "b_ba":              bf.get("ba",               0.248),
            "p_avg_velo":        pf.get("avg_fastball_velo",93.5),
            "p_whiff_pct":       pf.get("whiff_pct",        0.25),
            "p_zone_pct":        pf.get("zone_pct",         0.47),
            "p_k_pct":           pf.get("k_pct",            0.22),
            "p_bb_pct":          pf.get("bb_pct",           0.085),
            "park_factor":       ctx.get("park_factor",     100.0),
            "batter_hand_R":     int(hand == "R"),
            "batter_hand_S":     int(hand == "S"),
            "pitcher_hand_R":    int(ctx.get("pitcher_hand","R") == "R"),
            "h2h_pa":            ctx.get("h2h_pa",          0),
            "h2h_ba":            ctx.get("h2h_ba",          0.250),
            "b_season_woba":     ctx.get("b_season_woba",   0.315),
            "b_season_wrc_plus": ctx.get("b_season_wrc_plus",100.0),
            "p_season_era":      ctx.get("p_season_era",    4.0),
            "p_season_k_pct":    ctx.get("p_season_k_pct",  0.22),
            "batting_order":     ctx.get("batting_order",   5),
            "is_home":           int(ctx.get("is_home",     False)),
        }

    def _build_pitcher_row(self, pf: dict, ctx: dict) -> dict:
        return {
            "p_avg_velo":    pf.get("avg_fastball_velo", 93.5),
            "p_whiff_pct":   pf.get("whiff_pct",         0.25),
            "p_zone_pct":    pf.get("zone_pct",          0.47),
            "p_k_pct":       pf.get("k_pct",             0.22),
            "p_bb_pct":      pf.get("bb_pct",            0.085),
            "p_hr_per_9":    pf.get("hr_per_9",          1.3),
            "p_season_era":  ctx.get("p_season_era",     4.0),
            "p_season_fip":  ctx.get("p_season_fip",     4.0),
            "p_season_k_9":  ctx.get("p_season_k_9",     8.0),
            "opp_k_pct":     ctx.get("opp_k_pct",        0.22),
            "park_factor":   ctx.get("park_factor",      100.0),
            "is_home":       int(ctx.get("is_home",      True)),
            "days_rest":     ctx.get("days_rest",        4),
        }

    def _align(self, X: pd.DataFrame, features: list) -> pd.DataFrame:
        for col in features:
            if col not in X.columns:
                X[col] = 0.0
        return X[features].fillna(0.0)

    # -------------------------------------------------- population fallbacks

    def _pop_avg(self, target: str) -> float:
        avgs = {
            "hit_1plus":     0.280,
            "total_bases_1": 0.380,
            "total_bases_2": 0.210,
            "total_bases_3": 0.095,
            "run_1plus":     0.310,
            "rbi_1plus":     0.250,
            "walk_1plus":    0.125,
            "k_1plus":       0.310,
            "hr_1plus":      0.055,
            "sb_1plus":      0.040,
        }
        return avgs.get(target, 0.25)

    def _pop_avg_pitcher(self, target: str) -> float:
        avgs = {
            "k_4plus": 0.72,
            "k_5plus": 0.58,
            "k_6plus": 0.42,
            "k_7plus": 0.26,
            "k_8plus": 0.14,
            "qs":      0.45,
            "win":     0.50,
        }
        return avgs.get(target, 0.30)

    def _estimate_tb(self, results: dict) -> float:
        return (
            results.get("hit_1plus",     0.28) * 1.0 +
            results.get("total_bases_2", 0.21) * 0.5 +
            results.get("total_bases_3", 0.10) * 0.3 +
            results.get("hr_1plus",      0.055) * 1.5
        )

    # ---------------------------------------------------------- serialise

    def save(self, path: Optional[str] = None) -> str:
        path = path or str(MODEL_DIR / f"{self.version}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"PlayerPropsModel saved → {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "PlayerPropsModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"PlayerPropsModel loaded ← {path}")
        return obj


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def print_batter_props(
    player_name: str,
    pitcher_name: str,
    props: dict,
    weather_lean: str = "",
    umpire_k_factor: float = 1.0,
) -> None:
    print(f"\n  {'='*58}")
    print(f"  🏏  {player_name:<30} vs {pitcher_name}")
    if weather_lean:
        print(f"  {weather_lean}")
    print(f"  {'='*58}")

    etb = props.get("expected_tb", 0)
    print(f"  Expected Total Bases: {etb:.2f}")
    print()

    h1 = props.get("hit_1plus", 0)
    print(f"  HITS")
    print(f"    1+ Hits:        {h1*100:5.1f}%  {'🔥' if h1 > 0.35 else '📊'}")
    print()

    tb1 = props.get("total_bases_1", 0)
    tb2 = props.get("total_bases_2", 0)
    tb3 = props.get("total_bases_3", 0)
    print(f"  TOTAL BASES")
    print(f"    1+ TB:          {tb1*100:5.1f}%")
    print(f"    2+ TB:          {tb2*100:5.1f}%  {'🔥' if tb2 > 0.30 else '📊'}")
    print(f"    3+ TB:          {tb3*100:5.1f}%  {'🔥' if tb3 > 0.15 else '📊'}")
    print()

    run  = props.get("run_1plus",  0)
    rbi  = props.get("rbi_1plus",  0)
    bb   = props.get("walk_1plus", 0)
    k    = min(props.get("k_1plus", 0) * umpire_k_factor, 0.99)
    hr   = props.get("hr_1plus",   0)
    sb   = props.get("sb_1plus",   0)

    print(f"  OTHER PROPS")
    print(f"    1+ Runs:        {run*100:5.1f}%")
    print(f"    1+ RBI:         {rbi*100:5.1f}%")
    print(f"    1+ Walks:       {bb*100:5.1f}%")
    print(f"    1+ Strikeouts:  {k*100:5.1f}%")
    print(f"    1+ Home Runs:   {hr*100:5.1f}%  {'🔥' if hr > 0.10 else '📊'}")
    print(f"    1+ Stolen Base: {sb*100:5.1f}%")


def print_pitcher_props(
    pitcher_name: str,
    opponent: str,
    props: dict,
    weather_lean: str = "",
    umpire_k_factor: float = 1.0,
) -> None:
    exp_k     = props.get("expected_k", 0)
    exp_k_adj = round(exp_k * umpire_k_factor, 1)

    print(f"\n  {'='*58}")
    print(f"  ⚾  {pitcher_name:<30} vs {opponent}")
    if weather_lean:
        print(f"  {weather_lean}")
    print(f"  {'='*58}")
    print(
        f"  Expected Strikeouts: {exp_k_adj}"
        + (f" (ump adj {umpire_k_factor:.2f}x)" if umpire_k_factor != 1.0 else "")
    )
    print()

    print(f"  STRIKEOUT PROPS")
    for target in ["k_4plus", "k_5plus", "k_6plus", "k_7plus", "k_8plus"]:
        p    = min(props.get(target, 0) * umpire_k_factor, 0.99)
        info = PITCHER_PROP_TARGETS[target]
        hot  = "🔥" if p > 0.60 else ("📊" if p > 0.35 else "❄️ ")
        print(f"    {info['desc']:<18} {p*100:5.1f}%  {hot}")

    print()
    print(f"  GAME PROPS")
    qs  = props.get("qs",  0)
    win = props.get("win", 0)
    print(f"    Quality Start:     {qs*100:5.1f}%")
    print(f"    Win:               {win*100:5.1f}%")