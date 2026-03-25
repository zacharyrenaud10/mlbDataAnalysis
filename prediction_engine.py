"""
mlb_analytics/models/prediction_engine.py
------------------------------------------
Two models:
  1. BaseHitClassifier  — XGBoost binary classifier
     Predicts P(batter gets a base hit in a given PA / game)

  2. StrikeoutRegressor — XGBoost regressor
     Predicts expected pitcher strikeouts for a given game start

Both expose a sklearn-compatible fit() / predict() / predict_proba() API
and include SHAP-based explainability.
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBClassifier, XGBRegressor

MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature schemas  (ordered lists define column order fed to the model)
# ---------------------------------------------------------------------------

BASE_HIT_FEATURES = [
    # Batter rolling (last 5 games)
    "b_avg_ev",
    "b_avg_la",
    "b_hard_hit_pct",
    "b_k_pct",
    "b_bb_pct",
    "b_woba_rolling",
    # Pitcher rolling
    "p_avg_velo",
    "p_whiff_pct",
    "p_zone_pct",
    "p_k_pct",
    # Matchup / situational
    "park_factor",
    "batter_hand_R",      # one-hot
    "batter_hand_S",
    "pitcher_hand_R",
    # Career H2H
    "h2h_pa",
    "h2h_ba",
    # Seasonal baselines
    "b_season_woba",
    "b_season_wrc_plus",
    "p_season_era",
    "p_season_k_pct",
]

K_REGRESSOR_FEATURES = [
    # Pitcher rolling
    "p_avg_velo",
    "p_whiff_pct",
    "p_zone_pct",
    "p_k_pct",
    "p_bb_pct",
    "p_hr_per_9",
    # Seasonal
    "p_season_era",
    "p_season_fip",
    "p_season_k_9",
    "p_season_k_pct",
    # Opponent context
    "opp_team_k_pct_season",
    "park_factor",
    "is_home",
]


# ---------------------------------------------------------------------------
# Default hyperparameters (can be overridden by Optuna tuning)
# ---------------------------------------------------------------------------

DEFAULT_CLF_PARAMS: dict[str, Any] = {
    "n_estimators":     400,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "scale_pos_weight": 2.5,   # ~30% hit rate → re-balance
    "eval_metric":      "logloss",
    "use_label_encoder": False,
    "random_state":     42,
    "n_jobs":           -1,
}

DEFAULT_REG_PARAMS: dict[str, Any] = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha":        0.05,
    "reg_lambda":       1.0,
    "eval_metric":      "rmse",
    "random_state":     42,
    "n_jobs":           -1,
}


# ---------------------------------------------------------------------------
# Base Hit Classifier
# ---------------------------------------------------------------------------

class BaseHitClassifier:
    """
    XGBoost binary classifier with isotonic probability calibration.
    Target: 1 if batter recorded at least one hit, else 0.
    """

    def __init__(self, params: Optional[dict] = None, version: Optional[str] = None):
        self.params  = params or DEFAULT_CLF_PARAMS.copy()
        self.version = version or f"bh_clf_{datetime.now():%Y%m%d_%H%M}"
        self.features = BASE_HIT_FEATURES
        self._model: Optional[CalibratedClassifierCV] = None
        self.eval_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series]   = None,
        calibrate: bool = True,
    ) -> "BaseHitClassifier":
        X_train = self._align_features(X_train)
        logger.info(f"Training BaseHitClassifier | {len(X_train):,} rows, "
                    f"{y_train.mean():.3f} hit-rate")

        base_clf = XGBClassifier(**self.params)

        if X_val is not None:
            X_val_aligned = self._align_features(X_val)
            base_clf.fit(
                X_train, y_train,
                eval_set=[(X_val_aligned, y_val)],
                verbose=False,
            )
        else:
            base_clf.fit(X_train, y_train, verbose=False)

        if calibrate:
            logger.info("  Calibrating probabilities (isotonic) …")
            self._model = CalibratedClassifierCV(
                base_clf, method="isotonic", cv="prefit"
            )
            cal_X = X_val_aligned if X_val is not None else X_train
            cal_y = y_val         if y_val  is not None else y_train
            self._model.fit(cal_X, cal_y)
        else:
            self._model = base_clf  # type: ignore

        if X_val is not None and y_val is not None:
            self._compute_eval_metrics(X_val_aligned, y_val)

        return self

    # -------------------------------------------------------------- predict
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns shape (n, 2) array; [:, 1] is P(hit)."""
        assert self._model is not None, "Model not trained. Call fit() first."
        X = self._align_features(X)
        return self._model.predict_proba(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def predict_single(
        self,
        batter_features: dict,
        pitcher_features: dict,
        context: dict,
    ) -> dict[str, float]:
        """
        Convenience wrapper: pass raw feature dicts, get back a probability dict.

        Parameters
        ----------
        batter_features  : from get_batter_features_as_of()
        pitcher_features : from get_pitcher_features_as_of()
        context          : park_factor, batter_hand, pitcher_hand, h2h_pa, h2h_ba,
                           b_season_woba, b_season_wrc_plus, p_season_era, p_season_k_pct
        """
        row = {
            "b_avg_ev":         batter_features.get("avg_exit_velo"),
            "b_avg_la":         batter_features.get("avg_launch_angle"),
            "b_hard_hit_pct":   batter_features.get("hard_hit_pct"),
            "b_k_pct":          batter_features.get("k_pct"),
            "b_bb_pct":         batter_features.get("bb_pct"),
            "b_woba_rolling":   batter_features.get("woba"),
            "p_avg_velo":       pitcher_features.get("avg_fastball_velo"),
            "p_whiff_pct":      pitcher_features.get("whiff_pct"),
            "p_zone_pct":       pitcher_features.get("zone_pct"),
            "p_k_pct":          pitcher_features.get("k_pct"),
            "park_factor":      context.get("park_factor", 100.0),
            "batter_hand_R":    int(context.get("batter_hand") == "R"),
            "batter_hand_S":    int(context.get("batter_hand") == "S"),
            "pitcher_hand_R":   int(context.get("pitcher_hand") == "R"),
            "h2h_pa":           context.get("h2h_pa", 0),
            "h2h_ba":           context.get("h2h_ba", 0.250),
            "b_season_woba":    context.get("b_season_woba", 0.320),
            "b_season_wrc_plus":context.get("b_season_wrc_plus", 100.0),
            "p_season_era":     context.get("p_season_era", 4.0),
            "p_season_k_pct":   context.get("p_season_k_pct", 0.22),
        }
        X = pd.DataFrame([row])
        proba = self.predict_proba(X)[0, 1]
        return {"p_base_hit": float(proba)}

    # --------------------------------------------------------- eval metrics
    def _compute_eval_metrics(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        proba = self.predict_proba(X_val)[:, 1]
        preds = (proba >= 0.5).astype(int)
        self.eval_metrics = {
            "auc_roc":     float(roc_auc_score(y_val, proba)),
            "log_loss":    float(log_loss(y_val, proba)),
            "brier_score": float(brier_score_loss(y_val, proba)),
        }
        logger.info(f"  AUC={self.eval_metrics['auc_roc']:.4f}  "
                    f"LogLoss={self.eval_metrics['log_loss']:.4f}  "
                    f"Brier={self.eval_metrics['brier_score']:.4f}")

    # ------------------------------------------------- feature alignment
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure column order matches training; fill missing with 0."""
        for col in self.features:
            if col not in X.columns:
                X[col] = 0.0
        return X[self.features].fillna(0.0)

    # --------------------------------------------------------- serialise
    def save(self, path: Optional[str] = None) -> str:
        path = path or str(MODEL_DIR / f"{self.version}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved → {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "BaseHitClassifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Model loaded ← {path}")
        return obj

    # ---------------------------------------------------------- SHAP
    def explain(self, X: pd.DataFrame, max_display: int = 15) -> pd.DataFrame:
        """Return a DataFrame of mean |SHAP| values sorted descending."""
        X = self._align_features(X)
        try:
            raw_clf = (self._model.calibrated_classifiers_[0].estimator
                       if hasattr(self._model, "calibrated_classifiers_") else self._model)
            explainer = shap.TreeExplainer(raw_clf)
            shap_vals = explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            importance = (
                pd.DataFrame({"feature": self.features,
                               "mean_abs_shap": np.abs(shap_vals).mean(axis=0)})
                .sort_values("mean_abs_shap", ascending=False)
                .head(max_display)
            )
            return importance
        except Exception as exc:
            logger.warning(f"SHAP explain failed: {exc}")
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Strikeout Regressor
# ---------------------------------------------------------------------------

class StrikeoutRegressor:
    """
    XGBoost regressor predicting expected pitcher strikeouts
    for a given game start.
    """

    def __init__(self, params: Optional[dict] = None, version: Optional[str] = None):
        self.params  = params or DEFAULT_REG_PARAMS.copy()
        self.version = version or f"k_reg_{datetime.now():%Y%m%d_%H%M}"
        self.features = K_REGRESSOR_FEATURES
        self._model: Optional[XGBRegressor] = None
        self.eval_metrics: dict[str, float] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series]   = None,
    ) -> "StrikeoutRegressor":
        X_train = self._align_features(X_train)
        logger.info(f"Training StrikeoutRegressor | {len(X_train):,} rows  "
                    f"mean_K={y_train.mean():.2f}")

        self._model = XGBRegressor(**self.params)

        if X_val is not None:
            X_val_aligned = self._align_features(X_val)
            self._model.fit(
                X_train, y_train,
                eval_set=[(X_val_aligned, y_val)],
                verbose=False,
            )
            preds = self._model.predict(X_val_aligned)
            preds = np.clip(preds, 0, None)  # no negative Ks
            self.eval_metrics = {
                "mae":  float(mean_absolute_error(y_val, preds)),
                "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
            }
            logger.info(f"  MAE={self.eval_metrics['mae']:.3f}  "
                        f"RMSE={self.eval_metrics['rmse']:.3f}")
        else:
            self._model.fit(X_train, y_train, verbose=False)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self._model is not None, "Model not trained."
        X = self._align_features(X)
        preds = self._model.predict(X)
        return np.clip(preds, 0, None)

    def predict_single(
        self,
        pitcher_features: dict,
        context: dict,
    ) -> dict[str, float]:
        row = {
            "p_avg_velo":             pitcher_features.get("avg_fastball_velo"),
            "p_whiff_pct":            pitcher_features.get("whiff_pct"),
            "p_zone_pct":             pitcher_features.get("zone_pct"),
            "p_k_pct":                pitcher_features.get("k_pct"),
            "p_bb_pct":               pitcher_features.get("bb_pct"),
            "p_hr_per_9":             pitcher_features.get("hr_per_9"),
            "p_season_era":           context.get("p_season_era", 4.0),
            "p_season_fip":           context.get("p_season_fip", 4.0),
            "p_season_k_9":           context.get("p_season_k_9", 8.0),
            "p_season_k_pct":         context.get("p_season_k_pct", 0.22),
            "opp_team_k_pct_season":  context.get("opp_team_k_pct_season", 0.22),
            "park_factor":            context.get("park_factor", 100.0),
            "is_home":                int(context.get("is_home", False)),
        }
        X = pd.DataFrame([row])
        pred = float(self.predict(X)[0])
        return {"pred_k": pred}

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.features:
            if col not in X.columns:
                X[col] = 0.0
        return X[self.features].fillna(0.0)

    def save(self, path: Optional[str] = None) -> str:
        path = path or str(MODEL_DIR / f"{self.version}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved → {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "StrikeoutRegressor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Model loaded ← {path}")
        return obj


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------

def cross_validate_classifier(
    clf: BaseHitClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict[str, float]:
    """
    Stratified k-fold CV returning mean AUC, log-loss, and Brier score.
    Uses *time-aware* folds by preserving row order (no shuffle).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    aucs, losses, briers = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        Xt, Xv = X.iloc[train_idx], X.iloc[val_idx]
        yt, yv = y.iloc[train_idx], y.iloc[val_idx]

        fold_clf = BaseHitClassifier(params=clf.params.copy())
        fold_clf.fit(Xt, yt, calibrate=False)  # skip calibration per fold
        proba = fold_clf.predict_proba(Xv)[:, 1]

        aucs.append(roc_auc_score(yv, proba))
        losses.append(log_loss(yv, proba))
        briers.append(brier_score_loss(yv, proba))
        logger.debug(f"  Fold {fold}: AUC={aucs[-1]:.4f}")

    metrics = {
        "cv_auc_mean":    float(np.mean(aucs)),
        "cv_auc_std":     float(np.std(aucs)),
        "cv_logloss_mean":float(np.mean(losses)),
        "cv_brier_mean":  float(np.mean(briers)),
    }
    logger.info(f"CV results: {json.dumps(metrics, indent=2)}")
    return metrics
