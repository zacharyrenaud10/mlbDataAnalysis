"""
mlb_analytics/ev_engine.py
---------------------------
Compares model-predicted probabilities against FanDuel implied probabilities
to identify +EV (positive expected value) betting opportunities.

Core formula:
  EV% = (P_model × (decimal_odds - 1)) - (1 - P_model)
  Edge = P_model - P_implied_fair

Kelly fraction:
  f* = (P_model × decimal_odds - 1) / (decimal_odds - 1)
  Recommended unit: fraction_kelly × bank_roll × kelly_multiplier
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import pandas as pd
from loguru import logger

from mlb_analytics.db import engine
from mlb_analytics.ingestion.fanduel_scraper import (
    american_to_implied_prob,
    get_todays_props,
)
from mlb_analytics.models.prediction_engine import BaseHitClassifier, StrikeoutRegressor

# ---------------------------------------------------------------------------
# Thresholds  —  tune these to control bet selectivity
# ---------------------------------------------------------------------------
MIN_EDGE           = 0.04   # model_prob must exceed fair_implied by at least 4pp
MIN_EV_PCT         = 0.02   # bet must have ≥ 2 % expected value
MAX_KELLY          = 0.25   # never suggest more than 25 % of bankroll on one bet
KELLY_FRACTION     = 0.25   # use quarter-Kelly to reduce variance
CONFIDENCE_THRESHOLDS = {
    "HIGH":   0.08,   # edge ≥ 8 pp
    "MEDIUM": 0.05,   # edge ≥ 5 pp
    "LOW":    MIN_EDGE,
}

PROP_TYPE_TO_MODEL = {
    "batter_hits":         "base_hit",
    "batter_total_bases":  "base_hit",   # rough proxy; refine later
    "pitcher_strikeouts":  "strikeouts",
}


# ---------------------------------------------------------------------------
# Data class for a single opportunity
# ---------------------------------------------------------------------------
@dataclass
class EVOpportunity:
    game_date:     str
    player_name:   str
    prop_type:     str
    bet_side:      str          # "over" or "under"
    fanduel_line:  float
    fanduel_price: int          # American odds
    implied_prob:  float        # vig-removed book probability
    model_prob:    float        # AI predicted probability
    edge:          float        # model_prob - implied_prob
    ev_pct:        float        # expected value %
    kelly_fraction:float
    confidence:    str
    model_version: str = ""
    # Optional for display
    team:          str = ""
    opponent:      str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "game_date":      self.game_date,
            "player":         self.player_name,
            "team":           self.team,
            "prop":           self.prop_type,
            "side":           self.bet_side,
            "line":           self.fanduel_line,
            "fd_odds":        self.fanduel_price,
            "implied_prob%":  f"{self.implied_prob*100:.1f}",
            "model_prob%":    f"{self.model_prob*100:.1f}",
            "edge%":          f"{self.edge*100:.1f}",
            "ev%":            f"{self.ev_pct*100:.2f}",
            "kelly_f":        f"{self.kelly_fraction:.4f}",
            "confidence":     self.confidence,
        }


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------

def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal (European) odds."""
    if american > 0:
        return (american / 100.0) + 1.0
    else:
        return (100.0 / abs(american)) + 1.0


def compute_ev(model_prob: float, decimal_odds: float) -> float:
    """
    Expected value per unit bet.
      EV% = model_prob × (decimal_odds - 1) - (1 - model_prob)
    """
    return model_prob * (decimal_odds - 1.0) - (1.0 - model_prob)


def compute_kelly(model_prob: float, decimal_odds: float) -> float:
    """
    Full Kelly criterion fraction.
      f* = (b×p - q) / b   where b = decimal_odds - 1, q = 1 - p
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - model_prob
    return max(0.0, (b * model_prob - q) / b)


def confidence_label(edge: float) -> str:
    if edge >= CONFIDENCE_THRESHOLDS["HIGH"]:
        return "HIGH"
    if edge >= CONFIDENCE_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Player name → player_id resolver  (simple in-memory lookup)
# ---------------------------------------------------------------------------

def _build_name_id_map() -> dict[str, int]:
    """Load {full_name: player_id} mapping from the players table."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql("SELECT player_id, full_name FROM players", conn)
        return dict(zip(df["full_name"].str.lower(), df["player_id"]))
    except Exception as exc:
        logger.warning(f"Could not load player name map: {exc}")
        return {}


def _fuzzy_match_name(name: str, name_map: dict[str, int]) -> Optional[int]:
    """Simple lowercase exact-match; extend with rapidfuzz for production."""
    return name_map.get(name.lower())


# ---------------------------------------------------------------------------
# Feature lookup helpers
# ---------------------------------------------------------------------------

def _get_matchup_features(player_id: int, game_date: str) -> dict:
    """Pull the latest matchup feature row for a player before game_date."""
    try:
        from sqlalchemy import text
        query = text("""
            SELECT * FROM matchup_features
            WHERE  (batter_id = :pid OR pitcher_id = :pid)
              AND  game_date  <= :dt
            ORDER  BY game_date DESC
            LIMIT  1
        """)
        with engine.connect() as conn:
            row = pd.read_sql(query, conn, params={"pid": player_id, "dt": game_date})
        return row.iloc[0].to_dict() if not row.empty else {}
    except Exception:
        return {}


def _get_batter_rolling(player_id: int, game_date: str) -> dict:
    try:
        from sqlalchemy import text
        q = text("""
            SELECT * FROM batter_rolling_features
            WHERE  player_id = :pid AND as_of_date <= :dt
            ORDER  BY as_of_date DESC LIMIT 1
        """)
        with engine.connect() as conn:
            row = pd.read_sql(q, conn, params={"pid": player_id, "dt": game_date})
        return row.iloc[0].to_dict() if not row.empty else {}
    except Exception:
        return {}


def _get_pitcher_rolling(player_id: int, game_date: str) -> dict:
    try:
        from sqlalchemy import text
        q = text("""
            SELECT * FROM pitcher_rolling_features
            WHERE  player_id = :pid AND as_of_date <= :dt
            ORDER  BY as_of_date DESC LIMIT 1
        """)
        with engine.connect() as conn:
            row = pd.read_sql(q, conn, params={"pid": player_id, "dt": game_date})
        return row.iloc[0].to_dict() if not row.empty else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# EV Engine
# ---------------------------------------------------------------------------

class EVEngine:
    """
    Main +EV identification engine.

    Usage:
        engine = EVEngine.from_saved_models(clf_path="...", reg_path="...")
        opportunities = engine.run_daily()
        df = engine.to_dataframe(opportunities)
    """

    def __init__(
        self,
        clf: Optional[BaseHitClassifier]  = None,
        reg: Optional[StrikeoutRegressor] = None,
    ):
        self.clf = clf
        self.reg = reg
        self._name_id_map: dict[str, int] = {}

    @classmethod
    def from_saved_models(cls, clf_path: str, reg_path: str) -> "EVEngine":
        clf = BaseHitClassifier.load(clf_path)
        reg = StrikeoutRegressor.load(reg_path)
        logger.info("EVEngine: models loaded")
        return cls(clf=clf, reg=reg)

    # -------------------------------------------------------------- main run

    def run_daily(
        self,
        game_date: Optional[str] = None,
        min_edge: float = MIN_EDGE,
        min_ev: float = MIN_EV_PCT,
    ) -> list[EVOpportunity]:
        """
        Full daily pipeline:
          1. Fetch today's FanDuel props
          2. For each prop, generate a model prediction
          3. Compare to implied probability
          4. Return sorted list of +EV opportunities
        """
        game_date = game_date or date.today().isoformat()
        logger.info(f"Running EV Engine for {game_date}")

        # Step 1 — fetch odds
        props_df = get_todays_props()
        if props_df.empty:
            logger.warning("No props data returned — aborting.")
            return []
        logger.info(f"  {len(props_df)} prop rows to evaluate")

        # Step 2 — build player name map
        self._name_id_map = _build_name_id_map()

        opportunities: list[EVOpportunity] = []

        for _, prop in props_df.iterrows():
            try:
                opp = self._evaluate_prop(prop, game_date)
                if opp is not None and opp.edge >= min_edge and opp.ev_pct >= min_ev:
                    opportunities.append(opp)
            except Exception as exc:
                logger.debug(f"Error evaluating {prop.get('player_name')}: {exc}")

        # Sort by EV descending
        opportunities.sort(key=lambda x: x.ev_pct, reverse=True)
        logger.success(
            f"Found {len(opportunities)} +EV opportunities "
            f"(edge ≥ {min_edge*100:.0f}pp, EV ≥ {min_ev*100:.0f}%)"
        )
        return opportunities

    # --------------------------------------------------------- single prop eval

    def _evaluate_prop(
        self,
        prop: pd.Series,
        game_date: str,
    ) -> Optional[EVOpportunity]:
        """Evaluate a single prop row and return an EVOpportunity or None."""

        prop_type   = prop.get("prop_type", "")
        player_name = prop.get("player_name", "")
        model_key   = PROP_TYPE_TO_MODEL.get(prop_type)

        if model_key is None:
            return None   # unsupported market

        # Determine the bet side: we bet "over" when model_prob > implied_over
        over_price  = prop.get("over_price")
        under_price = prop.get("under_price")
        fair_over   = prop.get("fair_prob_over")
        fair_under  = prop.get("fair_prob_under")
        line        = prop.get("line", 0.5)

        if fair_over is None or fair_under is None:
            return None

        # Resolve player to internal ID
        player_id = _fuzzy_match_name(player_name, self._name_id_map)
        if player_id is None:
            logger.debug(f"Could not match player: {player_name}")
            # Continue with population-average features as fallback
            pass

        # Get features
        b_feat = _get_batter_rolling(player_id, game_date)  if player_id else {}
        p_feat = _get_pitcher_rolling(player_id, game_date) if player_id else {}
        ctx    = _get_matchup_features(player_id, game_date) if player_id else {}

        # Run model
        model_prob = self._get_model_prob(
            model_key, prop_type, line, b_feat, p_feat, ctx
        )
        if model_prob is None:
            return None

        # Compare over vs under
        # For "batter_hits": model_prob = P(hit >= line)  → over bet
        # We compare model vs the fair implied probability for the same side
        bet_side      = "over"
        implied_prob  = fair_over
        fd_price      = over_price

        if (1.0 - model_prob) - fair_under > model_prob - fair_over:
            bet_side     = "under"
            implied_prob = fair_under
            fd_price     = under_price
            model_prob   = 1.0 - model_prob

        if fd_price is None:
            return None

        edge    = model_prob - implied_prob
        decimal = american_to_decimal(int(fd_price))
        ev_pct  = compute_ev(model_prob, decimal)
        kelly   = min(compute_kelly(model_prob, decimal) * KELLY_FRACTION, MAX_KELLY)

        return EVOpportunity(
            game_date      = game_date,
            player_name    = player_name,
            prop_type      = prop_type,
            bet_side       = bet_side,
            fanduel_line   = line,
            fanduel_price  = int(fd_price),
            implied_prob   = round(implied_prob, 4),
            model_prob     = round(model_prob, 4),
            edge           = round(edge, 4),
            ev_pct         = round(ev_pct, 4),
            kelly_fraction = round(kelly, 4),
            confidence     = confidence_label(edge),
            model_version  = (self.clf.version if self.clf else ""),
        )

    # -------------------------------------------------------- model dispatch

    def _get_model_prob(
        self,
        model_key: str,
        prop_type: str,
        line: float,
        batter_features: dict,
        pitcher_features: dict,
        context: dict,
    ) -> Optional[float]:
        """
        Route to the right model and return P(outcome >= line) as a float [0,1].
        Falls back to population averages if features are missing.
        """
        try:
            if model_key == "base_hit":
                if self.clf is None:
                    return 0.28  # MLB average hit rate ~28 % fallback
                result = self.clf.predict_single(
                    batter_features or _population_batter_defaults(),
                    pitcher_features or _population_pitcher_defaults(),
                    context,
                )
                return result["p_base_hit"]

            elif model_key == "strikeouts":
                if self.reg is None:
                    return None
                result = self.reg.predict_single(
                    pitcher_features or _population_pitcher_defaults(),
                    context,
                )
                pred_k = result["pred_k"]
                # Convert regression output → P(strikeouts >= line)
                # Use a Poisson approximation
                return _poisson_p_over(pred_k, line)

        except Exception as exc:
            logger.debug(f"Model prediction failed: {exc}")
        return None

    # ---------------------------------------------------- output formatting

    @staticmethod
    def to_dataframe(opportunities: list[EVOpportunity]) -> pd.DataFrame:
        return pd.DataFrame([o.to_dict() for o in opportunities])

    @staticmethod
    def print_best_bets(opportunities: list[EVOpportunity], top_n: int = 10) -> None:
        print(f"\n{'='*70}")
        print(f"  🏆  TODAY'S BEST BETS  ({date.today()})  —  Top {top_n}")
        print(f"{'='*70}")
        for i, opp in enumerate(opportunities[:top_n], 1):
            icon = "🔥" if opp.confidence == "HIGH" else ("⚡" if opp.confidence == "MEDIUM" else "📌")
            print(
                f"{i:2}. {icon} [{opp.confidence}]  {opp.player_name:<22} "
                f"{opp.prop_type:<25}  {opp.bet_side.upper():<5} {opp.fanduel_line}  "
                f"  odds: {opp.fanduel_price:+4d}"
            )
            print(
                f"       Book: {opp.implied_prob*100:.1f}%  "
                f"Model: {opp.model_prob*100:.1f}%  "
                f"Edge: {opp.edge*100:.1f}pp  "
                f"EV: {opp.ev_pct*100:.2f}%  "
                f"Kelly: {opp.kelly_fraction*100:.2f}%"
            )
        print(f"{'='*70}\n")

    # ------------------------------------------- persist to DB

    def save_opportunities(self, opportunities: list[EVOpportunity]) -> None:
        if not opportunities:
            return
        df = pd.DataFrame([
            {
                "identified_at":  datetime.utcnow(),
                "game_date":      o.game_date,
                "player_name":    o.player_name,
                "market_type":    "player_prop",
                "prop_type":      o.prop_type,
                "bet_side":       o.bet_side,
                "fanduel_line":   o.fanduel_line,
                "fanduel_price":  o.fanduel_price,
                "implied_prob":   o.implied_prob,
                "model_prob":     o.model_prob,
                "edge":           o.edge,
                "ev_pct":         o.ev_pct,
                "kelly_fraction": o.kelly_fraction,
                "confidence":     o.confidence,
                "model_version":  o.model_version,
                "settled":        False,
            }
            for o in opportunities
        ])
        df.to_sql("ev_opportunities", con=engine, if_exists="append",
                  index=False, method="multi")
        logger.success(f"Saved {len(df)} EV opportunities to DB")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poisson_p_over(mean: float, line: float) -> float:
    """P(X > line) where X ~ Poisson(mean). Used to convert predicted K → probability."""
    import math
    if mean <= 0:
        return 0.0
    # P(X >= ceil(line)) = 1 - P(X <= floor(line))
    k_floor = int(math.floor(line))
    cdf = 0.0
    for k in range(0, k_floor + 1):
        cdf += (math.exp(-mean) * mean**k) / math.factorial(k)
    return float(1.0 - cdf)


def _population_batter_defaults() -> dict:
    """MLB population-average batter features (2023-2025 approximate)."""
    return {
        "avg_exit_velo":    88.5,
        "avg_launch_angle": 12.0,
        "hard_hit_pct":     0.38,
        "k_pct":            0.22,
        "bb_pct":           0.085,
        "woba":             0.315,
    }


def _population_pitcher_defaults() -> dict:
    """MLB population-average pitcher features (2023-2025 approximate)."""
    return {
        "avg_fastball_velo": 93.5,
        "whiff_pct":         0.25,
        "zone_pct":          0.47,
        "k_pct":             0.22,
        "bb_pct":            0.085,
        "hr_per_9":          1.3,
    }
