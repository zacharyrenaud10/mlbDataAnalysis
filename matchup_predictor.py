"""
matchup_predictor.py
--------------------
Interactive matchup analyzer with full prop predictions.
Also finds the most likely hit and HR candidates across all games.

Usage:
    python matchup_predictor.py              <- interactive game selector
    python matchup_predictor.py --all        <- all games today
    python matchup_predictor.py --hits       <- top hit candidates today
    python matchup_predictor.py --hr         <- top HR candidates today
    python matchup_predictor.py --pitcher "Webb" --pitcher-team SF --batting-team NYY
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import date
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import text

load_dotenv()

from mlb_analytics.db import engine
from mlb_analytics.models.player_props_model import (
    PlayerPropsModel,
    BATTER_PROP_TARGETS,
    PITCHER_PROP_TARGETS,
    print_batter_props,
    print_pitcher_props,
)

# ---------------------------------------------------------------------------
# Park factors
# ---------------------------------------------------------------------------
PARK_FACTORS = {
    "NYY": 103, "SF":  96,  "LAD": 100, "BOS": 104,
    "COL": 115, "MIN":  97, "HOU":  99, "ATL": 100,
    "NYM": 100, "CHC": 101, "PHI": 102, "SD":   94,
    "SEA":  95, "CLE":  98, "MIL":  99, "STL":  98,
    "TB":   97, "TOR": 100, "BAL": 102, "CIN": 104,
    "DET":  99, "KC":  100, "OAK":  97, "TEX": 101,
    "WSH":  99, "MIA":  95, "PIT":  99, "ARI": 100,
    "LAA": 100, "CWS":  99,
}

HAND_AVERAGES = {
    "R": {"avg_exit_velo": 88.2, "avg_launch_angle": 11.8, "hard_hit_pct": 0.374,
          "barrel_pct": 0.082, "k_pct": 0.228, "bb_pct": 0.082,
          "woba": 0.312, "ba": 0.248},
    "L": {"avg_exit_velo": 88.8, "avg_launch_angle": 12.4, "hard_hit_pct": 0.381,
          "barrel_pct": 0.088, "k_pct": 0.212, "bb_pct": 0.091,
          "woba": 0.323, "ba": 0.256},
    "S": {"avg_exit_velo": 88.5, "avg_launch_angle": 12.1, "hard_hit_pct": 0.377,
          "barrel_pct": 0.085, "k_pct": 0.220, "bb_pct": 0.086,
          "woba": 0.317, "ba": 0.252},
}

PITCHER_AVERAGES = {
    "avg_fastball_velo": 93.5, "whiff_pct": 0.25,
    "zone_pct": 0.47, "k_pct": 0.22,
    "bb_pct": 0.085, "hr_per_9": 1.3,
}


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def get_batter_rolling(player_name: str) -> dict:
    # Strip accents for database matching
    import unicodedata
    player_name = "".join(
        c for c in unicodedata.normalize("NFD", player_name)
        if unicodedata.category(c) != "Mn"
    )
    try:
        query = text("""
            SELECT brf.*
            FROM batter_rolling_features brf
            JOIN players p ON p.player_id = brf.player_id
            WHERE LOWER(p.full_name) LIKE :name
              AND brf.window_games = 5
            ORDER BY brf.as_of_date DESC
            LIMIT 1
        """)
        with engine.connect() as conn:
            row = pd.read_sql(query, conn,
                              params={"name": f"%{player_name.lower()}%"})
        if not row.empty:
            result = row.iloc[0].to_dict()
            # Blend in season barrel% and xSLG from Savant for better HR differentiation
            try:
                season_query = text("""
                    SELECT s.barrel_pct, s.xslg, s.avg_exit_velo, s.hard_hit_pct
                    FROM savant_batter_season s
                    JOIN players p ON p.player_id = s.player_id
                    WHERE LOWER(p.full_name) LIKE :name
                      AND s.season = 2026
                    LIMIT 1
                """)
                with engine.connect() as conn:
                    season = pd.read_sql(season_query, conn,
                                        params={"name": f"%{player_name.lower()}%"})
                if not season.empty:
                    s = season.iloc[0]
                    # Blend 60% rolling + 40% season for stability
                    if s["barrel_pct"] and float(s["barrel_pct"]) > 0:
                        result["barrel_pct"] = (
                            float(result.get("barrel_pct", 0.08) or 0.08) * 0.6 +
                            float(s["barrel_pct"]) * 0.4
                        )
                    if s["xslg"] and float(s["xslg"]) > 0:
                        result["xslg"] = float(s["xslg"])
                    if s["avg_exit_velo"] and float(s["avg_exit_velo"]) > 0:
                        result["avg_exit_velo"] = (
                            float(result.get("avg_exit_velo", 88.5) or 88.5) * 0.6 +
                            float(s["avg_exit_velo"]) * 0.4
                        )
            except Exception:
                pass
            return result
    except Exception as exc:
        logger.debug(f"DB lookup failed for {player_name}: {exc}")
    return {}


def get_pitcher_rolling(pitcher_name: str) -> dict:
    """
    Get rolling features for a pitcher, blended with season stats
    when available for a more accurate picture.
    """
    try:
        # First get rolling features
        query = text("""
            SELECT prf.*
            FROM pitcher_rolling_features prf
            JOIN players p ON p.player_id = prf.player_id
            WHERE LOWER(p.full_name) LIKE :name
              AND prf.window_games = 5
            ORDER BY prf.as_of_date DESC
            LIMIT 1
        """)
        with engine.connect() as conn:
            row = pd.read_sql(query, conn,
                              params={"name": f"%{pitcher_name.lower()}%"})

        if row.empty:
            return {}

        rolling = row.iloc[0].to_dict()
        player_id = rolling.get("player_id")

        # Try to blend with season stats
        if player_id:
            season_query = text("""
                SELECT k_pct, bb_pct, whiff_pct, xera, xwoba,
                       hard_hit_pct_allowed, avg_exit_velo_allowed
                FROM savant_pitcher_season
                WHERE player_id = :pid AND season = 2025
            """)
            with engine.connect() as conn:
                season = pd.read_sql(season_query, conn,
                                     params={"pid": int(player_id)})

            if not season.empty:
                s = season.iloc[0]
                # Blend: 40% rolling (recent form) + 60% season (true talent)
                rolling["k_pct"]    = (rolling.get("k_pct", 0.22) * 0.4 +
                                       float(s["k_pct"] or 0.22) * 0.6)
                rolling["bb_pct"]   = (rolling.get("bb_pct", 0.085) * 0.4 +
                                       float(s["bb_pct"] or 0.085) * 0.6)
                rolling["whiff_pct"]= (rolling.get("whiff_pct", 0.25) * 0.4 +
                                       float(s["whiff_pct"] or 0.25) * 0.6)
                rolling["xera"]     = float(s["xera"] or 4.0)
                rolling["xwoba_allowed"] = float(s["xwoba"] or 0.315)
                rolling["hard_hit_pct_allowed"] = float(
                    s["hard_hit_pct_allowed"] or 0.38)

        return rolling

    except Exception as exc:
        logger.debug(f"DB lookup failed for {pitcher_name}: {exc}")
    return {}


def load_model() -> PlayerPropsModel:
    model_dir = os.getenv("MODEL_DIR", "./models")
    files     = sorted(glob.glob(f"{model_dir}/props_*.pkl"), reverse=True)
    if files:
        return PlayerPropsModel.load(files[0])
    return PlayerPropsModel()


def get_weather_lean(home_team: str) -> str:
    try:
        with open("cache/weather_today.json", encoding="utf-8") as f:
            weather = json.load(f)
        w = weather.get(home_team, {})
        return w.get("betting_lean", "")
    except Exception:
        return ""


def get_umpire_k_factor(home_team: str) -> float:
    try:
        with open("cache/umpires_today.json", encoding="utf-8") as f:
            umpires = json.load(f)
        for gk, u in umpires.items():
            if u.get("home_team") == home_team:
                return float(u.get("k_factor", 1.0))
    except Exception:
        pass
    return 1.0


# ---------------------------------------------------------------------------
# Core prediction for one side of a matchup
# ---------------------------------------------------------------------------

def predict_one_side(
    pitcher_name: str,
    pitcher_team: str,
    batting_team: str,
    pitcher_hand: str,
    lineup: list[dict],
    model: PlayerPropsModel,
    show_pitcher_card: bool = True,
) -> list[dict]:
    park_factor  = PARK_FACTORS.get(pitcher_team, 100)
    weather_lean = get_weather_lean(pitcher_team)
    ump_k_factor = get_umpire_k_factor(pitcher_team)
    p_feat       = get_pitcher_rolling(pitcher_name) or PITCHER_AVERAGES.copy()

    if show_pitcher_card:
        p_context = {
            "park_factor": park_factor, "is_home": True,
            "days_rest": 4, "opp_k_pct": 0.22,
            "p_season_era":  p_feat.get("xera", 4.0),
            "p_season_fip":  p_feat.get("xera", 4.0),
            "p_season_k_9":  p_feat.get("k_pct", 0.22) * 27,
        }
        pitcher_props = model.predict_pitcher(p_feat, p_context)
        print_pitcher_props(
            pitcher_name, batting_team,
            pitcher_props, weather_lean, ump_k_factor,
        )

    print(f"\n  {'='*62}")
    print(f"  🏏  {batting_team} BATTING vs {pitcher_name}")
    print(f"  {'='*62}")

    all_props = []

    for batter in lineup:
        name  = batter.get("name", "Unknown")
        hand  = batter.get("bats", "R")
        order = batter.get("order", 5)
        pos   = batter.get("pos", "")

        b_feat  = get_batter_rolling(name) or \
                  HAND_AVERAGES.get(hand, HAND_AVERAGES["R"]).copy()
        context = {
            "park_factor":       park_factor,
            "batter_hand":       hand,
            "pitcher_hand":      pitcher_hand,
            "is_home":           batting_team == pitcher_team,
            "batting_order":     order,
            "h2h_pa":            0,
            "h2h_ba":            0.250,
            "b_season_woba":     b_feat.get("woba", 0.315),
            "b_season_wrc_plus": 100.0,
            "p_season_era":      p_feat.get("xera", 4.0),
            "p_season_k_pct":    p_feat.get("k_pct", 0.22),
        }

        props = model.predict_batter(b_feat, p_feat, context)
        print_batter_props(
            f"{order}. {name} ({pos})",
            pitcher_name, props,
            weather_lean if order == 1 else "",
            ump_k_factor,
        )

        for target, prob in props.items():
            if target == "expected_tb" or not isinstance(prob, float):
                continue
            if target in BATTER_PROP_TARGETS:
                info = BATTER_PROP_TARGETS[target]
                all_props.append({
                    "player":  name,
                    "team":    batting_team,
                    "prop":    info["desc"],
                    "market":  info["market"],
                    "line":    info["line"],
                    "prob":    min(prob, 0.99),
                    "order":   order,
                })

    return all_props


def print_top_bets(all_props: list[dict], top_n: int = 5) -> None:
    sorted_props = sorted(all_props, key=lambda x: x["prob"], reverse=True)
    print(f"\n{'='*62}")
    print(f"  🏆  TOP {top_n} MOST LIKELY OUTCOMES TO BET")
    print(f"{'='*62}")
    print(f"  {'#':<3} {'Player':<22} {'Prop':<22} {'Prob':>6}  {'Lean'}")
    print(f"  {'-'*60}")
    for i, p in enumerate(sorted_props[:top_n], 1):
        prob = p["prob"]
        icon = "🔥" if prob > 0.65 else ("✅" if prob > 0.50 else "📊")
        print(
            f"  {i:<3} {p['player']:<22} {p['prop']:<22} "
            f"{prob*100:5.1f}%  {icon}"
        )
    print()


def print_top_hit_candidates(all_props: list[dict], top_n: int = 3) -> None:
    hits = [p for p in all_props if p["market"] == "batter_hits"]
    hits.sort(key=lambda x: x["prob"], reverse=True)
    print(f"\n  🎯  TOP {top_n} MOST LIKELY TO GET A HIT:")
    for i, p in enumerate(hits[:top_n], 1):
        bar = "█" * int(p["prob"] * 20)
        print(f"  {i}. {p['player']:<22} {p['prob']*100:.1f}%  {bar}")
    print()


# ---------------------------------------------------------------------------
# Cross-game leaderboards
# ---------------------------------------------------------------------------

def _build_all_batter_props() -> list[dict]:
    """Shared helper — scan every confirmed lineup and return all prop predictions."""
    try:
        from daily_lineup import TODAYS_LINEUPS, TODAYS_STARTERS
    except ImportError:
        print("Run fetch_lineups.py first!")
        return []

    from fetch_lineups import get_todays_games, parse_game

    model     = load_model()
    results   = []
    games_raw = get_todays_games()
    games     = [parse_game(g) for g in games_raw if parse_game(g)]

    for game in games:
        home = game["home"]
        away = game["away"]
        for batting_team, pitching_team in [(away, home), (home, away)]:
            lineup  = TODAYS_LINEUPS.get(batting_team, [])
            if not lineup:
                continue
            sp      = TODAYS_STARTERS.get(pitching_team, {})
            sp_name = sp.get("name", "TBD")
            sp_hand = sp.get("hand", "R")
            p_feat  = get_pitcher_rolling(sp_name) or PITCHER_AVERAGES.copy()
            park_f  = PARK_FACTORS.get(home, 100)
            ump_k   = get_umpire_k_factor(home)
            weather = get_weather_lean(home)

            for batter in lineup:
                name   = batter.get("name", "")
                hand   = batter.get("bats", "R")
                b_feat = get_batter_rolling(name) or \
                         HAND_AVERAGES.get(hand, HAND_AVERAGES["R"]).copy()
                ctx = {
                    "park_factor":       park_f,
                    "batter_hand":       hand,
                    "pitcher_hand":      sp_hand,
                    "is_home":           batting_team == home,
                    "batting_order":     batter.get("order", 5),
                    "h2h_pa":            0,
                    "h2h_ba":            0.250,
                    "b_season_woba":     b_feat.get("woba", 0.315),
                    "b_season_wrc_plus": 100.0,
                    "p_season_era":      p_feat.get("xera", 4.0),
                    "p_season_k_pct":    p_feat.get("k_pct", 0.22),
                }
                props = model.predict_batter(b_feat, p_feat, ctx)
                results.append({
                    "player":   name,
                    "team":     batting_team,
                    "vs":       sp_name,
                    "game":     f"{away} @ {home}",
                    "hit_prob": props.get("hit_1plus",     0.28),
                    "tb2_prob": props.get("total_bases_2", 0.21),
                    "hr_prob":  min(props.get("hr_1plus", 0.055) * 1.8, 0.35),
                    "k_prob":   props.get("k_1plus",       0.31) * ump_k,
                    "bb_prob":  props.get("walk_1plus",    0.125),
                    "weather":  weather,
                })

    return results


def show_all_hit_candidates() -> None:
    """Rank all batters by hit probability across all games today."""
    all_batters = _build_all_batter_props()
    if not all_batters:
        return

    all_batters.sort(key=lambda x: x["hit_prob"], reverse=True)

    print(f"\n{'='*72}")
    print(f"  🎯  MOST LIKELY TO GET A BASE HIT TODAY  —  {date.today()}")
    print(f"{'='*72}")
    print(f"  {'#':<4} {'Player':<22} {'Team':<5} {'vs':<22} "
          f"{'Hit%':>6}  {'2+TB%':>6}  {'HR%':>5}")
    print(f"  {'-'*70}")

    for i, p in enumerate(all_batters[:20], 1):
        icon = "🔥" if p["hit_prob"] > 0.40 else \
               ("✅" if p["hit_prob"] > 0.35 else "📊")
        print(
            f"  {i:<4} {p['player']:<22} {p['team']:<5} "
            f"{p['vs']:<22} "
            f"{p['hit_prob']*100:5.1f}%  "
            f"{p['tb2_prob']*100:5.1f}%  "
            f"{p['hr_prob']*100:4.1f}%  {icon}"
        )

    print(f"\n  Showing top 20 of {len(all_batters)} confirmed lineup batters")
    print(f"  Hit% = P(1+ hits)  2+TB% = P(2+ total bases)  HR% = P(1+ HR)\n")


def show_prop_leaderboard(prop_key: str, title: str, pct_label: str,
                          icon: str = "📊", threshold: float = 0.20,
                          secondary: str = None) -> None:
    """Generic leaderboard for any batter prop."""
    all_batters = _build_all_batter_props()
    if not all_batters:
        return

    all_batters.sort(key=lambda x: x.get(prop_key, 0), reverse=True)
    top = [b for b in all_batters if b.get(prop_key, 0) >= threshold]
    if not top:
        top = all_batters[:15]

    print(f"\n{'='*65}")
    print(f"  {icon}  TOP {title} CANDIDATES TODAY  —  {date.today()}")
    print(f"{'='*65}")
    print(f"  {'#':<4} {'Player':<22} {'Team':<5} {'vs':<22} "
          f"{pct_label:>7}  {'Hit%':>6}")
    print(f"  {'-'*70}")

    for i, b in enumerate(top[:15], 1):
        val  = b.get(prop_key, 0) * 100
        hit  = b.get("hit_prob", 0) * 100
        sec  = f"  {b.get(secondary,0):.2f}TB" if secondary else ""
        print(f"  {i:<4} {b.get('player',''):<22} {b.get('team',''):<5} "
              f"{b.get('vs',''):<22} {val:>6.1f}%  {hit:>5.1f}%{sec}")

    confirmed = sum(1 for b in all_batters if b.get("lineup_confirmed"))
    print(f"\n  Showing top {min(15,len(top))} of {len(all_batters)} batters")
    print(f"{'='*65}\n")


def show_pitcher_k_leaderboard() -> None:
    """Rank today's starting pitchers by projected strikeouts."""
    try:
        from daily_lineup import TODAYS_STARTERS, TODAYS_LINEUPS, GAME_TIMES
    except ImportError:
        print("Run fetch_lineups.py first!")
        return

    model = load_model()

    # Build opponent lookup from starters — each team's opponent is whoever
    # shares the same game time
    game_times = {}
    try:
        game_times = GAME_TIMES
    except Exception:
        pass

    # Simple opponent lookup: pair teams that play each other
    # by finding which teams share game slots
    all_teams  = list(TODAYS_STARTERS.keys())
    opp_lookup = {}
    # Use game_times to pair opponents
    time_to_teams = {}
    for team, gt in game_times.items():
        if gt not in time_to_teams:
            time_to_teams[gt] = []
        time_to_teams[gt].append(team)
    for gt, teams in time_to_teams.items():
        if len(teams) == 2:
            opp_lookup[teams[0]] = teams[1]
            opp_lookup[teams[1]] = teams[0]

    pitcher_props = []
    for team, sp in TODAYS_STARTERS.items():
        sp_name = sp.get("name", "TBD")
        if sp_name == "TBD":
            continue

        p_feat = get_pitcher_rolling(sp_name) or PITCHER_AVERAGES.copy()
        opp    = opp_lookup.get(team, "???")

        ctx = {
            "park_factor":   PARK_FACTORS.get(team, 100),
            "is_home":       True,
            "days_rest":     4,
            "opp_k_pct":     0.22,
            "p_season_era":  p_feat.get("xera", 4.0),
            "p_season_fip":  p_feat.get("xera", 4.0),
            "p_season_k_9":  p_feat.get("k_pct", 0.22) * 27,
        }
        try:
            props = model.predict_pitcher(p_feat, ctx)
            pitcher_props.append({
                "pitcher": sp_name,
                "team":    team,
                "opp":     opp,
                "exp_k":   props.get("expected_k", 0),
                "k4":      props.get("k_4plus", 0),
                "k5":      props.get("k_5plus", 0),
                "k6":      props.get("k_6plus", 0),
                "k7":      props.get("k_7plus", 0),
                "k8":      props.get("k_8plus", 0),
            })
        except Exception:
            pass

    pitcher_props.sort(key=lambda x: x["exp_k"], reverse=True)

    print(f"\n{'='*72}")
    print(f"  ⚡  PITCHER K LEADERBOARD TODAY  —  {date.today()}")
    print(f"{'='*72}")
    print(f"  {'#':<4} {'Pitcher':<22} {'Team':<5} {'vs':<5} "
          f"{'ExpK':>5}  {'4K+':>5}  {'5K+':>5}  {'6K+':>5}  {'7K+':>5}")
    print(f"  {'-'*72}")

    for i, p in enumerate(pitcher_props[:15], 1):
        print(f"  {i:<4} {p['pitcher']:<22} {p['team']:<5} {p['opp']:<5} "
              f"{p['exp_k']:>5.1f}  "
              f"{p['k4']*100:>4.0f}%  "
              f"{p['k5']*100:>4.0f}%  "
              f"{p['k6']*100:>4.0f}%  "
              f"{p['k7']*100:>4.0f}%")

    print(f"\n{'='*72}\n")


def show_all_hr_candidates() -> None:
    """Rank all batters by HR probability across all games today."""
    all_batters = _build_all_batter_props()
    if not all_batters:
        return

    all_batters.sort(key=lambda x: x["hr_prob"], reverse=True)

    print(f"\n{'='*65}")
    print(f"  💣  MOST LIKELY TO HIT A HOME RUN TODAY  —  {date.today()}")
    print(f"{'='*65}")
    print(f"  {'#':<4} {'Player':<22} {'Team':<5} {'vs':<22} "
          f"{'HR%':>5}  {'Hit%':>6}  {'2+TB%':>6}")
    print(f"  {'-'*65}")

    for i, p in enumerate(all_batters[:15], 1):
        icon = "🔥" if p["hr_prob"] > 0.10 else \
               ("💣" if p["hr_prob"] > 0.07 else "📊")
        print(
            f"  {i:<4} {p['player']:<22} {p['team']:<5} "
            f"{p['vs']:<22} "
            f"{p['hr_prob']*100:4.1f}%  "
            f"{p['hit_prob']*100:5.1f}%  "
            f"{p['tb2_prob']*100:5.1f}%  {icon}"
        )

    print(f"\n  Showing top 15 of {len(all_batters)} confirmed lineup batters\n")


# ---------------------------------------------------------------------------
# Full matchup (both sides)
# ---------------------------------------------------------------------------

def predict_full_game(
    home_team: str,
    away_team: str,
    home_pitcher: str,
    away_pitcher: str,
    home_pitcher_hand: str = "R",
    away_pitcher_hand: str = "R",
    home_lineup: Optional[list] = None,
    away_lineup: Optional[list] = None,
) -> None:
    model        = load_model()
    park_factor  = PARK_FACTORS.get(home_team, 100)
    weather_lean = get_weather_lean(home_team)
    ump_k_factor = get_umpire_k_factor(home_team)
    today        = date.today().isoformat()

    print(f"\n{'='*62}")
    print(f"  ⚾  FULL GAME PREVIEW: {away_team} @ {home_team}")
    print(f"  📅  {today}  |  Park factor: {park_factor}")
    if weather_lean:
        print(f"  🌤️   Weather: {weather_lean}")
    if ump_k_factor != 1.0:
        k_label = "Tight zone +Ks" if ump_k_factor > 1.0 else "Wide zone -Ks"
        print(f"  ⚖️   Umpire: {ump_k_factor:.2f}x ({k_label})")
    print(f"{'='*62}")

    all_props = []

    print(f"\n\n  ── AWAY: {away_team} batting vs {home_pitcher} ──")
    away_props = predict_one_side(
        home_pitcher, home_team, away_team,
        home_pitcher_hand, away_lineup or [],
        model, show_pitcher_card=True,
    )
    all_props.extend(away_props)

    print(f"\n\n  ── HOME: {home_team} batting vs {away_pitcher} ──")
    home_props = predict_one_side(
        away_pitcher, away_team, home_team,
        away_pitcher_hand, home_lineup or [],
        model, show_pitcher_card=True,
    )
    all_props.extend(home_props)

    print_top_bets(all_props, top_n=5)
    print_top_hit_candidates(all_props, top_n=3)


# ---------------------------------------------------------------------------
# Interactive game selector
# ---------------------------------------------------------------------------

def interactive_selector() -> None:
    try:
        from daily_lineup import TODAYS_LINEUPS, TODAYS_STARTERS, GAME_TIMES
    except ImportError:
        try:
            from daily_lineup import TODAYS_LINEUPS, TODAYS_STARTERS
            GAME_TIMES = {}
        except ImportError:
            print("daily_lineup.py not found -- run fetch_lineups.py first!")
            return

    # Teams with no pitcher posted today
    no_pitcher_teams = set()
    try:
        from daily_lineup import NO_PITCHER_TEAMS
        no_pitcher_teams = set(NO_PITCHER_TEAMS)
    except ImportError:
        pass

    carried_teams = set()
    try:
        from daily_lineup import CARRIED_LINEUP_TEAMS
        carried_teams = set(CARRIED_LINEUP_TEAMS)
    except ImportError:
        pass

    from fetch_lineups import get_todays_games, parse_game

    games_raw = get_todays_games()
    games     = [parse_game(g) for g in games_raw if parse_game(g)]

    if not games:
        print("No games found from MLB API.")
        return

    model = load_model()

    print(f"\n{'='*72}")
    print(f"  ⚾  TODAY'S MLB GAMES — {date.today()}")
    print(f"{'='*72}")
    print(f"  {'#':<3} {'Matchup':<30} {'Pitchers':<35} {'Top Hit Candidates'}")
    print(f"  {'-'*70}")

    game_list = []
    for i, game in enumerate(games, 1):
        home = game.get("home", "")
        away = game.get("away", "")

        home_starter = TODAYS_STARTERS.get(home, {})
        away_starter = TODAYS_STARTERS.get(away, {})
        hp_name      = home_starter.get("name", "TBD")
        ap_name      = away_starter.get("name", "TBD")
        home_lineup  = TODAYS_LINEUPS.get(home, [])
        away_lineup  = TODAYS_LINEUPS.get(away, [])

        # Quick hit preview for away batters vs home pitcher
        hit_candidates = []
        p_feat   = get_pitcher_rolling(hp_name) or PITCHER_AVERAGES.copy()
        park_f   = PARK_FACTORS.get(home, 100)

        for batter in away_lineup[:6]:
            name   = batter.get("name", "")
            hand   = batter.get("bats", "R")
            b_feat = get_batter_rolling(name) or \
                     HAND_AVERAGES.get(hand, HAND_AVERAGES["R"]).copy()
            ctx = {
                "park_factor": park_f, "batter_hand": hand,
                "pitcher_hand": home_starter.get("hand", "R"),
                "is_home": False, "batting_order": batter.get("order", 5),
                "h2h_pa": 0, "h2h_ba": 0.250,
                "b_season_woba": b_feat.get("woba", 0.315),
                "b_season_wrc_plus": 100,
                "p_season_era":  p_feat.get("xera", 4.0),
                "p_season_k_pct": p_feat.get("k_pct", 0.22),
            }
            props = model.predict_batter(b_feat, p_feat, ctx)
            hit_candidates.append((name, props.get("hit_1plus", 0.28)))

        hit_candidates.sort(key=lambda x: x[1], reverse=True)
        top3 = ", ".join(
            f"{n.split()[-1]}({p*100:.0f}%)"
            for n, p in hit_candidates[:3]
        )

        matchup  = f"{away} @ {home}"
        pitchers = f"{ap_name} vs {hp_name}"

        # Lineup source indicators
        home_src = "(yest)" if home in carried_teams else ""
        away_src = "(yest)" if away in carried_teams else ""
        no_props = ""
        if home in no_pitcher_teams or away in no_pitcher_teams:
            no_props = " [no props]"
            top3 = "No pitcher posted"

        print(f"  {i:<3} {matchup:<30} {pitchers:<35} {top3}{no_props}")
        if home_src or away_src:
            note = []
            if away_src: note.append(f"{away}{away_src}")
            if home_src: note.append(f"{home}{home_src}")
            print(f"       Lineup: {', '.join(note)}")

        game_list.append({
            "home": home, "away": away,
            "home_pitcher":      hp_name,
            "away_pitcher":      ap_name,
            "home_pitcher_hand": home_starter.get("hand", "R"),
            "away_pitcher_hand": away_starter.get("hand", "R"),
            "home_lineup":       home_lineup,
            "away_lineup":       away_lineup,
        })

    print(f"\n  {'='*72}")
    print(f"  Enter game numbers (e.g. 1 3 5) | Enter = all games")
    print(f"  'h' = hit leaderboard | 'hr' = HR leaderboard | 'q' = quit")
    print(f"  {'='*72}")

    selection = input("\n  Your selection: ").strip().lower()

    if selection == "q":
        return
    if selection == "h":
        show_all_hit_candidates()
        return
    if selection == "hr":
        show_all_hr_candidates()
        return
    if selection == "" or selection == "all":
        selected = list(range(len(game_list)))
    else:
        try:
            selected = [int(x) - 1 for x in selection.split()
                        if x.isdigit() and 1 <= int(x) <= len(game_list)]
        except ValueError:
            print("Invalid selection")
            return

    for idx in selected:
        g = game_list[idx]
        predict_full_game(
            home_team         = g["home"],
            away_team         = g["away"],
            home_pitcher      = g["home_pitcher"],
            away_pitcher      = g["away_pitcher"],
            home_pitcher_hand = g["home_pitcher_hand"],
            away_pitcher_hand = g["away_pitcher_hand"],
            home_lineup       = g["home_lineup"],
            away_lineup       = g["away_lineup"],
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive MLB matchup predictor with prop betting lines"
    )
    parser.add_argument("--pitcher",      type=str)
    parser.add_argument("--pitcher-team", type=str)
    parser.add_argument("--batting-team", type=str)
    parser.add_argument("--all",  action="store_true",
                        help="Analyze all games today")
    parser.add_argument("--hits",  action="store_true",
                        help="Top hit candidates today")
    parser.add_argument("--hr",    action="store_true",
                        help="Top HR candidates today")
    parser.add_argument("--k",     action="store_true",
                        help="Top strikeout picks (batters most likely to K)")
    parser.add_argument("--ks",    action="store_true",
                        help="Pitcher K leaderboard (most Ks projected)")
    parser.add_argument("--tb",    action="store_true",
                        help="Top total bases candidates (2+TB)")
    parser.add_argument("--bb",    action="store_true",
                        help="Top walk candidates today")
    parser.add_argument("--sb",    action="store_true",
                        help="Top stolen base candidates today")
    parser.add_argument("--multi", action="store_true",
                        help="Top multi-hit candidates (2+ hits)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.hits:
        show_all_hit_candidates()
        return

    if args.hr:
        show_all_hr_candidates()
        return

    if args.k:
        show_prop_leaderboard("k_prob", "STRIKEOUT", "K%",
                              icon="🔥", threshold=0.30)
        return

    if args.ks:
        show_pitcher_k_leaderboard()
        return

    if args.tb:
        show_prop_leaderboard("tb2_prob", "2+ TOTAL BASES", "2+TB%",
                              icon="💥", threshold=0.25)
        return

    if args.bb:
        show_prop_leaderboard("bb_prob", "WALK", "BB%",
                              icon="👀", threshold=0.10)
        return

    if args.sb:
        show_prop_leaderboard("sb_prob", "STOLEN BASE", "SB%",
                              icon="💨", threshold=0.05)
        return

    if args.multi:
        show_prop_leaderboard("hit_prob", "MULTI-HIT (2+)", "Hit%",
                              icon="🔥", threshold=0.40)
        return

    if args.pitcher and args.pitcher_team and args.batting_team:
        model = load_model()
        try:
            from daily_lineup import TODAYS_LINEUPS, TODAYS_STARTERS
            lineup       = TODAYS_LINEUPS.get(args.batting_team, [])
            starter      = TODAYS_STARTERS.get(args.pitcher_team, {})
            pitcher_hand = starter.get("hand", "R")
        except ImportError:
            lineup       = []
            pitcher_hand = "R"

        if not lineup:
            from fetch_lineups import FALLBACK_ROSTERS
            lineup = FALLBACK_ROSTERS.get(
                args.batting_team, {}
            ).get("lineup", [])

        all_props = predict_one_side(
            args.pitcher, args.pitcher_team,
            args.batting_team, pitcher_hand,
            lineup, model,
        )
        print_top_bets(all_props, top_n=5)
        print_top_hit_candidates(all_props, top_n=3)
        return

    if args.all:
        try:
            from daily_lineup import TODAYS_LINEUPS, TODAYS_STARTERS
            from fetch_lineups import get_todays_games, parse_game
            games_raw = get_todays_games()
            games     = [parse_game(g) for g in games_raw if parse_game(g)]
            for game in games:
                home = game["home"]
                away = game["away"]
                predict_full_game(
                    home, away,
                    TODAYS_STARTERS.get(home, {}).get("name", "TBD"),
                    TODAYS_STARTERS.get(away, {}).get("name", "TBD"),
                    TODAYS_STARTERS.get(home, {}).get("hand", "R"),
                    TODAYS_STARTERS.get(away, {}).get("hand", "R"),
                    TODAYS_LINEUPS.get(home, []),
                    TODAYS_LINEUPS.get(away, []),
                )
        except ImportError:
            print("Run fetch_lineups.py first!")
        return

    interactive_selector()


if __name__ == "__main__":
    main()