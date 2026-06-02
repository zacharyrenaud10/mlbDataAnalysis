# -*- coding: utf-8 -*-
"""
parlay_builder.py
-----------------
Analyzes today's FanDuel moneylines using actual Statcast
rolling data, pitcher metrics, weather and umpire factors.

Usage:
    python parlay_builder.py
    python parlay_builder.py --legs 3
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import date

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import text

load_dotenv()

from mlb_analytics.db import engine

def load_team_ratings() -> dict:
    """Load Elo and Pythagorean ratings from cache."""
    try:
        with open("cache/team_ratings.json", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("teams", {})
    except Exception:
        return {}

TEAM_RATINGS = load_team_ratings()

# ---------------------------------------------------------------------------
# Team name mapping (Odds API full names -> our abbreviations)
# ---------------------------------------------------------------------------
TEAM_NAME_TO_ABB = {
    "New York Mets":         "NYM", "Pittsburgh Pirates":    "PIT",
    "Milwaukee Brewers":     "MIL", "Chicago White Sox":     "CWS",
    "Chicago Cubs":          "CHC", "Washington Nationals":  "WSH",
    "Baltimore Orioles":     "BAL", "Minnesota Twins":       "MIN",
    "Cincinnati Reds":       "CIN", "Boston Red Sox":        "BOS",
    "San Diego Padres":      "SD",  "Detroit Tigers":        "DET",
    "Houston Astros":        "HOU", "Los Angeles Angels":    "LAA",
    "Philadelphia Phillies": "PHI", "Texas Rangers":         "TEX",
    "St. Louis Cardinals":   "STL", "Tampa Bay Rays":        "TB",
    "Los Angeles Dodgers":   "LAD", "Arizona Diamondbacks":  "ARI",
    "Seattle Mariners":      "SEA", "Cleveland Guardians":   "CLE",
    "Toronto Blue Jays":     "TOR", "Athletics":             "OAK",
    "Miami Marlins":         "MIA", "Colorado Rockies":      "COL",
    "Atlanta Braves":        "ATL", "Kansas City Royals":    "KC",
    "New York Yankees":      "NYY", "San Francisco Giants":  "SF",
}

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

TEAM_OFFENSE_FALLBACK = {
    "LAD": 88, "NYY": 82, "ATL": 78, "PHI": 77,
    "HOU": 76, "BAL": 75, "CLE": 74, "MIL": 73,
    "TEX": 72, "SD":  72, "BOS": 71, "SEA": 70,
    "NYM": 70, "ARI": 69, "MIN": 68, "SF":  67,
    "TOR": 66, "CIN": 65, "DET": 64, "TB":  63,
    "CHC": 62, "STL": 61, "PIT": 55, "MIA": 54,
    "KC":  53, "LAA": 52, "WSH": 51, "OAK": 50,
    "CWS": 42, "COL": 40,
}

PITCHER_ERA_FALLBACK = {
    "Paul Skenes":        160, "Tarik Skubal":       158,
    "Zack Wheeler":       145, "Yoshinobu Yamamoto": 143,
    "Logan Webb":         138, "Cristopher Sanchez": 135,
    "Garrett Crochet":    132, "Logan Gilbert":      130,
    "Freddy Peralta":     125, "Hunter Brown":       122,
    "Tanner Bibee":       120, "Joe Ryan":           118,
    "Jacob Misiorowski":  115, "Andrew Abbott":      114,
    "Matthew Boyd":       110, "Nick Pivetta":       108,
    "Zac Gallen":         118, "Max Fried":          135,
    "Trevor Rogers":      105, "Nathan Eovaldi":     108,
    "Drew Rasmussen":     125, "Matthew Liberatore": 105,
    "Cade Cavalli":       100, "Shane Smith":         95,
}

CONFIDENCE_LABELS = {
    (75, 100): ("LOCK",        "A+"),
    (68,  75): ("STRONG",      "A"),
    (62,  68): ("LEAN",        "B+"),
    (55,  62): ("SLIGHT EDGE", "B"),
    (50,  55): ("COIN FLIP",   "C"),
}


def get_confidence_label(pct: float) -> tuple[str, str]:
    for (low, high), (label, grade) in CONFIDENCE_LABELS.items():
        if low <= pct < high:
            return label, grade
    return "COIN FLIP", "C"


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def get_platoon_woba(player_id: int, pitcher_hand: str) -> float:
    """
    Get batter's wOBA split vs LHP or RHP from Savant season stats.
    Falls back to overall wOBA if no split data available.
    """
    try:
        # Savant doesn't directly give platoon splits in our current schema
        # but we can approximate using the overall xwOBA and a handedness adjustment
        # LHB vs LHP typically drops ~30pts, LHB vs RHP is normal
        # RHB vs RHP typically drops ~15pts, RHB vs LHP is normal
        # This is a simplified model until we add full platoon split ingestion
        query = text("""
            SELECT xwoba, k_pct
            FROM savant_batter_season
            WHERE player_id = :pid
            ORDER BY season DESC
            LIMIT 1
        """)
        with engine.connect() as conn:
            row = pd.read_sql(query, conn, params={"pid": player_id})
        if row.empty:
            return 0.315

        xwoba = float(row.iloc[0]["xwoba"] or 0.315)
        # Apply handedness adjustment
        # This is approximate — full platoon splits need separate ingestion
        return xwoba
    except Exception:
        return 0.315


def get_team_rolling_stats(team_abb: str, opp_pitcher_hand: str = "R") -> dict:
    """
    Get average rolling stats for a team using confirmed lineup names.
    Blends 5-game rolling stats with 2025 season xwOBA for stability.
    Applies platoon adjustments based on opposing pitcher handedness.
    """
    lineup = []
    try:
        from daily_lineup import TODAYS_LINEUPS
        lineup = TODAYS_LINEUPS.get(team_abb, [])
    except ImportError:
        pass

    if not lineup:
        return {}

    rolling_stats = []
    season_xwobas = []

    try:
        for batter in lineup[:9]:
            name = batter.get("name", "")
            if not name:
                continue
            last  = name.split()[-1].lower()
            first = name.split()[0].lower()

            # Rolling features
            query = text("""
                SELECT brf.avg_exit_velo, brf.hard_hit_pct,
                       brf.k_pct, brf.bb_pct, brf.woba,
                       p.player_id
                FROM batter_rolling_features brf
                JOIN players p ON p.player_id = brf.player_id
                WHERE LOWER(p.full_name) LIKE :last
                  AND LOWER(p.full_name) LIKE :first
                  AND brf.window_games = 5
                ORDER BY brf.as_of_date DESC
                LIMIT 1
            """)
            with engine.connect() as conn:
                row = pd.read_sql(query, conn, params={
                    "last":  f"%{last}%",
                    "first": f"%{first}%",
                })

            if not row.empty:
                r = row.iloc[0].to_dict()
                rolling_stats.append(r)
                player_id = r.get("player_id")

                # Season xwOBA for stability
                if player_id:
                    sq = text("""
                        SELECT xwoba, avg_exit_velo, hard_hit_pct, k_pct,
                               barrel_pct
                        FROM savant_batter_season
                        WHERE player_id = :pid
                        ORDER BY season DESC
                        LIMIT 1
                    """)
                    with engine.connect() as conn:
                        srow = pd.read_sql(sq, conn, params={"pid": int(player_id)})
                    if not srow.empty:
                        raw_xwoba = float(srow.iloc[0]["xwoba"] or 0.315)
                        # Platoon adjustment
                        batter_hand = batter.get("bats", "R")
                        if batter_hand == "L" and opp_pitcher_hand == "L":
                            raw_xwoba *= 0.91  # LHB vs LHP penalty
                        elif batter_hand == "R" and opp_pitcher_hand == "L":
                            raw_xwoba *= 1.04  # RHB vs LHP slight boost
                        elif batter_hand == "S":
                            raw_xwoba *= 1.01  # switch hitters slight advantage
                        season_xwobas.append(raw_xwoba)
                        # Store barrel% for later use
                        barrel = srow.iloc[0].get("barrel_pct", None)
                        if barrel is not None and not pd.isna(barrel):
                            rolling_stats[-1]["season_barrel_pct"] = float(barrel) / 100.0

        if len(rolling_stats) >= 3:
            # Blend rolling wOBA with season xwOBA (40/60)
            roll_woba   = sum(s.get("woba", 0.315) or 0.315 for s in rolling_stats) / len(rolling_stats)
            season_woba = sum(season_xwobas) / len(season_xwobas) if season_xwobas else roll_woba
            blended_woba = roll_woba * 0.4 + season_woba * 0.6

            return {
                "avg_ev":       sum(s.get("avg_exit_velo", 88.5) or 88.5 for s in rolling_stats) / len(rolling_stats),
                "hard_hit_pct": sum(s.get("hard_hit_pct",  0.38) or 0.38 for s in rolling_stats) / len(rolling_stats),
                "k_pct":        sum(s.get("k_pct",         0.22) or 0.22 for s in rolling_stats) / len(rolling_stats),
                "bb_pct":       sum(s.get("bb_pct",       0.085) or 0.085 for s in rolling_stats) / len(rolling_stats),
                "woba":         blended_woba,
                "n_players":    len(rolling_stats),
            }
    except Exception as exc:
        logger.debug(f"Team rolling stats failed for {team_abb}: {exc}")
    return {}


# Direct player ID lookups for players not in players table
PLAYER_ID_MAP = {
    "shohei ohtani":      660271,
    "ohtani":             660271,
    "yoshinobu yamamoto": 673540,
    "yamamoto":           673540,
}


def get_pitcher_rolling_stats(pitcher_name: str) -> dict:
    """Get rolling + blended season stats for a starting pitcher.
    Filters to starter-quality appearances only (batters_faced >= 60)
    to avoid polluting stats with relief appearances.
    Falls back to direct player_id lookup for known players not in players table.
    """
    try:
        # First try direct ID lookup for known players
        name_lower = pitcher_name.lower()
        direct_id  = None
        for key, pid in PLAYER_ID_MAP.items():
            if key in name_lower or name_lower.split()[0] in key:
                direct_id = pid
                break

        if direct_id:
            query = text("""
                SELECT prf.*
                FROM pitcher_rolling_features prf
                WHERE prf.player_id = :pid
                  AND prf.window_games = 5
                  AND prf.batters_faced >= 40
                ORDER BY prf.as_of_date DESC
                LIMIT 1
            """)
            with engine.connect() as conn:
                row = pd.read_sql(query, conn, params={"pid": direct_id})
        else:
            query = text("""
                SELECT prf.*
                FROM pitcher_rolling_features prf
                JOIN players p ON p.player_id = prf.player_id
                WHERE LOWER(p.full_name) LIKE :name
                  AND prf.window_games = 5
                  AND prf.batters_faced >= 60
                ORDER BY prf.as_of_date DESC
                LIMIT 1
            """)
            with engine.connect() as conn:
                row = pd.read_sql(query, conn,
                                  params={"name": f"%{pitcher_name.lower().split()[0]}%"})
        if row.empty:
            return {}

        rolling   = row.iloc[0].to_dict()
        player_id = rolling.get("player_id")

        if player_id:
            sq = text("""
                SELECT k_pct, bb_pct, whiff_pct, xera,
                       avg_exit_velo_allowed, hard_hit_pct_allowed,
                       barrel_pct_allowed, put_away_pct
                FROM savant_pitcher_season
                WHERE player_id = :pid
                ORDER BY season DESC
                LIMIT 1
            """)
            with engine.connect() as conn:
                season = pd.read_sql(sq, conn, params={"pid": int(player_id)})
            if not season.empty:
                s = season.iloc[0]
                rolling["k_pct"]     = (rolling.get("k_pct", 0.22) * 0.4 +
                                        float(s["k_pct"] or 0.22) * 0.6)
                rolling["whiff_pct"] = (rolling.get("whiff_pct", 0.25) * 0.4 +
                                        float(s["whiff_pct"] or 0.25) * 0.6)
                rolling["xera"]      = float(s["xera"] or 4.0)

        return rolling

    except Exception as exc:
        logger.debug(f"Pitcher stats failed for {pitcher_name}: {exc}")
    return {}


def get_weather_lean(home_team: str) -> dict:
    try:
        with open("cache/weather_today.json", encoding="utf-8") as f:
            weather = json.load(f)
        return weather.get(home_team, {})
    except Exception:
        return {}


def get_injury_alerts() -> dict:
    """Load today's injury alerts from cache."""
    try:
        with open("cache/injuries_today.json", encoding="utf-8") as f:
            data = json.load(f)
        # Build lookup: team -> list of alerts
        alerts = {}
        for a in data.get("alerts", []):
            team = a.get("team", "")
            if team not in alerts:
                alerts[team] = []
            alerts[team].append(a)
        return alerts
    except Exception:
        return {}


def get_umpire_factor(home_team: str) -> float:
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
# Win probability model
# ---------------------------------------------------------------------------

def calculate_win_probability(
    home_team: str,
    away_team: str,
    home_pitcher: str,
    away_pitcher: str,
) -> tuple[float, float, dict]:
    factors = {}

    # Get pitcher handedness for platoon adjustments
    home_sp_hand = "R"
    away_sp_hand = "R"
    try:
        from daily_lineup import TODAYS_STARTERS
        home_sp_hand = TODAYS_STARTERS.get(home_team, {}).get("hand", "R")
        away_sp_hand = TODAYS_STARTERS.get(away_team, {}).get("hand", "R")
    except ImportError:
        pass

    home_off = get_team_rolling_stats(home_team, opp_pitcher_hand=away_sp_hand)
    away_off = get_team_rolling_stats(away_team, opp_pitcher_hand=home_sp_hand)

    def offense_score(stats: dict, fallback_team: str) -> float:
        fallback = float(TEAM_OFFENSE_FALLBACK.get(fallback_team, 55))
        if stats and stats.get("n_players", 0) >= 3:
            ev   = float(stats.get("avg_ev",       88.5) or 88.5)
            hh   = float(stats.get("hard_hit_pct",  0.38) or 0.38)
            woba = float(stats.get("woba",          0.315) or 0.315)
            k    = float(stats.get("k_pct",         0.22) or 0.22)
            # Wider normalization range to avoid clipping
            score = (
                (ev   - 83) / (97 - 83) * 30 +
                (hh   - 0.20) / (0.65 - 0.20) * 30 +
                (woba - 0.25) / (0.42 - 0.25) * 30 +
                (1 - k / 0.38) * 10
            )
            statcast_score = max(30.0, min(90.0, score))
            # Early season: weight fallback heavily until enough 2026 data
            # As season progresses this will naturally improve
            # After ~4 weeks flip to 60% Statcast / 40% fallback
            from datetime import date as _date
            season_start = _date(2026, 3, 25)
            days_played  = (_date.today() - season_start).days
            # Ramp from 10% Statcast on day 1 to 70% Statcast by day 60
            statcast_weight = min(0.70, max(0.10, days_played / 60 * 0.70))
            fallback_weight = 1.0 - statcast_weight
            return statcast_score * statcast_weight + fallback * fallback_weight
        return fallback

    home_off_score = offense_score(home_off, home_team)
    away_off_score = offense_score(away_off, away_team)
    factors["home_offense"] = round(home_off_score, 1)
    factors["away_offense"] = round(away_off_score, 1)
    factors["home_data"]    = "Statcast" if home_off.get("n_players", 0) >= 3 else "fallback"
    factors["away_data"]    = "Statcast" if away_off.get("n_players", 0) >= 3 else "fallback"

    home_pit = get_pitcher_rolling_stats(home_pitcher)
    away_pit = get_pitcher_rolling_stats(away_pitcher)

    def pitcher_score(stats: dict, fallback_name: str) -> float:
        if stats:
            k_pct = float(stats.get("k_pct",    0.22) or 0.22)
            whiff = float(stats.get("whiff_pct", 0.25) or 0.25)
            xera  = float(stats.get("xera",      4.50) or 4.50)
            # Cap xERA at realistic bounds — anything over 6.0 is replacement level
            xera  = min(xera, 6.0) if xera > 0 else 4.50
            score = (
                (k_pct - 0.10) / (0.38 - 0.10) * 40 +
                (whiff - 0.15) / (0.40 - 0.15) * 35 +
                (6.0 - xera)   / (6.0  - 1.5)  * 25
            )
            raw = max(25.0, min(90.0, score))
            # Blend with ERA+ fallback to prevent wild swings from small samples
            era_plus  = PITCHER_ERA_FALLBACK.get(fallback_name, 100)
            fallback  = max(30.0, min(90.0, era_plus / 2))
            return raw * 0.65 + fallback * 0.35
        era_plus = PITCHER_ERA_FALLBACK.get(fallback_name, 100)
        return max(30.0, min(90.0, era_plus / 2))

    home_pit_score = pitcher_score(home_pit, home_pitcher)
    away_pit_score = pitcher_score(away_pit, away_pitcher)
    factors["home_pitcher_score"] = round(home_pit_score, 1)
    factors["away_pitcher_score"] = round(away_pit_score, 1)
    factors["home_pitcher_data"]  = "Statcast" if home_pit else "fallback"
    factors["away_pitcher_data"]  = "Statcast" if away_pit else "fallback"

    park     = PARK_FACTORS.get(home_team, 100)
    park_adj = (park - 100) / 100.0 * 2.0
    factors["park_factor"] = park

    # Bullpen fatigue — tired bullpen = opponent offense boosted
    try:
        from data_fetchers.bullpen_fatigue import load_fatigue_cache
        fatigue = load_fatigue_cache()
        home_fatigue = fatigue.get(home_team, {}).get("penalty", 0.0)
        away_fatigue = fatigue.get(away_team, {}).get("penalty", 0.0)
        # If home bullpen is tired, away offense gets a boost
        home_off_score = home_off_score - (away_fatigue * 15)
        away_off_score = away_off_score - (home_fatigue * 15)
        factors["home_bullpen"] = fatigue.get(home_team, {}).get("status", "UNKNOWN")
        factors["away_bullpen"] = fatigue.get(away_team, {}).get("status", "UNKNOWN")
    except Exception:
        factors["home_bullpen"] = "UNKNOWN"
        factors["away_bullpen"] = "UNKNOWN"

    weather      = get_weather_lean(home_team)
    weather_lean = weather.get("betting_lean", "")
    weather_adj  = 0.0
    if "UNDER" in weather_lean or "pitcher" in weather_lean.lower():
        weather_adj = -1.0
    elif "OVER" in weather_lean or "hitter" in weather_lean.lower():
        weather_adj = 1.0
    factors["weather"] = weather_lean or "No data"

    # Home team wins when their offense > away pitcher
    # and their pitcher dominates away offense
    # Score differential — how much better is each team's offense vs opposing pitcher?
    # Positive = home team has advantage, negative = away team has advantage
    home_bat_edge  = home_off_score - away_pit_score   # home bats vs away SP
    away_bat_edge  = away_off_score - home_pit_score   # away bats vs home SP
    net_advantage  = home_bat_edge - away_bat_edge     # net edge for home team

    # Scale: 10 point net advantage ~ 5% win probability boost
    skill_component   = net_advantage * 0.5

    # Home field reduced to 1.0 to fix home bias
    # Data shows model was 60% on home picks but only 35% on away picks
    # Real MLB home field advantage is ~54% which is only ~1pp edge
    home_field        = 1.0 + park_adj + weather_adj

    home_advantage    = skill_component + home_field

    raw_prob  = 50.0 + home_advantage
    home_prob = max(20.0, min(80.0, raw_prob)) / 100.0
    away_prob = 1.0 - home_prob

    factors["home_advantage_score"] = round(home_advantage, 2)
    factors["data_source"] = (
        "Statcast rolling data"
        if (home_off.get("n_players", 0) >= 3 or away_off.get("n_players", 0) >= 3)
        else "2025 team ratings fallback"
    )

    return home_prob, away_prob, factors


# ---------------------------------------------------------------------------
# Ensemble model — runs 4 perspectives and combines
# ---------------------------------------------------------------------------

def calculate_win_probability_ensemble(
    home_team: str,
    away_team: str,
    home_pitcher: str,
    away_pitcher: str,
) -> tuple[float, float, dict]:
    """
    Runs 4 model perspectives and combines into a final prediction.
    High agreement = high confidence. High variance = skip.

    Perspectives:
      A — Statcast heavy (exit velo, hard hit%, barrel proxy)
      B — Pitcher dominant (K%, whiff%, xERA weighted 60%)
      C — Recent form only (rolling 5 games, no season blend)
      D — Season stats only (full season averages)
    """
    results = []
    perspectives = []

    home_off  = get_team_rolling_stats(home_team)
    away_off  = get_team_rolling_stats(away_team)
    home_pit  = get_pitcher_rolling_stats(home_pitcher)
    away_pit  = get_pitcher_rolling_stats(away_pitcher)

    park      = PARK_FACTORS.get(home_team, 100)
    park_adj  = (park - 100) / 100.0 * 2.0
    weather   = get_weather_lean(home_team)
    w_lean    = weather.get("betting_lean", "")
    w_adj     = -1.0 if "UNDER" in w_lean or "pitcher" in w_lean.lower() else                  1.0 if "OVER"  in w_lean or "hitter"  in w_lean.lower() else 0.0

    fallback_home = float(TEAM_OFFENSE_FALLBACK.get(home_team, 55))
    fallback_away = float(TEAM_OFFENSE_FALLBACK.get(away_team, 55))

    def win_prob_from_scores(h_off, a_off, h_sp, a_sp):
        net   = (h_off - a_sp) - (a_off - h_sp)
        skill = net * 0.5
        # Reduced from 2.0 to 1.0 to fix home bias
        hf    = 1.0 + park_adj + w_adj
        raw   = 50.0 + skill + hf
        return max(20.0, min(80.0, raw)) / 100.0

    def pitcher_score_b(stats, fallback_name=""):
        era_plus = PITCHER_ERA_FALLBACK.get(fallback_name, 100)
        fallback = max(30.0, min(90.0, era_plus / 2))
        if not stats:
            return fallback
        k     = float(stats.get("k_pct",    0.22) or 0.22)
        whiff = float(stats.get("whiff_pct", 0.25) or 0.25)
        xera  = float(stats.get("xera",      4.50) or 4.50)
        xera  = min(xera, 6.0) if xera > 0 else 4.50
        s = ((k-0.10)/(0.38-0.10)*40 +
             (whiff-0.15)/(0.40-0.15)*35 +
             (6.0-xera)/(6.0-1.5)*25)
        raw = max(25.0, min(90.0, s))
        return raw * 0.65 + fallback * 0.35


    # --- Perspective A: Statcast heavy ---
    if home_off and away_off:
        h_ev   = float(home_off.get("avg_ev",       88.5) or 88.5)
        a_ev   = float(away_off.get("avg_ev",       88.5) or 88.5)
        h_hh   = float(home_off.get("hard_hit_pct",  0.38) or 0.38)
        a_hh   = float(away_off.get("hard_hit_pct",  0.38) or 0.38)
        h_woba = float(home_off.get("woba",          0.315) or 0.315)
        a_woba = float(away_off.get("woba",          0.315) or 0.315)
        h_k    = float(home_off.get("k_pct",         0.22) or 0.22)
        a_k    = float(away_off.get("k_pct",         0.22) or 0.22)

        def sc_score(ev, hh, woba, k):
            s = ((ev-83)/(97-83)*35 + (hh-0.20)/(0.65-0.20)*35 +
                 (woba-0.25)/(0.42-0.25)*20 + (1-k/0.38)*10)
            return max(30.0, min(90.0, s))

        h_off_a = sc_score(h_ev, h_hh, h_woba, h_k)
        a_off_a = sc_score(a_ev, a_hh, a_woba, a_k)
        h_sp_a  = pitcher_score_b(home_pit, home_pitcher)
        a_sp_a  = pitcher_score_b(away_pit, away_pitcher)
        p_a = win_prob_from_scores(h_off_a, a_off_a,
                                   max(30, min(90, h_sp_a)),
                                   max(30, min(90, a_sp_a)))
        perspectives.append(("Statcast", p_a))

    # --- Perspective B: Pitcher dominant ---

    h_sp_b = pitcher_score_b(home_pit, home_pitcher)
    a_sp_b = pitcher_score_b(away_pit, away_pitcher)
    h_off_b = fallback_home * 0.5 + (float(home_off.get("woba", 0.315) or 0.315) * 100 if home_off else 50) * 0.5
    a_off_b = fallback_away * 0.5 + (float(away_off.get("woba", 0.315) or 0.315) * 100 if away_off else 50) * 0.5
    p_b = win_prob_from_scores(
        max(30, min(90, h_off_b)),
        max(30, min(90, a_off_b)),
        h_sp_b, a_sp_b
    )
    perspectives.append(("Pitcher", p_b))

    # --- Perspective C: Recent form (rolling only, no season blend) ---
    def rolling_offense(stats, fallback):
        if not stats or stats.get("n_players", 0) < 3:
            return fallback
        woba = float(stats.get("woba", 0.315) or 0.315)
        ev   = float(stats.get("avg_ev", 88.5) or 88.5)
        s = ((ev-83)/(97-83)*40 + (woba-0.25)/(0.42-0.25)*60)
        return max(30.0, min(90.0, s))

    h_off_c = rolling_offense(home_off, fallback_home)
    a_off_c = rolling_offense(away_off, fallback_away)
    h_sp_c  = pitcher_score_b(home_pit, home_pitcher)
    a_sp_c  = pitcher_score_b(away_pit, away_pitcher)
    p_c = win_prob_from_scores(
        h_off_c, a_off_c,
        max(30, min(90, h_sp_c)),
        max(30, min(90, a_sp_c))
    )
    perspectives.append(("RecentForm", p_c))

    # --- Perspective D: Season/fallback only ---
    h_sp_d = max(30.0, min(90.0, PITCHER_ERA_FALLBACK.get(home_pitcher, 100) / 2))
    a_sp_d = max(30.0, min(90.0, PITCHER_ERA_FALLBACK.get(away_pitcher, 100) / 2))
    p_d = win_prob_from_scores(fallback_home, fallback_away, h_sp_d, a_sp_d)
    perspectives.append(("SeasonFallback", p_d))

    # --- Perspective E: Elo ratings ---
    try:
        home_elo = TEAM_RATINGS.get(home_team, {}).get("elo", 1500)
        away_elo = TEAM_RATINGS.get(away_team, {}).get("elo", 1500)
        if home_elo and away_elo:
            elo_home_prob = 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400.0))
            # Add small home field advantage
            elo_home_prob = min(0.80, elo_home_prob + 0.02)
            perspectives.append(("Elo", elo_home_prob))
    except Exception:
        pass

    # --- Perspective F: Pythagorean win% ---
    try:
        home_pyth = TEAM_RATINGS.get(home_team, {}).get("pythag_wpct", 0.500)
        away_pyth = TEAM_RATINGS.get(away_team, {}).get("pythag_wpct", 0.500)
        home_luck = TEAM_RATINGS.get(home_team, {}).get("luck", 0.0)
        away_luck = TEAM_RATINGS.get(away_team, {}).get("luck", 0.0)
        if home_pyth and away_pyth:
            # Pythagorean matchup prob + luck correction (lucky teams get penalized)
            pyth_edge = (home_pyth - away_pyth) * 0.5
            luck_adj  = (away_luck - home_luck) * 0.1  # fade lucky teams
            pyth_prob = max(0.35, min(0.70, 0.50 + pyth_edge + luck_adj + 0.02))
            perspectives.append(("Pythagorean", pyth_prob))
    except Exception:
        pass

    # --- Combine ---
    probs   = [p for _, p in perspectives]
    mean_p  = sum(probs) / len(probs)
    variance= sum((p - mean_p)**2 for p in probs) / len(probs)
    std_dev = variance ** 0.5

    # Confidence penalty for high disagreement between models
    # std_dev > 0.08 means models strongly disagree — reduce confidence toward 50%
    if std_dev > 0.08:
        blend = 0.6  # pull toward 50% when models disagree
        mean_p = mean_p * blend + 0.5 * (1 - blend)

    home_prob = max(0.20, min(0.80, mean_p))
    # Reduce home field bias — model historically 60% on home picks but
    # only 35% on away picks, meaning home advantage is overweighted
    # Shrink probabilities slightly toward 50% to correct this
    home_prob = home_prob * 0.92 + 0.5 * 0.08

    # --- Streak + Run Differential Adjustment ---
    try:
        from mlb_analytics.db import engine
        from sqlalchemy import text as sqla_text
        import pandas as pd

        def get_team_streak(team, as_of=None):
            """Get win/loss streak and avg run diff from last 10 games via Statcast."""
            try:
                q = sqla_text("""
                    SELECT game_pk, game_date,
                           MAX(CASE WHEN inning_topbot = 'Bot' THEN pitcher_team END) as home_team,
                           MAX(CASE WHEN inning_topbot = 'Top' THEN pitcher_team END) as away_team,
                           MAX(bat_score) as home_score,
                           MAX(fld_score) as away_score
                    FROM statcast_pitches
                    WHERE (pitcher_team = :team OR batter_team = :team)
                      AND game_date <= :dt
                    GROUP BY game_pk, game_date
                    HAVING home_team IS NOT NULL AND away_team IS NOT NULL
                    ORDER BY game_date DESC
                    LIMIT 10
                """)
                dt = as_of or __import__('datetime').date.today().isoformat()
                with engine.connect() as conn:
                    df = pd.read_sql(q, conn, params={"team": team, "dt": dt})
                if df.empty:
                    return 0, 0
                wins = 0
                run_diffs = []
                for _, row in df.iterrows():
                    is_home = row["home_team"] == team
                    if is_home:
                        won = row["home_score"] > row["away_score"]
                        rd  = row["home_score"] - row["away_score"]
                    else:
                        won = row["away_score"] > row["home_score"]
                        rd  = row["away_score"] - row["home_score"]
                    wins += int(won)
                    run_diffs.append(rd)
                win_pct = wins / len(df)
                avg_rd  = sum(run_diffs) / len(run_diffs)
                return win_pct, avg_rd
            except Exception:
                return 0.5, 0

        home_wpct, home_rd = get_team_streak(home_team)
        away_wpct, away_rd = get_team_streak(away_team)

        # Streak adjustment: teams on hot streaks get small boost
        streak_adj = (home_wpct - away_wpct) * 0.03
        # Run diff adjustment: teams outscoring opponents get small boost
        rd_adj = (home_rd - away_rd) * 0.002
        total_adj = max(-0.04, min(0.04, streak_adj + rd_adj))
        home_prob = max(0.20, min(0.80, home_prob + total_adj))
    except Exception:
        pass

    away_prob = 1.0 - home_prob

    factors = {
        "home_offense":   round(fallback_home, 1),
        "away_offense":   round(fallback_away, 1),
        "home_pitcher_score": round(h_sp_b, 1),
        "away_pitcher_score": round(a_sp_b, 1),
        "park_factor":    park,
        "weather":        w_lean or "No data",
        "ensemble_perspectives": {n: round(p, 4) for n, p in perspectives},
        "ensemble_std_dev": round(std_dev, 4),
        "ensemble_agreement": "HIGH" if std_dev < 0.04 else
                               "MED"  if std_dev < 0.08 else "LOW",
        "data_source":    "Ensemble (Statcast+Pitcher+RecentForm+Season)",
        "home_advantage_score": round((home_prob - 0.5) * 100, 2),
    }

    return home_prob, away_prob, factors


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_moneylines() -> pd.DataFrame:
    from mlb_analytics.ingestion.fanduel_scraper import fetch_moneylines_api

    logger.info("Fetching moneylines from Odds API...")
    ml = fetch_moneylines_api()

    if ml.empty:
        logger.error("No moneyline data returned.")
        return pd.DataFrame()

    starters = {}
    try:
        from daily_lineup import TODAYS_STARTERS
        starters = TODAYS_STARTERS
    except ImportError:
        logger.warning("No lineup data -- run fetch_lineups.py first")

    rows = []
    for _, game in ml.iterrows():
        home_full = game["home_team"]
        away_full = game["away_team"]
        # Convert full names to abbreviations
        home      = TEAM_NAME_TO_ABB.get(home_full, home_full)
        away      = TEAM_NAME_TO_ABB.get(away_full, away_full)
        fair_home = game["fair_prob_home"]
        fair_away = game["fair_prob_away"]
        ml_home   = game["moneyline_home"]
        ml_away   = game["moneyline_away"]
        game_date = game["game_date"]

        home_sp = starters.get(home, {}).get("name", "TBD")
        away_sp = starters.get(away, {}).get("name", "TBD")

        model_home, model_away, factors = calculate_win_probability_ensemble(
            home, away, home_sp, away_sp
        )

        edge_home = model_home - fair_home
        edge_away = model_away - fair_away

        def calc_ev(model_p, decimal_odds):
            return model_p * (decimal_odds - 1) - (1 - model_p)

        if edge_home >= edge_away:
            best_side    = home_full
            best_abb     = home
            best_ml      = ml_home
            best_model_p = model_home
            best_fair_p  = fair_home
            best_edge    = edge_home
            opponent     = away_full
        else:
            best_side    = away_full
            best_abb     = away
            best_ml      = ml_away
            best_model_p = model_away
            best_fair_p  = fair_away
            best_edge    = edge_away
            opponent     = home_full

        # Away team confidence penalty — model historically 46% on away picks
        # Require stronger signal before picking an away team
        # Shrink away team probability 5% toward 50% to reflect this
        if best_abb == away:
            best_model_p = best_model_p * 0.95 + 0.5 * 0.05

        confidence_pct = best_model_p * 100
        label, grade   = get_confidence_label(confidence_pct)

        # BET/WATCH/SKIP filter based on 80 games of real data:
        # 70%+        = 100% win rate (bet anything)
        # 65-70%      = 100% win rate (bet anything)
        # 55-60% home = 66% win rate (bet home only)
        # 60-65%      = 40% win rate (NEVER BET - death zone)
        # Away <65%   = 46% win rate (never bet)
        is_home    = best_abb == home
        agreement  = factors.get("ensemble_agreement", "MED")
        bet_rating = "SKIP"

        if best_model_p >= 0.70:
            bet_rating = "BET"      # 100% historically, bet anything
        elif best_model_p >= 0.65:
            bet_rating = "BET"      # 100% historically, bet anything
        elif is_home and 0.55 <= best_model_p < 0.60:
            bet_rating = "BET"      # 66% home picks in this range
        elif is_home and 0.60 <= best_model_p < 0.65:
            bet_rating = "SKIP"     # 40% in this range = death zone
        elif not is_home and best_model_p >= 0.65:
            bet_rating = "WATCH"    # away only viable at 65%+
        elif is_home and best_model_p >= 0.52:
            bet_rating = "WATCH"    # low confidence home, monitor only

        # Check injury alerts
        injury_alerts = get_injury_alerts()
        home_injuries = injury_alerts.get(home, [])
        away_injuries = injury_alerts.get(away, [])
        has_pitcher_injury = any(
            a.get("pitcher") for a in home_injuries + away_injuries
        )
        # Downgrade bet rating if pitcher is injured
        if has_pitcher_injury and bet_rating == "BET":
            bet_rating = "SKIP"
        elif has_pitcher_injury and bet_rating == "WATCH":
            bet_rating = "SKIP"

        rows.append({
            "game_date":          game_date,
            "game":               f"{away_full} @ {home_full}",
            "pick":               best_side,
            "pick_abb":           best_abb,
            "opponent":           opponent,
            "home_sp":            home_sp,
            "away_sp":            away_sp,
            "fd_odds":            best_ml,
            "model_prob":         round(best_model_p, 4),
            "fair_prob":          round(best_fair_p,  4),
            "edge":               round(best_edge,    4),
            "confidence_pct":     round(confidence_pct, 1),
            "confidence":         label,
            "grade":              grade,
            "data_source":        factors.get("data_source", ""),
            "agreement":          factors.get("ensemble_agreement", ""),
            "bet_rating":         bet_rating,
            "is_home_pick":       is_home,
            "std_dev":            factors.get("ensemble_std_dev", 0),
            "weather":            factors.get("weather", ""),
            "home_offense":       factors.get("home_offense", 0),
            "away_offense":       factors.get("away_offense", 0),
            "home_pitcher_score": factors.get("home_pitcher_score", 0),
            "away_pitcher_score": factors.get("away_pitcher_score", 0),
        })

    df = pd.DataFrame(rows).sort_values("confidence_pct", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Parlay builder
# ---------------------------------------------------------------------------

def american_to_decimal(american: float) -> float:
    if american > 0:
        return (american / 100.0) + 1.0
    return (100.0 / abs(american)) + 1.0


def decimal_to_american(decimal: float) -> str:
    if decimal >= 2.0:
        return f"+{int((decimal - 1) * 100)}"
    return f"{int(-100 / (decimal - 1))}"


def find_best_parlays(df: pd.DataFrame, n_legs: int = 3) -> dict:
    """
    Three parlay tiers based on historical accuracy filters:
      SAFE   - only BET-rated games (home 55%+ = 85% historically)
      MIXED  - BET anchors + WATCH legs
      UPSIDE - max payout from highest edge games
    """
    all_rows    = df.to_dict("records")
    # Only BET-rated games — strict filter based on 80 games of real data
    # 60-65% range is explicitly excluded (40% win rate = death zone)
    bet_games   = [r for r in all_rows if r.get("bet_rating") == "BET"]
    watch_games = [r for r in all_rows if r.get("bet_rating") == "WATCH"
                   and r.get("model_prob", 0) >= 0.60
                   and r.get("is_home_pick", False)]
    mixed_pool  = bet_games + watch_games
    upside_pool = sorted(
        [r for r in all_rows if r.get("edge", 0) > 0.05],
        key=lambda x: x.get("edge", 0), reverse=True
    )

    def build_combos(pool, n, sort_by="prob"):
        if len(pool) < n:
            pool = all_rows[:max(n * 2, 6)]
        results = []
        for combo in itertools.combinations(pool, n):
            odds = 1.0
            prob = 1.0
            for leg in combo:
                odds *= american_to_decimal(leg["fd_odds"])
                prob *= leg["model_prob"]
            ev = prob * (odds - 1) - (1 - prob)
            results.append({"legs": combo, "odds": odds, "prob": prob, "ev": ev})
        key = {"prob":  lambda x: x["prob"],
               "ev":    lambda x: x["ev"],
               "odds":  lambda x: x["odds"]}[sort_by]
        results.sort(key=key, reverse=True)
        return results[:3]

    # SAFE: all BET-rated legs, sorted by win prob
    safe = build_combos(bet_games, n_legs, sort_by="prob")

    # MIXED: n-1 BET legs + 1 WATCH leg
    mixed_results = []
    if len(bet_games) >= n_legs - 1 and watch_games:
        for fav_combo in itertools.combinations(bet_games, n_legs - 1):
            for watch in watch_games:
                combo = fav_combo + (watch,)
                odds = prob = 1.0
                for leg in combo:
                    odds *= american_to_decimal(leg["fd_odds"])
                    prob *= leg["model_prob"]
                ev = prob * (odds - 1) - (1 - prob)
                mixed_results.append({"legs": combo, "odds": odds,
                                      "prob": prob, "ev": ev})
        mixed_results.sort(key=lambda x: x["ev"], reverse=True)
        mixed = mixed_results[:3]
    else:
        mixed = build_combos(mixed_pool, n_legs, sort_by="ev")

    # UPSIDE: highest edge games regardless of rating
    upside = build_combos(upside_pool, n_legs, sort_by="odds")

    return {"safe": safe, "mixed": mixed, "upside": upside}


def print_game_rankings(df: pd.DataFrame) -> None:
    print(f"\n{'='*80}")
    print(f"  GAME-BY-GAME RANKINGS  ({date.today()})  --  Statcast + FanDuel")
    print(f"{'='*80}")
    print(f"  {'#':<3} {'Matchup':<32} {'Model Favorite':<22} {'Fav%':>5}  "
          f"{'Value Pick':<20} {'Odds':<7} {'Edge':>6}  {'Data'}")
    print(f"  {'-'*80}")

    for i, row in enumerate(df.itertuples(), 1):
        agree      = getattr(row, "agreement", "")
        bet_rating = getattr(row, "bet_rating", "SKIP")
        agree_icon = "✓✓" if agree == "HIGH" else "✓ " if agree == "MED" else "? "

        bet_icon = "🟢 BET  " if bet_rating == "BET"   else                    "🟡 WATCH" if bet_rating == "WATCH" else                    "🔴 SKIP "

        # Model favorite is always the team with higher model prob
        fav_prob  = row.model_prob
        fav_team  = row.pick
        dog_team  = row.opponent
        dog_odds  = decimal_to_american(row.fd_odds)

        # If pick is the underdog (positive odds), swap display
        if row.fd_odds > 2.0:  # underdog
            fav_team  = row.opponent
            fav_prob  = 1.0 - row.model_prob
            dog_team  = row.pick

        matchup = f"{row.game.split(' @ ')[0][:13]} @ {row.game.split(' @ ')[1][:13]}"

        injury_note = ""
        try:
            injury_cache = get_injury_alerts()
            home_t = row.game.split("@ ")[-1].strip()[:3]
            away_t = row.game.split("@")[0].strip()[-3:]
            if injury_cache.get(home_t) or injury_cache.get(away_t):
                injury_note = " 🚑"
        except Exception:
            pass

        print(
            f"  {bet_icon} {matchup:<28} "
            f"{fav_team:<22} {fav_prob*100:4.1f}%  "
            f"{dog_team:<18} {dog_odds:<7} "
            f"{row.edge*100:+5.1f}pp {agree_icon}{injury_note}"
        )

    sc_count = len(df[df["data_source"].str.contains("Statcast", na=False)])
    fb_count = len(df) - sc_count
    print(f"\n  [SC] = Statcast ({sc_count} games)  [FB] = 2025 fallback ({fb_count} games)")
    print(f"  Model Favorite = team model thinks wins  Value Pick = best edge over FanDuel odds")


def _print_parlay_group(parlays: list, title: str, desc: str) -> None:
    """Print a group of parlays with a title."""
    print(f"\n  {'='*68}")
    print(f"  {title}")
    print(f"  {desc}")
    print(f"  {'='*68}")

    if not parlays:
        print("  Not enough qualifying games for this tier.")
        return

    medals = ["#1 GOLD", "#2 SILVER", "#3 BRONZE"]
    for i, p in enumerate(parlays):
        medal  = medals[i] if i < len(medals) else f"#{i+1}"
        ev_str = f"{p['ev']*100:+.1f}%"
        print(
            f"\n  {medal}  |  "
            f"Odds: {decimal_to_american(p['odds'])}  |  "
            f"Win Prob: {p['prob']*100:.1f}%  |  "
            f"EV: {ev_str}"
        )
        print(f"  {'-'*60}")
        for j, leg in enumerate(p["legs"], 1):
            src     = "[SC]" if "Statcast" in str(leg.get("data_source", "")) else "[FB]"
            weather = leg.get("weather", "")
            w_str   = f"  {weather[:25]}" if weather and weather != "No data" else ""
            print(
                f"  Leg {j}: {leg['pick']:<26} "
                f"{decimal_to_american(leg['fd_odds']):<8} "
                f"Model: {leg['confidence_pct']:.1f}%  "
                f"[{leg['grade']}]  {src}"
            )
            print(f"         Off: {leg['home_offense']:.0f}  "
                  f"SP: {leg['away_pitcher_score'] if leg.get('is_home_pick') else leg['home_pitcher_score']:.0f}  "
                  f"Edge: {leg['edge']*100:+.1f}pp{w_str}")

        if p["ev"] > 0:
            b     = p["odds"] - 1
            kelly = max(0, (b * p["prob"] - (1 - p["prob"])) / b)
            qk    = kelly * 0.25
            print(f"\n  Suggested stake: {qk*100:.2f}% of bankroll (quarter-Kelly)")
        else:
            print(f"\n  EV: {ev_str}  (negative EV but included for completeness)")


def print_parlay_recommendations(df: pd.DataFrame, n_legs: int) -> None:
    print(f"\n{'='*72}")
    print(f"  {n_legs}-LEG PARLAY RECOMMENDATIONS")
    print(f"{'='*72}")

    tiers = find_best_parlays(df, n_legs=n_legs)

    _print_parlay_group(
        tiers["safe"],
        "SAFE -- Highest Win Probability",
        "All strong model favorites. Most likely to cash."
    )
    _print_parlay_group(
        tiers["mixed"],
        "MIXED -- Anchor + Value Underdog",
        "Favorites anchored with one mispriced underdog for better payout."
    )
    _print_parlay_group(
        tiers["upside"],
        "UPSIDE -- Maximum Payout",
        "Underdog heavy. Low win prob but massive if it hits."
    )

    print(f"\n{'='*72}")
    print(f"  [SC] = Statcast model  [FB] = 2025 team ratings fallback")
    print(f"{'='*72}\n")


def apply_profit_boosts(df: pd.DataFrame, boosts: list[str]) -> pd.DataFrame:
    """
    Apply FanDuel profit boosts to specific teams.
    boosts format: ["PHI:25", "LAD:50"] means +25% on PHI, +50% on LAD

    A 25% profit boost on -110 odds effectively makes it -110 * 1.25 payout.
    """
    if not boosts:
        return df

    boost_map = {}
    for b in boosts:
        try:
            team, pct = b.split(":")
            boost_map[team.upper()] = float(pct) / 100.0
        except ValueError:
            continue

    df = df.copy()
    for idx, row in df.iterrows():
        team = row["pick_abb"]
        if team in boost_map:
            boost    = boost_map[team]
            dec_odds = row["fd_odds"]
            # Boost increases the payout on a win
            boosted_odds = 1.0 + (dec_odds - 1.0) * (1.0 + boost)
            df.at[idx, "fd_odds"] = boosted_odds
            # Recalculate edge and EV with boosted odds
            new_ev   = row["model_prob"] * (boosted_odds - 1) - (1 - row["model_prob"])
            new_edge = row["model_prob"] - (1.0 / boosted_odds)
            df.at[idx, "edge"] = new_edge
            print(f"  [BOOST +{boost*100:.0f}%] {row['pick']:<25} "
                  f"Odds: {decimal_to_american(dec_odds)} → "
                  f"{decimal_to_american(boosted_odds)}  "
                  f"EV: {new_ev*100:+.1f}%")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLB Parlay Builder -- powered by Statcast data"
    )
    parser.add_argument("--legs",  type=int, default=3)
    parser.add_argument("--boost", type=str, nargs="*", default=[],
                        help="Profit boosts e.g. --boost PHI:25 LAD:50")
    args = parser.parse_args()

    df = analyze_moneylines()
    if df.empty:
        return

    # Apply any profit boosts
    if args.boost:
        print(f"\n  Applying profit boosts:")
        df = apply_profit_boosts(df, args.boost)

    print_game_rankings(df)

    for n in [2, 3, 4]:
        print_parlay_recommendations(df, n_legs=n)

    print("\nQuick reference:")
    print("  SAFE   = highest chance to win, lower payout")
    print("  MIXED  = balanced risk/reward")
    print("  UPSIDE = long shot, big payout")
    print("  ✓✓ = all 4 models agree  ✓ = moderate agreement  ? = models disagree")


if __name__ == "__main__":
    main()
    # Auto-save predictions after every run
    try:
        import subprocess, sys
        subprocess.run([sys.executable, "save_predictions.py"], check=False)
    except Exception:
        pass