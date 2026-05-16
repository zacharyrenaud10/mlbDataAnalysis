# -*- coding: utf-8 -*-
"""
save_predictions.py
-------------------
Snapshots today's model predictions before games are played.
Run every morning after lineups post.

Saves to cache/predictions_YYYY-MM-DD.json

Then run track_results.py the next day to grade them and
feed the accuracy data back into model improvement.

Usage:
    python save_predictions.py
    python save_predictions.py --date 2026-03-31
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from loguru import logger

Path("cache").mkdir(exist_ok=True)


def save_todays_predictions(game_date: str) -> None:
    """
    Snapshot all of today's predictions:
    - Game win probabilities
    - Top hit candidates per game
    - Top HR candidates
    - Top K predictions
    - Pitcher K projections
    """
    from matchup_predictor import (
        load_model, get_batter_rolling, get_pitcher_rolling,
        HAND_AVERAGES, PITCHER_AVERAGES, PARK_FACTORS,
        get_weather_lean, get_umpire_k_factor,
    )
    from parlay_builder import calculate_win_probability
    from fetch_lineups import get_todays_games, parse_game

    try:
        from daily_lineup import TODAYS_LINEUPS, TODAYS_STARTERS
    except ImportError:
        logger.error("No daily_lineup.py found -- run fetch_lineups.py first")
        return

    model     = load_model()
    games_raw = get_todays_games(game_date)
    games     = [parse_game(g) for g in games_raw if parse_game(g)]

    predictions = {
        "date":         game_date,
        "model_version": str(getattr(model, "version", "unknown")),
        "games":        [],
        "batter_props": [],
        "pitcher_props": [],
    }

    for game in games:
        home = game["home"]
        away = game["away"]
        home_sp = TODAYS_STARTERS.get(home, {}).get("name", "TBD")
        away_sp = TODAYS_STARTERS.get(away, {}).get("name", "TBD")
        home_hand = TODAYS_STARTERS.get(home, {}).get("hand", "R")
        away_hand = TODAYS_STARTERS.get(away, {}).get("hand", "R")

        # Game win probability
        try:
            home_prob, away_prob, factors = calculate_win_probability(
                home, away, home_sp, away_sp
            )
        except Exception:
            home_prob, away_prob = 0.5, 0.5
            factors = {}

        predictions["games"].append({
            "matchup":        f"{away} @ {home}",
            "home":           home,
            "away":           away,
            "home_sp":        home_sp,
            "away_sp":        away_sp,
            "home_win_prob":  round(home_prob, 4),
            "away_win_prob":  round(away_prob, 4),
            "model_pick":     home if home_prob > away_prob else away,
            "confidence":     round(max(home_prob, away_prob), 4),
            "factors":        {k: v for k, v in factors.items()
                               if isinstance(v, (int, float, str))},
        })

        # Batter props for each lineup
        for batting_team, pitching_team, sp_name, sp_hand in [
            (away, home, home_sp, home_hand),
            (home, away, away_sp, away_hand),
        ]:
            lineup  = TODAYS_LINEUPS.get(batting_team, [])
            if not lineup:
                continue

            p_feat   = get_pitcher_rolling(sp_name) or PITCHER_AVERAGES.copy()
            park_f   = PARK_FACTORS.get(home, 100)
            ump_k    = get_umpire_k_factor(home)
            weather  = get_weather_lean(home)

            for batter in lineup:
                name   = batter.get("name", "")
                hand   = batter.get("bats", "R")
                order  = batter.get("order", 5)
                b_feat = get_batter_rolling(name) or \
                         HAND_AVERAGES.get(hand, HAND_AVERAGES["R"]).copy()

                ctx = {
                    "park_factor":       park_f,
                    "batter_hand":       hand,
                    "pitcher_hand":      sp_hand,
                    "is_home":           batting_team == home,
                    "batting_order":     order,
                    "h2h_pa":            0,
                    "h2h_ba":            0.250,
                    "b_season_woba":     b_feat.get("woba", 0.315),
                    "b_season_wrc_plus": 100.0,
                    "p_season_era":      p_feat.get("xera", 4.0),
                    "p_season_k_pct":    p_feat.get("k_pct", 0.22),
                }

                props = model.predict_batter(b_feat, p_feat, ctx)

                predictions["batter_props"].append({
                    "player":      name,
                    "team":        batting_team,
                    "vs_pitcher":  sp_name,
                    "batting_order": order,
                    "game":        f"{away} @ {home}",
                    # Key prop predictions
                    "hit_prob":    round(props.get("hit_1plus",     0), 4),
                    "hr_prob":     round(props.get("hr_1plus",      0), 4),
                    "tb2_prob":    round(props.get("total_bases_2", 0), 4),
                    "tb3_prob":    round(props.get("total_bases_3", 0), 4),
                    "k_prob":      round(props.get("k_1plus",       0) * ump_k, 4),
                    "bb_prob":     round(props.get("walk_1plus",    0), 4),
                    "sb_prob":     round(props.get("sb_1plus",      0), 4),
                    "exp_tb":      round(props.get("expected_tb",   0), 3),
                    # Features used (for model improvement)
                    "features": {
                        "avg_ev":       round(float(b_feat.get("avg_exit_velo", 0) or 0), 2),
                        "hard_hit_pct": round(float(b_feat.get("hard_hit_pct",  0) or 0), 3),
                        "woba":         round(float(b_feat.get("woba",          0) or 0), 3),
                        "k_pct":        round(float(b_feat.get("k_pct",         0) or 0), 3),
                        "p_k_pct":      round(float(p_feat.get("k_pct",         0) or 0), 3),
                        "p_xera":       round(float(p_feat.get("xera",          0) or 0), 2),
                        "park_factor":  park_f,
                        "ump_k":        round(ump_k, 3),
                    },
                })

            # Pitcher props
            from mlb_analytics.models.player_props_model import PITCHER_PROP_TARGETS
            p_context = {
                "park_factor": park_f, "is_home": pitching_team == home,
                "days_rest": 4, "opp_k_pct": 0.22,
                "p_season_era": p_feat.get("xera", 4.0),
                "p_season_fip": p_feat.get("xera", 4.0),
                "p_season_k_9": p_feat.get("k_pct", 0.22) * 27,
            }
            pitcher_props = model.predict_pitcher(p_feat, p_context)

            predictions["pitcher_props"].append({
                "pitcher":     sp_name,
                "team":        pitching_team,
                "vs_team":     batting_team,
                "game":        f"{away} @ {home}",
                "exp_k":       round(pitcher_props.get("expected_k", 0), 2),
                "k4_prob":     round(pitcher_props.get("k_4plus", 0), 4),
                "k5_prob":     round(pitcher_props.get("k_5plus", 0), 4),
                "k6_prob":     round(pitcher_props.get("k_6plus", 0), 4),
                "k7_prob":     round(pitcher_props.get("k_7plus", 0), 4),
                "qs_prob":     round(pitcher_props.get("qs",      0), 4),
                "features": {
                    "k_pct":    round(float(p_feat.get("k_pct",    0) or 0), 3),
                    "whiff_pct":round(float(p_feat.get("whiff_pct",0) or 0), 3),
                    "xera":     round(float(p_feat.get("xera",     0) or 0), 2),
                },
            })

    # Save to file
    out_path = f"cache/predictions_{game_date}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    n_batters  = len(predictions["batter_props"])
    n_games    = len(predictions["games"])
    n_pitchers = len(predictions["pitcher_props"])

    logger.success(f"Saved predictions for {game_date}")
    logger.info(f"  {n_games} games, {n_batters} batter props, "
                f"{n_pitchers} pitcher props")
    logger.info(f"  File: {out_path}")

    # Print quick summary
    print(f"\n{'='*60}")
    print(f"  PREDICTIONS SAVED -- {game_date}")
    print(f"{'='*60}")
    print(f"\n  Games:    {n_games}")
    print(f"  Batters:  {n_batters}")
    print(f"  Pitchers: {n_pitchers}")

    print(f"\n  TOP HR PICKS:")
    hr_picks = sorted(predictions["batter_props"],
                      key=lambda x: x["hr_prob"], reverse=True)[:5]
    for p in hr_picks:
        print(f"    {p['player']:<22} {p['team']:<5} "
              f"vs {p['vs_pitcher']:<20} HR: {p['hr_prob']*100:.1f}%")

    print(f"\n  TOP HIT PICKS:")
    hit_picks = sorted(predictions["batter_props"],
                       key=lambda x: x["hit_prob"], reverse=True)[:5]
    for p in hit_picks:
        print(f"    {p['player']:<22} {p['team']:<5} "
              f"vs {p['vs_pitcher']:<20} Hit: {p['hit_prob']*100:.1f}%")

    print(f"\n  TOP PITCHER K PICKS:")
    k_picks = sorted(predictions["pitcher_props"],
                     key=lambda x: x["exp_k"], reverse=True)[:5]
    for p in k_picks:
        print(f"    {p['pitcher']:<22} vs {p['vs_team']:<5} "
              f"ExpK: {p['exp_k']:.1f}  6+K: {p['k6_prob']*100:.0f}%")

    print(f"\n  Run tonight: python track_results.py --date {game_date}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save today's model predictions for later grading"
    )
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()

    game_date = args.date or date.today().isoformat()
    save_todays_predictions(game_date)


if __name__ == "__main__":
    main()
