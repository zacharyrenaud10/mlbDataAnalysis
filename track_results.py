# -*- coding: utf-8 -*-
"""
track_results.py
----------------
Grades model predictions against actual results.
Generates model improvement insights over time.

Run after games finish each night.

Usage:
    python track_results.py              <- grade yesterday
    python track_results.py --date 2026-03-31
    python track_results.py --summary    <- full record + calibration
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger
from sqlalchemy import text

from mlb_analytics.db import engine

RESULTS_FILE = "cache/prediction_results.json"
MLB_API      = "https://statsapi.mlb.com/api/v1"
HEADERS      = {"User-Agent": "Mozilla/5.0"}

Path("cache").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Fetch actual results
# ---------------------------------------------------------------------------

def fetch_game_results(game_date: str) -> list[dict]:
    try:
        resp = requests.get(
            f"{MLB_API}/schedule",
            params={"sportId": 1, "date": game_date,
                    "hydrate": "team,linescore"},
            headers=HEADERS, timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error(f"Failed to fetch results: {exc}")
        return []

    results = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue
            home       = game["teams"]["home"]["team"]["abbreviation"]
            away       = game["teams"]["away"]["team"]["abbreviation"]
            home_score = game["teams"]["home"].get("score", 0)
            away_score = game["teams"]["away"].get("score", 0)
            results.append({
                "date":       game_date,
                "home":       home,
                "away":       away,
                "home_score": home_score,
                "away_score": away_score,
                "winner":     home if home_score > away_score else away,
                "total_runs": home_score + away_score,
                "score":      f"{away_score}-{home_score}",
            })

    logger.info(f"Found {len(results)} final results for {game_date}")
    return results


def fetch_player_stats_from_statcast(game_date: str) -> pd.DataFrame:
    """Pull actual player stats from our Statcast DB."""
    try:
        query = text("""
            SELECT
                sp.batter_id,
                p.full_name,
                sp.game_pk,
                SUM(CASE WHEN sp.events IN ('single','double','triple','home_run')
                    THEN 1 ELSE 0 END) AS hits,
                SUM(CASE WHEN sp.events = 'home_run'
                    THEN 1 ELSE 0 END) AS hr,
                SUM(CASE WHEN sp.events = 'walk'
                    THEN 1 ELSE 0 END) AS bb,
                SUM(CASE WHEN sp.events IN ('strikeout','strikeout_double_play')
                    THEN 1 ELSE 0 END) AS k,
                SUM(CASE WHEN sp.events IN ('single','double','triple','home_run')
                    THEN 1 ELSE 0 END) +
                SUM(CASE WHEN sp.events = 'double'    THEN 1 ELSE 0 END) +
                SUM(CASE WHEN sp.events = 'triple'    THEN 2 ELSE 0 END) +
                SUM(CASE WHEN sp.events = 'home_run'  THEN 3 ELSE 0 END)
                    AS total_bases
            FROM statcast_pitches sp
            JOIN players p ON p.player_id = sp.batter_id
            WHERE sp.game_date = :gd
            GROUP BY sp.batter_id, p.full_name, sp.game_pk
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"gd": game_date})
        # Deduplicate — take best game per player if multiple
        df = df.sort_values("hits", ascending=False).drop_duplicates("full_name")
        return df
    except Exception as exc:
        logger.error(f"Statcast fetch failed: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Load/save results
# ---------------------------------------------------------------------------

def load_results() -> dict:
    try:
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"games": [], "props": [], "calibration": {}}


def save_results(data: dict) -> None:
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Advanced grading
# ---------------------------------------------------------------------------

def advanced_grade(pick_prob: float, actual_score: str,
                   model_pick: str, actual_winner: str,
                   home: str, away: str) -> dict:
    """
    Grade a pick beyond just win/loss:
    - Confidence calibration (was 70% pick closer than 55% pick?)
    - Blowout detection (did we pick a team that got crushed?)
    - Run differential vs expected
    - Cover quality (did the favorite cover by enough?)
    """
    correct = model_pick == actual_winner

    # Parse score
    try:
        parts = actual_score.split("-")
        away_score = int(parts[0])
        home_score = int(parts[1])
        winner_score = max(home_score, away_score)
        loser_score  = min(home_score, away_score)
        run_diff     = winner_score - loser_score
    except Exception:
        away_score = home_score = winner_score = loser_score = run_diff = 0

    # Grade quality
    if correct:
        if run_diff >= 5:
            quality = "DOMINANT"   # won big, model was right and confident
        elif run_diff >= 3:
            quality = "SOLID"      # clear win
        elif run_diff >= 1:
            quality = "NARROW"     # barely won, lucky
        else:
            quality = "TIE"
    else:
        if run_diff >= 5:
            quality = "CRUSHED"    # got destroyed, model badly wrong
        elif run_diff >= 3:
            quality = "BLOWN OUT"  # lost badly
        elif run_diff == 1:
            quality = "CLOSE"      # one run game, could go either way
        else:
            quality = "WRONG"

    # Confidence penalty — were we confidently wrong?
    if not correct and pick_prob >= 0.65:
        confidence_grade = "F"   # high confidence wrong = worst
    elif not correct and pick_prob >= 0.55:
        confidence_grade = "D"
    elif not correct:
        confidence_grade = "C-"  # low confidence wrong = expected
    elif correct and pick_prob >= 0.65:
        confidence_grade = "A"   # high confidence right = best
    elif correct and pick_prob >= 0.55:
        confidence_grade = "B"
    else:
        confidence_grade = "C+"  # low confidence right = lucky

    # Was it a blowout in the wrong direction?
    blowout_wrong = not correct and run_diff >= 5

    return {
        "correct":          correct,
        "quality":          quality,
        "confidence_grade": confidence_grade,
        "run_diff":         run_diff,
        "winner_score":     winner_score,
        "loser_score":      loser_score,
        "blowout_wrong":    blowout_wrong,
        "pick_prob":        pick_prob,
    }


# ---------------------------------------------------------------------------
# Grade game predictions
# ---------------------------------------------------------------------------

def grade_games(predictions: dict, results: list[dict]) -> list[dict]:
    graded = []
    result_map = {
        f"{r['away']}@{r['home']}": r for r in results
    }

    for g in predictions.get("games", []):
        matchup = g["matchup"].replace(" @ ", "@")
        result  = result_map.get(matchup)
        if not result:
            continue

        correct = g["model_pick"] == result["winner"]
        conf    = g["confidence"]

        graded.append({
            "matchup":       g["matchup"],
            "model_pick":    g["model_pick"],
            "model_prob":    conf,
            "actual_winner": result["winner"],
            "correct":       correct,
            "score":         result["score"],
            "home_sp":       g.get("home_sp", ""),
            "away_sp":       g.get("away_sp", ""),
        })

    return graded


# ---------------------------------------------------------------------------
# Grade prop predictions
# ---------------------------------------------------------------------------

def grade_props(predictions: dict, player_stats: pd.DataFrame) -> dict:
    if player_stats.empty or not predictions.get("batter_props"):
        return {}

    # Build lookup: player name -> actual stats
    stats_map = {}
    for _, row in player_stats.iterrows():
        name = row["full_name"]
        stats_map[name] = {
            "hits":        int(row["hits"]),
            "hr":          int(row["hr"]),
            "bb":          int(row["bb"]),
            "k":           int(row["k"]),
            "total_bases": int(row["total_bases"]),
        }

    graded_props = []
    # Calibration buckets: how often does X% prediction come true?
    calibration = {
        "hit":  {"buckets": {}, "n": 0, "correct": 0},
        "hr":   {"buckets": {}, "n": 0, "correct": 0},
        "tb2":  {"buckets": {}, "n": 0, "correct": 0},
    }

    for pred in predictions["batter_props"]:
        player  = pred["player"]
        actual  = stats_map.get(player)
        if not actual:
            continue  # player didn't play or not in DB yet

        hit_pred    = pred["hit_prob"]
        hr_pred     = pred["hr_prob"]
        tb2_pred    = pred["tb2_prob"]

        hit_result  = actual["hits"] >= 1
        hr_result   = actual["hr"] >= 1
        tb2_result  = actual["total_bases"] >= 2

        graded_props.append({
            "player":     player,
            "team":       pred["team"],
            "vs":         pred["vs_pitcher"],
            # Predictions
            "hit_pred":   hit_pred,
            "hr_pred":    hr_pred,
            "tb2_pred":   tb2_pred,
            "k_pred":     pred["k_prob"],
            "exp_tb":     pred["exp_tb"],
            # Actuals
            "hit_actual": hit_result,
            "hr_actual":  hr_result,
            "tb2_actual": tb2_result,
            "k_actual":   actual["k"] >= 1,
            "actual_tb":  actual["total_bases"],
            "actual_hits":actual["hits"],
            # Correct flags
            "hit_correct":hit_result == (hit_pred >= 0.5),
            "hr_correct": hr_result  == (hr_pred  >= 0.10),
            "tb2_correct":tb2_result == (tb2_pred >= 0.25),
        })

        # Calibration: bucket prediction into 5% bands
        for prop, pred_val, actual_val in [
            ("hit", hit_pred, hit_result),
            ("hr",  hr_pred,  hr_result),
            ("tb2", tb2_pred, tb2_result),
        ]:
            bucket = round(pred_val * 20) / 20  # 5% buckets
            cal    = calibration[prop]["buckets"]
            if bucket not in cal:
                cal[bucket] = {"predicted": 0, "actual": 0, "n": 0}
            cal[bucket]["predicted"] += pred_val
            cal[bucket]["actual"]    += int(actual_val)
            cal[bucket]["n"]         += 1
            calibration[prop]["n"]   += 1
            if actual_val:
                calibration[prop]["correct"] += 1

    return {
        "graded_props": graded_props,
        "calibration":  calibration,
    }


# ---------------------------------------------------------------------------
# Grade pitcher predictions
# ---------------------------------------------------------------------------

def grade_pitchers(predictions: dict, player_stats: pd.DataFrame) -> list[dict]:
    """Grade pitcher K predictions."""
    if player_stats.empty or not predictions.get("pitcher_props"):
        return []

    # Get actual pitcher K totals from statcast
    try:
        query = text("""
            SELECT
                p.full_name,
                SUM(CASE WHEN sp.events IN ('strikeout','strikeout_double_play')
                    THEN 1 ELSE 0 END) AS actual_k
            FROM statcast_pitches sp
            JOIN players p ON p.player_id = sp.pitcher_id
            WHERE sp.game_date = :gd
            GROUP BY p.full_name
        """)
        with engine.connect() as conn:
            pitcher_df = pd.read_sql(
                query, conn,
                params={"gd": predictions.get("date", date.today().isoformat())}
            )
        pitcher_map = dict(zip(pitcher_df["full_name"], pitcher_df["actual_k"]))
    except Exception:
        pitcher_map = {}

    graded = []
    for pred in predictions.get("pitcher_props", []):
        pitcher   = pred["pitcher"]
        actual_k  = pitcher_map.get(pitcher)
        if actual_k is None:
            continue
        graded.append({
            "pitcher":   pitcher,
            "vs":        pred["vs_team"],
            "exp_k":     pred["exp_k"],
            "actual_k":  int(actual_k),
            "k6_pred":   pred["k6_prob"],
            "k6_actual": int(actual_k) >= 6,
            "k6_correct":pred["k6_prob"] >= 0.40 and int(actual_k) >= 6 or
                         pred["k6_prob"] <  0.40 and int(actual_k) <  6,
            "k_error":   round(pred["exp_k"] - int(actual_k), 1),
        })

    return graded


# ---------------------------------------------------------------------------
# Model improvement insights
# ---------------------------------------------------------------------------

def generate_insights(all_results: dict) -> None:
    """
    Analyze patterns in wrong predictions to suggest model improvements.
    """
    all_props = []
    for day in all_results.get("props", []):
        all_props.extend(day.get("graded_props", []))

    if len(all_props) < 20:
        print("\n  Need more data for insights (20+ graded props)")
        return

    df = pd.DataFrame(all_props)

    print(f"\n  MODEL CALIBRATION INSIGHTS")
    print(f"  {'='*50}")

    # Hit calibration
    if "hit_pred" in df.columns and "hit_actual" in df.columns:
        df["hit_bucket"] = (df["hit_pred"] * 10).round() / 10
        cal = df.groupby("hit_bucket").agg(
            predicted=("hit_pred", "mean"),
            actual=("hit_actual", "mean"),
            n=("hit_pred", "count")
        ).reset_index()
        print(f"\n  HIT CALIBRATION (predicted vs actual rate):")
        for _, row in cal.iterrows():
            if row["n"] >= 5:
                diff = row["actual"] - row["predicted"]
                flag = " <-- OVERESTIMATING" if diff < -0.05 else \
                       " <-- UNDERESTIMATING" if diff > 0.05 else ""
                print(f"    {row['predicted']*100:.0f}% predicted: "
                      f"{row['actual']*100:.0f}% actual "
                      f"(n={row['n']:.0f}){flag}")

    # HR calibration
    if "hr_pred" in df.columns and "hr_actual" in df.columns:
        hr_rate = df["hr_actual"].mean()
        hr_pred_mean = df["hr_pred"].mean()
        diff = hr_rate - hr_pred_mean
        flag = "OVERESTIMATING" if diff < -0.01 else \
               "UNDERESTIMATING" if diff > 0.01 else "CALIBRATED"
        print(f"\n  HR CALIBRATION:")
        print(f"    Average predicted: {hr_pred_mean*100:.1f}%")
        print(f"    Average actual:    {hr_rate*100:.1f}%")
        print(f"    Status: {flag}")


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

def print_graded_results(
    game_date: str,
    game_grades: list[dict],
    prop_grades: dict,
    pitcher_grades: list[dict],
) -> None:

    print(f"\n{'='*65}")
    print(f"  RESULTS -- {game_date}")
    print(f"{'='*65}")

    # Games
    correct = sum(1 for g in game_grades if g["correct"])
    total   = len(game_grades)
    pct     = correct / total * 100 if total > 0 else 0

    # Advanced grades
    adv = {}
    for g in game_grades:
        home = g["matchup"].split(" @ ")[1].strip() if " @ " in g["matchup"] else ""
        away = g["matchup"].split(" @ ")[0].strip() if " @ " in g["matchup"] else ""
        adv[g["matchup"]] = advanced_grade(
            g["model_prob"], g["score"],
            g["model_pick"], g["actual_winner"],
            home, away
        )

    # Grade summary
    grades = [adv[g["matchup"]]["confidence_grade"] for g in game_grades]
    a_count = grades.count("A")
    b_count = grades.count("B")
    f_count = grades.count("F")
    blowouts = sum(1 for g in game_grades
                   if adv[g["matchup"]]["blowout_wrong"])

    print(f"\n  GAME PICKS ({correct}/{total} = {pct:.0f}%):")
    print(f"  Grade breakdown: "
          f"{a_count}x A  {b_count}x B  {f_count}x F  "
          f"{blowouts} blowout misses")
    print(f"  {'-'*65}")

    for g in game_grades:
        ag   = adv[g["matchup"]]
        icon = "[OK]" if g["correct"] else "[X] "
        cg   = ag["confidence_grade"]
        qual = ag["quality"]
        diff = ag["run_diff"]
        print(f"  {icon} [{cg}] {g['matchup']:<26} "
              f"Picked: {g['model_pick']:<5} ({g['model_prob']*100:.0f}%)  "
              f"Result: {g['actual_winner']} {g['score']}  "
              f"[{qual} +{diff}]" if g["correct"] else
              f"  {icon} [{cg}] {g['matchup']:<26} "
              f"Picked: {g['model_pick']:<5} ({g['model_prob']*100:.0f}%)  "
              f"Result: {g['actual_winner']} {g['score']}  "
              f"[{qual} -{diff}]")

    # Batter props
    gp = prop_grades.get("graded_props", [])
    if gp:
        hr_correct   = sum(1 for p in gp if p["hr_correct"])
        hit_correct  = sum(1 for p in gp if p["hit_correct"])
        n = len(gp)

        print(f"\n  PROP ACCURACY ({n} batters graded):")
        print(f"    Hit (>=50% threshold): {hit_correct}/{n} ({hit_correct/n*100:.0f}%)")
        print(f"    HR  (>=10% threshold): {hr_correct}/{n} ({hr_correct/n*100:.0f}%)")

        # Top HR predictions — did they hit?
        hr_preds = sorted(gp, key=lambda x: x["hr_pred"], reverse=True)[:5]
        print(f"\n  TOP HR PREDICTIONS:")
        for p in hr_preds:
            icon = "[OK]" if p["hr_actual"] else "[X] "
            hit_note = f"(got {p['actual_hits']} hit(s), {p['actual_tb']} TB)" \
                       if not p["hr_actual"] else "(HOMERED)"
            print(f"  {icon} {p['player']:<22} {p['hr_pred']*100:.1f}% pred  "
                  f"{hit_note}")

        # Top hit predictions — did they get one?
        hit_preds = sorted(gp, key=lambda x: x["hit_pred"], reverse=True)[:5]
        print(f"\n  TOP HIT PREDICTIONS:")
        for p in hit_preds:
            icon = "[OK]" if p["hit_actual"] else "[X] "
            print(f"  {icon} {p['player']:<22} {p['hit_pred']*100:.1f}% pred  "
                  f"actual: {p['actual_hits']} hit(s)")

    # Pitcher Ks
    if pitcher_grades:
        print(f"\n  PITCHER K PREDICTIONS:")
        for p in pitcher_grades:
            err  = p["k_error"]
            icon = "[OK]" if abs(err) <= 2 else "[X] "
            print(f"  {icon} {p['pitcher']:<22} "
                  f"predicted {p['exp_k']:.1f}K  actual {p['actual_k']}K  "
                  f"(err {err:+.1f})")

    print(f"\n{'='*65}\n")


def print_summary(data: dict) -> None:
    games_data = data.get("games", [])
    props_data = data.get("props", [])

    if not games_data:
        print("\n  No results tracked yet.")
        print("  Run: python track_results.py\n")
        return

    total   = sum(d["total"]   for d in games_data)
    correct = sum(d["correct"] for d in games_data)
    pct     = correct / total * 100 if total > 0 else 0

    print(f"\n{'='*65}")
    print(f"  MODEL RECORD SUMMARY  --  {date.today()}")
    print(f"{'='*65}")
    print(f"\n  Overall game picks: {correct}/{total} ({pct:.1f}%)")

    # By confidence tier
    all_graded = []
    for d in games_data:
        all_graded.extend(d.get("graded", []))

    if all_graded:
        high_conf = [g for g in all_graded if g["model_prob"] >= 0.60]
        mid_conf  = [g for g in all_graded if 0.55 <= g["model_prob"] < 0.60]
        low_conf  = [g for g in all_graded if g["model_prob"] < 0.55]

        def win_pct(games):
            if not games: return 0
            return sum(1 for g in games if g["correct"]) / len(games) * 100

        print(f"\n  By confidence:")
        print(f"    60%+ confidence: {win_pct(high_conf):.0f}% ({len(high_conf)} games)")
        print(f"    55-60%:          {win_pct(mid_conf):.0f}% ({len(mid_conf)} games)")
        print(f"    Under 55%:       {win_pct(low_conf):.0f}% ({len(low_conf)} games)")

    # Recent days
    print(f"\n  Last 7 days:")
    for day in games_data[-7:]:
        day_pct = day["correct"] / day["total"] * 100 if day["total"] > 0 else 0
        print(f"    {day['date']}:  {day['correct']}/{day['total']} ({day_pct:.0f}%)")

    # Prop accuracy summary
    all_props = []
    for d in props_data:
        all_props.extend(d.get("graded_props", []))

    if all_props:
        hr_correct  = sum(1 for p in all_props if p.get("hr_correct"))
        hit_correct = sum(1 for p in all_props if p.get("hit_correct"))
        n = len(all_props)
        print(f"\n  Prop accuracy ({n} total):")
        print(f"    Hit predictions: {hit_correct/n*100:.0f}%")
        print(f"    HR predictions:  {hr_correct/n*100:.0f}%")

    generate_insights(data)
    print(f"\n{'='*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade model predictions vs actual results"
    )
    parser.add_argument("--date",    type=str,
                        help="Date to grade (default: yesterday)")
    parser.add_argument("--summary", action="store_true",
                        help="Show full model record and calibration")
    args = parser.parse_args()

    data = load_results()

    if args.summary:
        print_summary(data)
        return

    game_date = args.date or (date.today() - timedelta(days=1)).isoformat()

    # Load saved predictions
    pred_file = f"cache/predictions_{game_date}.json"
    try:
        with open(pred_file, encoding="utf-8") as f:
            predictions = json.load(f)
        logger.info(f"Loaded predictions from {pred_file}")
    except FileNotFoundError:
        logger.warning(f"No saved predictions for {game_date}")
        logger.warning(f"Run 'python save_predictions.py --date {game_date}' first")
        logger.warning("Grading game picks only (no prop comparison)")
        predictions = {"games": [], "batter_props": [], "pitcher_props": []}

    # Fetch actual results
    results = fetch_game_results(game_date)
    if not results:
        print(f"\n  No final results for {game_date} yet.")
        print(f"  Games may still be in progress.\n")
        return

    # Fetch player stats
    player_stats = fetch_player_stats_from_statcast(game_date)
    if player_stats.empty:
        logger.info("No Statcast data yet -- pulling from MLB API")
        # Could add MLB API fallback here later

    # Grade everything
    game_grades    = grade_games(predictions, results)
    prop_grades    = grade_props(predictions, player_stats)
    pitcher_grades = grade_pitchers(predictions, player_stats)

    # Print results
    print_graded_results(game_date, game_grades, prop_grades, pitcher_grades)

    # Save to results file
    correct = sum(1 for g in game_grades if g["correct"])
    total   = len(game_grades)

    data["games"].append({
        "date":    game_date,
        "correct": correct,
        "total":   total,
        "graded":  game_grades,
    })
    if prop_grades:
        data["props"].append({
            "date":          game_date,
            "graded_props":  prop_grades.get("graded_props", []),
            "calibration":   prop_grades.get("calibration", {}),
        })

    save_results(data)
    logger.success(f"Results saved to {RESULTS_FILE}")
    logger.info("Run 'python track_results.py --summary' for full record")


if __name__ == "__main__":
    main()