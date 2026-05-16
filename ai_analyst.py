# -*- coding: utf-8 -*-
"""
ai_analyst.py
-------------
Rule-based morning betting brief. No API needed, 100% free.
Reads predictions, weather, results history and generates
a plain English analysis.

Usage:
    python ai_analyst.py                  <- morning brief
    python ai_analyst.py --parlay         <- best parlay explanation
    python ai_analyst.py --calibrate      <- model improvement report
    python ai_analyst.py --query "PHI"    <- quick team lookup
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

Path("cache").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Load context
# ---------------------------------------------------------------------------

def load_context() -> dict:
    ctx = {"date": date.today().isoformat()}

    # Predictions
    pred_file = f"cache/predictions_{ctx['date']}.json"
    try:
        with open(pred_file, encoding="utf-8") as f:
            ctx["predictions"] = json.load(f)
    except FileNotFoundError:
        ctx["predictions"] = {}

    # Weather
    try:
        with open("cache/weather_today.json", encoding="utf-8") as f:
            ctx["weather"] = json.load(f)
    except FileNotFoundError:
        ctx["weather"] = {}

    # Results history
    try:
        with open("cache/prediction_results.json", encoding="utf-8") as f:
            results = json.load(f)
        games = results.get("games", [])
        ctx["all_games"]  = games
        ctx["all_graded"] = []
        for day in games:
            ctx["all_graded"].extend(day.get("graded", []))
    except FileNotFoundError:
        ctx["all_games"]  = []
        ctx["all_graded"] = []

    # Lineups
    try:
        from daily_lineup import (
            TODAYS_STARTERS, TODAYS_LINEUPS,
            CARRIED_LINEUP_TEAMS, NO_PITCHER_TEAMS
        )
        ctx["starters"]      = TODAYS_STARTERS
        ctx["lineups"]       = TODAYS_LINEUPS
        ctx["carried_teams"] = list(CARRIED_LINEUP_TEAMS) \
                               if hasattr(CARRIED_LINEUP_TEAMS, "__iter__") else []
        ctx["no_pitcher"]    = list(NO_PITCHER_TEAMS) \
                               if hasattr(NO_PITCHER_TEAMS, "__iter__") else []
    except ImportError:
        ctx["starters"]      = {}
        ctx["lineups"]       = {}
        ctx["carried_teams"] = []
        ctx["no_pitcher"]    = []

    return ctx


# ---------------------------------------------------------------------------
# Model record helpers
# ---------------------------------------------------------------------------

def get_model_record(ctx: dict) -> dict:
    games = ctx.get("all_graded", [])
    if not games:
        return {}

    total   = len(games)
    correct = sum(1 for g in games if g["correct"])

    # By confidence tier
    tiers = {
        "70%+":   [g for g in games if g["model_prob"] >= 0.70],
        "65-70%": [g for g in games if 0.65 <= g["model_prob"] < 0.70],
        "60-65%": [g for g in games if 0.60 <= g["model_prob"] < 0.65],
        "55-60%": [g for g in games if 0.55 <= g["model_prob"] < 0.60],
        "<55%":   [g for g in games if g["model_prob"] < 0.55],
    }

    def pct(lst):
        if not lst: return None
        c = sum(1 for g in lst if g["correct"])
        return {"correct": c, "total": len(lst),
                "pct": round(c / len(lst) * 100, 1)}

    return {
        "overall": {"correct": correct, "total": total,
                    "pct": round(correct / total * 100, 1)},
        "by_tier": {k: pct(v) for k, v in tiers.items() if v},
        "best_tier": max(
            ((k, pct(v)) for k, v in tiers.items() if v and len(v) >= 3),
            key=lambda x: x[1]["pct"] if x[1] else 0,
            default=("N/A", None)
        ),
    }


def get_team_recent_record(team: str, ctx: dict, n: int = 10) -> dict:
    """How has the model done picking this team recently."""
    graded = [g for g in ctx.get("all_graded", [])
              if team in g.get("matchup", "")]
    recent = graded[-n:]
    if not recent:
        return {}
    correct = sum(1 for g in recent if g["correct"])
    return {"correct": correct, "total": len(recent),
            "pct": round(correct / len(recent) * 100, 1)}


# ---------------------------------------------------------------------------
# Game analysis helpers
# ---------------------------------------------------------------------------

def get_top_games(ctx: dict, min_conf: float = 0.65) -> list[dict]:
    games = ctx.get("predictions", {}).get("games", [])
    top   = [g for g in games if g.get("confidence", 0) >= min_conf]
    return sorted(top, key=lambda x: x.get("confidence", 0), reverse=True)


def get_weather_alerts(ctx: dict) -> list[str]:
    alerts = []
    weather = ctx.get("weather", {})
    for team, w in weather.items():
        lean = w.get("betting_lean", "")
        temp = w.get("temp_f", 70)
        wind = w.get("wind_mph", 0)
        if "UNDER" in lean or "pitcher" in lean.lower():
            alerts.append(f"{team}: {w.get('park','')[:20]} "
                          f"-- {temp}°F, {wind}mph -- PITCHER PARK today")
        elif "OVER" in lean or "hitter" in lean.lower():
            alerts.append(f"{team}: {w.get('park','')[:20]} "
                          f"-- {temp}°F, {wind}mph -- HITTER PARK today")
    return alerts


def get_stale_lineup_warnings(ctx: dict, top_games: list[dict]) -> list[str]:
    warnings = []
    carried  = ctx.get("carried_teams", [])
    for g in top_games:
        home = g.get("home", "")
        away = g.get("away", "")
        if home in carried:
            warnings.append(f"{home} lineup carried from yesterday -- verify before betting props")
        if away in carried:
            warnings.append(f"{away} lineup carried from yesterday -- verify before betting props")
    return warnings


def analyze_pitcher_matchup(home_sp: str, away_sp: str,
                             home: str, away: str, ctx: dict) -> str:
    """Generate a one-line pitcher matchup note."""
    from parlay_builder import get_pitcher_rolling_stats, PITCHER_ERA_FALLBACK
    home_stats = get_pitcher_rolling_stats(home_sp)
    away_stats = get_pitcher_rolling_stats(away_sp)

    home_era = PITCHER_ERA_FALLBACK.get(home_sp, 100)
    away_era = PITCHER_ERA_FALLBACK.get(away_sp, 100)

    home_k   = home_stats.get("k_pct", 0) if home_stats else 0
    away_k   = away_stats.get("k_pct", 0) if away_stats else 0

    notes = []
    if home_era >= 130:
        notes.append(f"{home_sp} is an elite arm (ERA+ {home_era})")
    elif home_era <= 90:
        notes.append(f"{home_sp} is a liability (ERA+ {home_era})")

    if away_era >= 130:
        notes.append(f"{away_sp} is an elite arm (ERA+ {away_era})")
    elif away_era <= 90:
        notes.append(f"{away_sp} is a liability (ERA+ {away_era})")

    if home_k >= 0.28:
        notes.append(f"{home_sp} is a strikeout machine ({home_k*100:.0f}% K rate)")
    if away_k >= 0.28:
        notes.append(f"{away_sp} is a strikeout machine ({away_k*100:.0f}% K rate)")

    return " | ".join(notes) if notes else "Average pitching matchup"


def get_avoid_list(ctx: dict) -> list[str]:
    """Games to avoid — low confidence, stale data, model disagreement."""
    games   = ctx.get("predictions", {}).get("games", [])
    carried = set(ctx.get("carried_teams", []))
    avoid   = []

    for g in games:
        conf     = g.get("confidence", 0)
        home     = g.get("home", "")
        away     = g.get("away", "")
        matchup  = g.get("matchup", "")
        std_dev  = g.get("std_dev", 0)
        agreement= g.get("agreement", "MED")

        reasons = []
        if conf < 0.55:
            reasons.append("coin flip (<55%)")
        if 0.55 <= conf < 0.60:
            reasons.append("model historically poor in this range (33%)")
        if agreement == "LOW" or std_dev > 0.08:
            reasons.append("models disagree (high variance)")
        if home in carried or away in carried:
            reasons.append("stale lineup data")

        if reasons:
            avoid.append(f"{matchup}: {', '.join(reasons)}")

    return avoid[:5]


def get_top_props(ctx: dict) -> dict:
    """Get top HR and hit predictions."""
    props    = ctx.get("predictions", {}).get("batter_props", [])
    top_hr   = sorted(props, key=lambda x: x.get("hr_prob",  0), reverse=True)[:5]
    top_hits = sorted(props, key=lambda x: x.get("hit_prob", 0), reverse=True)[:5]
    top_k    = ctx.get("predictions", {}).get("pitcher_props", [])
    top_k    = sorted(top_k, key=lambda x: x.get("exp_k", 0), reverse=True)[:3]
    return {"hr": top_hr, "hits": top_hits, "k": top_k}


# ---------------------------------------------------------------------------
# Morning brief
# ---------------------------------------------------------------------------

def morning_brief() -> None:
    ctx       = load_context()
    record    = get_model_record(ctx)
    top_games = get_top_games(ctx, min_conf=0.65)
    all_games = ctx.get("predictions", {}).get("games", [])
    weather   = get_weather_alerts(ctx)
    warnings  = get_stale_lineup_warnings(ctx, top_games)
    avoid     = get_avoid_list(ctx)
    props     = get_top_props(ctx)

    print(f"\n{'='*65}")
    print(f"  MLB BETTING BRIEF  --  {ctx['date']}")
    print(f"{'='*65}")

    # Model record
    if record:
        o = record["overall"]
        print(f"\n  MODEL RECORD: {o['correct']}/{o['total']} ({o['pct']}%)")
        print(f"  By confidence tier:")
        for tier, stats in record.get("by_tier", {}).items():
            if stats:
                bar = "✓" * int(stats["pct"] / 10) + "·" * (10 - int(stats["pct"] / 10))
                flag = " ← BET HERE" if stats["pct"] >= 65 and stats["total"] >= 3 else \
                       " ← AVOID"    if stats["pct"] < 50 and stats["total"] >= 3 else ""
                print(f"    {tier:<10} {stats['correct']}/{stats['total']} "
                      f"({stats['pct']}%)  {bar}{flag}")

    # Best bets
    print(f"\n  {'='*61}")
    print(f"  BEST BETS TODAY")
    print(f"  {'='*61}")

    if top_games:
        for i, g in enumerate(top_games, 1):
            conf     = g.get("confidence", 0) * 100
            pick     = g.get("model_pick", "")
            matchup  = g.get("matchup", "")
            home_sp  = g.get("home_sp", "TBD")
            away_sp  = g.get("away_sp", "TBD")
            home     = g.get("home", "")
            away     = g.get("away", "")
            agreement= g.get("agreement", "MED")
            agree_str= "ALL 4 MODELS AGREE" if agreement == "HIGH" else \
                       "3/4 models agree"    if agreement == "MED"  else \
                       "models DISAGREE -- risky"

            # Team recent record
            team_rec = get_team_recent_record(pick, ctx)
            rec_str  = f"  (model {team_rec['correct']}/{team_rec['total']} "  \
                       f"picking {pick} recently)" if team_rec else ""

            pitcher_note = analyze_pitcher_matchup(
                home_sp, away_sp, home, away, ctx
            )

            print(f"\n  #{i} {pick} -- {conf:.1f}% confidence")
            print(f"     Game:    {matchup}")
            print(f"     Pitchers:{away_sp} vs {home_sp}")
            print(f"     Pitcher: {pitcher_note}")
            print(f"     Ensemble:{agree_str}{rec_str}")

            # Weather for this game
            home_weather = ctx.get("weather", {}).get(home, {})
            if home_weather and home_weather.get("betting_lean", "No data") != "No data":
                print(f"     Weather: {home_weather.get('betting_lean','')}")
    else:
        print(f"\n  No games above 65% confidence today.")
        print(f"  Games closest to 65%:")
        near = sorted(all_games,
                      key=lambda x: x.get("confidence", 0),
                      reverse=True)[:3]
        for g in near:
            print(f"    {g.get('model_pick',''):<20} "
                  f"{g.get('confidence',0)*100:.1f}%  {g.get('matchup','')}")

    # Player props
    print(f"\n  {'='*61}")
    print(f"  TOP PROPS")
    print(f"  {'='*61}")

    if props["hr"]:
        print(f"\n  HR picks:")
        for p in props["hr"][:3]:
            weather_home = ctx.get("weather", {}).get(
                p.get("game", "").split("@ ")[-1][:3] if "@ " in p.get("game","") else "", {}
            )
            w_note = " 🌬️ wind out" if "hitter" in weather_home.get("betting_lean","").lower() else ""
            print(f"    {p['player']:<22} {p['hr_prob']*100:.1f}% HR  "
                  f"vs {p['vs_pitcher']}{w_note}")

    if props["hits"]:
        print(f"\n  Hit picks:")
        for p in props["hits"][:3]:
            print(f"    {p['player']:<22} {p['hit_prob']*100:.1f}% hit  "
                  f"vs {p['vs_pitcher']}")

    if props["k"]:
        print(f"\n  Pitcher K picks:")
        for p in props["k"][:3]:
            print(f"    {p['pitcher']:<22} {p['exp_k']:.1f} exp K  "f"vs {p['vs_team']}  "f"6K+: {p['k6_prob']*100:.0f}%")

    # Weather alerts
    if weather:
        print(f"\n  {'='*61}")
        print(f"  WEATHER ALERTS")
        print(f"  {'='*61}")
        for w in weather:
            print(f"    {w}")

    # Lineup warnings
    if warnings:
        print(f"\n  {'='*61}")
        print(f"  LINEUP WARNINGS")
        print(f"  {'='*61}")
        for w in warnings:
            print(f"    ⚠  {w}")

    # Avoid list
    if avoid:
        print(f"\n  {'='*61}")
        print(f"  GAMES TO AVOID")
        print(f"  {'='*61}")
        for a in avoid[:4]:
            print(f"    ✗  {a}")

    print(f"\n{'='*65}")
    print(f"  Run 'python ai_analyst.py --parlay' for parlay breakdown")
    print(f"  Run 'python ai_analyst.py --calibrate' for model insights")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Parlay explainer
# ---------------------------------------------------------------------------

def explain_parlay(skip=None, min_conf=0.65, n_legs=3) -> None:
    ctx       = load_context()
    top_games = get_top_games(ctx, min_conf=min_conf)
    # If not enough legs, lower threshold to fill
    if len([g for g in top_games if not any(s.upper() in g.get("matchup","").upper() for s in (skip or []))]) < n_legs:
        top_games = get_top_games(ctx, min_conf=0.625)

    # Filter skipped teams
    if skip:
        top_games = [g for g in top_games
                     if not any(s.upper() in g.get("matchup","").upper()
                                for s in skip)]

    if not top_games:
        print("\n  No qualifying games (65%+) for parlay today.")
        return

    legs = top_games[:n_legs]

    print(f"\n{'='*65}")
    print(f"  PARLAY BREAKDOWN  --  {ctx['date']}")
    print(f"{'='*65}")
    print(f"\n  Recommended legs:")

    combined_prob = 1.0
    for i, g in enumerate(legs, 1):
        prob     = g.get("confidence", 0.5)
        pick     = g.get("model_pick", "")
        matchup  = g.get("matchup", "")
        home_sp  = g.get("home_sp", "")
        away_sp  = g.get("away_sp", "")
        agreement= g.get("agreement", "MED")
        combined_prob *= prob

        print(f"\n  Leg {i}: {pick} ({prob*100:.1f}%)")
        print(f"    Matchup: {matchup}")
        print(f"    SP: {away_sp} @ {home_sp}")

        # Risk assessment
        if agreement == "HIGH":
            risk = "LOW RISK — all 4 models agree"
        elif agreement == "MED":
            risk = "MEDIUM RISK — most models agree"
        else:
            risk = "HIGH RISK — models disagree, consider dropping this leg"
        print(f"    Risk: {risk}")

        # Historical accuracy for this team
        team_rec = get_team_recent_record(pick, ctx)
        if team_rec and team_rec["total"] >= 3:
            print(f"    Model record picking {pick}: "f"{team_rec['correct']}/{team_rec['total']} "f"({team_rec['pct']}%)")

        # Weather
        home = g.get("home", "")
        home_w = ctx.get("weather", {}).get(home, {})
        lean   = home_w.get("betting_lean", "")
        if lean and lean != "No data":
            print(f"    Weather: {lean}")

    print(f"\n  Combined win probability: {combined_prob*100:.1f}%")

    # Correlation check
    homes = [g.get("home") for g in legs]
    aways = [g.get("away") for g in legs]
    all_teams = set(homes + aways)

    print(f"\n  Correlation notes:")
    # Same division games
    divisions = {
        "NL East":  {"NYM","PHI","ATL","MIA","WSH"},
        "NL Central":{"CHC","MIL","STL","CIN","PIT"},
        "NL West":  {"LAD","SF","SD","ARI","COL"},
        "AL East":  {"NYY","BOS","BAL","TOR","TB"},
        "AL Central":{"CLE","MIN","CWS","KC","DET"},
        "AL West":  {"HOU","TEX","SEA","LAA","OAK","ATH"},
    }
    for div, teams in divisions.items():
        overlap = all_teams & teams
        if len(overlap) >= 2:
            print(f"    Multiple {div} teams -- results may correlate")

    if combined_prob >= 0.25:
        print(f"\n  VERDICT: Solid parlay — {combined_prob*100:.1f}% win probability")
    elif combined_prob >= 0.15:
        print(f"\n  VERDICT: Risky parlay — consider 2-leg instead")
    else:
        print(f"\n  VERDICT: Long shot — only play with small stake")

    print(f"\n{'='*65}\n")


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------

def calibration_report() -> None:
    ctx    = load_context()
    record = get_model_record(ctx)
    graded = ctx.get("all_graded", [])

    if not graded:
        print("\n  No results data yet. Run track_results.py first.\n")
        return

    print(f"\n{'='*65}")
    print(f"  MODEL CALIBRATION REPORT")
    print(f"{'='*65}")

    # Overall
    o = record.get("overall", {})
    print(f"\n  Overall: {o.get('correct','?')}/{o.get('total','?')} "
          f"({o.get('pct','?')}%)")

    # Tier breakdown with recommendations
    print(f"\n  Confidence tier analysis:")
    tiers = record.get("by_tier", {})
    for tier, stats in tiers.items():
        if not stats:
            continue
        pct  = stats["pct"]
        n    = stats["total"]
        flag = ""
        if pct >= 68 and n >= 5:
            flag = "  ✓ RELIABLE — bet this tier"
        elif pct >= 55 and n >= 5:
            flag = "  ~ MARGINAL — proceed with caution"
        elif n >= 5:
            flag = "  ✗ AVOID — model underperforms here"
        elif n < 5:
            flag = "  (small sample)"
        print(f"    {tier:<10} {pct:.0f}%  (n={n}){flag}")

    # Home vs away bias
    home_picks = [g for g in graded if g["model_pick"] == g["matchup"].split(" @ ")[1].strip()
                  if " @ " in g["matchup"]]
    away_picks = [g for g in graded if g["model_pick"] == g["matchup"].split(" @ ")[0].strip()
                  if " @ " in g["matchup"]]

    if home_picks and away_picks:
        home_pct = sum(1 for g in home_picks if g["correct"]) / len(home_picks) * 100
        away_pct = sum(1 for g in away_picks if g["correct"]) / len(away_picks) * 100
        print(f"\n  Home vs Away bias:")
        print(f"    Picking home team: {home_pct:.0f}% ({len(home_picks)} picks)")
        print(f"    Picking away team: {away_pct:.0f}% ({len(away_picks)} picks)")
        if home_pct > away_pct + 10:
            print(f"    → Model better at picking home teams")
        elif away_pct > home_pct + 10:
            print(f"    → Model better at picking away teams")

    # Most wrong teams
    team_wrong = {}
    for g in graded:
        if not g["correct"]:
            pick = g["model_pick"]
            team_wrong[pick] = team_wrong.get(pick, 0) + 1
    if team_wrong:
        worst = sorted(team_wrong.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n  Teams model most often gets wrong:")
        for team, count in worst:
            print(f"    {team}: wrong {count} times")

    # Improvement suggestions
    print(f"\n  Improvement suggestions:")
    best_tier  = record.get("best_tier", ("N/A", None))
    worst_pct  = min((v["pct"] for v in tiers.values() if v and v["total"] >= 3),
                     default=50)

    if o.get("pct", 0) < 55:
        print(f"    1. Model needs more 2026 data — retrain weekly")
        print(f"    2. Increase fallback weight until 60+ games played")
    if worst_pct < 40:
        print(f"    3. Skip 55-65% picks entirely — historically losing money")
    print(f"    4. Run 'python train_props.py --seasons 2025 2026' weekly")
    print(f"    5. Run daily pipeline to keep rolling features fresh")
    print(f"       python -m mlb_analytics.ingestion.statcast_ingest "
          f"--start YESTERDAY --end TODAY")

    # Bet reset token parlay analysis

    print(f"\n{'='*65}\n")


# ---------------------------------------------------------------------------
# Team query
# ---------------------------------------------------------------------------

def team_query(query: str) -> None:
    ctx   = load_context()
    query = query.upper()

    print(f"\n{'='*65}")
    print(f"  TEAM LOOKUP: {query}")
    print(f"{'='*65}")

    # Find today's game
    games = ctx.get("predictions", {}).get("games", [])
    team_games = [g for g in games
                  if query in g.get("matchup", "").upper()]

    if team_games:
        g = team_games[0]
        print(f"\n  Today: {g['matchup']}")
        print(f"  Model pick: {g['model_pick']} ({g['confidence']*100:.1f}%)")
        print(f"  Pitchers: {g.get('away_sp','')} vs {g.get('home_sp','')}")
        print(f"  Agreement: {g.get('agreement','?')}")
    else:
        print(f"\n  {query} not found in today's games.")

    # Historical record
    rec = get_team_recent_record(query, ctx, n=20)
    if rec:
        print(f"\n  Model record picking {query} (last 20): "
              f"{rec['correct']}/{rec['total']} ({rec['pct']}%)")

    # Recent results involving this team
    recent = [g for g in ctx.get("all_graded", [])
              if query in g.get("matchup", "")][-5:]
    if recent:
        print(f"\n  Recent games:")
        for g in recent:
            icon = "[OK]" if g["correct"] else "[X] "
            print(f"    {icon} {g['matchup']:<30} "f"Picked {g['model_pick']} ({g['model_prob']*100:.0f}%)  "f"→ {g['actual_winner']} {g['score']}")

    # Bet reset token parlay analysis
    if reset_token and len(hr) >= n_legs:
        print(f"\n  BET RESET TOKEN ANALYSIS (${reset_token:.0f} token, {n_legs}-leg HR parlay):")
        print(f"  {'='*55}")

        top_legs = hr[:n_legs]
        # Combined probability all hit
        combined_prob = 1.0
        for p in top_legs:
            combined_prob *= p["hr_prob"]

        # Typical FanDuel HR parlay odds
        # Each HR leg ~+350 to +500, parlay multiplies
        # Estimate: 3-leg HR parlay pays roughly +2000 to +4000
        leg_decimal = [1 + (350/100) for _ in top_legs]  # assume +350 each
        parlay_decimal = 1.0
        for d in leg_decimal:
            parlay_decimal *= d
        payout_estimate = reset_token * (parlay_decimal - 1)

        # EV without reset token
        ev_normal = combined_prob * payout_estimate - (1 - combined_prob) * reset_token

        # EV WITH reset token
        # If any single leg fails, you get stake back
        # Simplified: prob of at least 1 hitting * partial value + prob all hit * full value
        prob_all_hit    = combined_prob
        prob_none_hit   = 1.0
        for p in top_legs:
            prob_none_hit *= (1 - p["hr_prob"])
        prob_partial    = 1 - prob_all_hit - prob_none_hit

        # With reset: lose nothing if miss, win big if hit
        ev_with_reset = prob_all_hit * payout_estimate - prob_none_hit * 0 - prob_partial * 0
        # Reset means you get $reset_token back on any loss
        ev_with_reset_adjusted = ev_with_reset  # net gain since stake returned on loss

        print(f"  Legs:")
        for i, p in enumerate(top_legs, 1):
            print(f"    {i}. {p['player']:<22} {p['hr_prob']*100:.1f}% HR prob")

        print(f"\n  Combined probability all hit: {combined_prob*100:.2f}%")
        print(f"  Estimated payout on ${reset_token:.0f}: ~${payout_estimate:.0f}")
        print(f"\n  WITHOUT reset token:")
        print(f"    EV: ${ev_normal:+.2f}")
        print(f"    Expected: lose ${reset_token:.0f} x {(1-combined_prob)*100:.1f}% of time")
        print(f"\n  WITH ${reset_token:.0f} reset token (stake returned on loss):")
        print(f"    EV: ${ev_with_reset_adjusted:+.2f}  <-- this is your real edge")
        print(f"    Risk: $0 (token covers stake if miss)")
        print(f"    Upside: ~${payout_estimate:.0f} if all {n_legs} HR")
        print(f"\n  VERDICT: ", end="")
        if ev_with_reset_adjusted > 0:
            print(f"PLACE THIS BET -- positive EV with reset token!")
        else:
            print(f"Marginal -- but reset token removes all downside risk")

    print(f"\n{'='*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def hr_analysis(reset_token: float = None, n_legs: int = 3) -> None:
    """Analyze today's HR predictions with weather and matchup context.
    If reset_token is set, calculates EV with bet reset protection.
    """
    ctx   = load_context()
    props = get_top_props(ctx)
    hr    = props.get("hr", [])

    if not hr:
        print("\n  No HR predictions available. Run save_predictions.py first.\n")
        return

    print(f"\n{'='*65}")
    print(f"  HR PREDICTION ANALYSIS -- {ctx['date']}")
    print(f"{'='*65}")
    print(f"\n  {'Player':<22} {'Team':<5} {'vs Pitcher':<22} {'HR%':>5} {'Hit%':>6} {'Note'}")
    print(f"  {'-'*75}")

    for i, p in enumerate(hr[:10], 1):
        player  = p["player"]
        team    = p["team"]
        pitcher = p["vs_pitcher"]
        hr_pct  = p["hr_prob"] * 100
        hit_pct = p["hit_prob"] * 100
        game    = p.get("game", "")
        home_t  = game.split("@ ")[-1].strip()[:3] if "@ " in game else ""
        weather = ctx.get("weather", {}).get(home_t, {})
        w_lean  = weather.get("betting_lean", "")
        temp    = weather.get("temp_f", 70)
        w_note  = "wind out" if "hitter" in w_lean.lower() else                   "cold"     if temp < 50 else                   "warm"     if temp > 75 else ""
        conf    = "FIRE" if hr_pct >= 9 else "BOMB" if hr_pct >= 7 else "----"
        print(f"  {i:<3} {player:<22} {team:<5} {pitcher:<22} "
              f"{hr_pct:>4.1f}% {hit_pct:>5.1f}% [{conf}] {w_note}")

    print(f"\n  FAIR ODDS (what FanDuel needs to offer for +EV):")
    for p in hr[:5]:
        prob      = p["hr_prob"]
        fair      = int((1 / prob - 1) * 100)
        print(f"    {p['player']:<22} {prob*100:.1f}% model  --> need +{fair} or better")

    k_props = props.get("k", [])
    if k_props:
        print(f"\n  PITCHER K PROPS:")
        for p in k_props[:5]:
            print(f"    {p['pitcher']:<22} vs {p['vs_team']:<5} "
                  f"ExpK:{p['exp_k']:.1f}  "
                  f"5K+:{p['k5_prob']*100:.0f}%  "
                  f"6K+:{p['k6_prob']*100:.0f}%  "
                  f"7K+:{p['k7_prob']*100:.0f}%")

    # Bet reset token parlay analysis
    if reset_token and len(hr) >= n_legs:
        print(f"\n  BET RESET TOKEN ANALYSIS (${reset_token:.0f} token, {n_legs}-leg HR parlay):")
        print(f"  {'='*55}")

        top_legs = hr[:n_legs]
        # Combined probability all hit
        combined_prob = 1.0
        for p in top_legs:
            combined_prob *= p["hr_prob"]

        # Typical FanDuel HR parlay odds
        # Each HR leg ~+350 to +500, parlay multiplies
        # Estimate: 3-leg HR parlay pays roughly +2000 to +4000
        leg_decimal = [1 + (350/100) for _ in top_legs]  # assume +350 each
        parlay_decimal = 1.0
        for d in leg_decimal:
            parlay_decimal *= d
        payout_estimate = reset_token * (parlay_decimal - 1)

        # EV without reset token
        ev_normal = combined_prob * payout_estimate - (1 - combined_prob) * reset_token

        # EV WITH reset token
        # If any single leg fails, you get stake back
        # Simplified: prob of at least 1 hitting * partial value + prob all hit * full value
        prob_all_hit    = combined_prob
        prob_none_hit   = 1.0
        for p in top_legs:
            prob_none_hit *= (1 - p["hr_prob"])
        prob_partial    = 1 - prob_all_hit - prob_none_hit

        # With reset: lose nothing if miss, win big if hit
        ev_with_reset = prob_all_hit * payout_estimate - prob_none_hit * 0 - prob_partial * 0
        # Reset means you get $reset_token back on any loss
        ev_with_reset_adjusted = ev_with_reset  # net gain since stake returned on loss

        print(f"  Legs:")
        for i, p in enumerate(top_legs, 1):
            print(f"    {i}. {p['player']:<22} {p['hr_prob']*100:.1f}% HR prob")

        print(f"\n  Combined probability all hit: {combined_prob*100:.2f}%")
        print(f"  Estimated payout on ${reset_token:.0f}: ~${payout_estimate:.0f}")
        print(f"\n  WITHOUT reset token:")
        print(f"    EV: ${ev_normal:+.2f}")
        print(f"    Expected: lose ${reset_token:.0f} x {(1-combined_prob)*100:.1f}% of time")
        print(f"\n  WITH ${reset_token:.0f} reset token (stake returned on loss):")
        print(f"    EV: ${ev_with_reset_adjusted:+.2f}  <-- this is your real edge")
        print(f"    Risk: $0 (token covers stake if miss)")
        print(f"    Upside: ~${payout_estimate:.0f} if all {n_legs} HR")
        print(f"\n  VERDICT: ", end="")
        if ev_with_reset_adjusted > 0:
            print(f"PLACE THIS BET -- positive EV with reset token!")
        else:
            print(f"Marginal -- but reset token removes all downside risk")

    print(f"\n{'='*65}\n")


def mega_parlay(stake: float = 1.0, skip: list = None) -> None:
    """Generate a mega parlay with every game. For fun only."""
    ctx       = load_context()
    all_games = ctx.get("predictions", {}).get("games", [])

    if skip:
        all_games = [g for g in all_games
                     if not any(s.upper() in g.get("matchup","").upper()
                                for s in skip)]

    if not all_games:
        print("\n  No games found.\n")
        return

    print(f"\n{'='*65}")
    print(f"  MEGA PARLAY -- {ctx['date']}  (${stake:.0f} for the memes)")
    print(f"{'='*65}")
    print(f"  Picking model favorite in every game:\n")

    combined_prob = 1.0
    combined_odds = 1.0
    legs = []

    for g in all_games:
        pick      = g.get("model_pick", "")
        prob      = g.get("confidence", 0.5)
        matchup   = g.get("matchup", "")
        home_sp   = g.get("home_sp", "")
        away_sp   = g.get("away_sp", "")
        combined_prob *= prob
        # Estimate decimal odds from model prob
        est_odds  = 1.0 / prob if prob > 0 else 2.0
        combined_odds *= est_odds
        legs.append((pick, prob, matchup))
        print(f"  {pick:<22} {prob*100:.0f}%  ({matchup})")

    # Estimate payout
    payout = stake * (combined_odds - 1)

    print(f"\n  Total legs: {len(legs)}")
    print(f"  Combined win probability: {combined_prob*100:.4f}%")
    print(f"  Estimated payout on ${stake:.0f}: ~${payout:,.0f}")
    print(f"  Implied odds: roughly +{int((combined_odds-1)*100):,}")
    print(f"\n  Realistic chance of hitting: about 1 in {int(1/combined_prob):,}")

    days_needed = int(1 / combined_prob)
    print(f"  At 1 bet per day you'd need ~{days_needed:,} days ({days_needed//365:,} years)")
    print(f"\n  VERDICT: Almost certainly loses. But if it hits... holy shit.")
    print(f"{'='*65}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLB betting analyst (free, no API needed)"
    )
    parser.add_argument("--parlay",    action="store_true",
                        help="Explain today's top parlay")
    parser.add_argument("--calibrate", action="store_true",
                        help="Model calibration report")
    parser.add_argument("--query",     type=str,
                        help="Look up a team (e.g. --query PHI)")
    parser.add_argument("--hr",        action="store_true",
                        help="Analyze today's HR predictions")
    parser.add_argument("--skip",      type=str, nargs="*", default=[],
                        help="Teams to skip (e.g. --skip NYM COL)")
    parser.add_argument("--mega",      action="store_true",
                        help="Generate a mega parlay with every game (for fun)")
    parser.add_argument("--stake",     type=float, default=1.0,
                        help="Stake amount for mega parlay (default $1)")
    parser.add_argument("--reset",     type=float, default=None,
                        help="Bet reset token value (e.g. --reset 10)")
    parser.add_argument("--legs",      type=int, default=3,
                        help="Number of parlay legs (default 3)")
    args = parser.parse_args()

    if args.mega:
        mega_parlay(stake=args.stake, skip=args.skip)
    elif args.parlay:
        explain_parlay(skip=args.skip, n_legs=3)
    elif args.calibrate:
        calibration_report()
    elif args.query:
        team_query(args.query)
    elif args.hr:
        hr_analysis(reset_token=args.reset, n_legs=args.legs)
    else:
        morning_brief()


if __name__ == "__main__":
    main()