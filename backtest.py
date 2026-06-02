from __future__ import annotations
import argparse
from datetime import date, timedelta
import requests
import pandas as pd
from loguru import logger

MLB_API = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_games_from_api(start, end):
    games = []
    start_dt = date.fromisoformat(start)
    end_dt   = date.fromisoformat(end)
    current  = start_dt

    while current <= end_dt:
        try:
            r = requests.get(f"{MLB_API}/schedule", params={
                "sportId": 1,
                "date": current.isoformat(),
                "hydrate": "linescore,probablePitcher",
            }, headers=HEADERS, timeout=10)
            data = r.json()
            for date_entry in data.get("dates", []):
                for game in date_entry.get("games", []):
                    status = game.get("status", {}).get("abstractGameState", "")
                    if status != "Final":
                        continue
                    linescore  = game.get("linescore", {})
                    home_score = linescore.get("teams", {}).get("home", {}).get("runs")
                    away_score = linescore.get("teams", {}).get("away", {}).get("runs")
                    home_team  = game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation")
                    away_team  = game.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation")
                    home_sp    = game.get("teams", {}).get("home", {}).get("probablePitcher", {}).get("fullName", "TBD")
                    away_sp    = game.get("teams", {}).get("away", {}).get("probablePitcher", {}).get("fullName", "TBD")
                    if home_score is None or away_score is None:
                        continue
                    if home_score == 0 and away_score == 0:
                        continue
                    games.append({
                        "game_date": current.isoformat(),
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_sp":   home_sp,
                        "away_sp":   away_sp,
                        "home_score": home_score,
                        "away_score": away_score,
                    })
        except Exception as e:
            logger.debug(f"API error for {current}: {e}")
        current += timedelta(days=1)

    df = pd.DataFrame(games)
    if not df.empty:
        df["home_win"] = df["home_score"] > df["away_score"]
        df["run_diff"] = df["home_score"] - df["away_score"]
    logger.info(f"Loaded {len(df):,} games from {start} to {end}")
    return df


def run_backtest(start, end, min_conf=0.55):
    # Import the actual parlay builder prediction function
    try:
        from parlay_builder import calculate_win_probability_ensemble
    except ImportError as e:
        print(f"Could not import parlay_builder: {e}")
        return

    games = get_games_from_api(start, end)
    if games.empty:
        print("No games found.")
        return

    results = []
    print(f"Running backtest on {len(games):,} games using parlay_builder model...")

    for i, row in games.iterrows():
        try:
            home_prob, away_prob, details = calculate_win_probability_ensemble(
                home_team    = row["home_team"],
                away_team    = row["away_team"],
                home_pitcher = row["home_sp"],
                away_pitcher = row["away_sp"],
            )
        except Exception as e:
            logger.debug(f"Prediction failed {row['away_team']}@{row['home_team']}: {e}")
            home_prob = 0.52

        if home_prob >= 0.5:
            pick_team = row["home_team"]
            pick_prob = home_prob
            pick_home = True
            correct   = bool(row["home_win"])
        else:
            pick_team = row["away_team"]
            pick_prob = 1 - home_prob
            pick_home = False
            correct   = not bool(row["home_win"])

        results.append({
            "game_date": row["game_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "pick_team": pick_team,
            "pick_prob": pick_prob,
            "pick_home": pick_home,
            "correct":   correct,
            "run_diff":  abs(row["run_diff"]),
        })

        if (i + 1) % 50 == 0:
            done = sum(1 for r in results if r["correct"])
            print(f"  Progress: {len(results):,}/{len(games):,} — {done/len(results)*100:.1f}%")

    df = pd.DataFrame(results)

    print(f"\n{'='*72}")
    print(f"  BACKTEST RESULTS — {start} to {end}")
    print(f"  Total games: {len(df):,}")
    print(f"{'='*72}")
    print(f"  Overall: {df['correct'].mean()*100:.1f}% ({df['correct'].sum()}/{len(df)})")

    print(f"\n  By confidence tier:")
    tiers = [("75%+",0.75,1.0),("70-75%",0.70,0.75),("65-70%",0.65,0.70),
             ("60-65%",0.60,0.65),("55-60%",0.55,0.60),("<55%",0.0,0.55)]
    for label, lo, hi in tiers:
        t = df[(df["pick_prob"] >= lo) & (df["pick_prob"] < hi)]
        if len(t) == 0:
            continue
        wr = t["correct"].mean()
        action = "BET" if wr >= 0.58 else ("WATCH" if wr >= 0.53 else "SKIP")
        print(f"  {label:<12} {wr*100:>5.1f}%  {t['correct'].sum()}/{len(t)}  {action}")

    home_df = df[df["pick_home"]]
    away_df = df[~df["pick_home"]]
    print(f"\n  Home picks: {home_df['correct'].mean()*100:.1f}% ({len(home_df)} picks)")
    print(f"  Away picks: {away_df['correct'].mean()*100:.1f}% ({len(away_df)} picks)")

    print(f"\n  Best teams to pick (min 10 games):")
    ts = df.groupby("pick_team").agg(wr=("correct","mean"), n=("correct","count")).reset_index()
    ts = ts[ts["n"] >= 10].sort_values("wr", ascending=False)
    for _, r in ts.head(5).iterrows():
        print(f"  {r['pick_team']:<5} {r['wr']*100:.1f}% ({r['n']} games)")

    print(f"\n  Worst teams to pick (min 10 games):")
    for _, r in ts.tail(5).iterrows():
        print(f"  {r['pick_team']:<5} {r['wr']*100:.1f}% ({r['n']} games)")

    conf_df = df[df["pick_prob"] >= min_conf]
    if len(conf_df) > 0:
        print(f"\n  {min_conf*100:.0f}%+ confidence only: {conf_df['correct'].mean()*100:.1f}% ({len(conf_df)} games)")

    print(f"\n{'='*72}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season",   type=int)
    parser.add_argument("--start",    type=str)
    parser.add_argument("--end",      type=str)
    parser.add_argument("--min-conf", type=float, default=0.55)
    args = parser.parse_args()

    if args.season:
        start = f"{args.season}-03-20"
        end   = f"{args.season}-10-01"
    elif args.start and args.end:
        start, end = args.start, args.end
    else:
        start = "2026-03-20"
        end   = date.today().isoformat()

    run_backtest(start, end, args.min_conf)


if __name__ == "__main__":
    main()
