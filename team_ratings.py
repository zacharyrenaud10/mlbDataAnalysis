# team_ratings.py
# ----------------
# Calculates Elo ratings and Pythagorean win% for all 30 MLB teams
# using 2026 game results from the MLB Stats API.
#
# Usage:
#     python team_ratings.py
#     python team_ratings.py --start 2026-03-20

from __future__ import annotations
import argparse
import json
import math
import requests
from datetime import date, timedelta
from pathlib import Path
from loguru import logger

Path("cache").mkdir(exist_ok=True)

MLB_API = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Starting Elo for all teams
DEFAULT_ELO = 1500.0
K_FACTOR    = 20.0  # how fast Elo updates


def expected_win(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(elo_a, elo_b, a_won, run_diff):
    """Update Elo with margin of victory multiplier."""
    exp_a  = expected_win(elo_a, elo_b)
    result = 1.0 if a_won else 0.0
    # Margin of victory multiplier (log scale, caps at ~2.0)
    mov    = math.log(abs(run_diff) + 1) * 0.5 + 1.0
    mov    = min(mov, 2.0)
    delta  = K_FACTOR * mov * (result - exp_a)
    return elo_a + delta, elo_b - delta


def fetch_season_results(start, end):
    """Fetch all final game results between two dates."""
    games  = []
    cur    = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)

    while cur <= end_dt:
        try:
            r = requests.get(f"{MLB_API}/schedule", params={
                "sportId": 1,
                "date":    cur.isoformat(),
                "hydrate": "linescore",
            }, headers=HEADERS, timeout=10)
            data = r.json()
            for d in data.get("dates", []):
                for game in d.get("games", []):
                    if game.get("status", {}).get("abstractGameState") != "Final":
                        continue
                    ls   = game.get("linescore", {})
                    hr   = game.get("teams",{}).get("home",{}).get("score")
                    ar   = game.get("teams",{}).get("away",{}).get("score")
                    TEAM_ID_MAP = {
                        133:"OAK",134:"PIT",135:"SD",136:"SEA",137:"SF",
                        138:"STL",139:"TB",140:"TEX",141:"TOR",142:"MIN",
                        143:"PHI",144:"ATL",145:"CWS",146:"MIA",147:"NYY",
                        158:"MIL",108:"LAA",109:"ARI",110:"BAL",111:"BOS",
                        112:"CHC",113:"CIN",114:"CLE",115:"COL",116:"DET",
                        117:"HOU",118:"KC",119:"LAD",120:"WSH",121:"NYM",
                    }
                    home_id = game.get("teams",{}).get("home",{}).get("team",{}).get("id")
                    away_id = game.get("teams",{}).get("away",{}).get("team",{}).get("id")
                    home = TEAM_ID_MAP.get(home_id)
                    away = TEAM_ID_MAP.get(away_id)
                    if hr is None or ar is None or not home or not away:
                        continue
                    if hr == 0 and ar == 0:
                        continue
                    games.append({
                        "date":       cur.isoformat(),
                        "home":       home,
                        "away":       away,
                        "home_runs":  hr,
                        "away_runs":  ar,
                        "home_win":   hr > ar,
                    })
        except Exception as e:
            logger.debug(f"API error {cur}: {e}")
        cur += timedelta(days=1)

    logger.info(f"Fetched {len(games):,} completed games")
    return games


def calculate_ratings(games):
    """Calculate Elo and Pythagorean ratings from game results."""
    elo      = {}
    runs_for = {}
    runs_ag  = {}
    wins     = {}
    losses   = {}

    for game in games:
        home = game["home"]
        away = game["away"]
        hr   = game["home_runs"]
        ar   = game["away_runs"]

        # Initialize
        for t in [home, away]:
            if t not in elo:
                elo[t]      = DEFAULT_ELO
                runs_for[t] = 0
                runs_ag[t]  = 0
                wins[t]     = 0
                losses[t]   = 0

        # Update Elo
        h_won = game["home_win"]
        rd    = abs(hr - ar)
        elo[home], elo[away] = update_elo(elo[home], elo[away], h_won, rd)

        # Update run totals
        runs_for[home] += hr
        runs_ag[home]  += ar
        runs_for[away] += ar
        runs_ag[away]  += hr

        # Update W/L
        if h_won:
            wins[home]   += 1
            losses[away] += 1
        else:
            wins[away]   += 1
            losses[home] += 1

    # Calculate Pythagorean win%
    pythag = {}
    for team in elo:
        rf = runs_for.get(team, 1)
        ra = runs_ag.get(team,  1)
        # Pythagorean formula (exponent 1.83 for baseball)
        pythag[team] = rf**1.83 / (rf**1.83 + ra**1.83)

    # Actual win%
    actual_wpct = {}
    for team in elo:
        w = wins.get(team, 0)
        l = losses.get(team, 0)
        actual_wpct[team] = w / max(w + l, 1)

    return elo, pythag, actual_wpct, runs_for, runs_ag, wins, losses


def save_ratings(elo, pythag, actual_wpct, runs_for, runs_ag, wins, losses):
    ratings = {}
    for team in elo:
        w = wins.get(team, 0)
        l = losses.get(team, 0)
        ratings[team] = {
            "elo":          round(elo[team], 1),
            "pythag_wpct":  round(pythag.get(team, 0.500), 4),
            "actual_wpct":  round(actual_wpct.get(team, 0.500), 4),
            "luck":         round(actual_wpct.get(team, 0.500) - pythag.get(team, 0.500), 4),
            "wins":         w,
            "losses":       l,
            "runs_for":     runs_for.get(team, 0),
            "runs_against": runs_ag.get(team, 0),
        }

    out = {
        "as_of":   date.today().isoformat(),
        "teams":   ratings,
    }
    with open("cache/team_ratings.json", "w") as f:
        json.dump(out, f, indent=2)

    # Print leaderboard
    sorted_teams = sorted(ratings.items(), key=lambda x: x[1]["elo"], reverse=True)
    print(f"\n{'='*65}")
    print(f"  MLB ELO RATINGS + PYTHAGOREAN WIN%  —  {date.today()}")
    print(f"{'='*65}")
    print(f"  {'#':<3} {'Team':<6} {'Elo':>6}  {'W-L':>7}  {'Act%':>5}  {'Pyth%':>5}  {'Luck':>6}")
    print(f"  {'-'*60}")
    for i, (team, r) in enumerate(sorted_teams, 1):
        luck_str = f"{r['luck']:+.3f}"
        print(f"  {i:<3} {team:<6} {r['elo']:>6.0f}  "
              f"{r['wins']:>3}-{r['losses']:<3}  "
              f"{r['actual_wpct']*100:>4.1f}%  "
              f"{r['pythag_wpct']*100:>4.1f}%  "
              f"{luck_str:>6}")
    print(f"\n  Luck = actual - pythagorean (positive = lucky, negative = unlucky)")
    print(f"  Saved to cache/team_ratings.json")
    print(f"{'='*65}\n")

    return ratings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2026-03-20")
    parser.add_argument("--end",   type=str, default=date.today().isoformat())
    args = parser.parse_args()

    logger.info(f"Fetching game results {args.start} → {args.end}...")
    games   = fetch_season_results(args.start, args.end)
    elo, pythag, actual_wpct, rf, ra, w, l = calculate_ratings(games)
    save_ratings(elo, pythag, actual_wpct, rf, ra, w, l)


if __name__ == "__main__":
    main()
