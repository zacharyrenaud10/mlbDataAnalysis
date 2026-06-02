from __future__ import annotations
import argparse
import os
import requests
from datetime import date
from dotenv import load_dotenv

load_dotenv()

API_KEY  = os.getenv("ODDS_API_KEY", "")
BASE_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

# Edit this list as you sign up for more books
MY_BOOKS = [
    "fanduel",
    "draftkings",
    # "betmgm",
    # "caesars",
    # "espnbet",
]

def decimal_to_american(dec):
    if dec >= 2.0:
        return f"+{int((dec - 1) * 100)}"
    else:
        return f"{int(-100 / (dec - 1))}"

def decimal_to_implied(dec):
    return 1 / dec if dec else 0

def fetch_odds(books):
    r = requests.get(BASE_URL, params={
        "apiKey":     API_KEY,
        "regions":    "us,uk,eu",
        "markets":    "h2h",
        "bookmakers": ",".join(books),
        "oddsFormat": "decimal",
    }, timeout=15)
    r.raise_for_status()
    remaining = r.headers.get("x-requests-remaining", "?")
    print(f"  Odds API — {remaining} requests remaining")
    return r.json()

def get_model_prob(team):
    try:
        import glob, json
        files = glob.glob("cache/predictions_*.json")
        if not files:
            return None
        latest = sorted(files)[-1]
        with open(latest, encoding="utf-8") as f:
            data = json.load(f)
        for game in data.get("games", []):
            if game.get("home") == team:
                return game.get("home_win_prob")
            if game.get("away") == team:
                return game.get("away_win_prob")
    except Exception:
        pass
    return None

def analyze(books, ev_only=False):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  BEST ODDS + EV FINDER  --  {date.today()}")
    print(f"  Books: {', '.join(books)}")
    print(sep)

    games = fetch_odds(books)
    if not games:
        print("  No games found.")
        return

    for game in games:
        home = game["home_team"]
        away = game["away_team"]
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue

        best     = {home: {"price": 0, "american": "", "book": ""},
                    away: {"price": 0, "american": "", "book": ""}}
        all_lines = {home: {}, away: {}}

        for book in bookmakers:
            bk = book["key"]
            for market in book.get("markets", []):
                if market["key"] != "h2h":
                    continue
                for outcome in market["outcomes"]:
                    team  = outcome["name"]
                    price = outcome["price"]
                    if team in best:
                        all_lines[team][bk] = decimal_to_american(price)
                        if price > best[team]["price"]:
                            best[team]["price"]    = price
                            best[team]["american"] = decimal_to_american(price)
                            best[team]["book"]     = bk

        home_prob    = get_model_prob(home)
        away_prob    = get_model_prob(away)
        home_implied = decimal_to_implied(best[home]["price"]) if best[home]["price"] else None
        away_implied = decimal_to_implied(best[away]["price"]) if best[away]["price"] else None

        home_ev = None
        away_ev = None
        if home_prob and home_implied and best[home]["price"]:
            home_ev = (home_prob * (best[home]["price"] - 1)) - (1 - home_prob)
        if away_prob and away_implied and best[away]["price"]:
            away_ev = (away_prob * (best[away]["price"] - 1)) - (1 - away_prob)

        has_ev = (home_ev and home_ev > 0) or (away_ev and away_ev > 0)
        if ev_only and not has_ev:
            continue

        print(f"\n  {away} @ {home}")
        print(f"  {'-' * 60}")

        for team, prob, ev, implied in [
            (home, home_prob, home_ev, home_implied),
            (away, away_prob, away_ev, away_implied),
        ]:
            if not best[team]["price"]:
                continue
            lines_str = "  ".join(f"{b}:{p}" for b, p in all_lines[team].items())
            best_str  = f"BEST: {best[team]['american']} @ {best[team]['book'].upper()}"
            prob_str  = f"Model: {prob*100:.1f}%" if prob else ""
            impl_str  = f"Implied: {implied*100:.1f}%" if implied else ""
            ev_str    = f"EV: {ev*100:+.1f}%" if ev is not None else ""
            ev_flag   = "  +EV!" if ev and ev > 0 else ""
            print(f"  {team:<28} {lines_str}")
            print(f"  {' '*28} {best_str}  {prob_str}  {impl_str}  {ev_str}{ev_flag}")

    print(f"\n{sep}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book",    type=str)
    parser.add_argument("--ev-only", action="store_true")
    args = parser.parse_args()
    books = [args.book] if args.book else MY_BOOKS
    analyze(books, ev_only=args.ev_only)

if __name__ == "__main__":
    main()
