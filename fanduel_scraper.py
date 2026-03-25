"""
mlb_analytics/ingestion/fanduel_scraper.py
------------------------------------------
Retrieves FanDuel MLB player props and moneylines.

Two strategies — use whichever matches your setup:
  Strategy A (recommended):  The Odds API (https://the-odds-api.com)
                              Clean JSON, no scraping, 500 free requests/month.
  Strategy B (fallback):     Selenium headless scrape of FanDuel directly.
                              Brittle but free.

Set ODDS_API_KEY in your .env to enable Strategy A.
"""

from __future__ import annotations

import os
import re
import time
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------
ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = os.getenv("ODDS_API_BASE", "https://api.the-odds-api.com/v4")
SPORT_KEY     = "baseball_mlb"
REGIONS       = "us"

# FanDuel's bookmaker key in The Odds API
FANDUEL_KEY   = "fanduel"

# American odds → implied probability
def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied (vig-inclusive) probability."""
    if odds is None:
        return float("nan")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def remove_vig(prob_over: float, prob_under: float) -> tuple[float, float]:
    """
    Remove the bookmaker's vig using the standard multiplicative method.
    Returns (fair_over, fair_under) probabilities that sum to 1.
    """
    total = prob_over + prob_under
    if total == 0:
        return 0.5, 0.5
    return prob_over / total, prob_under / total


# ---------------------------------------------------------------------------
# Strategy A — The Odds API  (preferred)
# ---------------------------------------------------------------------------

def _odds_api_get(endpoint: str, params: dict) -> Optional[dict]:
    """Low-level GET wrapper for The Odds API."""
    if not ODDS_API_KEY:
        raise EnvironmentError(
            "ODDS_API_KEY not set. Add it to your .env or use the Selenium scraper."
        )
    url = f"{ODDS_API_BASE}/{endpoint}"
    params["apiKey"] = ODDS_API_KEY
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        logger.debug(f"Odds API — {remaining} requests remaining")
        return resp.json()
    except requests.HTTPError as exc:
        logger.error(f"Odds API HTTP error: {exc.response.status_code} {exc.response.text}")
    except Exception as exc:
        logger.error(f"Odds API request failed: {exc}")
    return None


def fetch_moneylines_api() -> pd.DataFrame:
    """
    Pull current MLB moneylines for today's games (FanDuel only).
    Returns a DataFrame with one row per team.
    """
    data = _odds_api_get(
        f"sports/{SPORT_KEY}/odds",
        {"regions": REGIONS, "markets": "h2h",
         "bookmakers": FANDUEL_KEY, "dateFormat": "iso"},
    )
    if not data:
        return pd.DataFrame()

    rows = []
    for game in data:
        game_date = game["commence_time"][:10]
        home = game.get("home_team")
        away = game.get("away_team")

        fd_book = next(
            (b for b in game.get("bookmakers", []) if b["key"] == FANDUEL_KEY), None
        )
        if not fd_book:
            continue

        h2h = next(
            (m for m in fd_book["markets"] if m["key"] == "h2h"), None
        )
        if not h2h:
            continue

        prices = {o["name"]: o["price"] for o in h2h["outcomes"]}
        ml_home = prices.get(home)
        ml_away = prices.get(away)

        ip_home = american_to_implied_prob(ml_home)
        ip_away = american_to_implied_prob(ml_away)
        fair_home, fair_away = remove_vig(ip_home, ip_away)
        vig = (ip_home + ip_away) - 1.0

        rows.append({
            "scraped_at": datetime.utcnow(),
            "game_date": game_date,
            "home_team": home,
            "away_team": away,
            "market_type": "moneyline",
            "moneyline_home": ml_home,
            "moneyline_away": ml_away,
            "implied_prob_home": round(ip_home, 4),
            "implied_prob_away": round(ip_away, 4),
            "fair_prob_home": round(fair_home, 4),
            "fair_prob_away": round(fair_away, 4),
            "vig_pct": round(vig, 4),
        })

    df = pd.DataFrame(rows)
    logger.success(f"Fetched {len(df)} moneyline rows from Odds API")
    return df


def fetch_player_props_api(
    prop_types: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Pull player props for today's MLB games from FanDuel via The Odds API.

    `prop_types` filters to specific markets, e.g.:
        ["batter_hits", "pitcher_strikeouts", "batter_total_bases"]
    Pass None to fetch all available player prop markets.

    Returns a long-format DataFrame with one row per player/prop/side.
    """
    if prop_types is None:
        prop_types = [
            "batter_hits",
            "batter_total_bases",
            "pitcher_strikeouts",
            "pitcher_hits_allowed",
            "batter_home_runs",
            "batter_rbis",
        ]

    markets_str = ",".join(prop_types)

    # Fetch event IDs for today first
    events = _odds_api_get(
        f"sports/{SPORT_KEY}/events",
        {"dateFormat": "iso"},
    )
    if not events:
        return pd.DataFrame()

    today_str = date.today().isoformat()
    todays_events = [
        e for e in events
        if e.get("commence_time", "")[:10] == today_str
    ]
    logger.info(f"Found {len(todays_events)} MLB events today")

    all_rows: list[dict] = []

    for event in todays_events:
        event_id  = event["id"]
        game_date = event["commence_time"][:10]
        home      = event.get("home_team")
        away      = event.get("away_team")

        data = _odds_api_get(
            f"sports/{SPORT_KEY}/events/{event_id}/odds",
            {
                "regions": REGIONS,
                "markets": markets_str,
                "bookmakers": FANDUEL_KEY,
                "dateFormat": "iso",
            },
        )
        if not data:
            continue

        fd_book = next(
            (b for b in data.get("bookmakers", []) if b["key"] == FANDUEL_KEY), None
        )
        if not fd_book:
            continue

        for market in fd_book.get("markets", []):
            prop_type = market["key"]

            # Group outcomes into over/under pairs by player name
            player_outcomes: dict[str, dict] = {}
            for outcome in market.get("outcomes", []):
                name    = outcome.get("description", outcome.get("name", ""))
                side    = outcome.get("name", "").lower()          # "over" / "under"
                price   = outcome.get("price")
                line    = outcome.get("point")

                if name not in player_outcomes:
                    player_outcomes[name] = {"name": name, "line": line}
                player_outcomes[name][f"{side}_price"] = price
                player_outcomes[name][f"line"]          = line

            for player_name, po in player_outcomes.items():
                over_price  = po.get("over_price")
                under_price = po.get("under_price")
                ip_over     = american_to_implied_prob(over_price)  if over_price  else float("nan")
                ip_under    = american_to_implied_prob(under_price) if under_price else float("nan")
                if not (pd.isna(ip_over) or pd.isna(ip_under)):
                    fair_over, fair_under = remove_vig(ip_over, ip_under)
                    vig = (ip_over + ip_under) - 1.0
                else:
                    fair_over = fair_under = float("nan")
                    vig = float("nan")

                all_rows.append({
                    "scraped_at":         datetime.utcnow(),
                    "game_date":          game_date,
                    "home_team":          home,
                    "away_team":          away,
                    "market_type":        "player_prop",
                    "prop_type":          prop_type,
                    "player_name":        player_name,
                    "line":               po.get("line"),
                    "over_price":         over_price,
                    "under_price":        under_price,
                    "implied_prob_over":  round(ip_over,  4) if not pd.isna(ip_over)  else None,
                    "implied_prob_under": round(ip_under, 4) if not pd.isna(ip_under) else None,
                    "fair_prob_over":     round(fair_over, 4) if not pd.isna(fair_over) else None,
                    "fair_prob_under":    round(fair_under,4) if not pd.isna(fair_under) else None,
                    "vig_pct":            round(vig, 4) if not pd.isna(vig) else None,
                })

        time.sleep(0.2)  # polite to API

    df = pd.DataFrame(all_rows)
    logger.success(f"Fetched {len(df)} player prop rows")
    return df


# ---------------------------------------------------------------------------
# Strategy B — Selenium scraper  (fallback, use if no API key)
# ---------------------------------------------------------------------------

def fetch_player_props_selenium(headless: bool = True) -> pd.DataFrame:
    """
    Scrape FanDuel player props using Selenium.
    WARNING: FanDuel's layout changes frequently — treat this as a template
    and update the CSS selectors as needed.

    Requires: selenium, webdriver-manager (both in requirements.txt)
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        logger.error("selenium / webdriver-manager not installed. "
                     "Run: pip install selenium webdriver-manager")
        return pd.DataFrame()

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    rows: list[dict] = []

    try:
        url = "https://sportsbook.fanduel.com/navigation/mlb"
        logger.info(f"Navigating to {url}")
        driver.get(url)

        wait = WebDriverWait(driver, 20)
        # Wait for game links to load
        wait.until(EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "a[href*='/baseball/mlb/']")
        ))

        game_links = [
            el.get_attribute("href")
            for el in driver.find_elements(By.CSS_SELECTOR, "a[href*='/baseball/mlb/']")
        ]
        game_links = list(dict.fromkeys(game_links))  # deduplicate
        logger.info(f"Found {len(game_links)} game links")

        for game_url in game_links[:10]:  # cap at 10 for safety
            try:
                driver.get(game_url)
                time.sleep(2)

                # Navigate to Player Props tab
                prop_tabs = driver.find_elements(
                    By.XPATH, "//button[contains(text(),'Player Props') or contains(text(),'Props')]"
                )
                if not prop_tabs:
                    logger.debug(f"No props tab found: {game_url}")
                    continue
                prop_tabs[0].click()
                time.sleep(1.5)

                # --- Parse prop markets ---
                # NOTE: These selectors are illustrative and will need updating
                # when FanDuel changes their DOM. Use browser DevTools to update.
                prop_sections = driver.find_elements(
                    By.CSS_SELECTOR, "div[class*='MarketContainer'], div[class*='market-']"
                )

                for section in prop_sections:
                    try:
                        market_title = section.find_element(
                            By.CSS_SELECTOR, "h3, [class*='title'], [class*='header']"
                        ).text.strip()
                        buttons = section.find_elements(
                            By.CSS_SELECTOR, "button[class*='selection'], [class*='outcome']"
                        )
                        for btn in buttons:
                            text = btn.text.strip()
                            # Typical format: "Player Name\nO/U line\nOdds"
                            parts = [p.strip() for p in text.split("\n") if p.strip()]
                            if len(parts) >= 3:
                                rows.append({
                                    "scraped_at":  datetime.utcnow(),
                                    "game_date":   date.today().isoformat(),
                                    "game_url":    game_url,
                                    "market_type": "player_prop",
                                    "prop_type":   _normalise_market_name(market_title),
                                    "player_name": parts[0],
                                    "line_text":   parts[1],
                                    "price_text":  parts[2],
                                })
                    except Exception:
                        pass

            except Exception as exc:
                logger.warning(f"Error scraping {game_url}: {exc}")
            time.sleep(1)

    finally:
        driver.quit()

    df = pd.DataFrame(rows)
    if not df.empty:
        df = _parse_scraped_props(df)
    logger.success(f"Selenium scrape returned {len(df)} rows")
    return df


def _normalise_market_name(raw: str) -> str:
    """Map FanDuel display names to our internal prop_type keys."""
    mapping = {
        "hits":        "batter_hits",
        "strikeouts":  "pitcher_strikeouts",
        "total bases": "batter_total_bases",
        "home run":    "batter_home_runs",
        "rbi":         "batter_rbis",
    }
    raw_lower = raw.lower()
    for key, val in mapping.items():
        if key in raw_lower:
            return val
    return raw_lower.replace(" ", "_")


def _parse_scraped_props(df: pd.DataFrame) -> pd.DataFrame:
    """Parse line_text and price_text columns into typed fields."""
    # line_text examples: "Over 1.5", "Under 0.5"
    df["bet_side"] = df["line_text"].str.extract(r"(Over|Under)", flags=re.IGNORECASE)[0]
    df["line"]     = pd.to_numeric(
        df["line_text"].str.extract(r"([\d.]+)")[0], errors="coerce"
    )
    # price_text example: "-115", "+110"
    df["price"]    = pd.to_numeric(df["price_text"].str.replace("+", ""), errors="coerce")
    df["implied_prob"] = df["price"].apply(
        lambda x: american_to_implied_prob(int(x)) if not pd.isna(x) else None
    )
    return df


# ---------------------------------------------------------------------------
# Unified entry-point: auto-selects API vs Selenium
# ---------------------------------------------------------------------------

def get_todays_props() -> pd.DataFrame:
    """
    Fetch today's FanDuel player props.
    Uses The Odds API if ODDS_API_KEY is set, otherwise falls back to Selenium.
    """
    if ODDS_API_KEY:
        logger.info("Using The Odds API (Strategy A)")
        return fetch_player_props_api()
    else:
        logger.info("ODDS_API_KEY not set — using Selenium scraper (Strategy B)")
        return fetch_player_props_selenium()


def get_todays_moneylines() -> pd.DataFrame:
    if ODDS_API_KEY:
        return fetch_moneylines_api()
    logger.warning("Moneyline fetch requires ODDS_API_KEY (Selenium scraping not implemented "
                   "for moneylines). Set the key in .env.")
    return pd.DataFrame()
