"""
data_fetchers/weather.py
------------------------
Fetches game-time weather for every MLB ballpark using
Open-Meteo (free, no API key needed).

Automatically pulls game times from fetch_lineups.py.

Debug findings:
- MLB API returns Eastern time mislabeled as UTC
- Real UTC = API time + 3 hours
- Local park time = real UTC + park offset (from Open-Meteo)
- Example: API=00:05 + 3h = 03:05 UTC, SF offset=-7h, local=20:05 (8:05 PM PDT) correct

Usage:
    python data_fetchers/weather.py
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from typing import Optional

import requests
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Ballpark coordinates + roof status
# ---------------------------------------------------------------------------
BALLPARKS = {
    "NYY": {"name": "Yankee Stadium",           "lat": 40.8296, "lon": -73.9262, "roof": False},
    "BOS": {"name": "Fenway Park",              "lat": 42.3467, "lon": -71.0972, "roof": False},
    "SF":  {"name": "Oracle Park",              "lat": 37.7786, "lon": -122.3893,"roof": False},
    "LAD": {"name": "Dodger Stadium",           "lat": 34.0739, "lon": -118.2400,"roof": False},
    "CHC": {"name": "Wrigley Field",            "lat": 41.9484, "lon": -87.6553, "roof": False},
    "HOU": {"name": "Minute Maid Park",         "lat": 29.7572, "lon": -95.3556, "roof": True},
    "SEA": {"name": "T-Mobile Park",            "lat": 47.5914, "lon": -122.3325,"roof": True},
    "MIL": {"name": "American Family Field",    "lat": 43.0280, "lon": -87.9712, "roof": True},
    "TOR": {"name": "Rogers Centre",            "lat": 43.6414, "lon": -79.3894, "roof": True},
    "MIA": {"name": "loanDepot Park",           "lat": 25.7781, "lon": -80.2197, "roof": True},
    "ARI": {"name": "Chase Field",              "lat": 33.4453, "lon": -112.0667,"roof": True},
    "TB":  {"name": "Tropicana Field",          "lat": 27.7683, "lon": -82.6534, "roof": True},
    "ATL": {"name": "Truist Park",              "lat": 33.8908, "lon": -84.4678, "roof": False},
    "NYM": {"name": "Citi Field",               "lat": 40.7571, "lon": -73.8458, "roof": False},
    "PHI": {"name": "Citizens Bank Park",       "lat": 39.9061, "lon": -75.1665, "roof": False},
    "WSH": {"name": "Nationals Park",           "lat": 38.8730, "lon": -77.0074, "roof": False},
    "BAL": {"name": "Camden Yards",             "lat": 39.2838, "lon": -76.6217, "roof": False},
    "CLE": {"name": "Progressive Field",        "lat": 41.4962, "lon": -81.6852, "roof": False},
    "DET": {"name": "Comerica Park",            "lat": 42.3390, "lon": -83.0485, "roof": False},
    "MIN": {"name": "Target Field",             "lat": 44.9817, "lon": -93.2781, "roof": False},
    "CWS": {"name": "Guaranteed Rate Field",    "lat": 41.8300, "lon": -87.6339, "roof": False},
    "KC":  {"name": "Kauffman Stadium",         "lat": 39.0517, "lon": -94.4803, "roof": False},
    "TEX": {"name": "Globe Life Field",         "lat": 32.7473, "lon": -97.0845, "roof": True},
    "LAA": {"name": "Angel Stadium",            "lat": 33.8003, "lon": -117.8827,"roof": False},
    "OAK": {"name": "Oakland Coliseum",         "lat": 37.7516, "lon": -122.2005,"roof": False},
    "SD":  {"name": "Petco Park",               "lat": 32.7076, "lon": -117.1570,"roof": False},
    "COL": {"name": "Coors Field",              "lat": 39.7559, "lon": -104.9942,"roof": False},
    "STL": {"name": "Busch Stadium",            "lat": 38.6226, "lon": -90.1928, "roof": False},
    "PIT": {"name": "PNC Park",                 "lat": 40.4469, "lon": -80.0057, "roof": False},
    "CIN": {"name": "Great American Ball Park", "lat": 39.0979, "lon": -84.5082, "roof": False},
}

WIND_DIRECTIONS = {
    (0,   22):  "N",  (22,  67):  "NE", (67,  112): "E",
    (112, 157): "SE", (157, 202): "S",  (202, 247): "SW",
    (247, 292): "W",  (292, 337): "NW", (337, 360): "N",
}


def degrees_to_compass(deg: float) -> str:
    for (lo, hi), label in WIND_DIRECTIONS.items():
        if lo <= deg < hi:
            return label
    return "N"


def get_wind_impact(speed_mph: float, direction: str) -> str:
    if speed_mph < 5:
        return "🟡 Calm — minimal impact"
    elif speed_mph < 10:
        impact = "Light"
    elif speed_mph < 15:
        impact = "Moderate"
    else:
        impact = "Strong"
    out_dirs = {"E", "SE", "S"}
    in_dirs  = {"W", "NW", "N", "NE"}
    if direction in out_dirs:
        return f"🔴 {impact} wind OUT ({direction}) — favors HITTERS"
    elif direction in in_dirs:
        return f"🟢 {impact} wind IN ({direction}) — favors PITCHERS"
    else:
        return f"🟡 {impact} crosswind ({direction}) — neutral"


def get_temp_impact(temp: float) -> str:
    if temp < 45:
        return "🥶 Very cold — ball won't carry, strongly favors pitchers"
    elif temp < 55:
        return "❄️  Cool — slight pitcher advantage"
    elif temp < 70:
        return "✅ Ideal conditions"
    elif temp < 85:
        return "☀️  Warm — ball carries well"
    else:
        return "🔥 Hot — ball carries, favors hitters"


def get_rain_risk(precip: float) -> str:
    if precip >= 50:
        return f"🌧️  HIGH rain risk ({precip:.0f}%) — game delay possible, avoid betting"
    elif precip >= 25:
        return f"🌦️  Moderate rain risk ({precip:.0f}%) — monitor"
    else:
        return f"☀️  Low rain risk ({precip:.0f}%)"


def get_betting_lean(temp: float, wind_impact: str) -> str:
    pitcher_score = 0
    hitter_score  = 0
    if temp < 55:
        pitcher_score += 1
    elif temp > 80:
        hitter_score += 1
    if "PITCHER" in wind_impact:
        pitcher_score += 1
    elif "HITTER" in wind_impact:
        hitter_score += 1
    if pitcher_score > hitter_score:
        return "📉 Lean UNDER / pitcher props"
    elif hitter_score > pitcher_score:
        return "📈 Lean OVER / hitter props"
    else:
        return "⚖️  Neutral conditions"


def resolve_local_game_time(
    game_time_utc: str,
    utc_offset_seconds: int,
) -> str:
    """
    Convert MLB API game time to local park time.

    CONFIRMED via debug:
    - MLB API returns Eastern time mislabeled as UTC
    - Real UTC = API time + 3 hours
    - Local time = real UTC + park offset (from Open-Meteo utc_offset_seconds)

    Example verified for tonight:
    API=00:05 + 3h = 03:05 real UTC
    SF offset = -7h (PDT)
    03:05 - 7h = 20:05 = 8:05 PM PDT CORRECT
    """
    try:
        api_dt            = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
        park_offset_hours = utc_offset_seconds // 3600

        # MLB API is consistently 3h behind real UTC (returns ET as UTC)
        real_utc = api_dt + timedelta(hours=3)

        # Convert real UTC to local park time
        local_dt = real_utc + timedelta(hours=park_offset_hours)

        logger.info(
            f"  API={api_dt.strftime('%H:%M')} "
            f"→ realUTC={real_utc.strftime('%H:%M')} "
            f"→ local={local_dt.strftime('%H:%M')} "
            f"(park offset {park_offset_hours:+d}h)"
        )
        return local_dt.strftime("%Y-%m-%dT%H:%M")

    except Exception as exc:
        logger.warning(f"Could not parse game time {game_time_utc}: {exc}")
        return ""


def fetch_weather(
    team: str,
    game_date: Optional[str] = None,
    game_time_utc: Optional[str] = None,
) -> dict:
    """
    Fetch weather for a team's home ballpark at game time.

    Parameters
    ----------
    team          : team abbreviation e.g. 'SF', 'NYY'
    game_date     : YYYY-MM-DD (defaults to today)
    game_time_utc : UTC time string from MLB API
    """
    game_date = game_date or date.today().isoformat()

    # Normalize team abbreviation aliases
    TEAM_ALIASES = {
        "AZ":  "ARI",  # Arizona uses AZ in some APIs
        "ATH": "OAK",  # Athletics abbreviation change
    }
    team_key = TEAM_ALIASES.get(team, team)
    park     = BALLPARKS.get(team_key)

    if not park:
        logger.warning(f"No ballpark data for {team}")
        return {}

    if park["roof"]:
        return {
            "team":        team,
            "park":        park["name"],
            "roof":        True,
            "betting_lean":"🏟️  Indoor stadium — weather not a factor",
        }

    url    = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":         park["lat"],
        "longitude":        park["lon"],
        "hourly":           "temperature_2m,precipitation_probability,"
                            "windspeed_10m,winddirection_10m,relativehumidity_2m",
        "temperature_unit": "fahrenheit",
        "windspeed_unit":   "mph",
        "forecast_days":    3,
        "timezone":         "auto",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error(f"Weather fetch failed for {team}: {exc}")
        return {}

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])

    if not times:
        logger.error(f"No hourly data returned for {team}")
        return {}

    utc_offset_seconds = data.get("utc_offset_seconds", 0)

    if game_time_utc:
        target = resolve_local_game_time(game_time_utc, utc_offset_seconds)
        if not target:
            target = f"{game_date}T19:00"
    else:
        target = f"{game_date}T19:00"

    # Find closest matching hour in forecast
    idx = 0
    for i, t in enumerate(times):
        if t >= target:
            idx = i
            break

    temp     = hourly["temperature_2m"][idx]
    precip   = hourly["precipitation_probability"][idx]
    wind_spd = hourly["windspeed_10m"][idx]
    wind_dir = degrees_to_compass(hourly["winddirection_10m"][idx])
    humidity = hourly["relativehumidity_2m"][idx]

    wind_impact  = get_wind_impact(wind_spd, wind_dir)
    temp_impact  = get_temp_impact(temp)
    rain_risk    = get_rain_risk(precip)
    betting_lean = get_betting_lean(temp, wind_impact)

    return {
        "team":           team,
        "park":           park["name"],
        "roof":           False,
        "temp_f":         round(temp, 1),
        "precip_pct":     precip,
        "wind_mph":       round(wind_spd, 1),
        "wind_dir":       wind_dir,
        "humidity":       round(humidity, 1),
        "wind_impact":    wind_impact,
        "temp_impact":    temp_impact,
        "rain_risk":      rain_risk,
        "betting_lean":   betting_lean,
        "game_date":      game_date,
        "game_time_utc":  game_time_utc or "",
        "local_gametime": target,
    }


def fetch_all_weather(
    games: list[dict],
    game_date: Optional[str] = None,
) -> dict:
    """Fetch weather for all games."""
    import time
    results = {}
    gd      = game_date or date.today().isoformat()
    for game in games:
        home      = game.get("home", "")
        game_time = game.get("game_time", None)
        if not home:
            continue
        results[home] = fetch_weather(home, gd, game_time_utc=game_time)
        time.sleep(0.3)
    return results


def print_weather_report(weather: dict) -> None:
    if not weather:
        print("  No weather data available")
        return
    if weather.get("roof"):
        print(f"  🏟️  {weather['park']} — Indoor, weather N/A")
        return
    local_gt = weather.get("local_gametime", "")
    print(f"  🌤️  {weather['park']}"
          + (f"  (local game time: {local_gt})" if local_gt else ""))
    print(f"     Temp:    {weather['temp_f']:.0f}°F  —  {weather['temp_impact']}")
    print(f"     Wind:    {weather['wind_mph']:.0f} mph {weather['wind_dir']}"
          f"  —  {weather['wind_impact']}")
    print(f"     Rain:    {weather['rain_risk']}")
    print(f"     Humid:   {weather['humidity']:.0f}%")
    print(f"     Lean:    {weather['betting_lean']}")


if __name__ == "__main__":
    print("Fetching weather for all today's games...\n")

    from fetch_lineups import get_todays_games, parse_game

    today     = date.today().isoformat()
    games_raw = get_todays_games(today)
    games     = [parse_game(g) for g in games_raw if parse_game(g)]

    if not games:
        print("No games found from MLB API.")
    else:
        for game in games:
            home      = game["home"]
            away      = game["away"]
            game_time = game.get("game_time", "")
            print(f"  {away} @ {home}  ({game_time})")
            w = fetch_weather(home, today, game_time_utc=game_time)
            print_weather_report(w)
            print()