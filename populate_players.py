"""
populate_players.py
-------------------
Populates the players table with MLBAM IDs and names
by pulling from the MLB Stats API.

Usage:
    python populate_players.py
"""

from __future__ import annotations

import time
import unicodedata
import pandas as pd
import requests
from loguru import logger
from sqlalchemy import text

from mlb_analytics.db import engine

MLB_API = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def strip_accents(text: str) -> str:
    if not text:
        return text
    return "".join(
        c for c in unicodedata.normalize("NFD", str(text))
        if unicodedata.category(c) != "Mn"
    )


def _safe_hand(code: str, allow_switch: bool = False) -> str:
    """Ensure hand code satisfies DB CHECK constraints."""
    if code in ("L", "R"):
        return code
    if allow_switch and code == "S":
        return "S"
    return "R"


def get_all_player_ids() -> list[int]:
    with engine.connect() as conn:
        batters = pd.read_sql(
            text("SELECT DISTINCT batter_id as id FROM statcast_pitches "
                 "WHERE batter_id IS NOT NULL"), conn)
        pitchers = pd.read_sql(
            text("SELECT DISTINCT pitcher_id as id FROM statcast_pitches "
                 "WHERE pitcher_id IS NOT NULL"), conn)
    all_ids = set(batters["id"].tolist() + pitchers["id"].tolist())
    logger.info(f"Found {len(all_ids):,} unique player IDs in statcast data")
    return list(all_ids)


def fetch_player_info(player_id: int) -> dict:
    try:
        resp = requests.get(
            f"{MLB_API}/people/{player_id}",
            params={"hydrate": "currentTeam"},
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        people = resp.json().get("people", [])
        if not people:
            return {}

        p = people[0]
        return {
            "player_id":   player_id,
            "name_first":  strip_accents(p.get("firstName", "")),
            "name_last":   strip_accents(p.get("lastName",  "")),
            "bats":        _safe_hand(
                               p.get("batSide",   {}).get("code", "R"),
                               allow_switch=True
                           ),
            "throws":      _safe_hand(
                               p.get("pitchHand", {}).get("code", "R"),
                               allow_switch=False
                           ),
            "primary_pos": p.get("primaryPosition", {}).get("abbreviation", ""),
            "team":        p.get("currentTeam", {}).get("abbreviation", ""),
            "active":      p.get("active", True),
        }
    except Exception as exc:
        logger.debug(f"Failed to fetch player {player_id}: {exc}")
        return {}


def _insert_batch(records: list[dict]) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    df["name_first"] = df["name_first"].fillna("Unknown")
    df["name_last"]  = df["name_last"].fillna("Unknown")
    # Never insert full_name — it is a generated column in SQLite
    df = df.drop(columns=["full_name"], errors="ignore")

    try:
        df.to_sql("players", con=engine, if_exists="append",
                  index=False, method="multi")
    except Exception as exc:
        logger.debug(f"Batch insert issue: {exc}")
        for record in records:
            record.pop("full_name", None)
            try:
                pd.DataFrame([record]).to_sql(
                    "players", con=engine,
                    if_exists="append", index=False)
            except Exception:
                pass


def populate_players_table() -> None:
    player_ids = get_all_player_ids()
    logger.info(f"Fetching info for {len(player_ids):,} players...")

    with engine.connect() as conn:
        existing = pd.read_sql(text("SELECT player_id FROM players"), conn)
    existing_ids = set(existing["player_id"].tolist()) if not existing.empty else set()
    player_ids   = [pid for pid in player_ids if pid not in existing_ids]
    logger.info(f"  {len(existing_ids):,} already in DB, "
                f"fetching {len(player_ids):,} new")

    if not player_ids:
        logger.success("Players table already up to date!")
        _show_sample()
        return

    records = []
    for i, pid in enumerate(player_ids):
        info = fetch_player_info(int(pid))
        if info:
            records.append(info)

        if (i + 1) % 100 == 0:
            pct = (i + 1) / len(player_ids) * 100
            logger.info(f"  Progress: {i+1:,}/{len(player_ids):,} ({pct:.1f}%)")

        if len(records) >= 50:
            _insert_batch(records)
            records = []

        if (i + 1) % 10 == 0:
            time.sleep(0.1)

    if records:
        _insert_batch(records)

    with engine.connect() as conn:
        count = pd.read_sql(text("SELECT COUNT(*) as c FROM players"), conn)
    logger.success(f"Players table now has {count['c'].iloc[0]:,} players")
    _show_sample()


def _show_sample() -> None:
    with engine.connect() as conn:
        sample = pd.read_sql(text(
            "SELECT full_name, team, primary_pos "
            "FROM players ORDER BY RANDOM() LIMIT 10"
        ), conn)
    print("\nSample players (accents stripped):")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    logger.info("Populating players table from MLB Stats API...")
    populate_players_table()
    logger.success("Done! Now run: python train_props.py --seasons 2023 2024 2025")