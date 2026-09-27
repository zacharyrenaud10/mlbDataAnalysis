# MLB Predictive Analytics Engine

A full-season MLB game outcome and player performance prediction system built on Statcast data, ensemble machine learning, and live MLB API data. Predicts game winners, player prop outcomes, and pitcher strikeout totals with a daily automated pipeline.

**Full-season backtest (March–August 2026, ~2,070 games): 55% overall accuracy, 60.3% on high-confidence picks (55%+).**

---

## What It Does

- Pulls daily Statcast pitch-level data and Baseball Savant season stats
- Builds rolling 5-game batter and pitcher feature matrices
- Trains an ensemble of XGBoost, LightGBM, Random Forest, and Logistic Regression models daily
- Predicts game winners using a 6-model ensemble (Statcast, pitcher metrics, recent form, season ratings, Elo, Pythagorean win%)
- Predicts player prop outcomes: hits, total bases, home runs, walks, strikeouts
- Tracks real Elo ratings and Pythagorean win% for all 30 teams updated daily
- Fetches live sportsbook odds from FanDuel and DraftKings via the Odds API
- Backtests predictions against real historical results

---

## Architecture

```
MLB Stats API          Baseball Savant (Statcast)
     │                         │
     ▼                         ▼
fetch_lineups.py       statcast_ingest.py
(lineups, starters,    (pitch-level data,
 weather, umpires)      3.7M+ pitches)
     │                         │
     └──────────┬──────────────┘
                ▼
          SQLite Database
      (statcast_pitches, players,
    batter/pitcher rolling features,
       savant season stats)
                │
                ▼
        rolling_features.py
   (5-game rolling batter + pitcher
        feature matrices)
                │
                ▼
          train_props.py
  (XGBoost + LightGBM + Random Forest
   + Logistic Regression ensemble,
    time-series cross-validation,
      isotonic calibration)
                │
         ┌──────┴──────┐
         ▼             ▼
  parlay_builder.py  matchup_predictor.py
  (game winner        (player props:
   predictions,        HR, hits, TB,
   6-model ensemble,   Ks, walks)
   Elo + Pythagorean)
         │
         ▼
    best_odds.py
 (FanDuel + DraftKings
  odds comparison,
  pitcher K props)
```

---

## Model Details

### Game Winner (parlay_builder.py)
Six-model ensemble with post-processing:

| Model | Features |
|-------|----------|
| A — Statcast | Exit velocity, hard hit%, barrel% |
| B — Pitcher | K%, whiff%, xERA, CSW% |
| C — Recent form | Last 5 game rolling averages |
| D — Season ratings | Full season offensive/defensive ratings |
| E — Elo | Self-updating Elo ratings from season results |
| F — Pythagorean | Pythagorean win% + luck correction |

Post-ensemble adjustments: streak bonus, run differential, home/away shrinkage.

### Player Props (matchup_predictor.py)
Ensemble classifier for each prop type:
- XGBoost + Random Forest + LightGBM + Logistic Regression voting
- Calibrated with `CalibratedClassifierCV` (isotonic)
- Time-series 3-fold cross-validation
- AUC: 0.61–0.63 | CV-AUC: 0.59–0.60

Props predicted: `hit_1plus`, `total_bases_1/2/3`, `hr_1plus`, `walk_1plus`, `k_1plus`, pitcher K regressor.

---

## Backtest Results

Full-season backtest (March 26 – August 31, 2026 | 2,070 games):

| Confidence Tier | Win Rate | Games |
|----------------|----------|-------|
| 60–65% | **65.9%** | 88 |
| 55–60% | **59.4%** | 503 |
| 55%+ combined | **60.3%** | 595 |
| <55% | 52.9% | 1,475 |
| Overall | 55.0% | 2,070 |

Best teams by model accuracy: MIA (74.4%), ATL (67.4%), TB (66.1%), CIN (64.9%)

Worst teams: OAK (40.8%), DET (47.6%), MIN (47.1%)

---

## Setup

### Requirements
- Python 3.10+
- [Odds API key](https://the-odds-api.com/) (free tier works)

### Install

```bash
git clone https://github.com/zacharyrenaud10/mlbDataAnalysis.git
cd mlbDataAnalysis
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Add your ODDS_API_KEY to .env
```

### Build the database

```bash
# Pull Statcast data (this takes a while for multiple seasons)
python -m mlb_analytics.ingestion.statcast_ingest --start 2024-03-01 --end 2024-11-01
python -m mlb_analytics.ingestion.statcast_ingest --start 2025-03-01 --end 2025-11-01

# Pull Baseball Savant season stats
python -m mlb_analytics.ingestion.savant_season_stats --seasons 2024 2025

# Build rolling features
python -m mlb_analytics.run_pipeline --stages features --seasons 2024 2025

# Train models
python train_props.py --seasons 2024 2025
```

---

## Daily Usage

Run once after lineups post (~2–3 PM ET):

```bash
python morning.py                  # Full daily update
python morning.py --quick          # Skip retraining (faster)
python morning.py --picks-only     # Just show today's picks
```

Additional commands:

```bash
python backtest.py --start 2025-04-01 --end 2025-09-30   # Backtest accuracy
python best_odds.py                                        # Best ML odds
python best_odds.py --ks                                   # Pitcher K props
python matchup_predictor.py --hr                          # HR leaderboard
python matchup_predictor.py --ks                          # Pitcher K leaderboard
python track_results.py                                    # Grade yesterday
python track_results.py --summary                         # Full season record
python team_ratings.py                                     # Elo + Pythagorean ratings
```

---

## Project Structure

```
mlbDataAnalysis/
├── morning.py                  # Daily runner — does everything in order
├── parlay_builder.py           # Game winner predictions + parlay recommendations
├── matchup_predictor.py        # Player prop predictions (HR, hits, Ks, TB, BB)
├── train_props.py              # Ensemble model trainer
├── backtest.py                 # Historical accuracy backtesting
├── best_odds.py                # Multi-book odds comparison + K props
├── fetch_lineups.py            # MLB API lineup + starter fetcher
├── fetch_historical_lineups.py # Retroactive lineup cache builder
├── team_ratings.py             # Elo + Pythagorean ratings
├── track_results.py            # Grades predictions vs actual results
├── save_predictions.py         # Daily prediction snapshots
├── ai_analyst.py               # AI-powered parlay analysis
├── data_fetchers/
│   ├── weather.py              # Game-time weather conditions
│   ├── umpires.py              # Home plate umpire tendencies
│   └── bullpen_fatigue.py      # Reliever rest day tracking
└── mlb_analytics/
    ├── db.py                   # SQLite connection
    ├── run_pipeline.py         # Pipeline orchestration
    ├── features/
    │   └── rolling_features.py # 5-game rolling feature builder
    ├── ingestion/
    │   ├── statcast_ingest.py  # Statcast pitch data ingestion
    │   └── savant_season_stats.py # Baseball Savant season stats
    └── models/
        └── player_props_model.py  # Ensemble classifier + calibration
```

---

## Tech Stack

- **Language:** Python 3.12
- **ML:** XGBoost, LightGBM, scikit-learn (Random Forest, Logistic Regression, CalibratedClassifierCV)
- **Data:** pybaseball (Statcast), MLB Stats API, Baseball Savant, Odds API
- **Storage:** SQLite (~1GB, 3.7M+ pitches)
- **Key libraries:** pandas, numpy, requests, loguru, python-dotenv

---

## Environment Variables

```
ODDS_API_KEY=your_odds_api_key_here
```

Get a free key at [the-odds-api.com](https://the-odds-api.com/).
