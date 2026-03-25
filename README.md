# MLB Predictive Analytics Engine

A Python codebase for identifying +EV (Expected Value) opportunities in FanDuel MLB markets using Statcast data and XGBoost models.

---

## Project Structure

```
mlb_analytics/
├── schema.sql                          # Full PostgreSQL / SQLite schema
├── requirements.txt
├── .env.example                        # Copy → .env and fill in credentials
│
└── mlb_analytics/
    ├── db.py                           # SQLAlchemy engine + session
    ├── ev_engine.py                    # +EV calculator & best-bets output
    ├── run_pipeline.py                 # CLI pipeline runner
    │
    ├── ingestion/
    │   ├── statcast_ingest.py          # Statcast pull (pybaseball + chunked)
    │   ├── fangraphs_ingest.py         # FanGraphs seasonal stats
    │   └── fanduel_scraper.py          # FanDuel odds (Odds API or Selenium)
    │
    ├── features/
    │   └── rolling_features.py         # 5-game rolling averages (batter + pitcher)
    │
    └── models/
        └── prediction_engine.py        # XGBoost classifier + regressor
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set DATABASE_URL and ODDS_API_KEY
```

### 3. Initialise the database

```bash
python -c "from mlb_analytics.db import init_db; init_db('schema.sql')"
```

### 4. Run the full pipeline

```bash
# Ingest all three seasons + train + find today's best bets
python -m mlb_analytics.run_pipeline --all

# Or run stages individually:
python -m mlb_analytics.run_pipeline --stages ingest   --seasons 2025
python -m mlb_analytics.run_pipeline --stages features --window 5
python -m mlb_analytics.run_pipeline --stages train
python -m mlb_analytics.run_pipeline --stages ev       --top 15
```

### 5. Pull Statcast directly (standalone)

```bash
# Full 2025 season
python -m mlb_analytics.ingestion.statcast_ingest --season 2025

# Specific date range
python -m mlb_analytics.ingestion.statcast_ingest --start 2025-04-01 --end 2025-04-30

# Save to Parquet (skip DB write)
python -m mlb_analytics.ingestion.statcast_ingest --season 2025 \
       --parquet ./cache/statcast_2025.parquet --no-db
```

---

## Architecture

```
Baseball Savant (Statcast)          FanGraphs
        │                               │
        ▼                               ▼
 statcast_ingest.py            fangraphs_ingest.py
        │                               │
        └───────────┬───────────────────┘
                    ▼
               PostgreSQL / SQLite
          (statcast_pitches, fg_batter_season, …)
                    │
                    ▼
          rolling_features.py
    (batter_rolling_features, pitcher_rolling_features)
                    │
                    ▼
         prediction_engine.py
     ┌──────────────┴────────────────┐
     │                               │
BaseHitClassifier            StrikeoutRegressor
(XGBoost + isotonic cal.)    (XGBoost regressor)
P(base_hit)                  E[strikeouts]
     └──────────────┬────────────────┘
                    ▼
           fanduel_scraper.py
           (Odds API / Selenium)
                    │
                    ▼
              ev_engine.py
    Model P  vs  Implied P (vig-removed)
         Edge = Model P − Fair P
         EV%  = p × (decimal − 1) − (1−p)
         Kelly fraction (quarter-Kelly)
                    │
                    ▼
        ev_opportunities table
        best_bets_YYYY-MM-DD.csv
```

---

## The +EV Formula

```
Fair implied probability = vig-removed over / (over + under)

Edge   = Model P − Fair P
EV %   = Model P × (decimal_odds − 1) − (1 − Model P)
Kelly  = (b × p − q) / b    (b = decimal−1, q = 1−p)

Bet recommended when:
  Edge  ≥ 4 percentage points
  EV %  ≥ 2 %
  Kelly > 0
Stake  = Quarter-Kelly × bankroll   (conservative)
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Weekly chunks for Statcast pulls | Baseball Savant throttles large single requests |
| pybaseball disk cache enabled | Avoids re-downloading identical date ranges |
| Isotonic probability calibration | XGBoost outputs are poorly calibrated by default |
| Poisson approximation for K props | Converts regression output to an over/under probability |
| Quarter-Kelly sizing | Full Kelly is too aggressive for noisy sports predictions |
| `fair_prob` (vig-removed) comparison | Comparing to raw implied prob understates true edge |

---

## Next Steps / Roadmap

- [ ] Park factor table population (currently hardcoded defaults)
- [ ] Optuna hyperparameter tuning loop
- [ ] SHAP waterfall plots for explainability
- [ ] Historical EV tracking & P&L dashboard
- [ ] Lineup scraper (probable pitcher, batting order) for pre-game feature assembly
- [ ] Expand to total bases, RBI, HR props
- [ ] MLflow model registry integration
