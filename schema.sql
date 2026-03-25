-- =============================================================================
-- MLB Predictive Analytics Engine - Database Schema
-- Compatible with PostgreSQL 14+ / SQLite 3.35+
-- =============================================================================

-- -----------------------------------------------------------------------------
-- PLAYERS
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS players (
    player_id       INTEGER PRIMARY KEY,          -- MLB/MLBAM ID
    fangraphs_id    TEXT,
    name_first      TEXT NOT NULL,
    name_last       TEXT NOT NULL,
    full_name       TEXT GENERATED ALWAYS AS (name_first || ' ' || name_last) STORED,
    bats            CHAR(1) CHECK (bats IN ('L','R','S')),
    throws          CHAR(1) CHECK (throws IN ('L','R')),
    primary_pos     TEXT,
    team            TEXT,
    active          BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_players_name ON players(name_last, name_first);
CREATE INDEX IF NOT EXISTS idx_players_team ON players(team);

-- -----------------------------------------------------------------------------
-- TEAMS
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS teams (
    team_abbrev     TEXT PRIMARY KEY,             -- e.g. 'NYY', 'BOS'
    team_name       TEXT NOT NULL,
    league          CHAR(2) CHECK (league IN ('AL','NL')),
    division        TEXT,
    park_name       TEXT,
    park_factor     REAL DEFAULT 100.0            -- run-adjusted park factor
);

-- -----------------------------------------------------------------------------
-- GAMES
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS games (
    game_pk         INTEGER PRIMARY KEY,          -- Statcast game_pk
    game_date       DATE NOT NULL,
    season          INTEGER NOT NULL,
    home_team       TEXT REFERENCES teams(team_abbrev),
    away_team       TEXT REFERENCES teams(team_abbrev),
    home_score      INTEGER,
    away_score      INTEGER,
    game_type       CHAR(1) DEFAULT 'R',          -- R=Regular, P=Playoff, S=Spring
    venue_name      TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_games_date    ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_season  ON games(season);
CREATE INDEX IF NOT EXISTS idx_games_home    ON games(home_team);
CREATE INDEX IF NOT EXISTS idx_games_away    ON games(away_team);

-- -----------------------------------------------------------------------------
-- STATCAST PITCH-LEVEL DATA  (core raw table — potentially millions of rows)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS statcast_pitches (
    pitch_id            BIGSERIAL PRIMARY KEY,
    game_pk             INTEGER REFERENCES games(game_pk),
    game_date           DATE NOT NULL,
    season              INTEGER NOT NULL,

    -- Participants
    pitcher_id          INTEGER REFERENCES players(player_id),
    batter_id           INTEGER REFERENCES players(player_id),
    pitcher_team        TEXT,
    batter_team         TEXT,

    -- Pitch characteristics
    pitch_type          TEXT,                     -- FF, SL, CU, CH, etc.
    release_speed       REAL,                     -- mph
    release_spin_rate   REAL,                     -- rpm
    release_extension   REAL,                     -- ft
    pfx_x               REAL,                     -- horizontal break (ft)
    pfx_z               REAL,                     -- vertical break (ft)
    plate_x             REAL,
    plate_z             REAL,
    zone                INTEGER,                  -- 1-14 Statcast zone

    -- Outcome
    description         TEXT,                     -- called_strike, swinging_strike, ball, hit_into_play …
    type                TEXT,                     -- S, B, X
    events              TEXT,                     -- single, strikeout, home_run …
    bb_type             TEXT,                     -- ground_ball, fly_ball, line_drive, popup

    -- Batted ball metrics
    launch_speed        REAL,                     -- exit velo mph
    launch_angle        REAL,                     -- degrees
    hit_distance_sc     REAL,                     -- ft
    estimated_ba_using_speedangle REAL,
    estimated_woba_using_speedangle REAL,
    woba_value          REAL,

    -- Count / situation
    balls               INTEGER,
    strikes             INTEGER,
    outs_when_up        INTEGER,
    inning              INTEGER,
    inning_topbot       TEXT,
    on_1b               INTEGER,                  -- runner id or NULL
    on_2b               INTEGER,
    on_3b               INTEGER,
    stand               CHAR(1),                  -- batter handedness this PA
    p_throws            CHAR(1),

    -- Flags
    is_swing            BOOLEAN,
    is_whiff            BOOLEAN,
    is_in_zone          BOOLEAN,
    is_hard_hit         BOOLEAN GENERATED ALWAYS AS (launch_speed >= 95) STORED,

    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Partition-friendly indexes
CREATE INDEX IF NOT EXISTS idx_sp_game_date    ON statcast_pitches(game_date);
CREATE INDEX IF NOT EXISTS idx_sp_season       ON statcast_pitches(season);
CREATE INDEX IF NOT EXISTS idx_sp_pitcher      ON statcast_pitches(pitcher_id, game_date);
CREATE INDEX IF NOT EXISTS idx_sp_batter       ON statcast_pitches(batter_id, game_date);
CREATE INDEX IF NOT EXISTS idx_sp_events       ON statcast_pitches(events) WHERE events IS NOT NULL;

-- -----------------------------------------------------------------------------
-- FANGRAPHS BATTER SEASONAL STATS
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fg_batter_season (
    id              BIGSERIAL PRIMARY KEY,
    player_id       INTEGER REFERENCES players(player_id),
    season          INTEGER NOT NULL,
    team            TEXT,
    g               INTEGER,   pa              INTEGER,   ab              INTEGER,
    h               INTEGER,   doubles         INTEGER,   triples         INTEGER,
    hr              INTEGER,   rbi             INTEGER,   bb              INTEGER,
    hbp             INTEGER,   so              INTEGER,   sb              INTEGER,
    -- Rate stats
    avg             REAL,  obp             REAL,  slg             REAL,
    ops             REAL,  woba            REAL,  wrc_plus        REAL,
    -- Statcast-era metrics
    ev              REAL,   -- avg exit velo
    max_ev          REAL,
    la              REAL,   -- avg launch angle
    hard_hit_pct    REAL,
    barrel_pct      REAL,
    k_pct           REAL,
    bb_pct          REAL,
    sprint_speed    REAL,
    -- WAR
    war             REAL,
    UNIQUE (player_id, season, team)
);

CREATE INDEX IF NOT EXISTS idx_fg_batter_player ON fg_batter_season(player_id, season);

-- -----------------------------------------------------------------------------
-- FANGRAPHS PITCHER SEASONAL STATS
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fg_pitcher_season (
    id              BIGSERIAL PRIMARY KEY,
    player_id       INTEGER REFERENCES players(player_id),
    season          INTEGER NOT NULL,
    team            TEXT,
    g               INTEGER,   gs              INTEGER,   ip              REAL,
    w               INTEGER,   l               INTEGER,   sv              INTEGER,
    -- Standard
    era             REAL,  fip             REAL,  xfip            REAL,  siera   REAL,
    k_9             REAL,  bb_9            REAL,  hr_9            REAL,
    k_pct           REAL,  bb_pct          REAL,  k_bb            REAL,
    -- Pitch mix %
    fb_pct          REAL,  sl_pct          REAL,  ct_pct          REAL,
    cb_pct          REAL,  ch_pct          REAL,
    -- Statcast
    avg_fastball_velo REAL,
    whiff_pct       REAL,
    zone_pct        REAL,
    chase_pct       REAL,
    csw_pct         REAL,   -- called_strike + whiff %
    -- WAR
    war             REAL,
    UNIQUE (player_id, season, team)
);

CREATE INDEX IF NOT EXISTS idx_fg_pitcher_player ON fg_pitcher_season(player_id, season);

-- -----------------------------------------------------------------------------
-- ROLLING FEATURE STORE  (pre-computed, refreshed nightly)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS batter_rolling_features (
    id              BIGSERIAL PRIMARY KEY,
    player_id       INTEGER REFERENCES players(player_id),
    as_of_date      DATE NOT NULL,
    window_games    INTEGER NOT NULL DEFAULT 5,
    -- Rolling averages
    avg_exit_velo   REAL,
    avg_launch_angle REAL,
    hard_hit_pct    REAL,
    barrel_pct      REAL,
    k_pct           REAL,
    bb_pct          REAL,
    ba              REAL,
    obp             REAL,
    slg             REAL,
    woba            REAL,
    -- Sample size
    pa_in_window    INTEGER,
    games_in_window INTEGER,
    UNIQUE (player_id, as_of_date, window_games)
);

CREATE INDEX IF NOT EXISTS idx_brf_player_date ON batter_rolling_features(player_id, as_of_date);

CREATE TABLE IF NOT EXISTS pitcher_rolling_features (
    id                  BIGSERIAL PRIMARY KEY,
    player_id           INTEGER REFERENCES players(player_id),
    as_of_date          DATE NOT NULL,
    window_games        INTEGER NOT NULL DEFAULT 5,
    -- Rolling averages
    avg_fastball_velo   REAL,
    whiff_pct           REAL,
    zone_pct            REAL,
    chase_pct           REAL,
    k_pct               REAL,
    bb_pct              REAL,
    hr_per_9            REAL,
    era_in_window       REAL,
    -- Sample size
    batters_faced       INTEGER,
    games_in_window     INTEGER,
    UNIQUE (player_id, as_of_date, window_games)
);

CREATE INDEX IF NOT EXISTS idx_prf_player_date ON pitcher_rolling_features(player_id, as_of_date);

-- -----------------------------------------------------------------------------
-- MATCHUP FEATURES  (batter vs pitcher, pre-game)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS matchup_features (
    id              BIGSERIAL PRIMARY KEY,
    game_pk         INTEGER REFERENCES games(game_pk),
    game_date       DATE NOT NULL,
    batter_id       INTEGER REFERENCES players(player_id),
    pitcher_id      INTEGER REFERENCES players(player_id),
    -- Historical H2H (career)
    h2h_pa          INTEGER DEFAULT 0,
    h2h_hits        INTEGER DEFAULT 0,
    h2h_k           INTEGER DEFAULT 0,
    h2h_bb          INTEGER DEFAULT 0,
    h2h_ba          REAL,
    -- Batter rolling snapshot (at game time)
    b_avg_ev        REAL,
    b_avg_la        REAL,
    b_k_pct         REAL,
    b_bb_pct        REAL,
    b_woba_rolling  REAL,
    -- Pitcher rolling snapshot (at game time)
    p_avg_velo      REAL,
    p_whiff_pct     REAL,
    p_zone_pct      REAL,
    p_k_pct         REAL,
    -- Park / handedness
    park_factor     REAL,
    batter_hand     CHAR(1),
    pitcher_hand    CHAR(1),
    UNIQUE (game_pk, batter_id, pitcher_id)
);

-- -----------------------------------------------------------------------------
-- MODEL PREDICTIONS
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_predictions (
    id                  BIGSERIAL PRIMARY KEY,
    prediction_ts       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version       TEXT NOT NULL,
    game_pk             INTEGER REFERENCES games(game_pk),
    game_date           DATE NOT NULL,
    batter_id           INTEGER REFERENCES players(player_id),
    pitcher_id          INTEGER REFERENCES players(player_id),
    -- Base hit model
    p_base_hit          REAL,   -- probability 0-1
    p_base_hit_lo       REAL,   -- 90% CI lower
    p_base_hit_hi       REAL,   -- 90% CI upper
    -- Strikeout regression
    pred_k              REAL,   -- expected K count (pitcher)
    pred_k_lo           REAL,
    pred_k_hi           REAL,
    -- Metadata
    features_snapshot   JSONB,  -- full feature vector at prediction time
    UNIQUE (model_version, game_pk, batter_id, pitcher_id)
);

-- -----------------------------------------------------------------------------
-- FANDUEL ODDS  (scraped / API)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fanduel_odds (
    id              BIGSERIAL PRIMARY KEY,
    scraped_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    game_date       DATE NOT NULL,
    game_pk         INTEGER,                      -- matched post-scrape; nullable
    market_type     TEXT NOT NULL,                -- 'moneyline', 'player_prop'
    prop_type       TEXT,                         -- 'hits', 'strikeouts', 'total_bases', …
    team            TEXT,
    player_id       INTEGER REFERENCES players(player_id),
    player_name     TEXT,                         -- raw scraped name (pre-match)
    line            REAL,                         -- over/under line (props)
    over_price      INTEGER,                      -- American odds e.g. -115
    under_price     INTEGER,
    moneyline_home  INTEGER,
    moneyline_away  INTEGER,
    -- Derived
    implied_prob_over  REAL,                      -- computed from over_price
    implied_prob_under REAL,
    vig_pct         REAL                          -- book's margin
);

CREATE INDEX IF NOT EXISTS idx_fd_date       ON fanduel_odds(game_date);
CREATE INDEX IF NOT EXISTS idx_fd_player     ON fanduel_odds(player_id, game_date);
CREATE INDEX IF NOT EXISTS idx_fd_market     ON fanduel_odds(market_type, prop_type);

-- -----------------------------------------------------------------------------
-- EV OPPORTUNITIES  (+EV bets identified by the engine)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ev_opportunities (
    id                  BIGSERIAL PRIMARY KEY,
    identified_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    game_date           DATE NOT NULL,
    game_pk             INTEGER,
    player_id           INTEGER REFERENCES players(player_id),
    player_name         TEXT,
    market_type         TEXT,
    prop_type           TEXT,
    bet_side            TEXT,                     -- 'over' or 'under'
    fanduel_line        REAL,
    fanduel_price       INTEGER,                  -- American odds
    implied_prob        REAL,                     -- book implied probability (vig-removed)
    model_prob          REAL,                     -- AI predicted probability
    edge                REAL,                     -- model_prob - implied_prob
    ev_pct              REAL,                     -- expected value %
    kelly_fraction      REAL,                     -- full Kelly stake suggestion
    confidence          TEXT CHECK (confidence IN ('HIGH','MEDIUM','LOW')),
    model_version       TEXT,
    -- Resolution
    settled             BOOLEAN DEFAULT FALSE,
    outcome             TEXT,                     -- 'win', 'loss', 'push', 'void'
    actual_result       REAL                      -- actual stat value
);

CREATE INDEX IF NOT EXISTS idx_ev_date       ON ev_opportunities(game_date);
CREATE INDEX IF NOT EXISTS idx_ev_player     ON ev_opportunities(player_id);
CREATE INDEX IF NOT EXISTS idx_ev_settled    ON ev_opportunities(settled, game_date);

-- -----------------------------------------------------------------------------
-- MODEL REGISTRY  (track trained models)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_registry (
    id              SERIAL PRIMARY KEY,
    version         TEXT UNIQUE NOT NULL,
    model_type      TEXT NOT NULL,                -- 'base_hit_classifier', 'k_regressor'
    algorithm       TEXT,                         -- 'xgboost', 'lightgbm', …
    train_start     DATE,
    train_end       DATE,
    features        JSONB,
    hyperparams     JSONB,
    -- Eval metrics
    auc_roc         REAL,
    log_loss        REAL,
    rmse            REAL,
    mae             REAL,
    brier_score     REAL,
    artifact_path   TEXT,                         -- local or S3 path to .pkl
    is_production   BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- VIEWS  (convenience)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_todays_ev_bets AS
SELECT
    e.game_date,
    p.full_name          AS player,
    t.team_abbrev        AS team,
    e.prop_type,
    e.fanduel_line,
    e.bet_side,
    e.fanduel_price,
    ROUND(e.implied_prob * 100, 1)  AS implied_prob_pct,
    ROUND(e.model_prob  * 100, 1)   AS model_prob_pct,
    ROUND(e.edge        * 100, 1)   AS edge_pct,
    ROUND(e.ev_pct      * 100, 2)   AS ev_pct,
    e.confidence,
    e.kelly_fraction
FROM   ev_opportunities e
JOIN   players p ON p.player_id = e.player_id
LEFT JOIN teams t ON t.team_abbrev = p.team
WHERE  e.game_date = CURRENT_DATE
  AND  e.settled   = FALSE
  AND  e.edge      > 0.04
ORDER  BY e.ev_pct DESC;
