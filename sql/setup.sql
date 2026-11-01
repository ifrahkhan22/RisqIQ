
DROP TABLE IF EXISTS market;
CREATE TABLE market (
    date                           DATE,
    duoarea                        TEXT,
    area_name                      TEXT,
    product                        TEXT,
    product_name                   TEXT,
    process                        TEXT,
    process_name                   TEXT,
    series                         TEXT,
    series_description             TEXT,
    price_usd_per_barrel           DOUBLE PRECISION,
    units                          TEXT,
    daily_return_pct               DOUBLE PRECISION,
    annualized_volatility_20d_pct  DOUBLE PRECISION,
    prev_20d_std                   DOUBLE PRECISION,
    shock_score                    DOUBLE PRECISION,
    vol_shock                      BOOLEAN,
    large_move_5pct                BOOLEAN
);
COPY market FROM 'C:/Users/Public/risqiq/brent_market_features.csv' WITH (FORMAT csv, HEADER true);
 
-- 2. Model risk scores for the test period
DROP TABLE IF EXISTS risk_scores;
CREATE TABLE risk_scores (
    date         DATE,
    risk_prob    DOUBLE PRECISION,
    target_next5 DOUBLE PRECISION
);
COPY risk_scores FROM 'C:/Users/Public/risqiq/test_predictions.csv' WITH (FORMAT csv, HEADER true);
 
-- 3. Volatility forecasts and what actually happened
DROP TABLE IF EXISTS vol_forecasts;
CREATE TABLE vol_forecasts (
    date          DATE,
    forecast_vol  DOUBLE PRECISION,
    actual_vol    DOUBLE PRECISION
);
COPY vol_forecasts FROM 'C:/Users/Public/risqiq/volatility_forecasts.csv' WITH (FORMAT csv, HEADER true);
 
-- 4. Events with sources (typed in directly, no file needed)
DROP TABLE IF EXISTS events;
CREATE TABLE events (
    start_date  DATE,
    end_date    DATE,
    event       TEXT,
    source_url  TEXT,
    status      TEXT
);
INSERT INTO events VALUES
    ('2016-11-30', '2016-12-01', 'OPEC agrees a production cut in Vienna',
     'https://www.opec.org/pr-detail/458-30-nov-2016.html', 'sourced'),
    ('2020-03-06', '2020-03-09', 'OPEC+ talks collapse and Saudi-Russia price war',
     'https://www.spglobal.com/en/research-insights/articles/oil-prices-plunge-in-response-to-the-collapse-of-the-opec-alliance', 'sourced'),
    ('2022-03-01', '2022-03-02', 'Russia invasion of Ukraine pushes prices above 100 dollars',
     'https://www.eia.gov/TODAYinENERGY/detail.php?id=51498', 'sourced'),
    ('2026-03-01', '2026-03-31', 'Iran war and Hormuz disruption',
     'https://www.eia.gov/Todayinenergy/detail.php?id=67424', 'sourced');
 
-- 5. Quick check: expect about 2992 rows in market and 444 in risk_scores
SELECT 'market' AS table_name, COUNT(*) AS row_count FROM market
UNION ALL SELECT 'risk_scores', COUNT(*) FROM risk_scores
UNION ALL SELECT 'vol_forecasts', COUNT(*) FROM vol_forecasts
UNION ALL SELECT 'events', COUNT(*) FROM events;