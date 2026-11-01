-- Q1: Yearly risk summary
SELECT EXTRACT(YEAR FROM date)::int AS year,
       COUNT(*) AS trading_days,
       AVG(annualized_volatility_20d_pct) AS avg_volatility_pct,
       MAX(annualized_volatility_20d_pct) AS peak_volatility_pct,
       SUM(large_move_5pct::int) AS moves_5pct_or_more,
       SUM(vol_shock::int) AS vol_adjusted_shocks,
       MIN(daily_return_pct) AS worst_day_pct
FROM market
GROUP BY 1
ORDER BY 1;

-- Q2: Ten months with the most large moves (5 percent or more in one day)
SELECT TO_CHAR(date, 'YYYY-MM') AS month,
       SUM(large_move_5pct::int) AS large_moves,
       COUNT(*) AS trading_days
FROM market
WHERE daily_return_pct IS NOT NULL
GROUP BY 1
ORDER BY large_moves DESC, month
LIMIT 10;

-- Q3: What happened inside each sourced event window?
SELECT e.start_date, e.end_date, e.event,
       COUNT(*) AS trading_days,
       SUM(m.large_move_5pct::int) AS large_moves,
       MAX(ABS(m.daily_return_pct)) AS biggest_move_pct
FROM events e
JOIN market m ON m.date BETWEEN e.start_date AND e.end_date
WHERE e.status = 'sourced'
GROUP BY e.start_date, e.end_date, e.event
ORDER BY e.start_date;

-- Q4: Does a higher risk score mean more shocks? (group 5 = highest scores)
WITH grouped AS (
    SELECT risk_prob, target_next5,
           NTILE(5) OVER (ORDER BY risk_prob) AS score_group
    FROM risk_scores
)
SELECT score_group,
       COUNT(*) AS days,
       MIN(risk_prob) AS lowest_score,
       MAX(risk_prob) AS highest_score,
       100 * AVG(target_next5) AS pct_followed_by_shock
FROM grouped
GROUP BY score_group
ORDER BY score_group;

-- Q5: Volatility forecast vs reality by year (annualized, 15.87 is the square root of 252)
SELECT EXTRACT(YEAR FROM date)::int AS year,
       COUNT(*) AS days,
       15.87 * AVG(actual_vol) AS actual_vol,
       15.87 * AVG(forecast_vol) AS forecast_vol,
       100 * AVG(CASE WHEN forecast_vol < actual_vol THEN 1 ELSE 0 END) AS pct_days_forecast_too_low
FROM vol_forecasts
GROUP BY 1
ORDER BY 1;