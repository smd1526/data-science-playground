-- STDDEV is not available in sqlite3
-- Empy value is not matching NULL in sqlite3
-- SELECT MIN(radius), MAX(radius), AVG(radius), STDDEV(radius) FROM Planet WHERE kepler_name IS NULL;

SELECT MIN(radius), MAX(radius), AVG(radius) FROM Planet WHERE kepler_name IS "";