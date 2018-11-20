SELECT kepler_name, radius FROM Planet WHERE kepler_name IS NOT NULL AND status = 'CONFIRMED' AND radius BETWEEN 1 AND 3;
