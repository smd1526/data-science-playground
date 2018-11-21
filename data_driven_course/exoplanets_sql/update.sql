UPDATE Planet
SET kepler_name = NULL
WHERE UPPER(status) != 'CONFIRMED';

DELETE FROM Planet
WHERE radius < 0;

SELECT * FROM Planet;
