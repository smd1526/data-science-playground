SELECT p.koi_name, p.radius, s.radius
FROM Planet p
JOIN Star s USING(kepler_id)
WHERE p.kepler_id IN (
  SELECT s.kepler_id FROM Star s
  ORDER BY radius DESC
  LIMIT 5
);
