SELECT Star.radius, COUNT(Planet.koi_name)
From Star
JOIN Planet USING (kepler_id)
GROUP BY Star.kepler_id
HAVING COUNT(Planet.koi_name) > 1 AND Star.radius > 1
ORDER BY Star.radius DESC
