SELECT kepler_id, COUNT(koi_name)
FROM Planet 
GROUP BY kepler_id
HAVING COUNT(koi_name) > 1
ORDER BY COUNT(koi_name) desc;
