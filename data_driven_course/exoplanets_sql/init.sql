CREATE TABLE Star (
  kepler_id INTEGER NOT NULL,
  koi_name VARCHAR(20) NOT NULL,
  t_eff INTEGER,
  radius FLOAT(5),
  PRIMARY KEY (koi_name)
);

-- COPY Star (kepler_id, koi_name, t_eff, radius) FROM 'stars.csv' CSV;
.mode csv
.import test_data/stars.csv Star

CREATE TABLE Planet (
  kepler_id INTEGER NOT NULL,
  koi_name VARCHAR(20) NOT NULL,
  kepler_name VARCHAR(20),
  status VARCHAR(20) NOT NULL,
  period FLOAT,
  radius FLOAT,
  t_eq INTEGER,
  PRIMARY KEY (koi_name)
);

-- COPY Planet (kepler_id, koi_name, kepler_name, status, period, radius, t_eq) FROM 'planets.csv' CSV;
.mode csv
.import test_data/planets.csv Planet