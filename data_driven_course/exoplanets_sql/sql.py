import psycopg2
import numpy as np

def select_all(table):
	conn = psycopg2.connect(dbname='db', user='grok')
	cursor = conn.cursor() # The cursor is the object that interfaces with the database.

	cursor.execute('SELECT * FROM %s;' % table)
	return cursor.fetchall()

def column_stats(table, column):
	conn = psycopg2.connect(dbname='db', user='grok')
	cursor = conn.cursor()

	cursor.execute('SELECT %s FROM %s;' % (column, table))
	column = np.array(cursor.fetchall())

	return (np.mean(column), np.median(column))

def query(star_csv, planet_csv):
    joined = []
    star_data = np.loadtxt(star_csv, delimiter=',', usecols=(0,2))
    planet_data = np.loadtxt(planet_csv, delimiter=',', usecols=(0,5))
    star_data = star_data[star_data[:, 1] > 1, :]
    for planet in planet_data:
        for star in star_data:
            if planet[0] == star[0]:
                joined.append((planet[1]/star[1],))
    joined = np.array(joined)
    return joined[np.argsort(joined[:, 0]), :]
