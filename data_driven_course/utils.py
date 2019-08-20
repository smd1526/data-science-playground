import numpy as np
import sys
from astropy.io import fits
from helper import running_stats
import matplotlib.pyplot as plt
import time
import statistics
from astropy.coordinates import SkyCoord
from astropy import units as u

# ARRAYS
def array_tests():
	fluxes = np.array([23.3, 42.1, 2.0, -3.2, 55.6])
	print(np.mean(fluxes))
	print(np.size(fluxes))
	print(np.std(fluxes)) # standard deviation

	a = np.array([[1,2,3], [4,5,6]])  # 2x3 array

# FILE DATA
def file_data_test1():
	data = []
	for line in open('test_data/data.csv'):
	  data.append(line.strip().split(','))

	data = np.asarray(data, float)
	print(data)

# or...

def file_data_test2():
	data = np.loadtxt('test_data/data.csv', delimiter=',')
	print(data)

	print(np.mean(data).round(1))
	print(np.median(data).round(1))

def mean_datasets(files):
	data = None
  
	if files is None or len(files) == 0:
		return None
  
	for _file in files:
		if data is None:
			data = np.loadtxt(_file, delimiter=',')
		else:
			data += np.loadtxt(_file, delimiter=',')

	mean = data/len(files)
	return np.round(mean, 1)

def mean_datasets_test():
	print(mean_datasets(['test_data/data1.csv', 'test_data/data2.csv', 'test_data/data3.csv']))
	# >>> mean_datasets(['data1.csv', 'data2.csv', 'data3.csv'])
	# array([[ 11.   11.9  13. ]
	#        [  9.5   6.8   9.4]
	#        [  7.2  11.1  12.5]
	#        [  8.8   7.3   9.2]
	#        [ 16.6  10.6  10.3]])

	print(mean_datasets(['test_data/data4.csv', 'test_data/data5.csv', 'test_data/data6.csv']))
	# >>> mean_datasets(['data4.csv', 'data5.csv', 'data6.csv'])
	# array([[-2.9  2.6  0.6 -5.4]
	#        [-4.4 -0.7  0.7 -0.2]
	#        [-1.7  2.5 -8.7 -5.4]])


# FITS files
# Note: astropy requres Python 3.5
def fits_test():
	hdulist = fits.open('test_data/image0.fits')
	hdulist.info()
	data = hdulist[0].data
	print(data.shape)

	# Plot the 2D array
	plt.imshow(data, cmap=plt.cm.viridis)
	plt.xlabel('x-pixels (RA)')
	plt.ylabel('y-pixels (Dec)')
	plt.colorbar()
	plt.show()

def mean_fits(files):
	if files is None or len(files) == 0:
		return None

	data = None
	for _file in files:
		hdulist = fits.open(_file)
		if data is None:
			data = hdulist[0].data
		else:
			data += hdulist[0].data
		hdulist.close()
	return data/len(files)


# TIMING

def time_stat(func, size, ntrials):
	# the time to generate the random array should not be included
	data = np.random.rand(size)
	# modify this function to time func with ntrials times using a new random array each time
	start = time.perf_counter()
	for _ in range(ntrials):
		res = func(data)
	end = time.perf_counter() - start
	# return the average run time
	return end / ntrials

def time_test():
	print('{:.6f}s for statistics.mean'.format(time_stat(statistics.mean, 10**5, 10)))
	print('{:.6f}s for np.mean'.format(time_stat(np.mean, 10**5, 1000)))


# MEMORY USE

def mem_use_test():
	a = 3
	b = 3.123
	c = [a, b]
	d = []
	for obj in [a, b, c, d]:
		print(obj, sys.getsizeof(obj))

	a = np.array([])
	b = np.array([1, 2, 3])
	c = np.zeros(10**6)

	for obj in [a, b, c]:
		print('sys:', sys.getsizeof(obj), 'np:', obj.nbytes)

	a = np.zeros(5, dtype=np.int32)
	b = np.zeros(5, dtype=np.float64)

	for obj in [a, b]:
		print('nbytes         :', obj.nbytes)
		print('size x itemsize:', obj.size*obj.itemsize)


def median_fits(files):
	start = time.time()
	data = []
	for _file in files:
		hdu = fits.open(_file)
		data.append(hdu[0].data)
		hdu.close()

	data_stack = np.stack(data, axis=2)
	median = np.median(data_stack, axis=2)
	memory = data_stack.nbytes/1024
	stop = time.time() - start
	return median, stop, memory


# BINAPPROX

def median_bins_fits(files, B):
	mean, std = running_stats(files)
	dim = mean.shape
	left_bin = np.zeros(dim)
	bins = np.zeros((dim[0], dim[1], B))
	bin_width = 2*std/B

	for _file in files:
		hdulist = fits.open(_file)
		data = hdulist[0].data
		for i in range(dim[0]):
			for j in range(dim[1]):
				value = data[i, j]
				_mean = mean[i, j]
				_std = std[i, j]

				if value < _mean - _std:
					left_bin[i, j] += 1
				elif value < _mean + _std:
					_bin = int((value - (_mean-_std))/bin_width[i,j])
					bins[i, j, _bin] += 1

	return mean, std, left_bin, bins

# median = median_approx_fits(['test_data/image{}.fits'.format(str(i)) for i in range(11)], 4)
def median_approx_fits(files, B):
	mean, std, left_bin, bins = median_bins_fits(files, B)
	dim = mean.shape
	mid = (len(files)+1)/2
	bin_width = 2*std/B
	median = np.zeros(dim)

	for i in range(dim[0]):
		for j in range(dim[1]):
			count = left_bin[i, j]
			for _bin, bincount in enumerate(bins[i, j]):
				count += bincount
				if count >= mid:
					break
			median[i, j] = mean[i, j] - std[i, j] + bin_width[i, j]*(_bin + 0.5)
	return median

def hms2dec(hours, mins, secs):
	return (15*(hours + mins/60 + secs/(60*60)))

def dms2dec(deg, amins, asecs):
	if deg < 0:
		return -1*((-1*deg) + amins/60 + asecs/(60*60))
	return (deg + amins/60 + asecs/(60*60))

# [right ascension in HMS, declination in HMS]
# [  0.     4.    35.65 -47.    36.    19.1 ]
# bss.dat is table2.dat from BSS AT20G
def import_bss():
	result = []
	data = np.loadtxt('test_data/bss.dat', usecols=range(1,7))
	for x in range(0, len(data)):
		result.append((x+1, hms2dec(data[x][0], data[x][1], data[x][2]), dms2dec(data[x][3], data[x][4], data[x][5])))
	return result

# super.csv is truncated from SuperCOSMOS survey
def import_super():
	result = []
	data = np.loadtxt('test_data/super.csv', delimiter=',', skiprows=1, usecols=[0, 1])
	for x in range(0, len(data)):
		result.append((x+1, data[x][0], data[x][1]))
	return result


### Crossmatch implementations ###

def angular_dist_degrees_simple(_a1, _d1, _a2, _d2):
	a1 = np.radians(_a1)
	a2 = np.radians(_a2)
	d1 = np.radians(_d1)
	d2 = np.radians(_d2)
	x1 = np.sin(np.abs(d1-d2)/2)**2
	x2 = np.cos(d1)*np.cos(d2)*(np.sin(abs(a1-a2)/2)**2)
	return np.degrees(2*np.arcsin(np.sqrt(x1+x2)))

def angular_dist_test():
	print(angular_dist_degrees_simple(21.07, 0.1, 21.15, 8.2))
	print(angular_dist_degrees_simple(10.3, -3, 24.3, -29))

def find_closest_basic_original(cat, ra, dc):
	closest = (cat[0][0], angular_dist_degrees_simple(ra, dc, cat[0][1], cat[0][2]))
	for x in range(1, len(cat)):
		dist = angular_dist_degrees_simple(ra, dc, cat[x][1], cat[x][2])
		if dist < closest[1]:
			closest = (cat[x][0], dist)
	return closest

def test_find_closest():
	cat = import_bss()
	print(find_closest_basic(cat, 175.3, -32.5)) # (156, 3.7670580226469053)
	print(find_closest_basic(cat, 32.2, 40.7)) # (26, 57.729135775621295)

#1. Select object from bss cat
#2. Go through all super cat objects, find closest to bss object
#3. If close enough, record match.
#4. Repeat for all bss objects
# Return (list of matches, list of no matches)
# list of matches = tuples of (1st object, 2nd object, dist)
# list of unmatches = ids from bss_cat, unmatched
def crossmatch_basic_original(bss_cat, super_cat, max_dist):
	matches = []
	no_matches = []

	# bss_obj = (1, 1.1485416666666666, -47.60530555555556)
	for bss_obj in bss_cat:
		(closest, dist) = find_closest_basic_original(super_cat, bss_obj[1], bss_obj[2])
		if dist > max_dist:
			no_matches.append(bss_obj[0])
		else:
			matches.append((bss_obj[0], closest, dist))
	return (matches, no_matches)

def test_crossmatch():
	bss_cat = import_bss()
	super_cat = import_super()

	max_dist = 40/3600
	matches, no_matches = crossmatch_basic_original(bss_cat, super_cat, max_dist)
	print(matches[:3])
	print(no_matches[:3])
	print(len(no_matches))

	max_dist = 5/3600
	matches, no_matches = crossmatch(bss_cat, super_cat, max_dist)
	print(matches[:3])
	print(no_matches[:3])
	print(len(no_matches))


## CROSSMATCH TIMING ##

RA_INDEX = 0
DC_INDEX = 1

def crossmatch_basic(bss_cat, super_cat, max_dist):
	matches = []
	no_matches = []

	def find_closest_basic(cat, ra, dc):
		closest = (0, angular_dist_degrees_simple(ra, dc, cat[0][RA_INDEX], cat[0][DC_INDEX]))
		for x, obj in enumerate(cat):
			dist = angular_dist_degrees_simple(ra, dc, obj[RA_INDEX], obj[DC_INDEX])
			if dist < closest[1]:
				closest = (x, dist)
		return closest

	for i, bss_obj in enumerate(bss_cat):
		(closest, dist) = find_closest_basic(super_cat, bss_obj[RA_INDEX], bss_obj[DC_INDEX])
		if dist > max_dist:
			no_matches.append(i)
		else:
			matches.append((i, closest, dist))
	return (matches, no_matches)

# Calculate angular distance to single object
def angular_dist(a1, d1, a2, d2):
	x1 = np.sin(np.abs(d1-d2)/2)**2
	x2 = np.cos(d1)*np.cos(d2)*(np.sin(abs(a1-a2)/2)**2)
	return np.degrees(2*np.arcsin(np.sqrt(x1+x2)))

# converts bss_cat and super_cat to radians before distance calculations
def crossmatch_convert_first(bss_cat, super_cat, max_dist):

	def find_closest(cat, ra, dc):
		closest = (0, angular_dist(ra, dc, cat[0][RA_INDEX], cat[0][DC_INDEX]))
		for x in range(1, len(cat)):
			dist = angular_dist(ra, dc, cat[x][RA_INDEX], cat[x][DC_INDEX])
			if dist < closest[1]:
				closest = (x, dist)
		return closest

	matches = []
	no_matches = []

	bss_cat = np.radians(bss_cat)
	super_cat = np.radians(super_cat)

	for i, bss_obj in enumerate(bss_cat):
		(closest, dist) = find_closest(super_cat, bss_obj[RA_INDEX], bss_obj[DC_INDEX])
		if dist > max_dist:
			no_matches.append(i)
		else:
			matches.append((i, closest, dist))

	return (matches, no_matches)

# Calculate angular distance to array of objects
def angular_dist_array(ra, dec, ra2s, dec2s):
    x1 = np.sin(np.abs(dec-dec2s)/2)**2
    x2 = np.cos(dec)*np.cos(dec2s)*(np.sin(abs(ra-ra2s)/2)**2)
    return 2*np.arcsin(np.sqrt(x1+x2))

# Optimization: Calculate distance over all super_cat objects at once
# 	Uses numpy arrays which is faster (uses C optimizations)
# 	Uses the min of the resulting distances array
def crossmatch_with_dist_array(bss_cat, super_cat, max_dist):

	def find_closest(cat, ra, dec):
	    ra2s = cat[:,RA_INDEX]
	    dec2s = cat[:,DC_INDEX]
	    dists = angular_dist_array(ra, dec, ra2s, dec2s)
	    dists = np.degrees(dists)
	    min_dist_id = np.argmin(dists)
	    return (min_dist_id, dists[min_dist_id])

	matches = []
	no_matches = []

	bss_cat = np.radians(bss_cat)
	super_cat = np.radians(super_cat)

	for i, bss_obj in enumerate(bss_cat):
		(closest, dist) = find_closest(super_cat, bss_obj[RA_INDEX], bss_obj[DC_INDEX])
		if dist > max_dist:
			no_matches.append(i)
		else:
			matches.append((i, closest, dist))

	return (matches, no_matches)

# Optimization: Track declinations from second catalogue
# 	When dec2 > dec1 + max_dist: stop searching for closest object
def crossmatch_track_decs(bss_cat, super_cat, max_dist):

	def find_closest(cat, ra, dc, max_radius):
		dec2s = cat[:,DC_INDEX]
		dec2s_sort_order = np.argsort(dec2s)
		dec2s_sorted = np.sort(dec2s)
		closest = (dec2s_sort_order[0], angular_dist(ra, dc, cat[dec2s_sort_order[0]][RA_INDEX], dec2s_sorted[0]))
		for x in range(1, len(dec2s_sorted)):
			dc2 = dec2s_sorted[x]
			if dc2 > (dc+max_radius):
				return closest

		dist = angular_dist(ra, dc, cat[dec2s_sort_order[x]][RA_INDEX], dc2)
		if dist < closest[1]:
			closest = (dec2s_sort_order[x], dist)
		return closest

	matches = []
	no_matches = []

	bss_cat = np.radians(bss_cat)
	super_cat = np.radians(super_cat)

	for i, bss_obj in enumerate(bss_cat):
		(closest, dist) = find_closest(super_cat, bss_obj[RA_INDEX], bss_obj[DC_INDEX], max_dist)
		if dist > max_dist:
			no_matches.append(i)
		else:
			matches.append((i, closest, dist))
  
	return (matches, no_matches)

# Optimization: Only search for second catalogue objects with dec between [dec1-max_dist, dec1+max_dist]
# 	Use searchsorted to binary search for start and end indices
def crossmatch_limit_by_decs(bss_cat, super_cat, max_dist):

	def find_closest(cat, ra, dc, max_radius):
		dec2s = cat[:,DC_INDEX]
		dec2s_sort_order = np.argsort(dec2s)
		dec2s_sorted = np.sort(dec2s)
		start = np.searchsorted(dec2s_sorted, dc-max_radius, side='left')
		end = np.searchsorted(dec2s_sorted, dc+max_radius, side='right')

		closest = (dec2s_sort_order[start], angular_dist(ra, dc, cat[dec2s_sort_order[start]][RA_INDEX], dec2s_sorted[start]))
		for x in range(start+1, end):
			dist = angular_dist(ra, dc, cat[dec2s_sort_order[x]][RA_INDEX], cat[dec2s_sort_order[x]][DC_INDEX])
			if dist < closest[1]:
				closest = (dec2s_sort_order[x], dist)
		return closest

	matches = []
	no_matches = []

	bss_cat = np.radians(bss_cat)
	super_cat = np.radians(super_cat)

	for i, bss_obj in enumerate(bss_cat):
		(closest, dist) = find_closest(super_cat, bss_obj[RA_INDEX], bss_obj[DC_INDEX], max_dist)
		if dist > max_dist:
			no_matches.append(i)
		else:
			matches.append((i, closest, dist))

	return (matches, no_matches)

# Use Astropy's implementation of k-d trees to find matches
# Need to manually filter those matches outside the max distance
def crossmatch_with_kd_trees(bss_cat, super_cat, max_dist):
	sky_cat1 = SkyCoord(bss_cat*u.degree, frame='icrs')
	sky_cat2 = SkyCoord(super_cat*u.degree, frame='icrs')

	# closest_id and closest_dists give the matching object's row index in sky_cat2 and the distance to it  
	closest_ids, closest_dists, closest_dists3d = sky_cat1.match_to_catalog_sky(sky_cat2)

	no_matches = []
	matches = []
	for i, dist in enumerate(closest_dists):
		if dist.value > max_dist:
			no_matches.append(i)
		else:
			matches.append((i, closest_ids[i], dist.value))

	return (matches, no_matches)


def test_crossmatches():
	crossmatch_functions = [
							("Crossmatch Basic", crossmatch_basic),
							("Crossmatch Convert First", crossmatch_convert_first),
							("Crossmatch With Dist Array", crossmatch_with_dist_array),
							("Crossmatch Track Decs", crossmatch_track_decs),
							("Crossmatch Limit By Decs", crossmatch_limit_by_decs),
							("Crossmatch With K-D Trees", crossmatch_with_kd_trees)
							]
	cat1 = np.array([[180, 30], [45, 10], [300, -45]])
	cat2 = np.array([[180, 32], [55, 10], [302, -44]])
	# matches: [(0, 0, 2.0000000000000036), (2, 2, 1.7420109046547128)]
	# unmatched: [1]

	# def create_cat(n):
	# 	ras = np.random.uniform(0, 360, size=(n, 1))
	# 	decs = np.random.uniform(-90, 90, size=(n, 1))
	# 	return np.hstack((ras, decs))

	# np.random.seed(0)
	# cat1 = create_cat(10)
	# cat2 = create_cat(20)

	for i, (name, fnc) in enumerate(crossmatch_functions):
		print("**** Testing function %d: %s ****" % (i, name))
		start = time.perf_counter()
		matches, no_matches = fnc(cat1, cat2, 5);
		total_time = time.perf_counter() - start
		print("\tMatches: ", matches)
		print("\tNo matches: ", no_matches)
		print("\tTotal Time: ", total_time)


if __name__ == '__main__':
	test_crossmatches()
