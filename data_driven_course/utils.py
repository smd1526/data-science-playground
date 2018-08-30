import numpy as np
import sys
from astropy.io import fits
import matplotlib.pyplot as plt
import time
import statistics

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

if __name__ == '__main__':
	mem_use_test()


