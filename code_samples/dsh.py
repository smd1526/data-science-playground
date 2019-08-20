# Code samples/ideas copied/derived/modified from:
# Python Data Science Handbook (by Jake VanderPlas, https://jakevdp.github.io/PythonDataScienceHandbook/)

import numpy as np
import matplotlib.pyplot as plt
import seaborn

def create_from_pyarray():
	return np.array([1, 4, 2, 5, 3])

def creat_multid_array():
	return np.array([range(i, i+3) for i in [2, 4, 6]])

def create_zero_array():
	return np.zeros(10, dtype=int)

def create_ones_array():
	return np.ones((3,5), dtype=float)

def create_filled_array():
	return np.full((3,5), 3.14)

def create_ranged_array():
	return np.arange(0, 20, 2)

def create_random_array():
	return np.random.random((3,3))

def create_random_int_array():
	return np.random.randint(0, 10, (3,3))

def dump_array_attrs():
	np.random.seed(0)
	x = np.random.randint(10, size=(3,4,5))
	print("x ndim: ", x.ndim)
	print("x shape: ", x.shape)
	print("x size: ", x.size)
	print("x dtype: ", x.dtype)
	print("x itemsize: ", x.itemsize)
	print("x nbytes: ", x.nbytes)

# Note: slices are views, not copies. Modifying a subarray affects the original array
def slice_sample():
	x = np.arange(10)
	x[:5] # first five elements
	x[5:] # elements after index 5
	x[4:7] # middle subarray
	x[::2] # every other element
	x[1::2] # every other element starting at index 1

def multid_slice_sample():
	x = np.random.randint(10, size=(3,4))
	x[:2, :3] # two rows, three columns
	x[:, ::2] # all rows, every other column	
	x[:, 0] # first column, all rows
	x[0, :] # first row, all columns

def reshape_sample():
	x = np.arange(1, 10) # array([1, 2, 3, 4, 5, 6, 7, 8, 9])
	x.reshape(3,3) # Note: does not affect the shape of x
	# array([[1, 2, 3],
	# 		 [4, 5, 6],
	#		 [7, 8, 9]])

def newaxis_sample():
	x = np.array([1, 2, 3]) # array([1, 2, 3])
	x.reshape((1,3)) # array([[1, 2, 3]])
	x[np.newaxis, :] # array([[1, 2, 3]])

	x.reshape((3,1))
	# array([[1],
	#        [2],
	#        [3]])

	x[:, np.newaxis]
	# array([[1],
	#        [2],
	#        [3]])

def concat_sample():
	x = np.array([1, 2, 3])
	y = np.array([4, 5, 6])
	np.concatenate([x, y]) # array([1, 2, 3, 4, 5, 6])

	grid = np.array([[1, 2, 3], [4, 5, 6]])
	np.concatenate([grid, grid])
	# array([[1, 2, 3],
	#        [4, 5, 6],
	#        [1, 2, 3],
	#        [4, 5, 6]])

	np.concatenate([grid, grid], axis=1)
	# array([[1, 2, 3, 1, 2, 3],
	#        [4, 5, 6, 4, 5, 6]])

	np.vstack([x, grid])
	# array([[1, 2, 3],
	#        [1, 2, 3],
	#        [4, 5, 6]])

	z = np.array([[99], [99]])
	np.hstack([grid, z])
	# array([[ 1,  2,  3, 99],
	#        [ 4,  5,  6, 99]])

def split_sample():
	x = [1, 2, 3, 99, 99, 3, 2, 1]
	x1, x2, x3 = np.split(x, [3, 5])
	# array([1, 2, 3])
	# array([99, 99])
	# array([3, 2, 1])

	np.split(x, [3, 5])
	#[array([1, 2, 3]), array([99, 99]), array([3, 2, 1])]
	
	grid = np.arange(16).reshape((4,4))
	# array([[ 0,  1,  2,  3],
	#        [ 4,  5,  6,  7],
	#        [ 8,  9, 10, 11],
	#        [12, 13, 14, 15]])

	upper,lower = np.vsplit(grid, [2])
	# array([[0, 1, 2, 3],
	#        [4, 5, 6, 7]])
	# array([[ 8,  9, 10, 11],
	#        [12, 13, 14, 15]])

	left,right = np.hsplit(grid, [2])
	# array([[ 0,  1],
	#        [ 4,  5],
	#        [ 8,  9],
	#        [12, 13]])
	# array([[ 2,  3],
	#        [ 6,  7],
	#        [10, 11],
	#        [14, 15]])

def ufunc_sample():
	x = np.arange(4) # array([0, 1, 2, 3])
	x+5 # array([5, 6, 7, 8])
	x*2 # array([0, 2, 4, 6])
	x**2 # array([0, 1, 4, 9])

	x = np.array([-2, -1, 0, 1, 2])
	np.abs(x) # array([2, 1, 0, 1, 2])

	theta = np.linspace(0, np.pi, 3) # array([ 0.        ,  1.57079633,  3.14159265])
	np.sin(theta) # array([  0.00000000e+00,   1.00000000e+00,   1.22464680e-16])
	np.cos(theta) # array([  1.00000000e+00,   6.12323400e-17,  -1.00000000e+00])
	np.tan(theta) # array([  0.00000000e+00,   1.63312394e+16,  -1.22464680e-16])

	x = [-1, 0, 1]
	np.arcsin(x) # array([-1.57079633,  0.        ,  1.57079633])
	np.arccos(x) # array([ 3.14159265,  1.57079633,  0.        ])
	np.arctan(x) # array([-0.78539816,  0.        ,  0.78539816])

	x = [1, 2, 3]
	np.exp(x) # e^x array([  2.71828183,   7.3890561 ,  20.08553692])
	np.exp2(x) # 2^x array([ 2.,  4.,  8.])
	np.power(3, x) # 3^x array([ 3,  9, 27])

	x = [1, 2, 4, 10]
	np.log(x) # ln(x) array([ 0.        ,  0.69314718,  1.38629436,  2.30258509])
	np.log2(x) # array([ 0.        ,  1.        ,  2.        ,  3.32192809])
	np.log10(x) # array([ 0.        ,  0.30103   ,  0.60205999,  1.        ])

	x = np.arange(5)
	y = np.empty(5)
	np.multiply(x, 10, out=y)
	# y = array([  0.,  10.,  20.,  30.,  40.])

	# Uses compiled code in NumPy, so much faster than Python sum, min, max
	L = np.random.random(100)
	np.sum(L)
	np.min(L)
	np.max(L)

	M = np.random.randint(0,10,(3,4))
	# array([[8, 9, 5, 2],
	#        [5, 5, 3, 7],
	#        [3, 2, 8, 1]])
	M.sum() # 58
	# Note: The axis keyword specifies the dimension of the array that will be collapsed,
	#		rather than the dimension that will be returned. So specifying axis=0 means
	#		that the first axis will be collapsed: for 2D arrays, this means that values
	#		within each column will be aggregated
	M.sum(axis=0) # array([16, 16, 16, 10])
	M.sum(axis=1) # array([24, 20, 14])
	M.max(axis=0) # array([8, 9, 8, 7])
	M.max(axis=1) # array([9, 7, 8])
	M.mean() # 4.833333333333333
	# standard deviation
	M.std() # 2.5766041389567178
	# variance
	M.var() # 6.6388888888888893
	np.median(M) # 5.0
	# whether any elements are true
	np.any(M) # True
	# whether all elements are true
	np.all(M) # True

def masking_sample():
	x = np.array([1, 2, 3, 4, 5])
	# Results in Boolean array
	print(x < 3)
	print(x > 3)
	print(x == 3)
	print((2*x) == (x**2))

	x = np.random.randint(0,10,(3,4))
	print(x)
	print(np.count_nonzero(x < 6)) # count number of True entries
	print(np.sum(x < 6)) # False = 0 ; True = 1
	print(np.sum(x < 6, axis=1)) # [3 2 3] -> number of values < 6 for each row
	print(np.any(x > 8))
	print(np.all(x < 10))
	print(np.sum((2 < x) & (x < 7))) # other bitwise operators: |, ^, ~

	print(x[x < 5]) # get all values < 5: [2 2 1 0 3 1 3]

def fancy_index_sample():
	rand = np.random.RandomState(42)
	x = rand.randint(100, size=10)
	print(x) # [51 92 14 71 60 20 82 86 74 74]

	ind = [3, 7, 4]
	print(x[ind]) # [71 86 60]

	ind = np.array([[3, 7], [4, 5]])
	print(x[ind])
	# [[71 86]
 	#  [60 20]]

	X = np.arange(12).reshape((3, 4))
	print(X)
	# [[ 0  1  2  3]
 	#  [ 4  5  6  7]
 	#  [ 8  9 10 11]]

	row = np.array([0, 1, 2])
	col = np.array([2, 1, 3])
	print(X[row, col]) # [ 2  5 11]

	print(X[row[:, np.newaxis], col])
	# [[ 2  1  3]
 	#  [ 6  5  7]
 	#  [10  9 11]]

	print(row[:, np.newaxis] * col)
	# [[0 0 0]
 	#  [2 1 3]
 	#  [4 2 6]]

	print(X[2, [2, 0, 1]]) # [10  8  9]

	print(X[1:, [2, 0, 1]])
	# [[ 6  4  5]
	#  [10  8  9]]

def sort_sample():
	x = np.array([2, 1, 4, 3, 5])
	print(np.sort(x)) # [1 2 3 4 5]
	print(np.argsort(x)) # [1 0 3 2 4]

	rand = np.random.RandomState(42)
	X = rand.randint(0, 10, (4, 6))
	print(X)
	# [[6 3 7 4 6 9]
 	#  [2 6 7 4 3 7]
 	#  [7 2 5 4 1 7]
 	#  [5 1 4 0 9 5]]

	print(np.sort(X, axis=0)) # sort each column of X
	# [[2 1 4 0 1 5]
 	#  [5 2 5 4 3 7]
 	#  [6 3 7 4 6 7]
 	#  [7 6 7 4 9 9]]

	print(np.sort(X, axis=1)) # sort each row of X
	# [[3 4 6 6 7 9]
 	#  [2 3 4 6 7 7]
 	#  [1 2 4 5 7 7]
 	#  [0 1 4 5 5 9]]

	x = np.array([7, 2, 3, 1, 6, 5, 4])
	print(np.partition(x, 3)) # [2 1 3 4 6 5 7]


def k_nearest_neighbor_sample():
	rand = np.random.RandomState(42)
	X = rand.rand(10, 2)
	seaborn.set()
	plt.scatter(X[:, 0], X[:, 1], s=100)

	# print(X.shape) 	# (10, 2)
	# print(X[:,np.newaxis,:].shape) 	# (10, 1, 2)
	# print(X[np.newaxis,:,:].shape) 	# (1, 10, 2)

	# Find the distance between each pair of points
	# Squared distance between two points is the sum of the squared differences in each dimension
	# dist_sq = np.sum((X[:,np.newaxis,:] - X[np.newaxis,:,:])**2, axis=-1)
	# print(np.shape(dist_sq)) 	# (10, 10)

	# squared distance broken down:
	# for each pair of points, compute differences in their coordinates
	differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
	# print(differences.shape) 	# (10, 10, 2) = (i, j, k)
	# 10 elements, each a (10, 2) array of differences between X[i] and
	# every other point. ie. if i == j, then k = (0,0)

	# squared the coordinate differences
	sq_differences = differences ** 2
	# print(sq_differences.shape) 	# (10, 10, 2)

	# sum the coordinate differences to get the squared distance
	dist_sq = sq_differences.sum(-1)
	# print(dist_sq.shape) 	# (10, 10)

	# should be zeroes, since it is the distance between each point and itself
	# print(dist_sq.diagonal())

	# sort along each row
	nearest = np.argsort(dist_sq, axis=1)
	# leftmost column will give the indices of the nearest neighbors
	# increasing order of idx into X along each row of closeness to point X[i]
	# Note: leftmost column will be 0-9 since each point is closest to itself
	print(nearest)

	# If want k nearest neighbors:
	K = 2
	nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)
	print(nearest_partition)

	# print(nearest_partition[0, :K+1]) # [3 0 4] = closest idx's
	# draw lines from each point to its two nearest neighbors
	for i in range(X.shape[0]):
		for j in nearest_partition[i, :K+1]:
			print(X[j])
			print(X[i])
			return
			plt.plot(*zip(X[j], X[i]), color='black')

	# zip makes an iterator that aggregates elements from each of the passed iterables
	# 	Returns an iterator of tuples, where the i-th tuple contains the i-th element
	#	from each of the argument iterables. The iterator stops when the shortest
	# 	input iterable is exhausted
	# zip() with * can be used to unzip a list
	# >>> xj = [0.05808361, 0.86617615]
	# >>> xi = [0.37454012, 0.95071431]
	# >>> z = zip(xj, xi)
	# >>> list(z)
	# [(0.05808361, 0.37454012), (0.86617615, 0.95071431)]
	# >>> z2 = zip(*zip(xj, xi))
	# >>> list(z2)
	# [(0.05808361, 0.86617615), (0.37454012, 0.95071431)]

	plt.show()

def structured_arrays_sample():
	name = ['Alice', 'Bob', 'Cathy', 'Doug']
	age = [25, 45, 37, 19]
	weight = [55.0, 85.5, 68.0, 61.5]

	data = np.zeros(4, dtype={'names':('name', 'age', 'weight'), 'formats':('U10', 'i4', 'f8')})
	print(data.dtype) # [('name', '<U10'), ('age', '<i4'), ('weight', '<f8')]

	data['name'] = name
	data['age'] = age
	data['weight'] = weight
	print(data) # [('Alice', 25, 55. ) ('Bob', 45, 85.5) ('Cathy', 37, 68. ) ('Doug', 19, 61.5)]

	print(data['name']) # ['Alice' 'Bob' 'Cathy' 'Doug']
	print(data[0]) # ('Alice', 25, 55.)

	# can filter using boolean masking
	print(data[data['age'] < 30]['name']) # ['Alice' 'Doug']

	# Using RecordArrays to access fields as attributes
	data_rec = data.view(np.recarray)
	print(data_rec.age) # [25 45 37 19]

if __name__ == '__main__':
	structured_arrays_sample()