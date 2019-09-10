import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def compute_cost(X, y, theta):
	m = len(y)
	J = (((np.matmul(X, theta)-y)**2).sum())/(2*m)
	return J

# For univariate regression:
# X.shape = (97,2)
	# [1, 6.1]
	# [1, 5.5]
	# [1, 8.5]
# theta.shape = (2,1)
	# [0]
	# [0]
# y.shape = (97,1)

# j = 0:
# 	Xj = (97,1)
	# [1]
	# [1]
	# [1]

# j = 1:
# 	Xj = (97,1)
	# [6.1]
	# [5.5]
	# [8.5]

# np.matmul(X, theta) is (97,2) x (2,1), results in (97,1)

def gradient_descent(X, y, theta, alpha, num_iters):
	m = len(y)
	for i in range(0, num_iters):
		theta = theta - ((alpha/m)*np.matmul(X.transpose(), np.matmul(X, theta) - y))
	return theta

def run_GD_and_plot(X, y):
	m = len(y)
	theta = np.zeros((2,1))
	iterations = 1500
	alpha = 0.01

	theta = gradient_descent(X, y, theta, alpha, iterations)
	print(theta)
	plt.plot(X[:,1], np.matmul(X, theta), color='blue', label='Linear regression')

	return theta

def test_predictions(theta):
	predict1 = np.matmul(np.array([[1,3.5]]), theta)
	print(predict1)
	predict2 = np.matmul(np.array([[1,7]]), theta)
	print(predict2)

def plot_cost(X, y):
	# theta0 = np.linspace(-10, 10, 100)
	theta0 = np.linspace(10, -10, 100)
	# theta1 = np.linspace(-1, 4, 100)
	theta1 = np.linspace(4, -1, 100)
	J_vals = np.zeros((len(theta0), len(theta1)))
	for i in range(0, len(theta0)):
		for j in range(0, len(theta1)):
			J_vals[i,j] = compute_cost(X, y, np.array([theta0[i], theta1[j]]))

	fig = plt.figure()
	# ax = plt.axes(projection='3d')
	# ax.contour3D(theta0, theta1, J_vals, 500, cmap='binary')
	# plt.contour(theta0, theta1, J_vals)

	ax = plt.axes(projection='3d')
	ax.plot_surface(theta0, theta1, J_vals, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

	plt.figure(1)

def compute_cost_test(X, y):
	m = len(y)
	theta = np.zeros((2,1))
	cost = compute_cost(X, y, theta)
	print(cost)

	theta = np.array([[-1], [2]])
	cost = compute_cost(X, y, theta)
	print(cost)

def plot_data_univar(data):
	X = data[:,0]
	y = data[:,1]

	plt.scatter(X, y, marker='x', color='red', label='Training data')
	plt.xlabel('Population of City in 10,000s')
	plt.ylabel('Profit in $10,000s')

def test_data_univar():
	data = np.loadtxt('ex1data1.txt', delimiter=',')
	plt.figure(1)
	plot_data_univar(data)

	y = data[:,1,np.newaxis]
	m = len(y)
	X = np.column_stack([np.ones(m), data[:,0]])
	compute_cost_test(X, y)

	theta = run_GD_and_plot(X, y)
	test_predictions(theta)

	plot_cost(X, y)

	plt.legend()
	plt.show()

def feature_normalize(x):
	mu = x.mean(axis=0) 	# mean of each column = [2000.68085106    3.17021277]
	sigma = x.std(axis=0) 	# std of each column = [7.86202619e+02 7.52842809e-01]

	r = (x - mu) / sigma
	return (r, mu, sigma)

def test_data_multivar():
	data = np.loadtxt('ex1data2.txt', delimiter=',')
	(m, n) = data.shape 			# (47, 3) ; 47 samples, 2 features + 1 value (y)
	n -= 1 							# remove y
	X = data[:, :n] 				# all rows, all columns except the last
	y = data[:, n, np.newaxis] 		# all rows, last column ; make a vector
	m = len(y) 						# 47

	(X, mu, sigma) = feature_normalize(X)

	X = np.column_stack([np.ones(m), X])

	theta = np.zeros((n+1,1))
	iterations = 400 #1500
	alpha = 0.01

	theta = gradient_descent(X, y, theta, alpha, iterations)
	print(theta)

	cost = compute_cost(X, y, theta)
	print(cost)

def main():
	# test_data_univar()
	test_data_multivar()

if __name__ == '__main__':
	main()