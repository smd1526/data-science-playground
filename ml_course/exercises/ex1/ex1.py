import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def compute_cost(X, y, theta):
	m = len(y)
	J = (((np.matmul(X, theta)-y)**2).sum())/(2*m)
	return J

def gradient_descent(X, y, theta, alpha, num_iters):
	m = len(y)
	for i in range(0, num_iters):
		for j in range(len(theta)):
			Xj = X[:,j,np.newaxis]
			theta[j] = theta[j] - ((((np.matmul(X, theta)-y)*Xj).sum())*(alpha/m))
	return theta

def run(data):
	y = data[:,1,np.newaxis]
	m = len(y)
	X = np.column_stack([np.ones(m), data[:,0]])
	theta = np.zeros((2,1))
	iterations = 1500
	alpha = 0.01

	theta = gradient_descent(X, y, theta, alpha, iterations)
	print(theta)
	plt.plot(X[:,1], np.matmul(X, theta), color='blue', label='Linear regression')

	predict1 = np.matmul(np.array([[1,3.5]]), theta)
	print(predict1)
	predict2 = np.matmul(np.array([[1,7]]), theta)
	print(predict2)

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


def compute_cost_test(data):
	y = data[:,1,np.newaxis]
	m = len(y)
	X = np.column_stack([np.ones(m), data[:,0]])
	theta = np.zeros((2,1))
	cost = compute_cost(X, y, theta)
	print(cost)

	theta = np.array([[-1], [2]])
	cost = compute_cost(X, y, theta)
	print(cost)

def plot_data(data):
	X = data[:,0]
	y = data[:,1]

	plt.scatter(X, y, marker='x', color='red', label='Training data')
	plt.xlabel('Population of City in 10,000s')
	plt.ylabel('Profit in $10,000s')

def main():
	data = np.loadtxt('ex1data1.txt', delimiter=',')
	plt.figure(1)
	plot_data(data)
	# compute_cost_test(data)
	run(data)

	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()