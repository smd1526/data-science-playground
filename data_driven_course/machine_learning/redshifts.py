import numpy as np
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt

def data_test():
	data = np.load('sdss_galaxy_colors.npy')

	# print first row
	print(data[0])

	# print column labeled 'u'
	print(data['u'])
	print(data['redshift'])

def median_diff(predicted, actual):
	return np.median(np.abs(predicted[:] - actual[:]))

def get_features_targets(data):
	m = len(data)
	features = np.empty([len(data), 4])
	features[:, 0] = data['u'] - data['g']
	features[:, 1] = data['g'] - data['r']
	features[:, 2] = data['r'] - data['i']
	features[:, 3] = data['i'] - data['z']
	return (features, data['redshift'])

def test1():
	data = np.load('sdss_galaxy_colors.npy')

	features, targets = get_features_targets(data)
	print(len(features))
	print(len(targets))
	print(features[:2])
	print(targets[:2])

	dtr = DecisionTreeRegressor()
	dtr.fit(features, targets) # train the model
	predictions = dtr.predict(features)

	print(predictions[:4])

def validate_model(model, features, targets):
	if len(features) != len(targets):
		return None

	# split data into training and testing features and predictions
	split = features.shape[0]//2
	train_features = features[:split]
	test_features = features[split:]

	train_targets = targets[:split]
	test_targets = targets[split:]

	# train the model
	model.fit(train_features, train_targets)

	# get the predicted redshifts
	predictions = model.predict(test_features)

	# use median_diff to calculate accuracy
	return median_diff(predictions, test_targets)

def test2():
	data = np.load('sdss_galaxy_colors.npy')
	features, targets = get_features_targets(data)

	dtr = DecisionTreeRegressor()
	diff = validate_model(dtr, features, targets)
	print('Median difference: {:f}'.format(diff))

def test3():
	data = np.load('sdss_galaxy_colors.npy')

	cmap = plt.get_cmap('YlOrRd') # Get a contour map

	# Define our colour indexes u-g and r-i
	u_g = data['u'] - data['g']
	r_i = data['r'] - data['i']

	# Make a redshift array
	redshift = data['redshift']

	# Create the plot with plt.scatter and plt.colorbar
	plot = plt.scatter(u_g, r_i, s=0.5, lw=0, c=redshift, cmap=cmap)
	cb = plt.colorbar(plot)
	cb.set_label('Redshift')

	# Define your axis labels and plot title
	plt.xlabel('Colour index u-g')
	plt.ylabel('Colour index r-i')
	plt.title('Redshift (colour) u-g versus r-i')

	# Set any axis limits
	plt.xlim(-0.5, 2.5)
	plt.ylim(-0.5, 1)

	plt.show()

def main():
	# data_test()
	# test1()
	# test2()
	test3()

if __name__ == '__main__':
	main()