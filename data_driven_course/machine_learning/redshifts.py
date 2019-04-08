import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

def data_test():
	data = np.load('sdss_galaxy_colors.npy')

	# print first row
	print(data[0])

	# print column labeled 'u'
	print(data['u'])
	print(data['redshift'])

def median_diff(predicted, actual):
	return np.median(np.abs(predicted - actual))

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


def accuracy_by_treedepth(features, targets, depths):
	# split the data into testing and training sets
	split = features.shape[0]//2
	train_features, test_features = features[:split], features[split:]
	train_targets, test_targets = targets[:split], targets[split:]

	# initialize arrays to store the accuracies
	train_diffs = []
	test_diffs = []

	# loop through depths
	for depth in depths:
		# initialize model with the maximum depth
		dtr = DecisionTreeRegressor(max_depth=depth)

		# train the model using the training set
		dtr.fit(train_features, train_targets)

		# get the predictions for the training set and calculate their median_diff
		predictions = dtr.predict(train_features)
		train_diffs.append(median_diff(train_targets, predictions))

		# get the predictions for the testing set and calculate their median_diff
		predictions = dtr.predict(test_features)
		test_diffs.append(median_diff(test_targets, predictions))

	# return the accuracies for the training and testing sets
	return train_diffs, test_diffs

def test4():
	data = np.load('sdss_galaxy_colors.npy')
	features, targets = get_features_targets(data)

	tree_depths = [i for i in range(1, 36, 2)]

	train_med_diffs, test_med_diffs = accuracy_by_treedepth(features, targets, tree_depths)
	print("Depth with lowest median difference: {}".format(tree_depths[test_med_diffs.index(min(test_med_diffs))]))

	train_plot = plt.plot(tree_depths, train_med_diffs, label='Training set')
	test_plot = plt.plot(tree_depths, test_med_diffs, label='Validation set')
	plt.xlabel("Maximum Tree Depth")
	plt.ylabel("Median of Differences")
	plt.legend()
	plt.show()

def cross_validate_model(model, features, targets, k):
	kf = KFold(n_splits=k, shuffle=True)

	# initialise a list to collect median_diffs for each iteration of the loop below
	results = []

	for train_indices, test_indices in kf.split(features):
		train_features, test_features = features[train_indices], features[test_indices]
		train_targets, test_targets = targets[train_indices], targets[test_indices]

		# fit the model for the current set
		model.fit(train_features, train_targets)

		# predict using the model
		predictions = model.predict(test_features)

		# calculate the median_diff from predicted values and append to results array
		results.append(median_diff(predictions, test_targets))

	# return the list with your median difference values
	return results

def test5():
	data = np.load('./sdss_galaxy_colors.npy')
	features, targets = get_features_targets(data)

	# initialize model with a maximum depth of 19
	dtr = DecisionTreeRegressor(max_depth=19)

	# call your cross validation function
	diffs = cross_validate_model(dtr, features, targets, 10)

	# Print the values
	print('Differences: {}'.format(', '.join(['{:.3f}'.format(val) for val in diffs])))
	print('Mean difference: {:.3f}'.format(np.mean(diffs)))

def cross_validate_predictions(model, features, targets, k):
	kf = KFold(n_splits=k, shuffle=True)

	# declare an array for predicted redshifts from each iteration
	all_predictions = np.zeros_like(targets)

	for train_indices, test_indices in kf.split(features):
		# split the data into training and testing
		train_features, test_features = features[train_indices], features[test_indices]
		train_targets, test_targets = targets[train_indices], targets[test_indices]

		# fit the model for the current set
		model.fit(train_features, train_targets)

		# predict using the model
		predictions = model.predict(test_features)
		    
		# put the predicted values in the all_predictions array defined above
		all_predictions[test_indices] = predictions

	# return the predictions
	return all_predictions

def test6():
	data = np.load('./sdss_galaxy_colors.npy')
	features, targets = get_features_targets(data)

	# initialize model
	dtr = DecisionTreeRegressor(max_depth=19)

	# call your cross validation function
	predictions = cross_validate_predictions(dtr, features, targets, 10)

	# calculate and print the rmsd as a sanity check
	diffs = median_diff(predictions, targets)
	print('Median difference: {:.3f}'.format(diffs))

	# plot the results to see how well our model looks
	plt.scatter(targets, predictions, s=0.4)
	plt.xlim((0, targets.max()))
	plt.ylim((0, predictions.max()))
	plt.xlabel('Measured Redshift')
	plt.ylabel('Predicted Redshift')
	plt.show()

def split_galaxies_qsos(data):
	galaxies = data[data['spec_class'] == b'GALAXY']
	qsos = data[data['spec_class'] == b'QSO']
	return galaxies, qsos


def cross_validate_median_diff(data):
	features, targets = get_features_targets(data)
	dtr = DecisionTreeRegressor(max_depth=19)
	return np.mean(cross_validate_model(dtr, features, targets, 10))

def test7():
	data = np.load('./sdss_galaxy_colors.npy')
	galaxies, qsos= split_galaxies_qsos(data)

	# Here we cross validate the model and get the cross-validated median difference
	# The cross_validated_med_diff function is in "written_functions"
	galaxy_med_diff = cross_validate_median_diff(galaxies)
	qso_med_diff = cross_validate_median_diff(qsos)

	print("Median difference for Galaxies: {:.3f}".format(galaxy_med_diff))
	print("Median difference for QSOs: {:.3f}".format(qso_med_diff))

def main():
	# data_test()
	# test1()
	# test2()
	# test3()
	# test4()
	# test5()
	# test6()
	test7()

if __name__ == '__main__':
	main()