import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

def dump_data(index):
	data = np.load('galaxy_catalogue.npy')
	for name, value in zip(data.dtype.names, data[index]):
		print('{:10} {:.6}'.format(name, value))

def splitdata_train_test(data, fraction_training):
	train_num = int(len(data)*fraction_training)
	return data[:train_num], data[train_num:]

def test1():
	data = np.load('galaxy_catalogue.npy')
	fraction_training = 0.7
	training, testing = splitdata_train_test(data, fraction_training)
	print('Number data galaxies:', len(data))
	print('Train fraction:', fraction_training)
	print('Number of galaxies in training set:', len(training))
	print('Number of galaxies in testing set:', len(testing))

def generate_features_targets(data):
	targets = data['class']

	features = np.empty(shape=(len(data), 13))
	features[:, 0] = data['u-g']
	features[:, 1] = data['g-r']
	features[:, 2] = data['r-i']
	features[:, 3] = data['i-z']
	features[:, 4] = data['ecc']
	features[:, 5] = data['m4_u']
	features[:, 6] = data['m4_g']
	features[:, 7] = data['m4_r']
	features[:, 8] = data['m4_i']
	features[:, 9] = data['m4_z']

	# concentration in u filter
	features[:, 10] = data['petroR50_u'] / data['petroR90_u']
	# concentration in r filter
	features[:, 11] = data['petroR50_r'] / data['petroR90_r']
	# concentration in z filter
	features[:, 12] = data['petroR50_z'] / data['petroR90_z']

	return features, targets

def test2():
	data = np.load('galaxy_catalogue.npy')
	features, targets = generate_features_targets(data)
	print("Features shape:", features.shape)
	print("Targets shape:", targets.shape)

def dtc_predict_actual(data):
    np.random.seed(0)
    np.random.shuffle(data)
    training_set, testing_set = splitdata_train_test(data, 0.7)

    train_features, train_targets = generate_features_targets(training_set)
    test_features, test_targets = generate_features_targets(testing_set)

    dtc = DecisionTreeClassifier()
    dtc.fit(train_features, train_targets)
    predictions = dtc.predict(test_features)
    return predictions, test_targets

def test3():
	data = np.load('galaxy_catalogue.npy')
	predicted_class, actual_class = dtc_predict_actual(data)
	print("Some initial results...\n   predicted,  actual")
	for i in range(10):
		print("{}. {}, {}".format(i, predicted_class[i], actual_class[i]))

def calculate_accuracy(predicted, actual):
	correct = 0
	for i in range(0, len(predicted)):
		if predicted[i] == actual[i]:
			correct += 1
	return correct / len(predicted)

def test4():
	data = np.load('galaxy_catalogue.npy')
	features, targets = generate_features_targets(data)

	dtc = DecisionTreeClassifier()
	predicted = cross_val_predict(dtc, features, targets, cv=10)

	model_score = calculate_accuracy(predicted, targets)
	print("Our accuracy score:", model_score)

	class_labels = list(set(targets))
	model_cm = confusion_matrix(y_true=targets, y_pred=predicted, labels=class_labels)
	#plt.figure()
	#plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
	#plt.show()

def rf_predict_actual(data, n_estimators):
	features, targets = generate_features_targets(data)
	rfc = RandomForestClassifier(n_estimators=n_estimators)
	predicted = cross_val_predict(rfc, features, targets, cv=10)
	return predicted, targets

def test5():
	data = np.load('galaxy_catalogue.npy')

	number_estimators = 50 # Number of trees
	predicted, actual = rf_predict_actual(data, number_estimators)

	accuracy = calculate_accuracy(predicted, actual)
	print("Accuracy score:", accuracy)


	class_labels = list(set(actual))
	model_cm = confusion_matrix(y_true=actual, y_pred=predicted, labels=class_labels)

	# plot the confusion matrix using the provided functions.
	#plt.figure()
	#plot_confusion_matrix(model_cm, classes=class_labels, normalize=False)
	#plt.show()

def main():
	# dump_data(0)
	# test1()
	# test2()
	# test3()
	# test4()
	test5()

if __name__ == '__main__':
	main()