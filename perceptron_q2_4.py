#!/usr/bin/python

import csv
import numpy as np
import random

# Pretty printing of float values 
class prettyfloat(float):
	def __repr__(self):
		return "%0.2f" % self

# Read ionosphere csv file containing training data and populate array
def read_dataset(filename):
	with open(filename, 'rb') as csvfile:
		datasetreader = csv.reader(csvfile)
		for row in datasetreader:
			try :
				# Rearrange class label
				y = 1 if row[-1] == "g" else -1
				row.pop()
				row = map(float, row) 
				# Augment the feature vector
				row.append(1)
				# Append class label 
				trainset.append([row, y])
			except ValueError:
				pass;

# Online perceptron 
def online_perceptron(learning_rate, max_epoch_count, trainset):
	weights = [0.0 for i in range(len(trainset[0][0]))]
	iterations = 0
	epoch_count = 0
	change = True
	while change and epoch_count < max_epoch_count:
		change = False
		for feature in trainset:
			x = feature[0]
			y = feature[1]
			if np.dot(y, np.dot(weights, x)) <= 0:
				weights = np.add(weights, np.dot(learning_rate, np.dot(y, x)))
				iterations += 1
				change = True
		epoch_count += 1
#	print ("--> epoch = %d, iterations = %d" % (epoch_count, iterations))
	return weights

# Generate cross-validation test sets according to 10-fold cross validation
# Shuffle and split cross validation 
def cross_validation_generate(trainset, fold_count):
	test_dataset_folds = []
	trainset_c = list(trainset)
	fold_size = int(len(trainset) / fold_count)	
	for i in range(fold_count):
		current_fold = []
		while (len(current_fold) < fold_size):
			i = random.randrange(0, len(trainset_c))
			current_fold.append(trainset_c.pop(i))
		test_dataset_folds.append(current_fold)
	return test_dataset_folds

# Test data classifier prediction score
def test_prediction(weights, fold):
	score = 0
	for feature in fold:
		x = feature[0]
		y = feature[1]
		sign = 1 if np.dot(weights, x) >= 0 else -1
		predicted_label = 1 if sign >= 0 else -1
		if predicted_label == y:
			score += 1
	return float(score) / len(fold)

if __name__ == "__main__":

	# Read dataset from file and populate training set 
	trainset = []
	read_dataset("ionosphere.data")

	learning_rate = 1 
	max_epoch = [10, 15, 20, 25, 30, 35, 40, 45, 50]

	for epoch_count in max_epoch:
		print ("Learning rate = %.1f, Maximum epochs = %d" % (learning_rate, epoch_count))
		testset = cross_validation_generate(trainset, 10)
		val_scores = []
		for fold in testset:
			current_trainset = list(testset)
			current_trainset.remove(fold)
			current_trainset = sum(current_trainset, [])
			weights = online_perceptron(learning_rate, epoch_count, current_trainset)
			val_scores.append(test_prediction(weights, fold))

		print "\nValidation scores : "
		val_scores = map(prettyfloat, val_scores)
		print val_scores; print

		print ("Average accuracy : %.3f%%" % (float(sum(val_scores)) / len(val_scores) * 100))
		print "-----------------------------------------------------------\n"
