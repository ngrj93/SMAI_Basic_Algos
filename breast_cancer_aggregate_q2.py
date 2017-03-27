# @Author - Nagaraj Poti
# @Roll - 20162010
#!/usr/bin/python

import csv
import numpy as np
import random
import perceptron_q2_2 as vanilla
import matplotlib.pyplot as plt

# Pretty printing of float values 
class prettyfloat(float):
	def __repr__(self):
		return "%0.2f" % self

# Read breast cancer csv file containing training data and populate array
def read_dataset(filename):
	with open(filename, 'rb') as csvfile:
		datasetreader = csv.reader(csvfile)
		for row in datasetreader:
			try :
				row = map(float, row) 
				# Rearrange class label
				y = 1 if row[-1] == 2 else -1
				row.pop()
				# Augment the feature vector
				row.append(1)
				# Append class label 
				trainset.append([row, y])
			except ValueError:
				pass;

# Voted perceptron 
def voted_perceptron(learning_rate, max_epoch_count, trainset):
	weights = [0.0 for i in range(len(trainset[0][0]))]
	weight_history = []	
	iterations = 0
	current_votes = 1
	epoch_count = 0
	change = True
	while change and epoch_count < max_epoch_count:
		change = False
		for feature in trainset:
			x = feature[0]
			y = feature[1]
			if np.dot(y, np.dot(weights, x)) <= 0:
				weight_history.append([weights, current_votes])
				iterations += 1
				current_votes = 1
				weights = np.add(weights, np.dot(learning_rate, np.dot(y, x)))
				change = True
			else:
				current_votes += 1
		epoch_count += 1
#	print ("--> perceptron = voted, epoch = %d, iterations = %d" % (epoch_count, iterations))
	weight_history.append([weights, current_votes])
	return weight_history

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
		s = 0
		for weight in weights:
			w = weight[0]
			c = weight[1]
			sign = 1 if np.dot(w, x) >= 0 else -1
			s += c * sign	
		predicted_label = 1 if s >= 0 else -1
		if predicted_label == y:
			score += 1
	return float(score) / len(fold)

if __name__ == "__main__":

	# Read dataset from file and populate training set 
	trainset = []
	read_dataset("breast-cancer-wisconsin.data")

	learning_rate = 1
	max_epoch = [10, 15, 20, 25, 30, 35, 40, 45, 50] 

	# Average accuracy scores for both perceptrons
	voted_avg_acc = []
	vanilla_avg_acc = []

	for epoch_count in max_epoch:
		print ("Learning rate = %.1f, Maximum epochs = %d" % (learning_rate, epoch_count))
		testset = cross_validation_generate(trainset, 10)
		voted_scores = []
		vanilla_scores = []
		for fold in testset:
			current_trainset = list(testset)
			current_trainset.remove(fold)
			current_trainset = sum(current_trainset, [])
			weights = voted_perceptron(learning_rate, epoch_count, current_trainset)
			voted_scores.append(test_prediction(weights, fold))
			weights = vanilla.online_perceptron(learning_rate, epoch_count, current_trainset)
			vanilla_scores.append(vanilla.test_prediction(weights, fold))

		print "\nVoted perceptron validation scores : "
		voted_scores = map(prettyfloat, voted_scores)
		print voted_scores; print
		
		voted_avg_acc.append(float(sum(voted_scores)) / len(voted_scores) * 100)
		print ("Average accuracy : %.3f%%" % voted_avg_acc[-1])

		print "\nVanilla perceptron validation scores : "
		vanilla_scores = map(prettyfloat, vanilla_scores)
		print vanilla_scores; print

		vanilla_avg_acc.append(float(sum(vanilla_scores)) / len(vanilla_scores) * 100)
		print ("Average accuracy : %.3f%%" % vanilla_avg_acc[-1])
		print "-----------------------------------------------------------\n"

	# Plot accuracy values
	vot = plt.scatter([5 * i for i in range(2, 11)], voted_avg_acc, c = "red", label = "voted") 
	van = plt.scatter([5 * i for i in range(2, 11)], vanilla_avg_acc, c = "green", label = "vanilla")
	vot = plt.plot([5 * i for i in range(2, 11)], voted_avg_acc, c = "red") 
	van = plt.plot([5 * i for i in range(2, 11)], vanilla_avg_acc, c = "green")
	plt.suptitle("Vanilla perceptron vs Voted perceptron - Breast Cancer Dataset", fontsize = 18)
	plt.xlabel("Epochs", fontsize = 16)
	plt.ylabel("Accuracy %", fontsize = 16)
	plt.legend(loc = 'center left', shadow = True)
	plt.show()
