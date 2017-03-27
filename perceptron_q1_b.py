# @Author - Nagaraj Poti
# @Roll - 20162010
#!/usr/bin/python

import csv
import numpy as np
import sys
import matplotlib.pyplot as plt

# Read csv files containing training data and populate array
def read_dataset(filename, y):
	with open(filename, 'rb') as csvfile:
		datasetreader = csv.reader(csvfile)
		for row in datasetreader:
			row = map(float, row)
			# Augment the feature vector
			row.append(1)
			# Append class label
			temp = [row, y]
			trainset.append(temp)

# Online perceptron 
def online_perceptron(learning_rate, max_epoch_count, trainset):
	weights = [0.0 for i in range(len(trainset[0][0]))]
	iterations = 0
	epoch_count = 0
	change = True
	while change and epoch_count <= max_epoch_count:
		change = False
		for feature in trainset:
			x = feature[0]
			y = feature[1]
			if np.dot(y, np.dot(weights, x)) <= 0:
				weights = np.add(weights, np.dot(learning_rate, np.dot(y, x)))
				iterations += 1
				change = True
		epoch_count += 1
		print ("--> epoch = %d, iterations = %d" % (epoch_count, iterations))
	print
	return weights

# Read datasets from file and populate training set 
trainset = []
read_dataset("dataset_q1_2.csv", 1)
read_dataset("dataset_q1_3.csv", -1)

print "Training set : "
print trainset ; print

learning_rate = 0.1
max_epoch_count = 100
print ("Learning rate = %.1f, Maximum epochs = %d" % (learning_rate, max_epoch_count))

weights = online_perceptron(learning_rate, max_epoch_count, trainset)
sys.stdout.write("Final augmented weight vector : ")
print weights

# Plotting code for given input dataset
d1x1 = []; d1x2 = []; d2x1 = []; d2x2 = []
for feature in trainset:
	if feature[1] == 1:
		d1x1.append(feature[0][0])
		d1x2.append(feature[0][1])
	else:
		d2x1.append(feature[0][0])
		d2x2.append(feature[0][1])
c2 = plt.scatter(d1x1, d1x2, c = "red", label = "C2")
c3 = plt.scatter(d2x1, d2x2, c = "lightgreen", label = "C3")
plt.suptitle("C2 vs C3", fontsize = 20)
plt.xlabel("x1", fontsize = 16);
plt.ylabel("x2", fontsize = 16);

# Plotting code for linear classifier
x = np.array(range(-6, 10))
y = eval("x * (-weights[0] / weights[1]) - (weights[2] / weights[1])")
classifier = plt.plot(x, y, label = 'classifier')

plt.legend(loc = 'center left', shadow = True)
plt.show()
