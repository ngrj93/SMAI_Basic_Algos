# @Author - Nagaraj Poti
# @Roll - 20162010
#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys

# Pretty printing of float values
class prettyfloat(float):
	def __repr__(self):
		return "%0.2f" % self

# Widrow Hoff procedure
def least_squares(learning_rate, trainset, max_iterations):
	weights = [0.0 for i in range(len(trainset[0]))]
	cur_iterations = 1
	# Learning rate eita updates after each iteration
	eita = learning_rate 
	x = trainset[cur_iterations - 1]
  # Error 
	delta = np.dot(np.dot(eita, np.subtract(1, np.dot(weights, x))), x)
	while cur_iterations <= max_iterations and np.linalg.norm(delta) > 10e-9:
		weights = np.add(weights, delta)
		x = trainset[cur_iterations % len(trainset)]
		cur_iterations += 1
		eita = float(learning_rate) / cur_iterations
		delta = np.dot(np.dot(eita, np.subtract(1, np.dot(weights, x))), x)
	return weights

# Dataset 2 - Augmented vectors
class2_orig = [[-1,1,1],[0,0,1],[-1,-1,1],[1,0,1]]
class1 = [[3,3,1],[3,0,1],[2,1,1],[0,1.5,1]]
class2 = [[1,-1,-1],[0,0,-1],[1,1,-1],[-1,0,-1]]

learning_rate = 1
max_iterations = 10000
print ("Learning rate = %.1f" % learning_rate)

weights = least_squares(learning_rate, class1 + class2, max_iterations)
sys.stdout.write("\nFinal augmented weight vector : ")
print map(prettyfloat, weights) 

# Plotting code for given input dataset
d1x1 = []; d1x2 = []; d2x1 = []; d2x2 = []
for feature in class1:
	d1x1.append(feature[0])
	d1x2.append(feature[1])
for feature in class2_orig:
	d2x1.append(feature[0])
	d2x2.append(feature[1])
c1 = plt.scatter(d1x1, d1x2, c = "red", label = "C1")
c2 = plt.scatter(d2x1, d2x2, c = "lightgreen", label = "C2")
plt.suptitle("Least squares approach - C1 vs C2 - Dataset 2", fontsize = 20)
plt.xlabel("x1", fontsize = 16)
plt.ylabel("x2", fontsize = 16)

# Plotting code for linear classifier
x = np.array(range(-2, 4))
y = eval("x * (-weights[0] / weights[1]) - (weights[2] / weights[1])")
classifier = plt.plot(x, y, label = "classifier")

plt.legend(loc = 'center left', shadow = True)
plt.show()
