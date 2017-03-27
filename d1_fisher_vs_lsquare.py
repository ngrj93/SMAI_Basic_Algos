# @Author - Nagaraj Poti
# @Roll - 20162010
#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import perceptron_q2_2 as vanilla
import least_squares_q3_1 as lsq

# Pretty printing of float values
class prettyfloat(float):
	def __repr__(self):
		return "%0.2f" % self

#-----------------------------------------------------------------
# Dataset 1 - Augmented vectors - least square
class2_orig = [[-1,1,1],[0,0,1],[-1,-1,1],[1,0,1]]
class1 = [[3,3,1],[3,0,1],[2,1,1],[0,2,1]]
class2 = [[1,-1,-1],[0,0,-1],[1,1,-1],[-1,0,-1]]

learning_rate = 1
max_iterations = 10000
weights_lsq = lsq.least_squares(learning_rate, class1 + class2, max_iterations)
#-----------------------------------------------------------------

# Dataset 1
class1 = [[3,3],[3,0],[2,1],[0,2]]
class2 = [[-1,1],[0,0],[-1,-1],[1,0]]

# Compute mean vectors of each class 
mean_d1 = np.mean(class1, axis = 0)
mean_d2 = np.mean(class2, axis = 0)

# Compute within class scatter matrix 
scatter_data1 = np.dot((class1 - mean_d1).T, (class1 - mean_d1))
scatter_data2 = np.dot((class2 - mean_d2).T, (class2 - mean_d2))
scatter_within = scatter_data1 + scatter_data2
print "Within class scatter matrix : "
for l in scatter_within:
	print map(prettyfloat, l)

# Calculate weight vector
weights = np.dot(np.linalg.inv(scatter_within), (mean_d1 - mean_d2))
weights = weights / np.linalg.norm(weights)
print "\nUnit weight vector : "
print map(prettyfloat, weights)

#--------------------------------------------------------------------
# Plotting code for linear classifier by least square method
x = np.array(range(-2, 5))
y = eval("x * (-weights_lsq[0] / weights_lsq[1]) - (weights_lsq[2] / weights_lsq[1])")
plt.plot(x, y, 'k', label = "Least sq classifier")
#---------------------------------------------------------------------

# Plot dataset 
d1x1 = zip(*class1)[0]; d1x2 = zip(*class1)[1]
d2x1 = zip(*class2)[0]; d2x2 = zip(*class2)[1]
c1 = plt.scatter(d1x1, d1x2, c = "red", label = "C1")
c2 = plt.scatter(d2x1, d2x2, c = "green", label = "C2")
plt.suptitle("Fisher's discriminant vs Least square - C1 vs C2 - Dataset 1", fontsize = 18)
plt.xlabel("x1", fontsize = 16)
plt.ylabel("x2", fontsize = 16)

# Plot fisher's linear discriminant - dimension reduction
# Class 1 plot
scalars = np.dot(class1, weights)
points = [] 
for value in scalars:
	points.append(np.dot(value, weights))
total_points = [[[i[0],i[1],1],1] for i in points]
plt.plot(zip(*points)[0], zip(*points)[1], "rx", markersize = 10, mew = 2, label = "C1 projection")

for i in range(len(points)):
	plt.plot([class1[i][0], points[i][0]], [class1[i][1], points[i][1]], 'r--')

# Class 2 plot
scalars = np.dot(class2, weights)
points = []
for value in scalars:
	points.append(np.dot(value, weights))
total_points += [[[i[0],i[1],1],-1] for i in points]
plt.plot(zip(*points)[0], zip(*points)[1], "gx", markersize = 10, mew = 2, label = "C2 projection")

for i in range(len(points)):
	plt.plot([class2[i][0], points[i][0]], [class2[i][1], points[i][1]], 'g--')

x = np.array(range(-2, 5))
y = eval("(x - points[0][0]) * (weights[1] / weights[0]) + points[0][1]")
discriminant = plt.plot(x, y,'--', label = 'Fisher discriminant') 

# Apply online perceptron to obtain classifier line
w = vanilla.online_perceptron(0.1, 10000, total_points)

# Plotting code for linear classifier
x = np.array(range(-2, 5))
y = eval("x * (-w[0] / w[1]) - (w[2] / w[1])")
classifier = plt.plot(x, y, 'c', label = 'Fisher classifier')

plt.legend(loc = 'lower right', shadow = True)
plt.show()

