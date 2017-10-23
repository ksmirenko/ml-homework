import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('task_1_capital.txt', skip_header=1)
x_orig, y = np.hsplit(data, 2)
n = len(x_orig)
ones = np.ones((n, 1))
x = np.hstack((x_orig, ones))

# Formula: w∗ = (X^T * X)^−1 * X^T * y
x_tr = np.transpose(x)
w = np.dot(np.linalg.pinv(np.dot(x_tr, x)), x_tr).dot(y)

plt.scatter(x_orig, y)
max_x = max(x_orig)
plt.plot([0, max_x], [w[1], w[0] * max_x + w[1]], color='red')
plt.xlabel('Capital')
plt.ylabel('Rental')
plt.savefig('output.png')
