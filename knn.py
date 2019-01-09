import numpy as np
import math


def KNN(test, k):
    n = len(X_train)
    y_hat = 0
    distances = np.zeros((n, 2))
    for i in range(n):
        dist = math.sqrt(np.sum((test - X_train[i]) ** 2))
        distances[i] = [i, dist]
    distances = distances[distances[:,1].argsort()]
    for i in range(k):
        y_hat += Y_train[int(distances[i][0])]
    return y_hat/k