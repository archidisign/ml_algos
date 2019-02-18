import numpy as np

def linear_fit(X, Y):
    temp = inv(np.matmul(X.transpose(), X))
    temp = np.matmul(temp, X.transpose())
    return np.matmul(temp, Y)

def cross_valid_linear(data):
	MSE = [0]*5
	w = [0]*5
	new_data = np.copy(data)
	np.random.shuffle(new_data)
	stored_data = np.array_split(new_data, 5)
	for i in range(5):
	    temp = np.concatenate([stored_data[j] for j in range(5) if j != i], axis=0)
	    X_train, X_test= temp[:,:123], stored_data[i][:,:123]
	    Y_train, Y_test= temp[:,123], stored_data[i][:,123]
	    w[i] = linear_fit(X_train, Y_train)
	    yhat = np.matmul(X_test, w[i])
	    MSE[i] = np.mean((yhat - Y_test)**2)
	return MSE