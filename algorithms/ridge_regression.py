import numpy as np

def linear_fit_L2(X, Y, lambda_constant):
    n = np.size(X[0])
    lambda_matrix = np.identity(n)*lambda_constant
    temp = inv(np.matmul(X.transpose(), X) + lambda_matrix)
    temp = np.matmul(temp, X.transpose())
    return np.matmul(temp, Y)

def cross_valid_linear_L2(data):
	MSEs = np.zeros(shape=(5, 1000))
	parameters = x = [[[] for i in range(1000)] for j in range(5)]
	new_data2 = np.copy(data)
	np.random.shuffle(new_data2)
	stored_data = np.array_split(new_data, 5)
	for i in range(5):
	    temp = np.concatenate([stored_data[j] for j in range(5) if j != i], axis=0)
	    X_train, X_test= temp[:,:123], stored_data[i][:,:123]
	    Y_train, Y_test= temp[:,123], stored_data[i][:,123]
	    np.savetxt('CandC-train'+str(i+1)+'Q4_3.csv', temp, delimiter=",")
	    np.savetxt('CandC-test'+str(i+1) + 'Q4_3.csv', stored_data[i], delimiter=",")
	    for j in range(1000):
	        parameters[i][j] = linear_fit_L2(X_train, Y_train, 0.005*j)
	        MSEs[i][j] = np.mean((np.matmul(X_test, parameters[i][j]) - Y_test)**2)
	return MSEs