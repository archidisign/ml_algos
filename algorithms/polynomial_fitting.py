import numpy as np

def polyfit(X, Y, n):
    m = np.size(X)
    matrix = np.zeros(shape=(m, n+1))
    for i in range(n+1):
        matrix[:,i] = X**i
    return np.dot(inv(np.dot(matrix.transpose(), matrix)), np.dot(matrix.transpose(), Y))


def polyval(X, parameters):
    n = np.size(parameters)
    m = np.size(X)
    matrix = np.zeros(shape=(m, n))
    for i in range(n):
        matrix[:,i] = X**i
    return np.matmul(matrix, parameters)


def polyfit_lambda(X, Y, n, lambda_constant):
    m = np.size(X)
    matrix = np.zeros(shape=(m, n+1))
    for i in range(n+1):
        matrix[:,i] = X**i
    lambda_matrix = np.identity(n+1) * lambda_constant
    
    temp = inv(np.matmul(matrix.transpose(), matrix) + lambda_matrix)
    temp = np.matmul(temp, matrix.transpose())
    return np.matmul(temp, Y)


parameters = polyfit(data['Input'], data['Target'], 20)
prediction_train = polyval(data['Input'], parameters)
error = np.mean((prediction_train - data['Target'])**2)