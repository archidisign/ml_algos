import numpy as np

# With fixed learning rate
def SGD_LR(train, valid, n_epoch, step_size):
    n = len(train)
    parameters = [5, 5]
    error=float('inf')
    MSE=[0]*n_epoch
    for epoch in range(n_epoch):
        sum_error = 0
        np.random.shuffle(train)
        for i in range(n):
            yhat = parameters[0] + parameters[1]*train[i, 0]
            error = yhat - train[i, 1]
            sum_error += error**2
            parameters[0] += - step_size * error
            parameters[1] += - step_size * error * train[i, 0]
        MSE[epoch] = np.mean((parameters[0] + parameters[1]*valid[:,0] - valid[:,1])**2)
    return parameters, MSE


# Learning rate converges
def SGD_LR_sature(train, valid):
    n = len(train)
    step_size = float('1e-6')
    parameters = [5, 5]
    error=float('inf')
    MSE_train=[]
    MSE_valid=[]
    while abs(error)>0.0005:
        sum_error = 0
        np.random.shuffle(train)
        for i in range(n):
            yhat = parameters[0] + parameters[1]*train[i, 0]
            error = yhat - train[i, 1]
            sum_error += error**2
            parameters[0] += - step_size * error
            parameters[1] += - step_size * error * train[i, 0]
        MSE_train += [np.mean(( polyval(train[:,0], parameters) - train[:,1])**2)]
        MSE_valid += [np.mean(( polyval(valid[:,0], parameters) - valid[:,1])**2)]
    return parameters, MSE_train, MSE_valid


# Memorize the tried parameters
def SGD_LR_memo(train, valid, n_epoch, step_size):
    n = len(train)
    error=float('inf')
    MSE=[0]*n_epoch
    parameters = [5, 5]
    parameters_memo = np.zeros(shape=(n_epoch, 2))
    for epoch in range(n_epoch):
        sum_error = 0
        np.random.shuffle(train)
        for i in range(n):
            yhat = parameters[0] + parameters[1]*train[i, 0]
            error = yhat - train[i, 1]
            sum_error += error**2
            parameters[0] += - step_size * error
            parameters[1] += - step_size * error * train[i, 0]
        MSE[epoch] = np.mean(( polyval(valid[:,0], parameters) - valid[:,1])**2)
        parameters_memo[epoch,0]=parameters[0]
        parameters_memo[epoch,1]=parameters[1]
    return parameters_memo, MSE


parameters, MSE = SGD_LR(data.as_matrix(), data.as_matrix(), 5000, float('1e-6'))
parameters, MSE_train, MSE_valid = SGD_LR_sature(data.as_matrix(), data.as_matrix())
parameters_memo, MSE = SGD_LR_memo(data.as_matrix(), data.as_matrix(), 10, float('1e-2'))