import numpy as np

def find_coefficients(X_train, Y_train, X_test):
	# probabilities
	m = len(Y_train)
	countpos = np.count_nonzero(Y_train == 1)
	countneg = np.count_nonzero(Y_train == 0)
	pi = 1/m*countpos
	p_neg = pi
	p_pos = 1 - pi
	# Means
	sum_pos = np.zeros(20)
	sum_neg = np.zeros(20)
	for i in range(m):
	    xi = X_train[i, :]
	    yi = Y_train[i]
	    if yi == 1:
	        sum_pos += xi
	    else:
	        sum_neg += xi
	u1 = sum_pos/countpos
	u2 = sum_neg/countneg
	# Covariance matrix
	sqr_sum = np.zeros((20,20))
	for i in range(m):
	    xi = X_train[i,:]
	    yi = Y_train[i]
	    if yi == 1:
	        sqr = np.matmul(np.transpose(np.matrix(xi-u1)), np.matrix(xi-u1))
	    else:
	        sqr = np.matmul(np.transpose(np.matrix(xi-u2)), np.matrix(xi-u2))
	    sqr_sum += sqr
	covariance = 1/m*sqr_sum
	# Weights
	w = np.matmul(inv(covariance), (u1-u2))
	w0 = - 0.5*np.matmul(np.matmul(np.matrix(u1),inv(covariance)),np.transpose(np.matrix(u1))) + 0.5*np.matmul(np.matmul(np.matrix(u2),inv(covariance)),np.transpose(np.matrix(u2))) + math.log(p_pos/p_neg)
	return w, w0

def GDA(X_train, Y_train_X_test):
	w, w0 = find_coefficients(X_train, Y_train_X_test)
	a = np.matmul(np.matrix(w), np.transpose(np.matrix(X_test))) + w0
	prediction = 1/(1+np.exp(-1*a))
	return prediction