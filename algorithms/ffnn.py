def softmax(z):
    #Calculate exponent term first
    exp_scores = np.exp(z)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def softmax_loss(y,y_hat):
    # Clipping value
    minval = 0.000000000001
    # Number of samples
    m = y.shape[0]
    # Loss formula, note that np.sum sums up the entire matrix and therefore does the job of two sums from the formula
    loss = -1/m * np.sum(y * np.log(y_hat.clip(min=minval)))
    return loss

def loss_derivative(y,y_hat):
    return (y_hat-y)

def sigm(x):
    return 1/(1+np.exp(-x))
  
def sigm_derivative(x):
    return x*(1-x)

def tanh_derivative(x):
    return (1 - np.power(x, 2))

def accuracy_score(y_pred,y_true):
    count = 0
    for i in range(0,len(y_pred)):
    if y_pred[i] == y_true[i]:
        count = count + 1
    return count/len(y_pred)

from random import seed
from random import randrange

## cross-vaidation function, return tuples of validation set and training set
def cross_validation_split(dataset, folds = 5):
    meta_split = list()
    fold_size = int(len(dataset)/folds)
    for i in range(folds):
        #dataset_split = list()
        dataset_copy = list(dataset)
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        meta_split.append((fold,dataset_copy))
    return meta_split

# This is the forward propagation function
# For 1 hidden layer
def forward_prop_1(model,a0):
    #Start Forward Propagation
    
    # Load parameters from model
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Do the first Linear step 
    # Z1 is the input layer x times the dot product of the weights + our bias b
    z1 = a0.dot(W1) + b1
    
    # Put it through the first activation function
    a1 = sigm(z1)
    
    # Output layer
    z2 = a1.dot(W2) + b2
    
    # Output activation function
    a2 = softmax(z2)
    
    #Store all results in these values
    cache = {'a0':a0,'z1':z1,'a1':a1,'z2':z2,'a2':a2}
    return cache

  
# for 2 hidden layers
def forward_prop_2(model,a0):
    
    #Start Forward Propagation
    
    # Load parameters from model
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'],model['b3']
    
    # Do the first Linear step 
    # Z1 is the input layer x times the dot product of the weights + our bias b
    z1 = a0.dot(W1) + b1
    
    # Put it through the first activation function
    a1 = sigm(z1)
    
    # Second linear step
    z2 = a1.dot(W2) + b2
    
    # Second activation function
    a2 = sigm(z2)
    
    #Third linear step
    z3 = a2.dot(W3) + b3
    
    #For the Third linear activation function we use the softmax function, either the sigmoid of softmax should be used for the last layer
    a3 = softmax(z3)
    
    #Store all results in these values
    cache = {'a0':a0,'z1':z1,'a1':a1,'z2':z2,'a2':a2,'a3':a3,'z3':z3}
    return cache

# For 3 hidden layers
def forward_prop_3(model,a0):
    
    #Start Forward Propagation
    
    # Load parameters from model
    W1, b1, W2, b2, W3, b3, W4, b4 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'],model['b3'],model['W4'],model['b4']
    
    # Do the first Linear step 
    # Z1 is the input layer x times the dot product of the weights + our bias b
    z1 = a0.dot(W1) + b1
    
    # Put it through the first activation function
    a1 = sigm(z1)
    
    # Second linear step
    z2 = a1.dot(W2) + b2
    
    # Second activation function
    a2 = sigm(z2)
    
    # Third linear step
    z3 = a2.dot(W3) + b3
    
    # Third activation function
    a3 = sigm(z3)
    
    # Fourth linear step
    z4 = a3.dot(W4) + b4
    
    #For the Fourth linear activation function we use the softmax function, either the sigmoid of softmax should be used for the last layer
    a4 = softmax(z4)
    
    #Store all results in these values
    cache = {'a0':a0,'z1':z1,'a1':a1,'z2':z2,'a2':a2,'a3':a3,'z3':z3,'z4':z4,'a4':a4}
    return cache


# This is the BACKWARD PROPAGATION function

# For 1 hidden layer
def backward_prop_1(model,cache,y):

    # Load parameters from model
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Load forward propagation results
    a0,a1, a2 = cache['a0'],cache['a1'],cache['a2']
    
    # Get number of samples
    m = y.shape[0]
    
    # Calculate loss derivative with respect to output
    dz2 = loss_derivative(y=y,y_hat=a2)

    # Calculate loss derivative with respect to second layer weights
    dW2 = 1/m*(a1.T).dot(dz2) #dW2 = 1/m*(a1.T).dot(dz2) 
    
    # Calculate loss derivative with respect to second layer bias
    db2 = 1/m*np.sum(dz2, axis=0)
    
    dz1 = np.multiply(dz2.dot(W2.T),sigm_derivative(a1))
    
    dW1 = 1/m*np.dot(a0.T,dz1)
    
    db1 = 1/m*np.sum(dz1,axis=0)
    
    # Store gradients
    grads = {'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}
    return grads

# For 2 hidden layers
def backward_prop_2(model,cache,y):

    # Load parameters from model
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'],model['W3'],model['b3']
    
    # Load forward propagation results
    a0,a1, a2,a3 = cache['a0'],cache['a1'],cache['a2'],cache['a3']
    
    # Get number of samples
    m = y.shape[0]
    
    # Calculate loss derivative with respect to output
    dz3 = loss_derivative(y=y,y_hat=a3)

    # Calculate loss derivative with respect to second layer weights
    dW3 = 1/m*(a2.T).dot(dz3) #dW2 = 1/m*(a1.T).dot(dz2) 
    
    # Calculate loss derivative with respect to second layer bias
    db3 = 1/m*np.sum(dz3, axis=0)
    
    # Calculate loss derivative with respect to first layer
    dz2 = np.multiply(dz3.dot(W3.T) ,sigm_derivative(a2))
    
    # Calculate loss derivative with respect to first layer weights
    dW2 = 1/m*np.dot(a1.T, dz2)
    
    # Calculate loss derivative with respect to first layer bias
    db2 = 1/m*np.sum(dz2, axis=0)
    
    dz1 = np.multiply(dz2.dot(W2.T),sigm_derivative(a1))
    
    dW1 = 1/m*np.dot(a0.T,dz1)
    
    db1 = 1/m*np.sum(dz1,axis=0)
    
    # Store gradients
    grads = {'dW3':dW3, 'db3':db3, 'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}
    return grads

# For 3 hidden layers
def backward_prop_3(model,cache,y):

    # Load parameters from model
    W1, b1, W2, b2, W3, b3, W4, b4 = model['W1'], model['b1'], model['W2'], model['b2'],model['W3'],model['b3'],model['W4'],model['b4']
    
    # Load forward propagation results
    a0,a1, a2,a3,a4 = cache['a0'],cache['a1'],cache['a2'],cache['a3'],cache['a4']
    
    # Get number of samples
    m = y.shape[0]
    
    # Calculate loss derivative with respect to output
    dz4 = loss_derivative(y=y,y_hat=a4)

    # Calculate loss derivative with respect to third layer weights
    dW4 = 1/m*(a3.T).dot(dz4) #dW2 = 1/m*(a1.T).dot(dz2) 
    
    # Calculate loss derivative with respect to third layer bias
    db4 = 1/m*np.sum(dz4, axis=0)
    
    # Calculate loss derivative with respect to second layer
    dz3 = np.multiply(dz4.dot(W4.T) ,sigm_derivative(a3))
    
    # Calculate loss derivative with respect to second layer weights
    dW3 = 1/m*np.dot(a2.T, dz3)
    
    # Calculate loss derivative with respect to second layer bias
    db3 = 1/m*np.sum(dz3, axis=0)
    
    # Calculate loss derivative with respect to first layer
    dz2 = np.multiply(dz3.dot(W3.T) ,sigm_derivative(a2))
    
    # Calculate loss derivative with respect to first layer weights
    dW2 = 1/m*np.dot(a1.T, dz2)
    
    # Calculate loss derivative with respect to first layer bias
    db2 = 1/m*np.sum(dz2, axis=0)
    
    dz1 = np.multiply(dz2.dot(W2.T),sigm_derivative(a1))
    
    dW1 = 1/m*np.dot(a0.T,dz1)
    
    db1 = 1/m*np.sum(dz1,axis=0)
    
    # Store gradients
    grads = {'dW4':dW4,'db4':db4,'dW3':dW3, 'db3':db3, 'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}
    return grads

#TRAINING PHASE
# For 1 hidden layer
def initialize_parameters_1(nn_input_dim,nn_hdim,nn_output_dim):
    # First layer weights
    W1 = 2 *np.random.randn(nn_input_dim, nn_hdim) - 1
    
    # First layer bias
    b1 = np.zeros((1, nn_hdim))
    
    W2 = 2 * np.random.rand(nn_hdim, nn_output_dim) - 1
    b2 = np.zeros((1,nn_output_dim))
    
    # Package and return model
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model

# For 2 hidden layers
def initialize_parameters_2(nn_input_dim,nn_hdim,nn_output_dim):
    # First layer weights
    W1 = 2 *np.random.randn(nn_input_dim, nn_hdim) - 1
    
    # First layer bias
    b1 = np.zeros((1, nn_hdim))
    
    # Second layer weights
    W2 = 2 * np.random.randn(nn_hdim, nn_hdim) - 1
    
    # Second layer bias
    b2 = np.zeros((1, nn_hdim))
    W3 = 2 * np.random.rand(nn_hdim, nn_output_dim) - 1
    b3 = np.zeros((1,nn_output_dim))
       
    # Package and return model
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,'W3':W3,'b3':b3}
    return model

# For 3 hidden layers
def initialize_parameters_3(nn_input_dim,nn_hdim,nn_output_dim):
    # First layer weights
    W1 = 2 *np.random.randn(nn_input_dim, nn_hdim) - 1
    
    # First layer bias
    b1 = np.zeros((1, nn_hdim))
    
    # Second layer weights
    W2 = 2 * np.random.randn(nn_hdim, nn_hdim) - 1
    
    # Second layer bias
    b2 = np.zeros((1, nn_hdim))
    
    # Third layer
    W3 = 2 * np.random.randn(nn_hdim,nn_hdim) - 1
    b3 = np.zeros((1, nn_hdim))
    
    W4 = 2 * np.random.rand(nn_hdim, nn_output_dim) - 1
    b4 = np.zeros((1,nn_output_dim))
    
    
    # Package and return model
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,'W3':W3,'b3':b3,'W4':W4,'b4':b4}
    return model


# For 1 hidden layer1
def update_parameters_1(model,grads,learning_rate):
    # Load parameters
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Update parameters
    W1 -= learning_rate * grads['dW1']
    b1 -= learning_rate * grads['db1']
    W2 -= learning_rate * grads['dW2']
    b2 -= learning_rate * grads['db2']
    
    # Store and return parameters
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model

# For 2 hidden layers
def update_parameters_2(model,grads,learning_rate):
    # Load parameters
    W1, b1, W2, b2,b3,W3 = model['W1'], model['b1'], model['W2'], model['b2'],model['b3'],model["W3"]
    
    # Update parameters
    W1 -= learning_rate * grads['dW1']
    b1 -= learning_rate * grads['db1']
    W2 -= learning_rate * grads['dW2']
    b2 -= learning_rate * grads['db2']
    W3 -= learning_rate * grads['dW3']
    b3 -= learning_rate * grads['db3']
    
    # Store and return parameters
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3':W3,'b3':b3}
    return model

# For 3 hidden layers
def update_parameters_3(model,grads,learning_rate):
    # Load parameters
    W1, b1, W2, b2,b3,W3,b4,W4 = model['W1'], model['b1'], model['W2'], model['b2'],model['b3'],model["W3"],model['b4'],model['W4']
    
    # Update parameters
    W1 -= learning_rate * grads['dW1']
    b1 -= learning_rate * grads['db1']
    W2 -= learning_rate * grads['dW2']
    b2 -= learning_rate * grads['db2']
    W3 -= learning_rate * grads['dW3']
    b3 -= learning_rate * grads['db3']
    W4 -= learning_rate * grads['dW4']
    b4 -= learning_rate * grads['db4']
    
    # Store and return parameters
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3':W3,'b3':b3,'W4':W4,'b4':b4}
    return model

# For 1 hidden layer
def predict_1(model, x):
    # Do forward pass
    c = forward_prop_1(model,x)
    #get y_hat
    y_hat = np.argmax(c['a2'], axis=1)
    return y_hat

# For 2 hidden layers
def predict_2(model, x):
    # Do forward pass
    c = forward_prop_2(model,x)
    #get y_hat
    y_hat = np.argmax(c['a3'], axis=1)
    return y_hat

# For 3 hidden layers
def predict_3(model, x):
    # Do forward pass
    c = forward_prop_3(model,x)
    #get y_hat
    y_hat = np.argmax(c['a4'], axis=1)
    return y_hat


# For 1 hidden layer
def train_1(model,X_,y_,learning_rate, epochs=20000, print_loss=False):
    ##acc = 0
    # Gradient descent. Loop over epochs
    for i in range(0, epochs):

        # Forward propagation
        cache = forward_prop_1(model,X_)
        #a1, probs = cache['a1'],cache['a2']
        # Backpropagation
        
        grads = backward_prop_1(model,cache,y_)
        # Gradient descent parameter update
        # Assign new parameters to the model
        model = update_parameters_1(model=model,grads=grads,learning_rate=learning_rate)
    
        # Pring loss & accuracy every 100 iterations
        ##if i == epochs-1:
            ##a2 = cache['a2']
            #print('Loss after iteration',i,':',softmax_loss(y_,a2))
            ##y_hat = predict_1(model,X_)
            ##y_true = y_.argmax(axis=1)
            #print('Accuracy after iteration',i,':',accuracy_score(y_pred=y_hat,y_true=y_true)*100,'%')
            ##acc = accuracy_score(y_pred=y_hat,y_true=y_true)
    return model

# For 2 hidden layers
def train_2(model,X_,y_,learning_rate, epochs=20000, print_loss=False):
    ##acc = 0
    # Gradient descent. Loop over epochs
    for i in range(0, epochs):

        # Forward propagation
        cache = forward_prop_2(model,X_)
        #a1, probs = cache['a1'],cache['a2']
        # Backpropagation
        
        grads = backward_prop_2(model,cache,y_)
        # Gradient descent parameter update
        # Assign new parameters to the model
        model = update_parameters_2(model=model,grads=grads,learning_rate=learning_rate)
    
        # Pring loss & accuracy every 100 iterations
        ##if i == epochs-1:
            ##a3 = cache['a3']
            #print('Loss after iteration',i,':',softmax_loss(y_,a3))
            ##y_hat = predict_2(model,X_)
            ##y_true = y_.argmax(axis=1)
            #print(i)
            #print('Accuracy after iteration',i,':',accuracy_score(y_pred=y_hat,y_true=y_true)*100,'%')
            ##acc = accuracy_score(y_pred=y_hat,y_true=y_true)
            ##print(acc)
    return model

# For 3 hidden layers
def train_3(model,X_,y_,learning_rate, epochs=20000, print_loss=False):
    ##acc = 0
    # Gradient descent. Loop over epochs
    for i in range(0, epochs):

        # Forward propagation
        cache = forward_prop_3(model,X_)
        #a1, probs = cache['a1'],cache['a2']
        # Backpropagation
        
        grads = backward_prop_3(model,cache,y_)
        # Gradient descent parameter update
        # Assign new parameters to the model
        model = update_parameters_3(model=model,grads=grads,learning_rate=learning_rate)
    
        # Pring loss & accuracy every 1000 iterations
        ##if i == epochs-1:
            ##a4 = cache['a4']
            #print('Loss after iteration',i,':',softmax_loss(y_,a4))
            ##y_hat = predict_3(model,X_)
            ##y_true = y_.argmax(axis=1)
            #print('Accuracy after iteration',i,':',accuracy_score(y_pred=y_hat,y_true=y_true)*100,'%')
            ##acc = accuracy_score(y_pred=y_hat,y_true=y_true)
    return model

np.random.seed(0)
# Model selection based on cross-validation
l = cross_validation_split(dataset=np.c_[train_images_x,train_y_grid])  ## 5 different splits
colx = train_images_x.shape[1]   ## number of features in dataset
coly = train_y_grid.shape[1]   ## number of classes in dataset
# 5 different splits as below
a_val_x = np.asarray(l[0][0])[:,0:colx]    ## first split
a_val_y = np.asarray(l[0][0])[:,colx:(colx+coly)]
a_tra_x = np.asarray(l[0][1])[:,0:colx]
a_tra_y = np.asarray(l[0][1])[:,colx:(colx+coly)]
b_val_x = np.asarray(l[1][0])[:,0:colx]    ## second split
b_val_y = np.asarray(l[1][0])[:,colx:(colx+coly)]
b_tra_x = np.asarray(l[1][1])[:,0:colx]
b_tra_y = np.asarray(l[1][1])[:,colx:(colx+coly)]
c_val_x = np.asarray(l[2][0])[:,0:colx]    ## third split
c_val_y = np.asarray(l[2][0])[:,colx:(colx+coly)]
c_tra_x = np.asarray(l[2][1])[:,0:colx]
c_tra_y = np.asarray(l[2][1])[:,colx:(colx+coly)]
d_val_x = np.asarray(l[3][0])[:,0:colx]    ## fourth split
d_val_y = np.asarray(l[3][0])[:,colx:(colx+coly)]
d_tra_x = np.asarray(l[3][1])[:,0:colx]
d_tra_y = np.asarray(l[3][1])[:,colx:(colx+coly)]
e_val_x = np.asarray(l[4][0])[:,0:colx]    ## fifth split
e_val_y = np.asarray(l[4][0])[:,colx:(colx+coly)]
e_tra_x = np.asarray(l[4][1])[:,0:colx]
e_tra_y = np.asarray(l[4][1])[:,colx:(colx+coly)]

units = [180,160,140,130,120,100,50]    ## list of numbers of nodes
rate = [0.08, 0.1, 0.2]    ## list of learning rate

## Cross-validation for 1 hidden layers
for u in units:
  #print('units:',i)
  for r in rate:
    #print('rate:',j)
    #u = units[i]  ## num of nodes
    #r = rate[j]   ## learning rate
    scores = np.zeros(5)  ## array to store accuracy for 5 splits
    
    model_a = initialize_parameters_1(nn_input_dim=colx, nn_hdim= u, nn_output_dim= coly)
    model_a = train_1(model_a,a_tra_x,a_tra_y,learning_rate=r,epochs=4500)
    y_hat_a = predict_1(model_a,a_val_x)
    y_true_a = a_val_y.argmax(axis=1)
    scores[0] = accuracy_score(y_pred=y_hat_a,y_true=y_true_a)
    
    model_b = initialize_parameters_1(nn_input_dim=colx, nn_hdim= u, nn_output_dim= coly)
    model_b = train_1(model_b,b_tra_x,b_tra_y,learning_rate=r,epochs=4500)
    y_hat_b = predict_1(model_b,b_val_x)
    y_true_b = b_val_y.argmax(axis=1)
    scores[1] = accuracy_score(y_pred=y_hat_b,y_true=y_true_b)
    
    model_c = initialize_parameters_1(nn_input_dim=colx, nn_hdim= u, nn_output_dim= coly)
    model_c = train_1(model_c,c_tra_x,c_tra_y,learning_rate=r,epochs=4500)
    y_hat_c = predict_1(model_c,c_val_x)
    y_true_c = c_val_y.argmax(axis=1)
    scores[2] = accuracy_score(y_pred=y_hat_c,y_true=y_true_c)
    
    model_d = initialize_parameters_1(nn_input_dim=colx, nn_hdim= u, nn_output_dim= coly)
    model_d = train_1(model_d,d_tra_x,d_tra_y,learning_rate=r,epochs=4500)
    y_hat_d = predict_1(model_d,d_val_x)
    y_true_d = d_val_y.argmax(axis=1)
    scores[3] = accuracy_score(y_pred=y_hat_d,y_true=y_true_d)
    
    model_e = initialize_parameters_1(nn_input_dim=colx, nn_hdim= u, nn_output_dim= coly)
    model_e = train_1(model_e,e_tra_x,e_tra_y,learning_rate=r,epochs=4500)
    y_hat_e = predict_1(model_e,e_val_x)
    y_true_e = e_val_y.argmax(axis=1)
    scores[4] = accuracy_score(y_pred=y_hat_e,y_true=y_true_e)
    print('1 hidden layer, units:',u,',rate:',r,',accuracy:',np.mean(scores))

## Cross-validation for 2 hidden layers
for u in units:
  #print('units:',i)
  for r in rate:
    #print('rate:',j)
    #u = units[i]  ## num of nodes
    #r = rate[j]   ## learning rate
    scores = np.zeros(5)  ## array to store accuracy for 5 splits
    
    model_a = initialize_parameters_2(nn_input_dim=colx, nn_hdim= u, nn_output_dim= coly)
    model_a = train_2(model_a,a_tra_x,a_tra_y,learning_rate=r,epochs=4500)
    y_hat_a = predict_2(model_a,a_val_x)
    y_true_a = a_val_y.argmax(axis=1)
    scores[0] = accuracy_score(y_pred=y_hat_a,y_true=y_true_a)
    
    model_b = initialize_parameters_2(nn_input_dim=colx, nn_hdim= u, nn_output_dim= coly)
    model_b = train_2(model_b,b_tra_x,b_tra_y,learning_rate=r,epochs=4500)
    y_hat_b = predict_2(model_b,b_val_x)
    y_true_b = b_val_y.argmax(axis=1)
    scores[1] = accuracy_score(y_pred=y_hat_b,y_true=y_true_b)
    
    model_c = initialize_parameters_2(nn_input_dim=colx, nn_hdim= u, nn_output_dim= coly)
    model_c = train_2(model_c,c_tra_x,c_tra_y,learning_rate=r,epochs=4500)
    y_hat_c = predict_2(model_c,c_val_x)
    y_true_c = c_val_y.argmax(axis=1)
    scores[2] = accuracy_score(y_pred=y_hat_c,y_true=y_true_c)
    
    model_d = initialize_parameters_2(nn_input_dim=colx, nn_hdim= u, nn_output_dim= coly)
    model_d = train_2(model_d,d_tra_x,d_tra_y,learning_rate=r,epochs=4500)
    y_hat_d = predict_2(model_d,d_val_x)
    y_true_d = d_val_y.argmax(axis=1)
    scores[3] = accuracy_score(y_pred=y_hat_d,y_true=y_true_d)
    
    model_e = initialize_parameters_2(nn_input_dim=colx, nn_hdim= u, nn_output_dim= coly)
    model_e = train_2(model_e,e_tra_x,e_tra_y,learning_rate=r,epochs=4500)
    y_hat_e = predict_2(model_e,e_val_x)
    y_true_e = e_val_y.argmax(axis=1)
    scores[4] = accuracy_score(y_pred=y_hat_e,y_true=y_true_e)
    print('2 hidden layers, units:',u,',rate:',r,',accuracy:',np.mean(scores))

## Cross-validation for 3 hidden layers
for u in units:
  #print('units:',i)
  for r in rate:
    #print('rate:',j)
    #u = units[i]  ## num of nodes
    #r = rate[j]   ## learning rate
    scores = np.zeros(5)  ## array to store accuracy for 5 splits
    
    model_a = initialize_parameters_3(nn_input_dim=colx, nn_hdim= u, nn_output_dim=coly)
    model_a = train_3(model_a,a_tra_x,a_tra_y,learning_rate=r,epochs=4500)
    y_hat_a = predict_3(model_a,a_val_x)
    y_true_a = a_val_y.argmax(axis=1)
    scores[0] = accuracy_score(y_pred=y_hat_a,y_true=y_true_a)
    
    model_b = initialize_parameters_3(nn_input_dim=colx, nn_hdim= u, nn_output_dim= coly)
    model_b = train_3(model_b,b_tra_x,b_tra_y,learning_rate=r,epochs=4500)
    y_hat_b = predict_3(model_b,b_val_x)
    y_true_b = b_val_y.argmax(axis=1)
    scores[1] = accuracy_score(y_pred=y_hat_b,y_true=y_true_b)
    
    model_c = initialize_parameters_3(nn_input_dim=colx, nn_hdim= u, nn_output_dim=coly)
    model_c = train_3(model_c,c_tra_x,c_tra_y,learning_rate=r,epochs=4500)
    y_hat_c = predict_3(model_c,c_val_x)
    y_true_c = c_val_y.argmax(axis=1)
    scores[2] = accuracy_score(y_pred=y_hat_c,y_true=y_true_c)
    
    model_d = initialize_parameters_3(nn_input_dim=colx, nn_hdim= u, nn_output_dim= coly)
    model_d = train_3(model_d,d_tra_x,d_tra_y,learning_rate=r,epochs=4500)
    y_hat_d = predict_3(model_d,d_val_x)
    y_true_d = d_val_y.argmax(axis=1)
    scores[3] = accuracy_score(y_pred=y_hat_d,y_true=y_true_d)
    
    model_e = initialize_parameters_3(nn_input_dim=colx, nn_hdim= u, nn_output_dim=coly)
    model_e = train_3(model_e,e_tra_x,e_tra_y,learning_rate=r,epochs=4500)
    y_hat_e = predict_3(model_e,e_val_x)
    y_true_e = e_val_y.argmax(axis=1)
    scores[4] = accuracy_score(y_pred=y_hat_e,y_true=y_true_e)
    print('3 hidden layers, units:',u,',rate:',r,',accuracy:',np.mean(scores))

## time ~~ 36min
## predict test data using 1 layer, 160 nodes, sigmoid fn, and 0.2 learning rate
model_final = initialize_parameters_1(nn_input_dim=416,nn_hdim=160,nn_output_dim=31)
model_final = train_1(model_final,train_images_x,train_y_grid,learning_rate=0.2,epochs=4500)
y_prediction = predict_1(model_final,test_images_x)
## compute training accuracy
y_pred_train = predict_1(model_final,train_images_x)
y_true_train = train_y_grid.argmax(axis=1)
score_train = accuracy_score(y_pred=y_pred_train,y_true=y_true_train)
## print the training accuracy
print(score_train)