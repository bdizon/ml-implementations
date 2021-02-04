'''
Bianca Dizon
CSE 5523 HW 5
'''
import numpy as np
import argparse
import random
from math import log

# create parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument('-A', action="store", dest="a", type=str)
parser.add_argument('-y', action="store", dest="y", type=str)
parser.add_argument('-ln', action="store", dest="l", type=int)
parser.add_argument('-un', action="store", dest="u", type=str)
parser.add_argument('-a', action="store", dest="f", type=str)
parser.add_argument('-ls', action="store", dest="s", type=str)
parser.add_argument('-out', action="store", dest="o", type=str)
parser.add_argument('-lr', action="store", dest="lr", type=float)
parser.add_argument('-nepochs', action="store", dest="e", type=int)
parser.add_argument('-bs', action="store", dest="b", type=int)
parser.add_argument('-tol', action="store", dest="t", type=float)
args = parser.parse_args()
# assign arguments to variables to use
train_file = str(args.a)
target_file = str(args.y)
layers = int(args.l)
str_units = str(args.u)
units = list(map(int, str_units.split(','))) # turn str into a list of ints
activation = str(args.f)
loss_func = str(args.s)
output_file = str(args.o)
learning_rate = float(args.lr)
epochs = int(args.e)
batch_size = int(args.b)
tol = float(args.t)

if len(units) != layers: # check if the length of the units list is equal to the number of hidden layers 
    print("Error: the length of the units array is not the same length as the number of hidden layers.")
    

# read in training data and targets from input files
x = np.genfromtxt(train_file, delimiter=' ')    # x has the training data
y = np.genfromtxt(target_file, delimiter=' ')   # y has the target data
# print(y)

# create nn architecture
network = []    # list of dictionaries containing info about the input layer sizes and output layer sizes and activation function of each layer
arch = {}       # dictionaries containing info
arch['input_layer'] = 5   # 5 because of the 5 values per line from train_file
arch['output_layer'] = units[0]     # first hidden layer size
arch['activation'] = activation 
network.append(arch)
for i in range(len(units)):
    arch = {}
    arch['input_layer'] = units[i]    #add the dimensions of the input layer
    if i == len(units)-1:
        arch['output_layer'] = 1    #one layer for last "output" layer
    else:
         arch['output_layer'] = units[i+1]    #add the dimensions of the hidden "output" layer
    arch['activation'] = activation     #specify the activation function
    # print(arch)
    network.append(arch)    #add layer information to the whole architecture
#     print("**************")
# print(network)

# populate parameters(weights and biases) for each layer with random values
def init_parameters(network):
    np.random.seed(100)      # random numbers from 0 to 50
    parameters = {}      # dictionary of the weights and biases per layer
    for index, layer in enumerate(network):     
        input_size = layer["input_layer"]     # size of the input layer
        # print(input_size)                   
        output_size = layer["output_layer"]   # size of the outer layer
        parameters['w' + str(index)] = np.random.randn(
            output_size, input_size) * 0.1      # create an array of weight values for layer, size of inner array = input_size, size of outer array = output_size
        parameters['b' + str(index)] = np.random.randn(
            output_size, 1) * 0.1   # create an array of biases for layer, size of array = output_size
    return parameters
# parameters = init_parameters(network)
# print(parameters)

# activation functions
def sigmoid(x):     # sigmoid function for forward pass
    return 1/(1+np.exp(-x))

def tanh(x):    #  tanh function for forward pass
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def sigmoid_backward(dA, Z):    # sigmoid function for backward pass
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def tanh_backward(dA, Z):   # tanh function for backward pass
    tan = tanh(Z)
    return dA * (1-np.square(tan))

# loss functions
def sse(y_pred, y_true):    # mse or sse loss function
    # print("SSE")
    return np.sqrt(((y_pred - y_true) ** 2).mean())
   
def ce(y_pred, y_true):     # cross entropy loss function
    # print("CE")    
    # sum_score = 0.0
    # for i in range(len(y_true)):
    #     for j in range(len(y_true[i])):
    #         sum_score += y_true[i][j] * y_pred[i][j]
    #         mean_sum_score = 1.0 / len(y_true) * sum_score
    # return abs(mean_sum_score) 
    ce = np.mean((y_pred) * y_true) 
    return abs(ce)    # return ce

# get loss depending on the loss_func chosen
def loss_value(y_pred, y_true, loss_func):
    if loss_func == "SSE":  # chosen function is sse
        loss_use = sse
    elif loss_func == "CE":     # chosen function is ce
        loss_use = ce
    else:
        raise Exception('Error: Loss function chosen is not available. Loss functions implemented: SSE and CE')
    value = loss_use(y_pred, y_true)
    # print(value)
    return value

# single layer forward propagation
def single_layer_forward_propagation(old_A, new_w, new_b, activation):
    z_value = np.dot(new_w, old_A) + new_b # input value for activation func
    # Z_curr = np.dot( old_A, W_curr) + b_curr

    # print(old_A.shape)
    if activation == "tanh":
        activation_func = tanh
    elif activation == "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Error: unknown activation function')
    # activation_func = sigmoid
    return activation_func(z_value), z_value # use activation function

# full forward propagation
def forward_propagation(x, params_values, nn_architecture):
    memory = {} # keep memory of a and z values
    new_A = x
    
    for index, layer in enumerate(nn_architecture):
        old_A = new_A
        activation_func = layer["activation"] # choose activation fucntion
        new_w = params_values["w" + str(index)] # weights for layer
        new_b = params_values["b" + str(index)] # biases for layer
        new_A, Z_curr = single_layer_forward_propagation(old_A, new_w, new_b, activation_func)
        
        memory["A" + str(index)] = old_A # save a and z values  in memory
        memory["Z" + str(index)] = Z_curr
       
    return new_A, memory

# single layer backward propation
def single_layer_backward_propagation(new_dA, new_w, new_b, Z_curr, old_A, activation):
    
    if activation == "tanh":
        backward_activation_func = tanh_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Error: unknown activation function')
    
    new_dZ = backward_activation_func(new_dA, Z_curr) # run loss function for backward prop
    new_dw = np.dot(new_dZ, old_A.T) / old_A.shape[1] #matrix w derivative
    new_db = np.sum(new_dZ, axis=1, keepdims=True) / old_A.shape[1] #matrix b derivative
    old_dA = np.dot(new_w.T, new_dZ)# matrix old a derivative

    return old_dA, new_dw, new_db

# backwards propogation function
def backward_propagation(y_pred, y_true, memory, params_values, nn_architecture):
    gradients = {}
    y_true = y_true.reshape(y_pred.shape) #target and prediction are the same shape
    old_dA = - (np.divide(y_true, y_pred)-np.divide(1-y_true, 1-y_pred)); # gradient descent algorithm
    
    for index, layer in reversed(list(enumerate(nn_architecture))):
        activation_function = layer["activation"]
     
        W_curr = params_values["w" + str(index)]    #get weight values
        b_curr = params_values["b" + str(index)]    #get biases
        new_dA = old_dA
        old_A = memory["A" + str(index)]       #get a values
        Z_curr = memory["Z" + str(index)]
                
        old_dA, new_dw, db_curr = single_layer_backward_propagation(
            new_dA, W_curr, b_curr, Z_curr, old_A, activation_function)    # perform single backward prop
        gradients["dw" + str(index)] = new_dw   # update gradients
        gradients["db" + str(index)] = db_curr
    # print(gradients)
    return gradients


def update(params_values, gradients, nn_architecture, learning_rate):
    for index, layer in enumerate(nn_architecture):
        params_values["w" + str(index)] -= learning_rate * gradients["dw" + str(index)]       # change the weights 
        params_values["b" + str(index)] -= learning_rate * gradients["db" + str(index)]     #change the parameters
    #print(params_values)
    return params_values;

# create batches for training and testing data
def batch(xdata, ydata, batch_size):
    batchx_array = []
    batchy_array = []
    for i in range(batch_size):
        num = random.randint(0,xdata.shape[0]-1)    # choose random variable for index of train and testing arrays
        newx_batch = np.array(xdata[num])       # choose array of index num to be included into new x train batch
        batchx_array.append(newx_batch)
        newy_batch = np.array(ydata[num])        # choose array of index num to be included into new y train batch
        batchy_array.append(newy_batch)
    batchx_array = np.array(batchx_array)         # make sure they are numpy arrays
    batchy_array = np.array(batchy_array)
    return batchx_array, batchy_array


def train(x, y, nn_architecture, epochs, learning_rate, batch_size, loss_func, tol):
    params_values = init_parameters(nn_architecture)    # set up parameter values (weights and biases) for each layer
    loss_list= []   # all the loss values through batch iterations
    # print(X.shape)
    # print(Y.shape)
    old_loss = 20000000     # keep track of old loss to compare to new loss for tol comparison
    for i in range(epochs):
        x_batch, y_batch=batch(x, y, batch_size)    # create batch
        x_batch = np.transpose(x_batch)     # transpose train data
        y_batch = np.transpose(y_batch.reshape((y_batch.shape[0], 1)))  # transpose target data
        # print(x_batch.shape)
        # print(y_batch.shape)
        y_pred, cashe = forward_propagation(x_batch, params_values, nn_architecture)    # forward propagate
        new_loss = loss_value(y_pred, y_batch, loss_func)   # get loss value
        # print(new_loss)
        loss_list.append(new_loss)

        gradients = backward_propagation(y_pred, y_batch, cashe, params_values, nn_architecture)    # backward propogate
        params_values = update(params_values, gradients, nn_architecture, learning_rate)        # update the parameter values for all layers
        
        if abs(new_loss - old_loss) <= tol:     # check if the difference between batch losses is less than or equal to
            return params_values, new_loss, i+1, loss_list  # stop training
        else:
           old_loss = new_loss  # update loss for old batch
    return params_values, new_loss, i+1, loss_list
    # return params_values, cost_history
    # , accuracy_history

# print(x)
# print(x.shape)
# print(y.shape)
params, cost, num_iter, loss_list= train(x,y,network,epochs,learning_rate, batch_size, loss_func,tol)
# # params, cost= train(x,y,network,1000,0.001)
# print(params)
print(cost)
# print(num_iter)
print(len(loss_list))
# print(loss_list)

# output weights to output file x_file
file_out = open(output_file, "w")
for i in range(len(loss_list)):
    file_out.write(str(loss_list[i])) # write each loss on a separate line
    if i < len(loss_list)-1:
        file_out.write("\n")
file_out.close()