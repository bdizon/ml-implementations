import pandas as pd
import numpy as np
import argparse
'''
	Author: Bianca Dizon
'''

# create parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument('-A', action="store", dest="a", type=str)
parser.add_argument('-y', action="store", dest="y", type=str)
parser.add_argument('-beta', action="store", dest="b", type=float)
parser.add_argument('-x', action="store", dest="xo", type=str)
parser.add_argument('-lr', action="store", dest="l", type=float)
parser.add_argument('-maxiters', action="store", dest="m", type=float)
parser.add_argument('-tol', action="store", dest="t", type=float)
args = parser.parse_args()
# assign arguments to variables to use
train_file = str(args.a)
target_file = str(args.y)
beta = float(args.b)
x_file = str(args.xo)
lr = float(args.l)
maxiters = float(args.m)
tol = float(args.t)

'''
#name the input files
training_file = "train_data1.txt"
targets_file = "train_target1.txt"

#assign values
lr = 0.2 #learning rate
beta = 0.1 #beta value
maxiters = 100 #max number of iterations allowed
tol = 0.01 #max difference between change in gradients of your learnable linear weights
x_file = "output_file.txt" #output filename 
'''
# read in training data and targets from input files
x = np.genfromtxt(train_file, delimiter=' ')
y = np.genfromtxt(target_file, delimiter=' ')


# initialize weights
w = [beta,beta] # new weights
old_w = [beta,beta] # old weights used for comparison

# perform stochastic gradient descent
def sgd():
	n = 0 # iterate count
	while n < maxiters: # stop sgd if iterate over max iterate
		for i in range(len(x)): # number of arrays of x
			for j in range(len(x[i])): # number of values in array x[i]
				old_w[0] = w[0] # save old weight0 value
				old_w[1] = w[1] # save old weight1 value
				# perform sgd calculatios to find new weights
				w[0] = w[0]-lr*(2*(w[0]+(w[1]*x[i][j])-y[j]))
				w[1] = w[1]-lr*((2*x[i][j])*(w[0]+(w[1]*x[i][j])-y[j]))
				# figure out if change in gradients of learnable linear weights is less than tol
				diff1 = old_w[0] - w[0]
				if diff1 < 0: # make sure to get absolute value of difference
					diff1 = diff1 * -1
				diff2 = old_w[1] - w[1]
				if diff2 < 0: # make sure to get absolute value of difference
					diff2 = diff2 * -1
				sum_diff = diff1 + diff2 # sum differences of weights
				# if change in gradients of learnable linear weights is less than tol stop sgd
				if sum_diff < tol:
					return
		n += 1

sgd() # perform sgd
print(w)

# output weights to output file x_file
file_out = open(x_file, "w")
file_out.write(str(w))
file_out.close()