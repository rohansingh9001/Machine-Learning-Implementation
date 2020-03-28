'''
Sample dataset acquired from :

https://people.sc.fsu.edu/~jburkardt/datasets/regression/regression.html

Legend:
=======

In the graph generated as a result of running this program,

Red Dots : Input data in the form of a scatter plot.
Black Line : Randomised initial line as a starting point if gradient descent.
Blue Line : Final result of gradient descent.
'''

__author__ = 'Rohan Singh, singh.77@iitj.ac.in'

__version__ = '0.2'

import numpy as np
import matplotlib.pyplot as plt
import random

# Not needed as of now but is usefull in debugging.
from mpl_toolkits.mplot3d import axes3d


def printmat(name, matrix):
    '''
    Prints matrix in a easy to read form with
    dimension and label.

    Parameters
    ==========
    name:
        data type : str
        The name displayed in the output.

    matrix:
        data type : numpy array
        The matrix to be displayed.
    '''
    print('matrix ' + name + ':', matrix.shape)
    print(matrix, '\n')


def readdata(file):
    '''
    Read the training data from a file in the same directory.

    Parameters
    ==========
    file:
        data type : str
        Name of the file to be read with extension.

    Example
    =======

    If the training data is stored in "dataset.txt" use

    >>> readdata('dataset.txt')

    '''
    A = np.genfromtxt(file)

    global X, Y, M, N

    X = np.hstack((A[:, 1:2], np.ones((A.shape[0], 1))))
    M, N = X.shape

    Y = A[:, -1]
    Y.shape = (1, M)


def activation(X, W):
    '''
    Returns X * W.
    Where,
    X is the parameter matrix
    W is the weight matrix

    effectively executes - y = mx + c
    m - first weight
    c - bias

    Parameters
    ==========
    X:
        data type : numpy array
        Parameter Matrix
    W:
        data type : numpy array
        Weight Matrix

    Example
    =======

    If the training data is stored in "dataset.txt" use

    >>> X = np.array([[5, 1]])
    >>> W = np.array([[2], [5]])
    >>> activation()
    15

    '''
    return np.matmul(X, W)


# Loss Function not needed as of now.
def loss(X, W, Y):
    pass


# Hardcoded Derivative. Autodifferentiate needed.
def dm(X, Y, W):
    '''
    Returns d/dx(loss).
    Where,
    X is the parameter matrix
    Y is expected output vector
    W is the weight matrix

    Current hardcoded formula is:

    d(loss) =  1
    -------   --- * (y' - y) * x
      dx       M

    Where y' = mx + c

    Parameters
    ==========
    X:
        data type : numpy array
        Parameter Matrix
    Y:
        data type : numpy array
        Expected Output Matrix
    W:
        data type : numpy array
        Weight Matrix

    '''
    return (np.matmul((activation(X, W).transpose() - Y), X).transpose()) / M


# Generate Weights
def generate_weights():
    '''
    Generates a Matrix of weights according to the
    input data.
    '''
    global W
    W = np.random.rand(N, 1)


# Plots the data and the initial line with random parameters.
def initplot():
    '''
    PLots the initial conditions

    The input data in terms of
    a scatter plot of red dots.

    The random initial line which is
    the starting point of gradient
    descent as a black line.
    '''
    plt.xlabel('parameter')
    plt.ylabel('output')

    plt.scatter(X[:, 0], Y, c='r')
    plt.plot(X[:, 0], activation(X, W), c='k')


# Normal gradient descent execution.
def gradient_descent():
    '''
    The simple gradient descent execution.

    W = W - a * dm

    a - learning rate
    dm - derivative of loss function wrt x (parameter)

    'steps' is the total number
    of iterations before
    we stop gradient descent.
    '''
    global W
    for step in range(steps):
        W = W - a * dm(X, Y, W)


# Stochastic gradient descent execution.
def gradient_descent_stochastic():

    global W

    for step in range(steps):
        # Choose a random data point to calculate gradient descent
        i = random.randint(0, M-1)
        x = X[i, :]
        y = Y[:, i]
        x.shape = (1, N)
        y.shape = (1, 1)
        W = W - a * dm(x, y, W)


def gradient_descent_mini_batch(batchsize=5):

    global W

    for step in range(steps):

        index = [random.randint(0, M-1) for i in range(batchsize)]
        x = X[index, :]
        y = Y[:, index]
        x.shape = (batchsize, N)
        y.shape = (1, batchsize)
        W = W - a * dm(x, y, W)


# Variable required for gradient descent algorithms with momentum.
Vp = 0


# Gradient Descent with momentum.
# Vp is previous change Vc is current change.
def gradient_descent_momentum(batchsize=5):

    gamma = 0.9

    global W
    global Vp

    for step in range(steps):

        index = [random.randint(0, M-1) for i in range(batchsize)]
        x = X[index, :]
        y = Y[:, index]
        x.shape = (batchsize, N)
        y.shape = (1, batchsize)

        Vc = gamma * Vp + a * dm(x, y, W)

        W = W - Vc

        Vp = Vc


def gradient_descent_nesterov_accelerated(batchsize=5):

    gamma = 0.9

    global W
    global Vp

    for step in range(steps):

        index = [random.randint(0, M-1) for i in range(batchsize)]
        x = X[index, :]
        y = Y[:, index]
        x.shape = (batchsize, N)
        y.shape = (1, batchsize)

        Vc = gamma * Vp + a * dm(x, y, W - gamma * Vp)

        W = W - Vc

        Vp = Vc


# Variable required for gradient descent algorithms which include adagrad, RMSprop and adaDelta.
S = 0


# Adagrad implementation.
def gradient_descent_adagrad(batchsize=5):

    epsilon = 0.00000001

    global W
    global S

    for step in range(steps):

        index = [random.randint(0, M-1) for i in range(batchsize)]
        x = X[index, :]
        y = Y[:, index]
        x.shape = (batchsize, N)
        y.shape = (1, batchsize)

        d = dm(x, y, W)
        S += d * d
        W = W - a / np.sqrt(S + epsilon) * d


# AdaDelta/RMSProp implementation.
def gradient_descent_adadelta(batchsize=5):

    epsilon = 0.00000001
    gamma = 0.9

    global W
    global S

    for step in range(steps):

        index = [random.randint(0, M-1) for i in range(batchsize)]
        x = X[index, :]
        y = Y[:, index]
        x.shape = (batchsize, N)
        y.shape = (1, batchsize)

        d = dm(x, y, W)
        S = gamma * S + (1 - gamma) * d * d
        W = W - a / np.sqrt(S + epsilon) * d

 # Variable required for gradient descent algorithms which include adagrad, RMSprop and adaDelta.
V = 0
S = 0
Vc = 0
Sc = 0

# AdaDelta/RMSProp implementation.


def gradient_descent_adam(batchsize=5):

    epsilon = 0.00000001
    b1 = 0.9
    b2 = 0.999

    global W
    global V, Vc
    global S, Sc

    for step in range(steps):

        index = [random.randint(0, M-1) for i in range(batchsize)]
        x = X[index, :]
        y = Y[:, index]
        x.shape = (batchsize, N)
        y.shape = (1, batchsize)

        d = dm(x, y, W)

        V = b1 * V + (1 - b1) * d
        S = b2 * S + (1 - b2) * d * d

        Vc = V / (1 - (b1))
        Sc = S / (1 - (b2))

        W = W - a / (np.sqrt(Sc) + epsilon) * Vc


# Plots the final output and shows the graph.
def finalplot():
    '''
    Plots the final conditions.

    The final output line in the form of
    a blue line.
    '''
    plt.plot(X[:, 0], activation(X, W), 'b')
    plt.show()


if __name__ == '__main__':

    # Define parameters related to gradient descent
    steps = 2000
    a = 0.01

    readdata('dataset1.txt')

    generate_weights()

    initplot()

    gradient_descent_adam()

    printmat('W', W)

    finalplot()
