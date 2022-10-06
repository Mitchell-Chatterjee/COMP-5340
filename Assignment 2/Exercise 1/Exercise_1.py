import torch
import math
import random

# We are assuming that neurons can be broken into two inputs each
class Neuron:
    def __init__(self, operation, inputs):
        self.operation = operation
        self.current_activation = 0
        self.inputs = inputs

    def activate(self, x):
        if not x:
            self.current_activation = self.operation(self.inputs)
        else:
            self.current_activation = self.operation(self.inputs + [x])
        return self.current_activation

# A class to store neurons per layer
class Layer:
    def __init__(self, neurons):
        self.neurons = neurons

    def calculate(self, x):
        activations = []
        for neuron in self.neurons:
            activations.append(neuron.activate(x))
        return activations


class Network:
    def __init__(self, dim):
        # Initializing the weights
        self.A = [[random.random()] * dim] * dim
        self.B = [[random.random()] * dim] * dim
        self.C = [[random.random()] * dim] * dim

        # Defining the neurons
        self.y = Neuron(matrix_mult, self.A)
        self.v = Neuron(matrix_mult, self.B)
        self.u = Neuron(sigmoid, [self.y])
        self.z = Neuron(matrix_add, [self.u, self.v])
        self.w = Neuron(matrix_mult, [self.z, self.C])

        # Defining layers
        self.layer_1 = Layer([self.y, self.v])
        self.layer_2 = Layer(self.u)
        self.layer_3 = Layer(self.z)
        self.layer_4 = Layer(self.w)


    def forward(self, x):
        # Add the output from each layer to a dictionary
        # TODO: Issue with how inputs to each layer are used
        x = self.layer_1.calculate(x)
        x = self.layer_2.calculate(x)
        x = self.layer_3.calculate(x)
        x = self.layer_4.calculate(x)

        return x



# Note the first variable, is the variable we are taking the derivative wrt.
def derivative(x, y, operation):
    if operation == "*":
        return y
    if operation == "+":
        return 1

def sigmoid(z):
    rows_z , cols_z = len(z), len(z[0])
    for i in range(rows_z):
        for j in range(cols_z):
            z[i][j] = 1.0/(1.0+math.exp(-z[i][j]))
    return z

def sigmoid_dv(z):
    return sigmoid(z)*(1-sigmoid(z))

def matrix_mult(inputs):
    A, B = inputs
    rows_a, cols_a = len(A), len(A[0])
    rows_b, cols_b = len(B), len(B[0])

    # Assume that we are always passing them in the correct order. No swapping.
    if cols_a != rows_b:
        raise Exception('Incompatible matrices.')

    C = [[0]*cols_b]*rows_a
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(rows_b):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matrix_add(inputs):
    A, B = inputs
    rows_a, cols_a = len(A), len(A[0])

    C = [ [0] * cols_a ] * rows_a
    for i in range(rows_a):
        for j in range(cols_a):
            C[i][j] = A[i][j] + B[i][j]


if __name__ == '__main__':
    print('here')



    # Test code for matrix mult
    # A = [[1, 2, 3],
    #      [1, 2, 3]]
    # B = [[1, 2, 3, 4],
    #      [1, 2, 3, 4],
    #      [1, 2, 3, 4]]
    #
    # print(matrix_mult(A, B))