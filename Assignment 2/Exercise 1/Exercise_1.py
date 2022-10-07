import torch
import math
import random


# We are assuming that neurons can be broken into two inputs each
class Neuron:
    def __init__(self, operation, input_keys, label):
        self.operation = operation
        self.current_activation = 0
        # These are keys to a dictionary
        self.input_keys = input_keys
        self.label = label

    def activate(self, x):
        # Get the inputs from the dictionary
        inputs = [x[key] for key in self.input_keys]
        # Calculate the activation using the inputs
        self.current_activation = self.operation(inputs)
        return self.current_activation


# A class to store neurons per layer
class Layer:
    # We initialize the layer with all non-neuron inputs it may have
    def __init__(self, neurons, inputs):
        self.neurons = neurons
        self.activations = {}
        # These are standard input values that are not neurons
        self.inputs = inputs

    def calculate(self, prev_layer, input):
        # Merge the two dictionaries and add the input
        x = {**self.inputs, **prev_layer, "x": input}

        # Now calculate the values in this layer
        for neuron in self.neurons:
            self.activations[neuron.label] = neuron.activate(x)
        return self.activations


class Network:
    def __init__(self, dim):
        # Initializing the weights
        self.A = [[random.random()] * dim] * dim
        self.B = [[random.random()] * dim] * dim
        self.C = [[random.random()] * dim] * dim

        # Defining the neurons.
        # Each neuron has an operation, a set of input keys, and a label.
        self.y = Neuron(matrix_mult, ["A", "x"], "y")
        self.v = Neuron(matrix_mult, ["B", "x"], "v")
        self.u = Neuron(sigmoid, ["y"], "u")
        self.z = Neuron(matrix_add, ["u", "v"], "z")
        self.w = Neuron(matrix_mult, ["z", "C"], "w")

        # Defining layers
        self.layer_1 = Layer([self.y], {"A": self.A})
        self.layer_2 = Layer([self.u, self.v], {"B": self.B})
        self.layer_3 = Layer([self.z], {})
        self.layer_4 = Layer([self.w], {"C": self.C})

    def forward(self, input):
        # Add the output from each layer to a dictionary
        # TODO: Issue with how inputs to each layer are used
        # We initialize the dictionary with the standard values

        x = self.layer_1.calculate({}, input)
        x = self.layer_2.calculate(x, input)
        x = self.layer_3.calculate(x, input)
        x = self.layer_4.calculate(x, input)

        return x


# Note the first variable, we are taking the derivative wrt x.
def derivative(x, y, operation):
    if operation == matrix_mult:
        return y
    elif operation == matrix_add:
        return [1]*len(x)
    elif operation == sigmoid:
        return sigmoid_dv(x)


def sigmoid(z):
    rows_z , cols_z = len(z), len(z[0])
    for i in range(rows_z):
        for j in range(cols_z):
            z[i][j] = 1.0/(1.0+math.exp(-z[i][j]))
    return z


def sigmoid_dv(z):
    rows_z, cols_z = len(z), len(z[0])
    for i in range(rows_z):
        for j in range(cols_z):
            z[i][j] = sigmoid(z[i][j])*(1-sigmoid(z[i][j]))
    return z


# Assuming x is R^k and A, B, C are KxK
def matrix_mult(inputs):
    A, B = inputs
    try:
        len(A[0])
    except: # swap if we have them in the wrong order
        A, B = B, A

    rows_a, cols_a = len(A), len(A[0])
    rows_b = len(B)

    # Assume that we are always passing them in the correct order. No swapping.
    if cols_a != rows_b:
        raise Exception('Incompatible matrices.')

    C = [0]*rows_a
    for i in range(rows_a):
        for j in range(rows_b):
            C[j] += A[i][j] * B[j]
    return C


def matrix_add(inputs):
    A, B = inputs
    rows_a, cols_a = len(A), len(A[0])

    C = [0] * cols_a
    for i in range(rows_a):
        for j in range(cols_a):
            C[j] = A[i][j] + B[j]
    return C


# This is the squared Euclidean Norm
def calculate_loss(output):
    return sum([elem ** 2 for elem in output])


if __name__ == '__main__':
    N = 5
    data = []
    for i in range(N):
        data.append([random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)])

    network = Network(len(data[0]))
    for x in data:
        network.forward(x)

    output = network.forward(data[0])
    print(output)
    print(calculate_loss(output['w']))



    # Test code for matrix mult
    # A = [[1, 2, 3],
    #      [1, 2, 3]]
    # B = [[1, 2, 3, 4],
    #      [1, 2, 3, 4],
    #      [1, 2, 3, 4]]
    #
    # print(matrix_mult(A, B))