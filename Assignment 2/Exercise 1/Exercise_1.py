import torch
import math
import random


class Graph:
    """Represents a computational graph
    """

    def __init__(self):
        """Construct Graph"""
        self.operations = []
        self.placeholders = []
        self.variables = []

    def as_default(self):
        global _default_graph
        _default_graph = self


class Placeholder:
    """Represents a placeholder node that has to be provided with a value
       when computing the output of a computational graph
    """

    def __init__(self):
        """Construct placeholder
        """
        self.consumers = []

        # Append this placeholder to the list of placeholders in the currently active default graph
        _default_graph.placeholders.append(self)


class Variable:
    def __init__(self, initial_value = None):
        self.value = initial_value
        self.consumers = []

        # Append this variable to the list of variables in the currently active default graph
        _default_graph.variables.append(self)


# We are assuming that neurons can be broken into two inputs each
class Node:
    def __init__(self, operation, input_nodes=[]):
        self.operation = operation

        # Initialize list of consumers (i.e. nodes that receive this operation's output as input)
        self.consumers = []

        # Append this operation to the list of consumers of all input nodes
        for input_node in input_nodes:
            input_node.consumers.append(self)

        # Append this operation to the list of operations in the currently active default graph
        _default_graph.operations.append(self)

    def compute(self):
        return self.operation(self.inputs)



# A class to store neurons per layer
class Layer:
    # We initialize the layer with all non-neuron inputs it may have
    def __init__(self, neurons):
        self.neurons = neurons
        self.activations = {}

    def calculate(self, activation_dict):
        # Now calculate the values in this layer
        for neuron in self.neurons:
            self.activations[neuron.label] = neuron.compute(activation_dict)
        # Carry forward any values from the previous layer we may need
        return {**activation_dict, **self.activations}

    def error(self, activation_dict, gradient_dict):
        # Calculate the gradient here
        # The gradient for the previous is calculated here
        # The
        for neuron in self.neurons:
            neuron.error(activation_dict, gradient_dict)


# Note the first variable, we are taking the derivative wrt x.
def derivative(z, operation):
    if operation == matrix_mult:
        x, y = z
        return y
    elif operation == matrix_add:
        x, y = z
        return [1]*len(x)
    elif operation == sigmoid:
        return sigmoid_dv(z[0])
    elif operation == squared:
        return 2 * z[0]


def squared(x):
    return x * x


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

    # We may need to reshape them

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


def euclidean_norm(x):
    return math.sqrt(sum([elem ** 2 for elem in x]))


# This is the squared Euclidean Norm
def calculate_loss(output):
    return sum([elem ** 2 for elem in output])


if __name__ == '__main__':
    # N = 5
    # data = []
    # for i in range(N):
    #     data.append([random.randint(9, 10), random.randint(9, 10), random.randint(9, 10)])
    #
    # network = Network(len(data[0]))
    # # for x in data:
    # #     network.forward(x)

    # Create a new graph
    Graph().as_default()

    # Create variables
    A = Variable([[1, 0], [0, -1]])
    b = Variable([1, 1])

    # Create placeholder
    x = Node([])

    # Create hidden node y
    y = matmul(A, x)

    # Create output node z
    z = add(y, b)



    # Test code for matrix mult
    # A = [[1, 2, 3],
    #      [1, 2, 3]]
    # B = [[1, 2, 3, 4],
    #      [1, 2, 3, 4],
    #      [1, 2, 3, 4]]
    #
    # print(matrix_mult(A, B))