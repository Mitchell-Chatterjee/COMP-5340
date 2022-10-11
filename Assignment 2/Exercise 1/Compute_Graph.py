import math
import numpy as np


def getRowsAndCols(a):
    rows, cols = len(a), 0
    try:
        cols = len(a[0])
    except:
        pass

    return rows, cols


def convertListToMatrix(a):
    return [[elem] for elem in a]


def transpose(a):
    rows, cols = getRowsAndCols(a)
    z = []
    for _ in range(rows):
        z.append([0])

    if cols == 0:
        for i in range(rows):
            z[i][0] = a[i]
    else:
        z = []
        for i in range(cols):
            z.append([0]*rows)

        for i in range(rows):
            for j in range(cols):
                z[j][i] = a[i][j]

    return z


# Computational nodes
class Node:
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.consumers = []
        for input_node in input_nodes:
            input_node.consumers.append(self)

        _default_graph.nodes.append(self)

    def compute(self):
        pass


# Computes matrix additions
class add(Node):
    def __init__(self, x, y):
        super().__init__([x, y])

    # Assuming x and y are the same size
    def compute(self, x_value, y_value):
        rows_x, cols_x = getRowsAndCols(x_value)
        rows_y, cols_y = getRowsAndCols(y_value)

        # Convert a list to a matrix
        if cols_x == 0:
            x_value = convertListToMatrix(x_value)
        if cols_y == 0:
            y_value = convertListToMatrix(y_value)

        rows_x, cols_x = getRowsAndCols(x_value)
        rows_y, cols_y = getRowsAndCols(y_value)

        # Build new matrix
        C = []
        for _ in range(rows_x):
            C.append([0] * cols_x)

        for i in range(rows_x):
            for j in range(cols_x):
                C[i][j] = x_value[i][j] + y_value[i][j]
        return C

    def gradient(self, grad):
        return [grad, grad]


class error(Node):
    def __init__(self, w):
        super().__init__([w])

    def compute(self, w):
        return w.T.dot(w)
        # return matmul.compute(self, transpose(w), w)

    # Gradient of the loss wrt to the loss is 1
    def gradient(self, grad):
        return 1


# Assuming x is R^k and A, B, C are KxK
class matmul(Node):
    def __init__(self, a, b):
        super().__init__([a, b])

    def compute(self, a_value, b_value):
        return a_value.dot(b_value)

    # def compute(self, a_value, b_value):
    #     rows_a, cols_a = getRowsAndCols(a_value)
    #     rows_b, cols_b = getRowsAndCols(b_value)
    #
    #     # Convert a list to a matrix
    #     if cols_a == 0:
    #         a_value = convertListToMatrix(a_value)
    #     if cols_b == 0:
    #         b_value = convertListToMatrix(b_value)
    #
    #     rows_a, cols_a = getRowsAndCols(a_value)
    #     rows_b, cols_b = getRowsAndCols(b_value)
    #
    #     if cols_a != rows_b:
    #         b_value = transpose(b_value)
    #         rows_b, cols_b = getRowsAndCols(b_value)
    #
    #     # Build new matrix
    #     C = []
    #     for _ in range(rows_a):
    #         C.append([0] * cols_b)
    #
    #     for i in range(rows_a):
    #         for j in range(cols_b):
    #             for k in range(rows_b):
    #                 C[i][j] += a_value[i][k] * b_value[k][j]
    #     return C

    def gradient(self, grad):
        if type(grad) != np.array:
            grad = np.array(grad)

        A = self.inputs[0]
        B = self.inputs[1]

        return [grad.dot(B.T), A.T.dot(grad)]


# Computes sigmoid on a matrix
class sigmoid(Node):
    def __init__(self, z):
        super().__init__([z])

    def compute(self, z):
        rows_z, cols_z = getRowsAndCols(z)

        # Convert a list to a matrix
        if cols_z == 0:
            z = convertListToMatrix(z)

        rows_z, cols_z = getRowsAndCols(z)

        for i in range(rows_z):
            for j in range(cols_z):
                z[i][j] = 1.0 / (1.0 + math.exp(-z[i][j]))
        return z

    def gradient(self, grad):
        sigmoid = self.output

        return grad * sigmoid * (1 - sigmoid)


class UnsetNode:
    def __init__(self):
        self.consumers = []
        _default_graph.unsetNodes.append(self)


# Input variables
class Variable:
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.consumers = []

        _default_graph.variables.append(self)


class Graph:
    def __init__(self):
        self.nodes = []
        self.unsetNodes = []
        self.variables = []

    def as_default(self):
        global _default_graph
        _default_graph = self
        return self

    def forward(self, last_node, activation_dict={}):
        # Post order traversal gives us the correct execution sequence of the nodes
        postorder_traversal = []
        traverse_postorder(last_node, postorder_traversal)

        # Iterate all nodes to determine their value
        for node in postorder_traversal:
            node_type = type(node)
            if node_type == UnsetNode:
                # Set the value to the value from activation_dict
                node.output = activation_dict[node]
            elif node_type == Variable:
                # Set the node value to the variable's value attribute
                node.output = node.value
            else:  # Node
                # Get the input values for this operation from the output values of the input nodes
                node.inputs = [input_node.output for input_node in node.input_nodes]

                # Compute the output of this operation
                node.output = node.compute(*node.inputs)

            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)

        # Return the requested node value
        return last_node.output

    def backward(self, loss_node):
        loss_node.compute()


def traverse_postorder(node, postorder_traversal):
    if isinstance(node, Node):
        for val in node.input_nodes:
            traverse_postorder(val, postorder_traversal)
    postorder_traversal.append(node)