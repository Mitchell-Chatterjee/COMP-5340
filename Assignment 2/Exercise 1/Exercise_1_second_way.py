# Nodes
class Node:
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes

        # Initialize list of consumers (i.e. nodes that receive this operation's output as input)
        self.consumers = []

        # Append this operation to the list of consumers of all input nodes
        for input_node in input_nodes:
            input_node.consumers.append(self)

        # Append this operation to the list of nodes in the currently active default graph
        _default_graph.nodes.append(self)

    def compute(self):
        pass


# Computes matrix additions
class add(Node):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_value, y_value):
        return x_value + y_value


# Computes matrix multiplications
class matmul(Node):
    def __init__(self, a, b):
        super().__init__([a, b])

    def compute(self, a_value, b_value):
        return a_value.dot(b_value)


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


import numpy as np


class Session:
    def forward(self, last_node, activation_dict={}):
        # Post order traversal gives us the correct execution sequence of the nodes
        postorder_traversal = []
        traverse_postorder(last_node, postorder_traversal)

        # Iterate all nodes to determine their value
        for node in postorder_traversal:

            if type(node) == UnsetNode:
                # Set the value to the value from activation_dict
                node.output = activation_dict[node]
            elif type(node) == Variable:
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


def traverse_postorder(node, postorder_traversal):
    if isinstance(node, Node):
        for val in node.input_nodes:
            traverse_postorder(val, postorder_traversal)
    postorder_traversal.append(node)


# Create a new graph
Graph().as_default()

# Create variables
A = Variable([[1, 0], [0, -1]])
b = Variable([1, 1])

# Create placeholder
x = UnsetNode()

# Create hidden node y
y = matmul(A, x)

# Create output node z
z = add(y, b)

session = Session()
output = session.forward(z, {
    x: [1, 2]
})
print(output)

# Continue work from here: https://www.codingame.com/playgrounds/9487/deep-learning-from-scratch---theory-and-implementation/gradient-descent-and-backpropagation