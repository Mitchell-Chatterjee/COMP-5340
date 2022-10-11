import Compute_Graph as cpg
import random
import numpy as np
from queue import Queue

'''
Note this work was largely aided by the following links.

https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/
https://www.codingame.com/playgrounds/9487/deep-learning-from-scratch---theory-and-implementation/
'''

################### Gradient Descent ###################
LEARNING_RATE = 0.01


class GradientDescent(cpg.Node):
    def __init__(self, error):
        super().__init__([error])

    def compute(self):
        # Compute gradients
        gradient_table = compute_gradients(self.input_nodes[0])

        # Iterate all variables
        for node in gradient_table:
            if type(node) == cpg.Variable:
                # Retrieve gradient for this variable
                grad = gradient_table[node]

                # Take a step in the negative direction of the gradient
                node.value -= LEARNING_RATE * grad

########################################################


################ Backprop ##############################


def compute_gradients(loss):
    # gradient_table[node] will contain the gradient of the loss w.r.t. the node's output
    gradient_table = {}

    # The gradient of the loss with respect to the loss is just 1
    gradient_table[loss] = 1

    # Perform a breadth-first search, backwards from the loss
    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()

        # We do not compute on the loss
        if node != loss:
            # Compute the gradient of the loss with respect to this node's output
            gradient_table[node] = 0

            # Iterate all consumers
            for consumer in node.consumers:

                # Retrieve the gradient of the loss w.r.t. consumer's output
                lossgrad_wrt_consumer_output = gradient_table[consumer]

                # Get the gradient of the loss with respect to all of consumer's inputs
                lossgrads_wrt_consumer_inputs = consumer.gradient(lossgrad_wrt_consumer_output)

                if len(consumer.input_nodes) == 1:
                    # If there is a single input node to the consumer, lossgrads_wrt_consumer_inputs is a scalar
                    gradient_table[node] += lossgrads_wrt_consumer_inputs

                else:
                    # Otherwise, lossgrads_wrt_consumer_inputs is an array of gradients for each input node

                    # Retrieve the index of node in consumer's inputs
                    node_index_in_consumer_inputs = consumer.input_nodes.index(node)

                    # Get the gradient of the loss with respect to node
                    lossgrad_wrt_node = lossgrads_wrt_consumer_inputs[node_index_in_consumer_inputs]

                    # Add to total gradient
                    gradient_table[node] += lossgrad_wrt_node

        # Append each input node to the queue
        if hasattr(node, "input_nodes"):
            for input_node in node.input_nodes:
                if not input_node in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    # Return gradients for each visited node
    return gradient_table
########################################################


K = 3

# Create a new graph
graph = cpg.Graph().as_default()

# Create variables
A, B, C = cpg.Variable([[1]*K] * K), cpg.Variable([[1]*K] * K), cpg.Variable([[1]*K] * K)

# Create placeholder
x = cpg.UnsetNode()

# Create hidden node y, u, v and z
y = cpg.matmul(A, x)
u = cpg.sigmoid(y)
v = cpg.matmul(B, x)
z = cpg.add(u, v)

# Create output node w
w = cpg.matmul(C, z)

# This is the loss
Error = cpg.error(w)

minimizer = GradientDescent(Error)

activation_dict = {
    x: [2] * K
}

for step in range(100):
    loss = graph.forward(Error, activation_dict)
    if step % 10 == 0:
        print("Step:", step, " Loss:", loss)
    graph.backward(minimizer)