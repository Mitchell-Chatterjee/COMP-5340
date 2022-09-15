import random
import math
import torch
import numpy

# Constants
MU = 0.0


# Could also define this as a lambda function

def getValue(x_vals):
    return torch.tensor([math.cos(2 * math.pi * x) for x in x_vals])

def getData(data_size, st_dev):
    # Draw normal values first
    normal_vals = numpy.random.normal(MU, st_dev, data_size)
    # Return the (X,Y) tuples
    return [getValue(random.random(), normal_vals[i]) for i in range(data_size)]


def getDataTensor(data_size, st_dev):
    x_vals = torch.rand(data_size, requires_grad=True)
    y_vals = torch.tensor(getValue(x_vals) + torch.normal(mean=MU, std=st_dev, size=(1, data_size))[0],
                          requires_grad=True)

    return x_vals, y_vals


# Note polynomial is function
def getMSE(dataset, theta):
    # Rewrite using torch
    x_vals, y_vals = dataset
    x_matrix = torch.tensor([[x ** i for i in range(len(theta))] for x in x_vals])

    loss = torch.pow(torch.sub(x_matrix.matmul(theta), y_vals), 2)
    return sum(loss)/(len(dataset))


def getLoss(x, y, theta, degree):
    loss = torch.tensor([x ** i for i in range(degree)])
    return torch.sub(theta.matmul(loss), y) ** 2


# We use mini-batched SGD, with default batch size of half the size of the dataset
def fitData(degree, train_data, st_dev, iterations, learning_rate=0.01, batch_size=0.5):
    batch_size = batch_size or (len(train_data) // 2)
    degree = degree + 1

    # Start with random theta
    # We add an extra degree for the constant term
    theta = torch.randn(degree, requires_grad=True)

    # We then need to convert our x-value to a compatible vector

    for _ in range(iterations):
        loss = getMSE(train_data, theta)
        loss.backward()

        print(theta.grad)

    # Returns theta, e_in, e_out
    # return theta, getMSE(train_data, theta), getMSE(getData(2000, st_dev), theta)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_data = getDataTensor(10, 0.5)

    fitData(1, train_data, 0.5, 10, learning_rate=0.01, batch_size=0.5)
