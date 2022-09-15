import random
import math
import torch
import numpy

# Constants
MU = 0.0


# Could also define this as a lambda function

def getValue(x_vals):
    return torch.tensor([math.cos(2 * math.pi * x) for x in x_vals])


def getDataTensor(data_size, st_dev):
    x_vals = torch.rand(data_size, requires_grad=True)
    y_vals = torch.tensor(getValue(x_vals) + torch.normal(mean=MU, std=st_dev, size=(1, data_size))[0],
                          requires_grad=True)

    return x_vals, y_vals


# Note polynomial is function
def getMSE(dataset, theta):
    # Rewrite using torch
    x_vals, y_vals = dataset
    # Convert x_values to a compatible vector to multiply with theta
    x_matrix = torch.tensor([[x ** i for i in range(len(theta))] for x in x_vals])

    loss = torch.pow(torch.sub(x_matrix.matmul(theta), y_vals), 2)
    return sum(loss)/(len(x_vals))


def getLoss(x, y, theta, degree):
    loss = torch.tensor([x ** i for i in range(degree)])
    return torch.sub(theta.matmul(loss), y) ** 2


# We use regular Gradient Descent
def fitData(degree, train_data, st_dev, iterations, learning_rate=0.01, batch_size=0.5):
    batch_size = batch_size or (len(train_data) // 2)
    degree = degree + 1

    # Start with random theta
    # We add an extra degree for the constant term
    theta = torch.randn(degree, requires_grad=True)

    for _ in range(iterations):
        loss = getMSE(train_data, theta)
        loss.backward()

        # Updating theta
        theta = torch.tensor(theta - learning_rate*theta.grad, requires_grad=True)

    # Returns theta, e_in, e_out
    return theta, getMSE(train_data, theta), getMSE(getDataTensor(2000, st_dev), theta)


def experiment(training_size, degree, variance):
    st_dev = math.sqrt(variance)
    theta_bar, e_in_bar, e_out_bar = 0, 0, 0

    for _ in range(50):
        training_data = getDataTensor(training_size, st_dev)
        theta, e_in, e_out = fitData(10, train_data, 0.5, iterations=100, learning_rate=0.01, batch_size=0.5)

        theta_bar = theta_bar + theta
        e_in_bar = e_in_bar + e_in
        e_out_bar = e_out_bar + e_out
    theta_bar = theta_bar / 50
    e_in_bar = e_in_bar / 50
    e_out_bar = e_out_bar / 50


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_data = getDataTensor(1000, 0.5)

    theta, E_in, E_out = fitData(100, train_data, 0.5, iterations=100, learning_rate=0.01, batch_size=0.5)
    print(theta, E_in, E_out, sep="\n")
