import math
import torch
import matplotlib.pyplot as plt

# Constants
MU = 0.0
NUMBER_OF_TRIALS = 50
BATCH_SIZE = 50
LARGE_DATA_SIZE = 1000
ITERATIONS = 25
WEIGHT_DECAY = 1

cuda = torch.device('cuda')

# Experiment constants
TRAINING_SET_SIZES = [2, 5, 10, 20, 50, 100, 200]
VARIANCES = [0.01, 0.1, 1]


def getValue(x_vals):
    return torch.tensor([math.cos(2 * math.pi * x) for x in x_vals])


def getData(data_size, st_dev):
    x_vals = torch.rand(data_size, requires_grad=True, device=cuda)
    y_vals = torch.tensor(getValue(x_vals) + torch.normal(mean=MU, std=st_dev, size=(1, data_size))[0],
                          requires_grad=True, device=cuda)
    return x_vals, y_vals


def getBatch(dataset, batch_size):
    x_vals, y_vals = dataset
    if batch_size > len(x_vals):
        return x_vals, y_vals

    # We take a random permutation matrix with values from 0 to len(x_vals)-1
    # We then take only the first 0 to batch_size-1 entries
    p_mat = torch.randperm(len(x_vals))[:batch_size]
    return torch.tensor([x_vals[i] for i in p_mat], device=cuda), torch.tensor([y_vals[i] for i in p_mat], device=cuda)


def getMSE(dataset, theta):
    x_vals, y_vals = dataset
    # Convert x_values to a compatible matrix to multiply with theta
    x_matrix = torch.tensor([[x ** i for i in range(len(theta))] for x in x_vals], device=cuda)

    # Compute loss and take the average
    loss = torch.pow(torch.sub(x_matrix.matmul(theta), y_vals), 2)

    return sum(loss)/(len(x_vals))


# We use Mini-Batched Stochastic Gradient Descent when the sample size is larger than 50
def fitData(degree, train_data, st_dev, iterations, learning_rate=0.01, batch_size=BATCH_SIZE, weight_decay=0):

    # Start with random theta
    theta = torch.randn(degree, requires_grad=True, device=cuda)

    for _ in range(iterations):
        # Compute the MSE
        # This is regular gradient descent.
        loss = getMSE(train_data, theta) + weight_decay*(theta.matmul(theta))

        # In order to implement SGD uncomment the following line and comment out the preceding line.
        # loss = getMSE(getBatch(train_data, batch_size), theta) + weight_decay*(theta.matmul(theta))

        # Compute the gradient
        loss.backward()

        # Updating theta
        theta = torch.tensor(theta - learning_rate*theta.grad, requires_grad=True, device=cuda)

    # Returns theta, e_in, e_out
    return theta, getMSE(train_data, theta).item(), getMSE(getData(LARGE_DATA_SIZE, st_dev), theta).item()


def experiment(training_size, degree, variance):
    # We increase the value of degree by 1 to account for the constant term
    degree = degree + 1
    st_dev = math.sqrt(variance)

    theta_bar, e_in_bar, e_out_bar = torch.zeros(degree, device=cuda), 0, 0

    # Loop over M trials, M = 50
    for _ in range(NUMBER_OF_TRIALS):
        training_data = getData(training_size, st_dev)
        theta, e_in, e_out = fitData(degree=degree, train_data=training_data, st_dev=st_dev, iterations=ITERATIONS,
                                     learning_rate=0.01, batch_size=BATCH_SIZE, weight_decay=WEIGHT_DECAY)

        theta_bar = theta_bar.add(theta)
        e_in_bar = e_in_bar + e_in
        e_out_bar = e_out_bar + e_out

    # Average e_in, e_out and theta
    theta_bar = torch.div(theta_bar, NUMBER_OF_TRIALS)
    e_in_bar = round(e_in_bar / NUMBER_OF_TRIALS, 5)
    e_out_bar = round(e_out_bar / NUMBER_OF_TRIALS, 5)

    # Generate a new large dataset and test theta_bar
    e_bias = round(getMSE(getData(LARGE_DATA_SIZE, st_dev), theta_bar).item(), 5)

    return e_in_bar, e_out_bar, e_bias


if __name__ == '__main__':
    print(torch.cuda.get_device_name(0))
    variance_e_in, variance_e_out, variance_e_bias = [], [], []
    degree_e_in, degree_e_out, degree_e_bias = [], [], []

    for var in VARIANCES:
        for degree in range(0, 21):
            e_in_array, e_out_array, e_bias_array = [], [], []
            for size in TRAINING_SET_SIZES:
                e_in_bar, e_out_bar, e_bias = experiment(training_size=size, degree=degree, variance=var)

                e_in_array.append(e_in_bar)
                e_out_array.append(e_out_bar)
                e_bias_array.append(e_bias)

            degree_e_in.append(e_in_array)
            degree_e_out.append(e_out_array)
            degree_e_bias.append(e_bias_array)

        variance_e_in.append(degree_e_in)
        variance_e_out.append(degree_e_out)
        variance_e_bias.append(degree_e_bias)

    # Note that the code for plotting the results has been removed in order to reduce bloat
