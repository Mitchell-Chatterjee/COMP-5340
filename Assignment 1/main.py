import math
import torch
import matplotlib.pyplot as plt

# Constants
MU = 0.0
NUMBER_OF_TRIALS = 50
BATCH_SIZE = 50
LARGE_DATA_SIZE = 1000
ITERATIONS = 10
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

    p_mat = torch.randperm(batch_size)
    return torch.tensor([x_vals[i] for i in p_mat], device=cuda), torch.tensor([y_vals[i] for i in p_mat], device=cuda)


def getMSE(dataset, theta):
    x_vals, y_vals = dataset
    # Convert x_values to a compatible matrix to multiply with theta
    x_matrix = torch.tensor([[x ** i for i in range(len(theta))] for x in x_vals], device=cuda)

    # Compute loss and take the average
    loss = torch.pow(torch.sub(x_matrix.matmul(theta), y_vals), 2)

    return sum(loss)/(len(x_vals))


# We use Stochastic Gradient Descent when the sample size is larger than 50
def fitData(degree, train_data, st_dev, iterations, learning_rate=0.01, batch_size=BATCH_SIZE):

    # Start with random theta
    theta = torch.randn(degree, requires_grad=True, device=cuda)

    for _ in range(iterations):
        # Compute the MSE
        dataset = getBatch(train_data, batch_size)
        loss = getMSE(dataset, theta)
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
                                     learning_rate=0.01, batch_size=BATCH_SIZE)

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


def plot(e_in_array, e_out_array, e_bias_array, x_values, x_label, title, ax):
    ax.plot(x_values, e_in_array, 'g')
    ax.plot(x_values, e_out_array, 'b')
    ax.plot(x_values, e_bias_array, 'r')

    ax.set_xlabel(x_label)
    ax.set_title(title)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle('Experiment')

    variance_e_in, variance_e_out, variance_e_bias = [], [], []
    degree_e_in, degree_e_out, degree_e_bias = [], [], []

    # First loop goes from degree 0 to 20
    for var in VARIANCES:
        for degree in range(0, 21):
            e_in_array, e_out_array, e_bias_array = [], [], []
            for size in TRAINING_SET_SIZES:
                e_in_bar, e_out_bar, e_bias = experiment(training_size=size, degree=degree, variance=var)

                e_in_array.append(e_in_bar)
                e_out_array.append(e_out_bar)
                e_bias_array.append(e_bias)

                # Plot 2: Increasing degree with maximal sample size
                if var == VARIANCES[0] and size == TRAINING_SET_SIZES[-1]:
                    degree_e_in.append(e_in_bar)
                    degree_e_out.append(e_out_bar)
                    degree_e_bias.append(e_bias)

                # Plot 3: Increasing variance with maximal degree and sample size
                if degree == 20 and size == TRAINING_SET_SIZES[-1]:
                    variance_e_in.append(e_in_bar)
                    variance_e_out.append(e_out_bar)
                    variance_e_bias.append(e_bias)

            # Plot 1: Increasing sample size with maximal degree
            if degree == 20 and var == VARIANCES[0]:
                plot(e_in_array, e_out_array, e_bias_array, x_values=TRAINING_SET_SIZES,
                     x_label="Training Set Size",
                     title=('Increasing sample size with degree %s and variance %s' % (degree, var)), ax=ax1)

    plot(degree_e_in, degree_e_out, degree_e_bias, x_values=range(0,21), x_label="Model Complexity (Degree)",
         title=('Increasing model complexity with sample size %s and variance %s' % (TRAINING_SET_SIZES[-1], VARIANCES[0])), ax=ax2)
    plot(variance_e_in, variance_e_out, variance_e_bias, x_values=VARIANCES, x_label="Variance",
         title=('Increasing variance with sample size %s and degree %s' % (TRAINING_SET_SIZES[-1], 20)), ax=ax3)

    ax1.legend(['E_in', 'E_out', 'E_bias'])
    ax2.legend(['E_in', 'E_out', 'E_bias'])
    ax3.legend(['E_in', 'E_out', 'E_bias'])

    plt.show()