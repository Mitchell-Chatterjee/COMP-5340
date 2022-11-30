import torch
import torchvision
import torch.nn as nn


# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64


def download_cifar10(train_batch_size=25, test_batch_size=100, num_workers=2):
    """
    Downloads the ImageNet dataset.
    :param num_of_classes: The number of classes to select from the dataset.
    :param train_batch_size: Specifies the training batch size.
    :param test_batch_size: Specifies the test batch size.
    :param num_workers: Specifies the number of workers to load data. Modify this value based on your system resources.
    :return: Returns the value of filter_data_by_classes.
    """
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    return trainset, testset


def download_mnist(train_batch_size=25, test_batch_size=100, num_workers=2):
    """
    Downloads the MNIST dataset.
    :param num_of_classes: The number of classes to select from the dataset.
    :param train_batch_size: Specifies the training batch size.
    :param test_batch_size: Specifies the test batch size.
    :param num_workers: Specifies the number of workers to load data. Modify this value based on your system resources.
    :return: Returns the value of filter_data_by_classes.
    """
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

    return trainset, testset


def create_data_loaders(train_set, test_set):
    """
    Returns data loaders based on the training, validation and testing set passed in.
    :param train_set: The training data.
    :param validation_set: The validation data.
    :param test_set: The testing data.
    :return:
    """
    training_loaders = torch.utils.data.DataLoader(train_set, batch_size=100,
                                                   shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100,
                                              shuffle=True, num_workers=2)
    return training_loaders, test_loader


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs