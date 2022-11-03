# Loading datasets
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PGD import projected_gradient_descent


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    print(ConvNet())

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=25,
                                               shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100,
                                              shuffle=True, num_workers=1)
    print("Done importing data.")

    net = ConvNet()
    CUDA=torch.cuda.is_available()
    if CUDA:
        net.cuda()

    # Let's first define our device as the first visible cuda device if we have
    # CUDA available:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    print(torch.cuda.get_device_name(device=device))

    # 4. Train the network
    print("\n\n\n**************Training Phase**************")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    accuracy_values=[]
    epoch_number=[]

    # Training loss and accuracy arrays
    training_loss_regular = []
    training_loss_perturbed = []
    training_accuracy_regular = []
    training_accuracy_perturbed = []

    for epoch in range(10):  # loop over the dataset multiple times. Here 10 means 10 epochs
        running_loss_adv = 0.0
        running_loss_reg = 0.0
        perturbed_data = torch.empty()

        # This is the training loop
        for i, (inputs,labels) in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            if CUDA:
              inputs = inputs.cuda()
              labels = labels.cuda()
            else:
              inputs = inputs.cpu()
              labels = labels.cpu()

            x_adv = projected_gradient_descent(net, inputs, labels, loss_fn, num_steps=20,
                                               step_size=0.02, epsilon=0.3, delta=torch.rand(inputs.shape, device=device))

            # Append the perturbed data for later use in testing
            perturbed_data.cat(x_adv)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            reg_outputs = net(inputs)
            adv_outputs = net(x_adv)

            # Regular loss
            loss_reg = loss_fn(reg_outputs, labels)

            # Adversarial loss for training
            loss_adv = loss_fn(adv_outputs, labels)
            loss_adv.backward()
            optimizer.step()

            # Aggregate losses
            running_loss_adv += loss_adv.item()
            running_loss_reg += loss_reg.item()

        # print statistics
        print('[epoch%d] adversarial loss: %.3f' %
              (epoch + 1, running_loss_adv / len(inputs)))
        print('[epoch%d] regular loss: %.3f' %
              (epoch + 1, running_loss_reg / len(inputs)))

        correct = 0
        total = 0

        # This part calculates the training accuracy and training loss on the original training images
        with torch.no_grad():
            for images, labels in train_loader:
                if CUDA:
                  images = images.cuda()
                  labels = labels.cuda()
                else:
                  images = images.cpu()
                  labels =labels.cpu()

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                if CUDA:
                  correct += (predicted.cpu()==labels.cpu()).sum().item()
                else:
                  correct += (predicted==labels).sum().item()

            Train_Accuracy_Reg = 100 * correct / total;
            epoch_number += [epoch+1]
            accuracy_values += [Train_Accuracy_Reg]
            print('Epoch=%d Training Accuracy Regular=%.3f' %
                      (epoch + 1, Train_Accuracy_Reg))

        # Record the training losses and accuracies in an array
        training_loss_regular.append(round(running_loss_reg / len(inputs), 3))
        training_loss_perturbed.append(round(running_loss_adv / len(inputs), 3))
        training_accuracy_regular.append(Train_Accuracy_Reg)
        training_accuracy_perturbed.append()

    print('Finished Training')