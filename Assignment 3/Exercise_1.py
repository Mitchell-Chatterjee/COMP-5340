# Loading datasets
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PGD import train, test


PATH = ["first.pt", "second.pt", "third.pt"]


class PGDSettings:
    def __init__(self, num_steps, eps, step_size, targeted=False):
        self.num_steps = num_steps
        self.eps = eps
        self.step_size = step_size
        self.targeted = targeted


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    pgd_settings = [PGDSettings(num_steps=20, eps=0.3, step_size=0.02),
                    PGDSettings(num_steps=1, eps=0.3, step_size=0.5),
                    PGDSettings(num_steps=20, eps=0.3, step_size=0.02, targeted=True)]
    print(ConvNet())

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=25,
                                               shuffle=True, num_workers=4)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100,
                                              shuffle=True, num_workers=4)
    print("Done importing data.")

    models = [ConvNet() for _ in range(3)]

    CUDA=torch.cuda.is_available()
    if CUDA:
        for model in models:
            model.cuda()

    # Let's first define our device as the first visible cuda device if we have
    # CUDA available:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    print(torch.cuda.get_device_name(device=device))

    # Train the network
    print("\n\n\n**************Training Phase**************")
    for model, path, setting in zip(models, PATH, pgd_settings):
        print(f"\n\nTraining model with the following settings\n"
              f"num_steps={setting.num_steps}, step_size={setting.step_size}, eps={setting.eps}, targeted={setting.targeted}")
        # We either train the model or load it from memory if it has already been trained
        if os.path.exists(path):
            # Load the model
            model.load_state_dict(torch.load(path))
            model.eval()
        else:
            model = train(model, train_loader, CUDA, device, num_steps=setting.num_steps,
                          step_size=setting.step_size, epsilon=setting.eps, targeted=setting.targeted)
            # Save the model
            torch.save(model.state_dict(), path)
    print('Finished Training')

    # Evaluate the model
    print("\n\n\n**************Testing Phase**************")
    test_step_size = 0.01
    test_num_steps = 40
    test_radii = (0, 0.1, 0.2, 0.3, 0.45)
    image_flag = True
    for model in models:
        test(model=model, CUDA=CUDA, device=device, test_loader=test_loader,
             num_steps=test_num_steps, step_size=test_step_size, test_radii=test_radii, image_flag=image_flag)
        image_flag = False

    print('Finished Testing')