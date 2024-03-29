import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def projected_gradient_descent(model, x, y, loss_fn, num_steps, step_size, epsilon, delta,
                               clamp=(0, 1), y_target=None):

    # Step 1: Perturb the elements in the batch
    x_hat = x.clone().detach().requires_grad_(True).to(x.device)
    targeted = y_target is not None

    # This creates a tensor of the same size as x_hat with values (-epsilon, +epsilon)
    delta = (epsilon * (-2)) * delta + epsilon
    x_hat = torch.add(x_hat, delta)

    for i in range(num_steps):
        # Step 2: Update x_hat
        x_old = x_hat.clone().detach().requires_grad_(True)

        prediction = model(x_old)
        loss = loss_fn(prediction, y_target if targeted else y)
        loss.backward()

        with torch.no_grad():
            gradients = step_size * x_old.grad.sign()
        
        z = x_old - gradients if targeted else x_old + gradients
        
        # Step 3: Project z to the L-inf ball
        x_hat = torch.min(torch.max(z, x - epsilon), x + epsilon)

        x_hat.clamp(*clamp)

    return x_hat.detach()


def train(net, train_loader, CUDA, device, num_steps=1, step_size=0.02, epsilon=0.3, targeted=False):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epoch_number = []

    # Training loss and accuracy arrays
    training_loss_regular = []
    training_loss_perturbed = []
    training_accuracy_regular = []
    training_accuracy_perturbed = []

    for epoch in range(10):  # loop over the dataset multiple times. Here 10 means 10 epochs
        running_loss_adv = 0.0
        running_loss_reg = 0.0

        # Place to store the adversarial data
        perturbed_data = None
        perturbed_data_labels = None

        # This is the training loop
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                inputs = inputs.cpu()
                labels = labels.cpu()

            # Regular loss
            with torch.no_grad():
                reg_outputs = net(inputs)
                loss_reg = loss_fn(reg_outputs, labels)

                # Get the second most likely label if this is a targeted attack
                _, second_prediction = (None, None) if not targeted else \
                    torch.kthvalue(reg_outputs.data, reg_outputs.shape[1]-1, dim=1)

            x_adv = projected_gradient_descent(net, inputs, labels, loss_fn, num_steps=num_steps, step_size=step_size,
                                               epsilon=epsilon, delta=torch.rand(inputs.shape, device=device),
                                               y_target=second_prediction)

            # Append the perturbed data for later use in testing
            if perturbed_data is not None:
                perturbed_data = torch.cat((perturbed_data, x_adv), 0)
                perturbed_data_labels = torch.cat((perturbed_data_labels, labels), 0)
            else:
                perturbed_data = x_adv
                perturbed_data_labels = labels

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            adv_outputs = net(x_adv)

            # Adversarial loss for training
            loss_adv = loss_fn(adv_outputs, labels)
            loss_adv.backward()
            optimizer.step()

            # Aggregate losses
            running_loss_adv += loss_adv.item()
            running_loss_reg += loss_reg.item()

        # print statistics
        print('[epoch%d] regular loss: %.3f' %
              (epoch + 1, running_loss_reg / len(inputs)))
        print('[epoch%d] adversarial loss: %.3f' %
              (epoch + 1, running_loss_adv / len(inputs)))

        with torch.no_grad():
            # This part calculates the training accuracy and training loss on the original training images
            correct = 0
            total = 0
            for images, labels in train_loader:
                if CUDA:
                    images = images.cuda()
                    labels = labels.cuda()
                else:
                    images = images.cpu()
                    labels = labels.cpu()

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                if CUDA:
                    correct += (predicted.cpu() == labels.cpu()).sum().item()
                else:
                    correct += (predicted == labels).sum().item()
            Train_Accuracy_Reg = 100 * correct / total;

            # This part calculates the training accuracy and training loss on the perturbed training images
            correct = 0
            for i in range(25, perturbed_data_labels.shape[0], 25):
                images, labels = perturbed_data[(i - 25):i], perturbed_data_labels[(i - 25):i]
                if CUDA:
                    images = images.cuda()
                    labels = labels.cuda()
                else:
                    images = images.cpu()
                    labels = labels.cpu()

                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                if CUDA:
                    correct += (predicted.cpu()==labels.cpu()).sum().item()
                else:
                    correct += (predicted==labels).sum().item()
            Train_Accuracy_Adv = 100 * correct / perturbed_data_labels.shape[0];

            epoch_number += [epoch + 1]
            print('Epoch=%d Training Accuracy Regular=%.3f' %
                  (epoch + 1, Train_Accuracy_Reg))
            print('Epoch=%d Training Accuracy Adversarial=%.3f' %
                  (epoch + 1, Train_Accuracy_Adv))

        # Clear up the gpu memory by deleting the perturbed data tensor
        del perturbed_data
        torch.cuda.empty_cache()

        # Record the training losses and accuracies in an array
        training_loss_regular.append(round(running_loss_reg / len(inputs), 3))
        training_loss_perturbed.append(round(running_loss_adv / len(inputs), 3))
        training_accuracy_regular.append(Train_Accuracy_Reg)
        training_accuracy_perturbed.append(Train_Accuracy_Adv)

    # Plot the training results and return the net
    plot_training_results(epoch_number, training_loss_regular, training_loss_perturbed, training_accuracy_regular, training_accuracy_perturbed)
    return net


def test(model, CUDA, device, test_loader, num_steps, step_size, test_radii, image_flag):
    loss_fn = nn.CrossEntropyLoss()

    Test_Accuracy_Adv = []
    Test_Accuracy_Adv_Targeted = []

    for test_eps in test_radii:
        # This part calculates the training accuracy and training loss on the original training images
        correct_adv, correct_adv_targeted = 0, 0
        total = 0
        for images, labels in test_loader:
            if CUDA:
                images = images.cuda()
                labels = labels.cuda()
            else:
                images = images.cpu()
                labels = labels.cpu()
            # Regular loss
            reg_outputs = model(images)

            # Get the second most likely label if this is a targeted attack
            _, second_prediction = torch.kthvalue(reg_outputs.data, reg_outputs.shape[1] - 1, dim=1)

            # Perturb the images
            x_adv = projected_gradient_descent(model, images, labels, loss_fn, num_steps=num_steps, step_size=step_size,
                                               epsilon=test_eps, delta=torch.rand(images.shape, device=device),
                                               y_target=None)
            x_adv_targeted = projected_gradient_descent(model, images, labels, loss_fn, num_steps=num_steps,
                                                        step_size=step_size, epsilon=test_eps,
                                                        delta=torch.rand(images.shape, device=device),
                                                        y_target=second_prediction)
            with torch.no_grad():
                # TODO: show some examples
                # If this is the first time through this loop, let's show a picture
                if image_flag and correct_adv == 0:
                    show_image(x_adv[1], x_adv_targeted[1], second_prediction[1].data.item(), test_eps, num_steps, step_size)

                if CUDA:
                    x_adv = x_adv.cuda()
                    x_adv_targeted = x_adv_targeted.cuda()
                else:
                    x_adv = x_adv.cpu()
                    x_adv_targeted = x_adv_targeted.cpu()

                outputs_adv = model(x_adv)
                outputs_adv_targeted = model(x_adv_targeted)

                _, predicted_adv = torch.max(outputs_adv.data, 1)
                _, predicted_adv_targeted = torch.max(outputs_adv_targeted.data, 1)

                total += labels.size(0)
                if CUDA:
                    correct_adv += (predicted_adv.cpu() == labels.cpu()).sum().item()
                    correct_adv_targeted += (predicted_adv_targeted.cpu() == labels.cpu()).sum().item()
                else:
                    correct_adv += (predicted_adv == labels).sum().item()
                    correct_adv_targeted += (predicted_adv_targeted == labels).sum().item()
        Test_Accuracy_Adv.append(round(100 * correct_adv / total, 3));
        Test_Accuracy_Adv_Targeted.append(round(100 * correct_adv_targeted / total, 3))

    plot_testing_results(test_radii, Test_Accuracy_Adv, Test_Accuracy_Adv_Targeted)


def plot_training_results(epoch_number, training_loss_regular, training_loss_perturbed, training_accuracy_regular, training_accuracy_perturbed):

    '''
    ################## Plotting the Training Loss ##################
    '''
    plt.plot(epoch_number, training_loss_regular, label='Training_Loss_Reg')
    plt.plot(epoch_number, training_loss_perturbed, label='Training_Loss_Adv')

    # Add a legend
    plt.legend()

    # Add labels
    plt.xlabel("Epoch Number")
    plt.ylabel("Training Loss")

    # Show the plot
    plt.show()

    '''
    ################## Plotting the Training Accuracy ##################
    '''
    # Plot the data
    plt.plot(epoch_number, training_accuracy_regular, label='Training_Acc_Reg')
    plt.plot(epoch_number, training_accuracy_perturbed, label='Training_Acc_Adv')

    # Add a legend
    plt.legend()

    # Add labels
    plt.xlabel("Epoch Number")
    plt.ylabel("Training Accuracy")

    # Show the plot
    plt.show()


def plot_testing_results(test_radii, Test_Accuracy_Adv, Test_Accuracy_Adv_Targeted):
    '''
    ################## Plotting the Testing Accuracy ##################
    '''
    # Plot the data
    plt.plot(test_radii, Test_Accuracy_Adv, label='Test_Acc_Adv', marker='o')
    plt.plot(test_radii, Test_Accuracy_Adv_Targeted, label='Test_Acc_Adv_Targeted', marker='o')

    # Add a legend
    plt.legend()

    # Add labels
    plt.xlabel("Epsilon Values")
    plt.ylabel("Training Accuracy")

    # Show the plot
    plt.show()


def show_image(x_adv, x_adv_targeted, target_label, test_eps, num_steps, step_size):
    temp_x_adv = x_adv[0].cpu()
    temp_x_adv_targeted = x_adv_targeted[0].cpu()

    # create figure
    txt = f"test_eps={test_eps}, num_steps={num_steps}, step_size={step_size}"
    fig = plt.figure(figsize=(10, 7))
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)

    # setting values to rows and column variables
    rows = 1
    cols = 2

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, cols, 1)
    plt.imshow(temp_x_adv, cmap='gray')
    plt.axis('off')
    plt.title("Non-Targeted")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, cols, 2)
    plt.imshow(temp_x_adv_targeted, cmap='gray')
    plt.axis('off')
    plt.title(f"Targeted={target_label}")

    plt.show()
