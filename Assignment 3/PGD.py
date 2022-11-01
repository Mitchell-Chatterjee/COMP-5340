import torch


def projected_gradient_descent(model, x, y, loss_fn, num_steps, step_size, epsilon,
                               clamp=(0, 1), y_target=None):

    # Step 1: Perturb the elements in the batch
    x_hat = x.clone().detach().requires_grad_(True).to(x.device)
    targeted = y_target is not None

    # This creates a tensor of the same size as x_hat with values (-epsilon, +epsilon)
    delta = (epsilon * (-2)) * torch.rand(x_hat.shape[0], x_hat.shape[1]) + epsilon
    x_hat = torch.add(x_hat, delta)

    for i in range(num_steps):
        # Step 2: Update x_hat
        x_old = x_hat.clone().detach().requires_grad_(True)

        prediction = model(x_hat)
        loss = loss_fn(prediction, y_target if targeted else y)
        loss.backward()

        with torch.no_grad():
            gradients = step_size * x_old.grad.sign()
        
        z = x_old - gradients if targeted else x_old + gradients
        
        # Step 3: Project z to the L-inf ball
        x_hat = torch.min(torch.max(z, x - epsilon), x + epsilon)

        x_hat.clamp(*clamp)

    return x_hat.detach()
