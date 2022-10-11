import torch
import torch.nn.functional as F

K = 3
LEARNING_RATE = 0.01

# Initialization
x = torch.ones(K)  # input tensor
A, B, C = torch.ones(K, K, requires_grad=True), torch.ones(K, K, requires_grad=True), torch.ones(K, K, requires_grad=True)

for step in range(30):
    y = torch.matmul(x, A)
    v = torch.matmul(x, B)
    u = torch.sigmoid(y)
    z = torch.add(u, v)

    w = torch.matmul(z, C)
    empty = torch.zeros_like(w)

    # Using MSE with zero gives us (1/(len(w)))*||w||^2
    loss = torch.nn.functional.mse_loss(w, empty, reduction='sum')
    loss.backward(retain_graph=True)
    print("Step:", step, " Loss:", loss.item())

    # Step backwards
    A = torch.tensor(torch.sub(A, LEARNING_RATE * A.grad), requires_grad=True)
    B = torch.tensor(torch.sub(B, LEARNING_RATE * B.grad), requires_grad=True)
    C = torch.tensor(torch.sub(C, LEARNING_RATE * C.grad), requires_grad=True)