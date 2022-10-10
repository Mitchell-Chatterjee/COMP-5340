import torch
import torch.nn.functional as F

K = 3

x = torch.ones(K)  # input tensor
A, B, C = torch.rand(K, K, requires_grad=True), torch.rand(K, K, requires_grad=True), torch.rand(K, K, requires_grad=True)
y = torch.matmul(x, A)
v = torch.matmul(x, B)
u = torch.sigmoid(y)
z = torch.add(u, v)

w = torch.matmul(z, C)
empty = torch.zeros_like(w)

# Using MSE with zero gives us ||w||^2
loss = torch.nn.functional.mse_loss(w, empty)

loss.backward()

print(A.grad)
print(B.grad)
print(C.grad)