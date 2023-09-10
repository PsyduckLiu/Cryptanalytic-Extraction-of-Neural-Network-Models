import sys
import torch
import numpy as np

# get parameter list
sizes = list(map(int,sys.argv[1].split("-")))

# Hyperparameters
input_dim = sizes[0]
output_dim = sizes[1]

# generate a random weight matrix A
A = torch.rand(size=(output_dim, input_dim))
# generate a random bias vector B
B = torch.rand(output_dim)
# generate a random input vector x
x = torch.rand(input_dim, requires_grad=True)
print(f'Weight Matrix is: {A}')
print(f'Bias Vector is: {B}')

# compute the output of the network
y = torch.matmul(A,x) + B

# compute the gradient of the y with respect to the input
y.backward(torch.ones_like(y), retain_graph=True)
print(f'The sum of the column of weights is: {x.grad}')
x.grad.zero_()

# interate all basic grad_vectors, which correspond to the columns of the weight matrix, within a loop
# generate a gradient matrix G
G = torch.zeros(size=(output_dim, input_dim))

for i in range(output_dim):
    v = torch.zeros(output_dim, dtype=torch.float)
    v[i] = 1.0
    y.backward(v, retain_graph=True)
    G[i] = x.grad
    x.grad.zero_()

print(f'Gradient Matrix is: {G}')

# For an adversary only has some inputs and corresponding outputs
# He can also recover the weight matrix A and the bias vertor B
A_hat = torch.zeros(size=(output_dim, input_dim))
B_hat = torch.zeros(output_dim)
for i in range(input_dim):
    with torch.no_grad():
        x_hat = x.clone()
        x_hat[i] += 1.0
        print(x_hat)
        
        # get the output of the network
        y_hat = torch.matmul(A, x_hat) + B

        A_hat[:,i] = (y_hat - y) / 1.0

B_hat = y - torch.matmul(A_hat, x)
print(f'Recovered Weight Matrix is: {A_hat}')
print(f'Recovered Bias Vector is: {B_hat}')