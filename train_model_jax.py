import sys
import os
import numpy as np
import jax
import jax.example_libraries.optimizers
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def matmul(a,b,c):
    """
    Calculate a*b+c
    """
    if c is None:
        c = jax.numpy.zeros(1)

    return jax.numpy.dot(a,b)+c

def run(x, A, B):
    """
    Run the neural network forward on the input x using the matrix A,B.
    """
    for i, (op, a, b) in enumerate(zip(ops, A, B)):
        x = op(x, a, b)
        if i < len(sizes)-2:
            x = x*(x>0)

    return x

def loss(params, inputs, targets):
    """
    Calculate the mean loss for optimization.
    """
    logits = run(inputs, params[0], params[1])
    res = (targets-logits.flatten())**2
    return jax.numpy.mean(res)

# generate a random number from the seed
if len(sys.argv) > 2:
    seed = int(sys.argv[2])
else:
    seed = 211
np.random.seed(seed)

# parameter list
sizes = list(map(int,sys.argv[1].split("-")))
# parameter tuple
dimensions = [tuple([x]) for x in sizes]
# parameter list
neuron_count = sizes
# function list
ops = [matmul]*(len(sizes)-1)

# weights list
A = []
# bias list
B = []
# a = dimension[i-1] = row number = dimension of input, b = dimension[i] = column number = dimension of output
# Initialize with a standard gaussian initialization.
for i, (op, a, b) in enumerate(zip(ops, sizes, sizes[1:])):
    A.append(np.random.normal(size=(a,b))/(b**.5))
    B.append(np.zeros((b,)))
params = [A,B]

# generate random training data
SAMPLES = 20
X = np.random.normal(size=(SAMPLES, sizes[0]))
Y = np.array(np.random.normal(size=SAMPLES)>0,dtype=np.float32)

# optimize the parameters
optimizer_init_function, optimzier_update_function, get_params_function = jax.example_libraries.optimizers.adam(step_size = 3e-4)
loss_grad = jax.grad(loss)

@jax.jit
def update(i, opt_state, batch_x, batch_y):
    params = get_params_function(opt_state)
    return optimzier_update_function(i, loss_grad(params, batch_x, batch_y), opt_state)
opt_state = optimizer_init_function(params)

# Train loop.
step = 0
BS = 4
for i in range(100):
    if i%10 == 0:
        print('loss', loss(params, X, Y))

    for j in range(0,SAMPLES,BS):
        batch_x = X[j:j+BS]
        batch_y = Y[j:j+BS]

        # gradient descent!
        opt_state = update(step, opt_state, batch_x, batch_y)
        params = get_params_function(opt_state)

        step += 1
        
# Save the model.
np.save("models/" + str(seed) + "_A1_" + "-".join(map(str,sizes)), params[0][0])
np.save("models/" + str(seed) + "_A2_" + "-".join(map(str,sizes)), params[1][0])
np.save("models/" + str(seed) + "_B1_" + "-".join(map(str,sizes)), params[0][1])
np.save("models/" + str(seed) + "_B2_" + "-".join(map(str,sizes)), params[1][1])

# A1 = onp.load("models/" + str(seed) + "_A1_" + sys.argv[1] + ".npy")