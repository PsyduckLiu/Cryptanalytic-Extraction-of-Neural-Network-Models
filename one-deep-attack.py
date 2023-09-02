import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple neural network class
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim,bias=True)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x

# get parameter list
sizes = list(map(int,sys.argv[1].split("-")))

# Hyperparameters
input_dim = sizes[0]
hidden_dim = sizes[1]
output_dim = sizes[2]

# Create an instance of the model
loaded_model = SimpleNN(input_dim, hidden_dim, output_dim)

# Load the saved state dictionary
loaded_model.load_state_dict(torch.load("models/trained_model_" + sys.argv[1] + ".pth"))
loaded_model.eval()  # Set the model to evaluation mode

print("Model loaded.")

# Attack Process
def binary_search(offset=None, direction=None, low=-1e3, high=1e3):
    """
    A slow but correct implementation of binary search to identify critical points.
    Just performs binary searcon from [low,high] splitting down the middle any time
    we determine it's not a linear function.

    In practice we perform binary search between the points
    (offset + direction * low)  ->  (offset + direction * high)
    """

    if offset is None:
        offset = torch.normal(mean=0.,std=0.01,size=(input_dim,))
    if direction is None:
        direction = torch.normal(mean=0.,std=0.01,size=(input_dim,))
    
    print("offset is:", offset)
    print("direction is:", direction)

    relus = []

    def search(low, high):
        mid = (low+high)/2

        with torch.no_grad():
            f_low = loaded_model(offset+direction*low)
            f_mid = loaded_model(offset+direction*mid)
            f_high = loaded_model(offset+direction*high)

        # if torch.abs(f_mid - (f_high + f_low)/2) < 1e-8*((high-low)**.5):
        if np.abs(f_mid - (f_high + f_low)/2)/(high-low) < 1e-8:
        # if np.abs(f_mid - (f_high + f_low)/2) < 1e-8:
            # In a linear region
            return
        elif high-low < 1e-6:
            # In a non-linear region and the interval is small enough
            if len(relus) == 0:
                relus.append(offset + direction*mid)
                print(low,high)
                print("f_low",f_low)
                print("f_mid",f_mid)
                print("f_high",f_high)
                print(torch.abs(f_mid - (f_high + f_low)/2))
                print(torch.abs(f_mid - (f_high + f_low)/2)/(high-low))
                print(torch.abs(f_mid - (f_high + f_low)/2)/((high-low)**.5))
            else:
                add_point = True
                for point in relus:
                    euclidean_distance = torch.norm(point - offset - direction*mid, p=2)
                
                    if euclidean_distance < 0.5:
                        add_point = False
                
                if add_point:
                    print("f_low",f_low)
                    print("f_mid",f_mid)
                    print("f_high",f_high)
                    print(np.abs(f_mid - (f_high + f_low)/2))
                    print(np.abs(f_mid - (f_high + f_low)/2)/(high-low))
                    print(torch.abs(f_mid - (f_high + f_low)/2)/((high-low)**.5))
                    relus.append(offset + direction*mid)
            return

        search(low, mid)
        # if len(relus) > 0:
            # we're done because we just want the left-most solution; don't do more searching
            # return
        search(mid, high)

    search(low, high)

    return relus

# critical_points = binary_search()
critical_points = binary_search(offset=torch.tensor([-0.0196, -0.0019,  0.0074, -0.0221],dtype = torch.float32), direction = torch.tensor([-0.0062, -0.0085,  0.0054,  0.0013],dtype = torch.float32), low=-1e3, high=1e3)
# critical_points = binary_search(offset=torch.tensor([ 0.0254, -0.0059,  0.0048,  0.0029,  0.0129, -0.0063]), direction = torch.tensor([-0.0012,  0.0153,  0.0280, -0.0050,  0.0057,  0.0084]))
# critical_points = [torch.tensor([-0.0619, -0.0925,  0.3997, -0.0861]),torch.tensor([-0.2873, -0.4128,  1.6692, -0.3590])]
print("critical point is:", critical_points)

# get fc1 from loaded_model
fc1 = loaded_model.fc1
fc2 = loaded_model.fc2
for point in critical_points:
    result = torch.matmul(fc1.weight, point) + fc1.bias
    print(result)

print(fc1.weight)
print(fc1.bias)
print(fc2.weight)
print(fc2.bias)

A1_hat = torch.ones(size=(hidden_dim, input_dim))
B1_hat = torch.zeros(hidden_dim)
epsilon = 0.01
i = 0
for point in critical_points:
    alpha = 0.0
    for j in range(input_dim):
        e_j = torch.zeros(input_dim)
        e_j[j] = 1.0

        # if i==1:
        #     alpha_plus = 0.0
        #     alpha_minus = loaded_model(point) - loaded_model(point-epsilon*e_j)
        # else:
        #     alpha_plus = loaded_model(point+epsilon*e_j) - loaded_model(point)
        #     alpha_minus = 0.0

        alpha_plus = loaded_model(point+epsilon*e_j) - loaded_model(point)
        alpha_minus = loaded_model(point) - loaded_model(point-epsilon*e_j)
    
        # if abs(alpha_plus)<abs(alpha_minus):
        #     alpha_plus = 0.0
        # else:
        #     alpha_minus = 0.0

        print(alpha_plus,alpha_minus,alpha_plus-alpha_minus,alpha)
        if j==0:
            alpha = alpha_plus - alpha_minus
        
        A1_hat[i,j] = (alpha_plus - alpha_minus) / alpha
        # print(alpha_plus-alpha_minus,alpha,A1_hat[i,j])
    B1_hat[i] = -torch.matmul(A1_hat[i], point)
    i += 1    

# B1_hat = B1_hat.t()
print(f'Recovered Weight Matrix is: {A1_hat}')
print(f'Recovered Bias Vector is: {B1_hat}')

# A1_hat = torch.tensor([[1.0000, -0.9615, 3.7258, 2.9035],[1.0000, -0.4286, -0.0203, 0.9974]])
A1_hat = torch.tensor([[0.1232, -0.1185,  0.4591,  0.3578],[0.4098, -0.1761, -0.0092,  0.4088]])
B1_hat = torch.tensor([-0.1550,  0.2071])
# for point in critical_points:
#     B1_hat = -torch.matmul(A1_hat, point)
#     print(torch.matmul(fc1.weight, point) + fc1.bias)
#     print(torch.matmul(A1_hat, point) + B1_hat)
print(f'Recovered Weight Matrix is: {A1_hat}')
print(f'Recovered Bias Vector is: {B1_hat}')



# Define a simple neural network class
class ZeroNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ZeroNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x

# Create an instance of the model
recovered_hidden_model = ZeroNN(input_dim, hidden_dim)
recovered_hidden_model.fc1.weight = nn.Parameter(A1_hat)
recovered_hidden_model.fc1.bias = nn.Parameter(B1_hat)

random_input_status = True
random_input = torch.rand(input_dim)
random_hidden = recovered_hidden_model(random_input)
for value in random_hidden:
    if value < 0.0:
        random_input_status = False
while not random_input_status:
    random_input_status = True
    random_input = torch.rand(input_dim)
    random_hidden = recovered_hidden_model(random_input)
    for value in random_hidden:
        if value < 0.0:
            random_input_status = False

random_output = loaded_model(random_input)
print(random_input)
print(random_hidden)
print(random_output)

# tensor([0.2503, 0.6028, 0.4449, 0.5687])
# tensor([2.2787, 0.8652], grad_fn=<ReluBackward0>)
# tensor([0.1984], grad_fn=<AddBackward0>)

A2_hat = torch.zeros(size=(output_dim, hidden_dim))
B2_hat = torch.zeros(output_dim)
for i in range(hidden_dim):
    with torch.no_grad():
        random_hidden_hat = random_hidden.clone()
        random_hidden_hat[i] += 1.0

        random_output_hat_input = torch.linalg.lstsq(A1_hat, (random_hidden_hat-B1_hat).t()).solution
        random_output_hat = loaded_model(random_output_hat_input.t())

        A2_hat[:,i] = (random_output_hat - random_output) / 1.0

B2_hat = random_output - torch.matmul(A2_hat, random_hidden)
print(f'Recovered Weight Matrix is: {A2_hat}')
print(f'Recovered Bias Vector is: {B2_hat}')

# Define a simple neural network class
class RecoveredNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RecoveredNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim,bias=True)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x

# Create an instance of the model
recovered_model = RecoveredNN(input_dim, hidden_dim, output_dim)
recovered_model.fc1.weight = nn.Parameter(A1_hat)
recovered_model.fc1.bias = nn.Parameter(B1_hat)
recovered_model.fc2.weight = nn.Parameter(A2_hat)
recovered_model.fc2.bias = nn.Parameter(B2_hat)

# Use the trained model for predictions
# test_input_tensor = torch.tensor([0.5652, 0.6091, 0.2224, 0.7096, 0.6958, 0.4827, 0.9137, 0.3462, 0.3952, 0.5003],dtype=torch.float32)
test_input_tensor = torch.tensor([0.5652, 0.6091, 0.2224, 0.7096],dtype=torch.float32)

# Make a prediction
with torch.no_grad():
    prediction = loaded_model(test_input_tensor)
    prediction_hat = recovered_model(test_input_tensor)

# Print the prediction result
print("Prediction result is:", prediction)
print("Recovered Prediction result is:", prediction_hat)

random_input = torch.rand(1, input_dim)
# Make a prediction
with torch.no_grad():
    prediction = loaded_model(random_input)
    prediction_hat = recovered_model(random_input)

# Print the prediction result
print("Prediction result is:", prediction)
print("Recovered Prediction result is:", prediction_hat)