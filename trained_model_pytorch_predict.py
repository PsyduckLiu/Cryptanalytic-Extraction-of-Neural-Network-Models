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

# Use the trained model for predictions
# random_input = torch.rand(1, input_dim)
test_input_tensor = torch.tensor([[0.5584, 0.2044, 0.2384, 0.6188]],dtype=torch.float32)

# Make a prediction
with torch.no_grad():
    prediction = loaded_model(test_input_tensor)

# Print the prediction result
print("Prediction result is:", prediction)