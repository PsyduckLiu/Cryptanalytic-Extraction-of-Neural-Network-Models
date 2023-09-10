import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# get parameter list
sizes = list(map(int,sys.argv[1].split("-")))

# Hyperparameters
input_dim = sizes[0]
hidden_dim = sizes[1]
output_dim = sizes[2]
learning_rate = 0.001
epochs = 200

# generate a random number from the seed
if len(sys.argv) > 2:
    seed = int(sys.argv[2])
else:
    seed = 42
np.random.seed(seed)

# Create a simple dataset
SAMPLES = 10000
X = np.random.rand(SAMPLES, sizes[0])
y = np.array(np.random.normal(size=(SAMPLES,1))>0,dtype=np.float32)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
print(X_tensor.shape)
print(y_tensor.shape)

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

# Initialize the model
model = SimpleNN(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()  # Clear gradients
    outputs = model(X_tensor)  # Forward pass
    loss = criterion(outputs, y_tensor)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("Training finished!")

# save the model
torch.save(model.state_dict(), "models/trained_model_" + sys.argv[1] + ".pth")
print("Model saved.")

# Print weight matrices
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f'Layer: {name}, Shape: {param.shape}')
        print(param)
    elif 'bias' in name:
        print(f'Bias Vector of Layer {name}:')
        print(param)

# Use the trained model for predictions
# Generate a random input tensor
random_input = torch.rand(1, input_dim)
print("Random input is:", random_input)

# test_input_tensor = torch.tensor([0.5652, 0.6091, 0.2224, 0.7096, 0.6958, 0.4827, 0.9137, 0.3462, 0.3952, 0.5003],dtype=torch.float32)
# test_input_tensor = torch.tensor([0.5652, 0.6091, 0.2224, 0.7096],dtype=torch.float32)

# Make a prediction
with torch.no_grad():
    prediction = model(random_input)

# Print the prediction result
print("Prediction result is:", prediction)
