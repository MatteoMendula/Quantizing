import torch
import torch.nn as nn
import torch.optim as optim

# Define two simple linear regression models
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # Single input feature and single output feature

    def forward(self, x):
        return self.linear(x)

# Create instances of the models
model1 = LinearRegression()
model2 = LinearRegression()

# Define your input and target tensors
input_tensor = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
target_tensor = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer (SGD with learning rate 0.01 and momentum 0.9)
optimizer = optim.SGD(
    [
        {'params': model1.parameters()},
        {'params': model2.parameters()}
    ],
    lr=0.01,
    momentum=0.9
)

# Training loop
for epoch in range(100):
    # Forward pass
    output_tensor1 = model1(input_tensor)
    output_tensor2 = model2(input_tensor)
    
    # Compute the loss
    loss1 = criterion(output_tensor1, target_tensor)
    loss2 = criterion(output_tensor2, target_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()  # Zero the gradients
    loss1.backward()  # Compute the gradients for model1
    loss2.backward()  # Compute the gradients for model2
    optimizer.step()  # Update the models' parameters
    
    # Print the losses for monitoring the training progress
    print(f"Epoch {epoch+1}, Loss 1: {loss1.item()}, Loss 2: {loss2.item()}")

# Test the trained models
test_input = torch.tensor([[5.0], [6.0]])
predicted_output1 = model1(test_input)
predicted_output2 = model2(test_input)
print(f"Predicted output from model 1: {predicted_output1}")
print(f"Predicted output from model 2: {predicted_output2}")
