import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from custom_swish import Swish

# Define a small neural network using the custom Swish operation
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.swish = Swish() # Using our custom Swish layer
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.swish(x)
        x = self.fc2(x)
        return x

def train():
    # Hyperparameters
    input_size = 10
    hidden_size = 20
    output_size = 1
    learning_rate = 0.01
    num_epochs = 100

    # Dummy data
    inputs = torch.randn(100, input_size)
    targets = torch.randn(100, output_size)

    # Model, Loss, Optimizer
    model = SimpleNet(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print("Starting training with Custom Swish Activation...")
    print("-" * 40)
    
    loss_history = []

    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("-" * 40)
    print("Training complete.")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), loss_history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs using Custom Swish')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig('loss_curve.png')
    print("Loss curve saved as 'loss_curve.png'")
    # plt.show() # Uncomment to show plot if running in a GUI environment

if __name__ == "__main__":
    train()