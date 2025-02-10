import torch
import torch.nn as nn
import torch.optim as optim

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SiameseNetwork, self).__init__()
        # Define the layers for the sub-network (same for both inputs)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output similarity score (0 or 1)

    def forward_once(self, x):
        # Define the forward pass for one of the inputs
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, input1, input2):
        # Pass both inputs through the network
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # Calculate the absolute difference between the outputs
        distance = torch.abs(output1 - output2)
        similarity = torch.sigmoid(distance)
        return distance, similarity

# Initialize the model with the correct input_dim (768 for BERT embeddings)
model = SiameseNetwork(input_dim=768, hidden_dim=128)  # Update input_dim to 768
