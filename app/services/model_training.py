import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from app.repositories.model_repository import ModelRepository
from app.utils.config import Config

class ModelTraining:
    def __init__(self, model, config: Config):
        """
        Initialize the ModelTraining class with the model and configuration.
        :param model: The pre-trained or custom model to train.
        :param config: Configuration object containing training settings.
        """
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        self.model.to(self.device)

        # Set up the optimizer (Adam is a good choice for this task)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def contrastive_loss(self, output1, output2, label, margin=1.0):
        """
        Contrastive loss function for Siamese networks.
        :param output1: Output of the first input pair.
        :param output2: Output of the second input pair.
        :param label: True label (1 for similar, 0 for dissimilar).
        :param margin: Margin for dissimilar pairs.
        :return: Computed contrastive loss.
        """
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + 
                          (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
        return loss

    def train(self, train_data: torch.Tensor, train_labels: torch.Tensor, epochs: int = 10):
        """
        Train the Siamese model.
        :param train_data: Tensor of training data (pairs of text embeddings).
        :param train_labels: Tensor of labels (1 for similar, 0 for dissimilar).
        :param epochs: Number of training epochs.
        :return: None
        """
        # Convert to DataLoader
        train_dataset = TensorDataset(train_data[0], train_data[1], train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0

            for data1, data2, labels in train_loader:
                data1, data2, labels = data1.to(self.device), data2.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass: Get the similarity score
                output1, output2 = self.model(data1, data2)

                # Calculate loss: Compare similarity with labels
                loss = self.contrastive_loss(output1, output2, labels)   # Use similarity directly in the loss

                # Backward pass
                loss.backward()

                # Optimize the weights
                self.optimizer.step()

                running_loss += loss.item()

            # Print loss every epoch
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

            # Save model checkpoint every few epochs
            if (epoch + 1) % self.config.save_every == 0:
                self.save_model_checkpoint(epoch)

    def save_model_checkpoint(self, epoch: int):
        """
        Save the model checkpoint.
        :param epoch: Current epoch number.
        :return: None
        """
        checkpoint_path = f"{self.config.model_save_path}/checkpoint_epoch_{epoch + 1}.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    def evaluate(self, test_data: torch.Tensor, test_labels: torch.Tensor):
        """
        Evaluate the model on the test set.
        :param test_data: Tensor of test data (pairs of text embeddings).
        :param test_labels: Tensor of test labels (1 for similar, 0 for dissimilar).
        :return: accuracy of the model on the test set.
        """
        self.model.eval()  # Set model to evaluation mode
        test_data1, test_data2 = test_data[0].to(self.device), test_data[1].to(self.device)
        test_labels = test_labels.to(self.device)

        with torch.no_grad():
            # Forward pass
            output1, output2 = self.model(test_data1, test_data2)
            euclidean_distance = nn.functional.pairwise_distance(output1, output2)

            # Predict similar or dissimilar based on the distance
            predicted = (euclidean_distance < self.config.threshold).float()

            correct_preds = (predicted == test_labels).sum().item()
            total_preds = test_labels.size(0)

        accuracy = correct_preds / total_preds * 100
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy
