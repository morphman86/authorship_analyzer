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
        
        # Set up the criterion (loss function) and optimizer
        self.criterion = nn.CrossEntropyLoss()  # Adjust based on your model's output type
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def train(self, train_data: torch.Tensor, train_labels: torch.Tensor, epochs: int = 10):
        """
        Train the model.
        :param train_data: Tensor of training data.
        :param train_labels: Tensor of training labels.
        :param epochs: Number of training epochs.
        :return: None
        """
        # Convert to DataLoader
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            correct_preds = 0
            total_preds = 0

            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Optimize the weights
                self.optimizer.step()

                running_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct_preds / total_preds * 100
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

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
        :param test_data: Tensor of test data.
        :param test_labels: Tensor of test labels.
        :return: accuracy of the model on the test set.
        """
        self.model.eval()  # Set model to evaluation mode
        test_data, test_labels = test_data.to(self.device), test_labels.to(self.device)

        with torch.no_grad():
            outputs = self.model(test_data)
            _, predicted = torch.max(outputs, 1)
            correct_preds = (predicted == test_labels).sum().item()
            total_preds = test_labels.size(0)

        accuracy = correct_preds / total_preds * 100
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy
