import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from app.repositories.model_repository import ModelRepository
from app.utils.config import Config

class ModelTraining:
    def __init__(self, model, config: Config):
        """
        Initialize the ModelTraining class with the model and configuration.
        """
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Set up optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def contrastive_loss(self, output1, output2, label, margin=1.0):
        """
        Contrastive loss function for Siamese networks.
        """
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + 
                          (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
        return loss

    def tokenize_batch(self, texts):
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return encoded["input_ids"], encoded["attention_mask"]

    def train(self, input_ids1, input_ids2, attention_mask1, attention_mask2, train_labels, epochs: int = 10):
        """
        Train the Siamese model.
        :param input_ids1: Tensor of input IDs for the first text in the pair.
        :param input_ids2: Tensor of input IDs for the second text in the pair.
        :param attention_mask1: Tensor of attention masks for the first text in the pair.
        :param attention_mask2: Tensor of attention masks for the second text in the pair.
        :param train_labels: Tensor of labels (1 for similar, 0 for dissimilar).
        :param epochs: Number of training epochs.
        :return: None
        """
        # Convert to DataLoader
        train_dataset = TensorDataset(input_ids1, attention_mask1, input_ids2, attention_mask2, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()  # Set model to training mode
            running_loss = 0.0

            for emb1_ids, emb1_mask, emb2_ids, emb2_mask, labels in train_loader:
                emb1_ids, emb1_mask, emb2_ids, emb2_mask, labels = emb1_ids.to(self.device), emb1_mask.to(self.device), \
                                                                    emb2_ids.to(self.device), emb2_mask.to(self.device), \
                                                                    labels.to(self.device)

                # Forward pass
                output1, output2 = self.model(emb1_ids, emb1_mask, emb2_ids, emb2_mask)

                # Compute contrastive loss
                loss = self.contrastive_loss(output1, output2, labels)
                running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

            if (epoch + 1) % self.config.save_every == 0:
                self.save_model_checkpoint(epoch)

    def save_model_checkpoint(self, epoch):
        checkpoint_path = f"{self.config.model_save_path}/checkpoint_epoch_{epoch + 1}.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    def load_model_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        print(f"Model loaded from {checkpoint_path}")

    def evaluate(self, test_data, test_labels):
        """
        Evaluate the model.
        """
        self.model.eval()
        test_input_ids1, test_attention_mask1 = self.tokenize_batch(test_data[0])
        test_input_ids2, test_attention_mask2 = self.tokenize_batch(test_data[1])

        test_dataset = TensorDataset(test_input_ids1, test_attention_mask1, 
                                     test_input_ids2, test_attention_mask2, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)

        correct_preds, total_preds = 0, 0

        with torch.no_grad():
            for input_ids1, mask1, input_ids2, mask2, labels in test_loader:
                input_ids1, mask1 = input_ids1.to(self.device), mask1.to(self.device)
                input_ids2, mask2 = input_ids2.to(self.device), mask2.to(self.device)
                labels = labels.to(self.device)

                output1, output2 = self.model(input_ids1, input_ids2)

                euclidean_distance = nn.functional.pairwise_distance(output1, output2)
                
                predicted = (euclidean_distance < self.config.threshold).float()
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

        accuracy = correct_preds / total_preds * 100
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy
