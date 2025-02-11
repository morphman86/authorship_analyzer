import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import BertTokenizer
from app.repositories.model_repository import ModelRepository
from app.utils.config import Config

class ModelTraining:
    def __init__(self, model, config: Config):
        """
        Initialize the ModelTraining class with the model and configuration.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config

        # Optimizer & Mixed Precision
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def contrastive_loss(self, output1, output2, label, margin=1.0):
        """
        Contrastive loss function for Siamese networks.
        """
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = (1 - label) * euclidean_distance.pow(2) + label * torch.clamp(margin - euclidean_distance, min=0.0).pow(2)
        return loss.mean()

    def train(self, input_ids1, input_ids2, attention_mask1, attention_mask2, train_labels, epochs=10, patience=3):
        """
        Train the Siamese model with early stopping and AMP for faster computation.
        """
        print(f"Training on {len(train_labels)} samples for {epochs} epochs with patience {patience}.")
        
        # DataLoader with multiple workers
        train_dataset = TensorDataset(input_ids1, attention_mask1, input_ids2, attention_mask2, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        best_val_loss, patience_counter = float("inf"), 0

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
            for emb1_ids, emb1_mask, emb2_ids, emb2_mask, labels in progress_bar:
                emb1_ids, emb1_mask, emb2_ids, emb2_mask, labels = map(lambda x: x.to(self.device, non_blocking=True), 
                                                                        (emb1_ids, emb1_mask, emb2_ids, emb2_mask, labels))

                self.optimizer.zero_grad()
                
                # Automatic Mixed Precision (AMP)
                with torch.amp.autocast(enabled=torch.cuda.is_available()):
                    output1, output2 = self.model(emb1_ids, emb1_mask, emb2_ids, emb2_mask)
                    loss = self.contrastive_loss(output1, output2, labels, self.config.margin)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item()

                # Only update progress bar every 10 iterations to reduce overhead
                if progress_bar.n % 10 == 0:
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = running_loss / len(train_loader)
            val_loss = self.evaluate(train_loader)

            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if (epoch + 1) % self.config.save_every == 0:
                    self.save_model_checkpoint(epoch)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. No improvement in {patience} epochs.")
                break            

    def save_model_checkpoint(self, epoch):
        checkpoint_path = f"{self.config.model_save_path}/checkpoint_epoch_{epoch + 1}.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    def evaluate(self, data_loader):
        """
        Evaluate the model using contrastive loss.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for input_ids1, mask1, input_ids2, mask2, labels in data_loader:
                input_ids1, mask1, input_ids2, mask2, labels = map(lambda x: x.to(self.device, non_blocking=True),
                                                                    (input_ids1, mask1, input_ids2, mask2, labels))
                with torch.amp.autocast(enabled=torch.cuda.is_available()):
                    output1, output2 = self.model(input_ids1, mask1, input_ids2, mask2)
                    loss = self.contrastive_loss(output1, output2, labels, self.config.margin)

                total_loss += loss.item()

        return total_loss / len(data_loader)
