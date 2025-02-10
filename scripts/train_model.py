# scripts/train_model.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.repositories.dataset_loader import load_data
from app.services.model_training import ModelTraining
from app.utils.config import Config
from app.utils.siamese_network import SiameseNetwork
import torch
import pickle

def load_bert_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def create_pairs(embeddings, authors):
    pairs = []
    labels = []
    
    # Assuming embeddings is a list and authors is a list of labels
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            pairs.append((embeddings[i], embeddings[j]))
            if authors[i] == authors[j]:
                labels.append(1)  # Same author
            else:
                labels.append(0)  # Different authors
    return pairs, labels

def main():
    config = Config()
    
    # Load BERT embeddings (Ensure they are in a compatible format for PyTorch)
    embeddings = load_bert_embeddings(config.bert_embeddings_path)
    
    # Load text data and authors (this should return texts and corresponding authors)
    texts, authors = load_data(config)
    
    # Create pairs of embeddings and corresponding labels
    pairs, labels = create_pairs(embeddings, authors)
    
    # Convert pairs and labels to PyTorch tensors
    pair_embeddings_1 = torch.stack([torch.tensor(pair[0]) for pair in pairs])  # Convert each embedding pair to tensor
    pair_embeddings_2 = torch.stack([torch.tensor(pair[1]) for pair in pairs])  # Convert each embedding pair to tensor
    # Remove the extra dimension (i.e., squeeze the tensor)
    pair_embeddings_1 = pair_embeddings_1.squeeze(1)  # Shape: [batch_size, embedding_size]
    pair_embeddings_2 = pair_embeddings_2.squeeze(1)  # Shape: [batch_size, embedding_size]

    train_labels = torch.tensor(labels)  # Convert labels to tensor
    
    # Initialize model and training class
    model = SiameseNetwork(input_dim=768, hidden_dim=128)
    model_training = ModelTraining(model, config)
    
    # Train the model
    model_training.train((pair_embeddings_1, pair_embeddings_2), train_labels, epochs=config.epochs)

if __name__ == "__main__":
    main()
