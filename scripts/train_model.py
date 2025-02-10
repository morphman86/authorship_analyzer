import os
import sys

import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.repositories.dataset_loader import load_data
from app.services.model_training import ModelTraining
from app.utils.config import Config
from app.utils.siamese_network import SiameseNetwork
import torch
import pickle
from transformers import BertTokenizer

def load_bert_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def create_pairs(texts, authors):
    pairs = []
    labels = []
    
    # Create pairs of texts from the same author and different authors
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            pairs.append((texts[i], texts[j]))  # Store the raw text pairs
            if authors[i] == authors[j]:
                labels.append(1)  # Same author
            else:
                labels.append(0)  # Different authors
    return pairs, labels

def tokenize_batch(texts, tokenizer, max_length=512):
    # Tokenize the batch and create attention masks
    encoding = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return encoding['input_ids'], encoding['attention_mask']

def load_data(config):
    # Assuming `config` contains the file path or other necessary info
    # Read the CSV file using pandas (or any other method)
    df = pd.read_csv(config.input_csv)  # Adjust according to your data format
    
    # Extract the texts and authors from the DataFrame
    texts = df['text'].tolist()  # 'text' column contains the text data
    authors = df['author'].tolist()  # 'author' column contains the author names

    return texts, authors

def main():
    config = Config()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Initialize BERT tokenizer
    
    # Load text data and authors
    texts, authors = load_data(config)
    
    # Load precomputed BERT embeddings
    embeddings = load_bert_embeddings(config.output_embeddings_path)
    
    # Create pairs of text embeddings and corresponding labels
    pairs, labels = create_pairs(texts, authors)
    
    # Tokenize the raw text pairs
    pair_texts_1 = [pair[0] for pair in pairs]
    pair_texts_2 = [pair[1] for pair in pairs]
    
    # Ensure both are lists of strings
    assert isinstance(pair_texts_1, list) and all(isinstance(text, str) for text in pair_texts_1), "pair_texts_1 must be a list of strings."
    assert isinstance(pair_texts_2, list) and all(isinstance(text, str) for text in pair_texts_2), "pair_texts_2 must be a list of strings."
    
    input_ids1, attention_mask1 = tokenize_batch(pair_texts_1, tokenizer)
    input_ids2, attention_mask2 = tokenize_batch(pair_texts_2, tokenizer)
    
    # Convert labels to tensor
    train_labels = torch.tensor(labels)  # Convert labels to tensor
    
    # Initialize model and training class
    model = SiameseNetwork(hidden_dim=128)
    model_training = ModelTraining(model, config)
    
    # Train the model using embeddings
    model_training.train(input_ids1, input_ids2, attention_mask1, attention_mask2, train_labels, epochs=config.epochs)


if __name__ == "__main__":
    main()
