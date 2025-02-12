import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.model_training import ModelTraining
from app.utils.config import Config
from app.utils.siamese_network import SiameseNetwork

def load_bert_embeddings(file_path):
    """Load precomputed BERT embeddings efficiently."""
    return np.load(file_path)

def create_pairs(texts, authors):
    """Create positive and negative text pairs."""
    pairs, labels = [], []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            pairs.append((texts[i], texts[j]))
            labels.append(1 if authors[i] == authors[j] else 0)
    return pairs, torch.tensor(labels, dtype=torch.float32)  # Convert once to tensor

from torch.nn.utils.rnn import pad_sequence

def batch_tokenize(texts, tokenizer, batch_size=64, max_length=512):
    """Tokenize text in batches and ensure consistent tensor sizes."""
    input_ids, attention_masks = [], []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing", unit="batch"):
        batch = texts[i : i + batch_size]
        encodings = tokenizer(batch, padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
        
        input_ids.extend(encodings["input_ids"])
        attention_masks.extend(encodings["attention_mask"])
    
    # Pad all tensors to the longest sequence length in the batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return input_ids, attention_masks


def tokenize_pairs(pairs, tokenizer, batch_size=64, max_length=512):
    """Tokenize text pairs in batches."""
    texts1, texts2 = zip(*pairs)  # Unzip pairs
    input_ids1, attention_mask1 = batch_tokenize(list(texts1), tokenizer, batch_size, max_length)
    input_ids2, attention_mask2 = batch_tokenize(list(texts2), tokenizer, batch_size, max_length)
    return input_ids1, attention_mask1, input_ids2, attention_mask2

def load_data(file_path):
    """Load text and author data from CSV."""
    df = pd.read_csv(file_path, usecols=["text", "author"])  # Load only required columns
    return df['text'].tolist(), df['author'].tolist()

def main():
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.bert_base_model_name)

    print("Loading data...")
    texts, authors = load_data(config.input_csv)
    print(f"Loaded {len(texts)} texts. Generating pairs...")

    pairs, labels = create_pairs(texts, authors)
    print(f"Generated {len(pairs)} pairs. Tokenizing...")

    input_ids1, attention_mask1, input_ids2, attention_mask2 = tokenize_pairs(pairs, tokenizer)
    print("Tokenization complete. Initializing model...")

    model = SiameseNetwork(hidden_dim=128)
    model_training = ModelTraining(model, config)

    print(f"Training model for {config.epochs} epochs...")
    model_training.train(input_ids1, input_ids2, attention_mask1, attention_mask2, labels, epochs=config.epochs, patience=config.data_loss_patience)
    print("Training complete.")

if __name__ == "__main__":
    main()
