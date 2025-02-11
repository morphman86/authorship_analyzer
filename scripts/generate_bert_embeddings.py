import numpy as np
import os
import pandas as pd
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.utils.config import Config

class TextDataset(Dataset):
    """Custom dataset to handle text data efficiently for batching."""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in inputs.items()}  # Remove batch dimension

def load_text_data(file_path, text_column):
    """Load the text data from CSV."""
    df = pd.read_csv(file_path)
    return df[text_column].tolist()

def generate_bert_embeddings(texts, model_name, tokenizer_name, max_length=512, batch_size=32):
    """Generate BERT embeddings for a list of texts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    model = BertModel.from_pretrained(model_name).to(device)
    model.eval()

    dataset = TextDataset(texts, tokenizer, max_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    with torch.no_grad():
        progress_bar = tqdm(data_loader, total=len(data_loader), leave=False)
        for batch in progress_bar:
            # Move batch to the correct device
            batch = {key: val.to(device) for key, val in batch.items()}
            
            # Get model outputs
            outputs = model(**batch)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Extract CLS token embedding
            
            embeddings.append(cls_embeddings)

    # Concatenate all batch embeddings into a single numpy array
    embeddings = np.vstack(embeddings)
    return embeddings

def save_embeddings(embeddings, output_path):
    """Save the generated embeddings to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists
    np.save(output_path, embeddings)  # Save embeddings as a binary file (more space-efficient)

def main():
    print("Current Working Directory:", os.getcwd())
    config = Config()
    print(f"Loading input data from {config.input_csv}")
    texts = load_text_data(config.input_csv, config.text_column)
    print(f"Generating embeddings, model: {config.bert_model_name}, length: {config.max_seq_length}")
    embeddings = generate_bert_embeddings(texts, config.bert_base_model_name, config.bert_tokenizer_name, config.max_seq_length)
    print(f"Saving embeddings: file {config.output_embeddings_path}")
    save_embeddings(embeddings, config.output_embeddings_path)
    print(f"BERT embeddings saved to {config.output_embeddings_path}")

if __name__ == "__main__":
    main()
