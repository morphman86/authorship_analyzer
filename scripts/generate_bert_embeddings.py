import sys
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import pickle
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.utils.config import Config

def load_text_data(file_path, text_column):
    df = pd.read_csv(file_path)
    return df[text_column].tolist()

def generate_bert_embeddings(texts, model_name, max_length=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)

    embeddings = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Extract CLS token embedding
            embeddings.append(cls_embedding)

    return embeddings

def save_embeddings(embeddings, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)

def main():
    print("Current Working Directory:", os.getcwd())
    config = Config()
    print(f"Loading input data from {config.input_csv}")
    texts = load_text_data(config.input_csv, config.text_column)
    print(f"Generating embeddings, model: {config.bert_model_name}, length: {config.max_seq_length}")
    embeddings = generate_bert_embeddings(texts, config.bert_model_name, config.max_seq_length)
    print(f"Saving embeddings: file {config.output_embeddings_path}")
    save_embeddings(embeddings, config.output_embeddings_path)
    print(f"BERT embeddings saved to {config.output_embeddings_path}")

if __name__ == "__main__":
    main()
