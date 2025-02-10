# app/repositories/dataset_loader.py

import pickle
import pandas as pd

def load_data(config):
    # Load text data
    data = pd.read_csv(config.input_csv)
    texts = data[config.text_column].values
    authors = data['author'].values

    # Load BERT embeddings from the pickle file
    with open(config.bert_embeddings_path, 'rb') as f:
        bert_embeddings = pickle.load(f)
    
    # Ensure that the number of embeddings matches the number of texts
    assert len(bert_embeddings) == len(texts), "Mismatch between texts and embeddings count."
    
    # We assume bert_embeddings is a list of embeddings with the same order as the texts
    # The data (embeddings) is returned as pairs of texts and embeddings
    embeddings = bert_embeddings  # BERT embeddings are already in the correct order

    # Preprocess text data if necessary (e.g., tokenization), or directly return the embeddings
    return embeddings, authors
