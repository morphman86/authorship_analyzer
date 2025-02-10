# app/utils/config.py

class Config:
    def __init__(self):
        # Hyperparameters for model training
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 10
        self.save_every = 2
        
        # Paths for saving and loading models
        self.model_save_path = "models/"
        self.vectorizer_save_path = "models/tfidf_vectorizer.pkl"
        
        # Preprocessing settings
        self.max_seq_length = 512  # Max sequence length for text input
        self.embedding_dim = 300   # Dimensionality of word embeddings (e.g., Word2Vec, GloVe)
        
        # Example for BERT-related settings (if using BERT embeddings)
        self.bert_model_name = "bert-base-uncased"
        self.bert_tokenizer_name = "bert-base-uncased"
        
        # Directory settings for raw and processed data
        self.raw_data_dir = "data/raw/"
        self.processed_data_dir = "data/processed/"
        self.embeddings_dir = "data/embeddings/"
        
        # Model configuration (e.g., for Siamese Network)
        self.margin = 0.2  # Margin for the Siamese network loss function
        self.l2_reg = 0.0001  # L2 regularization term

    def __repr__(self):
        return f"Config(learning_rate={self.learning_rate}, batch_size={self.batch_size}, " \
               f"epochs={self.epochs}, model_save_path='{self.model_save_path}')"
