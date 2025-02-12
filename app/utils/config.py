# app/utils/config.py

class Config:
    def __init__(self):
        # Hyperparameters for model training
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 2
        self.save_every = 1

        # Paths for saving and loading models
        self.model_save_path = "models/"
        self.vectorizer_save_path = "models/tfidf_vectorizer.pkl"

        # Preprocessing settings
        self.max_seq_length = 512  # Max sequence length for text input
        self.embedding_dim = 300   # Dimensionality of word embeddings (e.g., Word2Vec, GloVe)

        # BERT Model Settings
        self.bert_model_name = "bert-base-uncased-siamese.npy"
        self.bert_base_model_name = "bert-base-uncased"
        self.checkpoint_model_name = "checkpoint_epoch_1.pth"
        self.bert_tokenizer_name = "bert-base-uncased"
        self.bert_class_name = "bert-base-uncased-siamese"

        # Directory settings for raw and processed data
        self.raw_data_dir = "data/raw/"
        self.processed_data_dir = "data/processed/"
        self.embeddings_dir = "data/embeddings/"

        # File paths for dataset and embeddings
        self.input_csv = f"{self.raw_data_dir}dataset.csv"
        self.text_column = "text"  # Change this if needed
        self.output_embeddings_path = f"{self.embeddings_dir}{self.bert_model_name}"
        self.bert_embeddings_path = f"{self.embeddings_dir}{self.bert_model_name}"
        self.trained_model_path = f"{self.model_save_path}{self.checkpoint_model_name}"

        # Model configuration (e.g., for Siamese Network)
        self.margin = 0.2  # Margin for the Siamese network loss function
        self.l2_reg = 0.0001  # L2 regularization term
        self.threshold = 0.5  # Threshold for classification decision
        self.device = "cuda"

    def __repr__(self):
        return f"Config(learning_rate={self.learning_rate}, batch_size={self.batch_size}, " \
               f"epochs={self.epochs}, model_save_path='{self.model_save_path}')"
