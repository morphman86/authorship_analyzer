# app/utils/config.py

class Config:
    def __init__(self):
        # ============================
        # Model Training Hyperparameters
        # ============================
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 10
        self.save_every = 2
        self.margin = 0.2
        self.l2_reg = 0.0001
        self.threshold = 0.5
        self.data_loss_patience = 3

        # ============================
        # Model Training Preprocessing Settings
        # ============================
        self.max_seq_length = 512
        self.embedding_dim = 300
        self.text_column = "text"

        # ============================
        # File Names
        # ============================
        self.bert_model_name = "bert-base-uncased-siamese.npy"
        self.bert_base_model_name = "bert-base-uncased"
        self.checkpoint_model_name = "checkpoint_epoch_1.pth"
        self.bert_tokenizer_name = "bert-base-uncased"
        self.bert_class_name = "bert-base-uncased-siamese"
        self.input_csv_name = "dataset.csv"

        # ============================
        # File Save Paths
        # ============================
        self.model_save_path = "models/"
        self.raw_data_dir = "data/raw/"
        self.processed_data_dir = "data/processed/"
        self.embeddings_dir = "data/embeddings/"

        # ============================
        # Dataset & Embeddings Paths
        # ============================
        self.input_csv = f"{self.raw_data_dir}{self.input_csv_name}"
        self.output_embeddings_path = f"{self.embeddings_dir}{self.bert_model_name}"
        self.bert_embeddings_path = f"{self.embeddings_dir}{self.bert_model_name}"
        self.trained_model_path = f"{self.model_save_path}{self.checkpoint_model_name}"

        # ============================
        # Device Configuration
        # ============================
        self.device = "cuda"  # Device for model training (e.g., 'cpu' or 'cuda')
        
    def __repr__(self):
        return f"Config(learning_rate={self.learning_rate}, batch_size={self.batch_size}, " \
               f"epochs={self.epochs}, model_save_path='{self.model_save_path}')"
