import os
import sys
import torch
from app.services.model_training import ModelTraining  # Import ModelTraining class
from app.repositories.model_repository import ModelRepository  # Assuming model loading is handled here
from app.utils.config import Config
from transformers import BertTokenizer, BertModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.utils.siamese_network import SiameseNetwork  # Import your SiameseNetwork model

class ModelPrediction:
    def __init__(self, config: Config):
        """
        Initialize the ModelPrediction class with the configuration.
        :param config: Configuration object containing settings like model path, etc.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model and tokenizer from the configuration
        self.model = self.load_trained_model()

        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_tokenizer_name)
        self.model.to(self.device)
        self.model.eval()

    def model_class_mapping(self, class_name):
        """
        Return the model class based on the specified BERT class name.
        :param class_name: Name of the BERT class.
        :return: BERT model class.
        """
        classmap = {
            "bert-base-uncased-siamese": SiameseNetwork,  # Use the custom SiameseBERT model for comparison
            # Add other BERT models here if needed
            "bert-base-uncased": BertModel,
            # "bert-large-uncased": BertModel,
            # "bert-for-sequence-classification": BertForSequenceClassification,
        }
        return classmap.get(class_name)
      
    def load_trained_model(self):
        """
        Load the trained model from the given path.
        :return: Loaded model.
        """
        model_repo = ModelRepository(self.config.model_save_path)  # Pass the model save path from config
        model_class = self.model_class_mapping(self.config.bert_class_name)

        if not model_class:
            raise ValueError(f"Model class for {self.config.bert_class_name} not found.")
        
        model = model_class()  # Initialize the SiameseNetwork class

        # Load the model state dictionary from the checkpoint
        checkpoint_path = os.path.join(self.config.model_save_path, self.config.checkpoint_model_name)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device), strict=False)

        model.to(self.device)
        model.eval()

        return model


    def generate_bert_embeddings(self, text, max_length=512):
        """
        Generate BERT embeddings for the input text using the tokenizer.
        :param text: Input text to generate embeddings for.
        :param max_length: Maximum sequence length for tokenization.
        :return: BERT embeddings for the input text.
        """
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model.bert(**inputs)  # Use the model's BERT component to generate embeddings
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        return cls_embedding

    def predict_similarity(self, text_1, text_2):
        """
        Predict the similarity between two texts using the trained model.
        :param text_1: First text to compare.
        :param text_2: Second text to compare.
        :return: Similarity score (probability).
        """
        # Tokenize and generate BERT inputs (input_ids and attention_mask)
        inputs_1 = self.tokenizer(text_1, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        inputs_2 = self.tokenizer(text_2, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

        inputs_1 = {key: val.to(self.device) for key, val in inputs_1.items()}
        inputs_2 = {key: val.to(self.device) for key, val in inputs_2.items()}

        # Extract input_ids and attention_mask for both texts
        input_ids_1, attention_mask_1 = inputs_1['input_ids'], inputs_1['attention_mask']
        input_ids_2, attention_mask_2 = inputs_2['input_ids'], inputs_2['attention_mask']

        # Get similarity score using the Siamese model
        output1, output2 = self.model(input_ids_1, input_ids_2, attention_mask_1, attention_mask_2)

        # Calculate cosine similarity between the two output embeddings
        cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
        
        return cosine_similarity.item()  # Return similarity score as a scalar value
