import torch
from app.utils.config import Config
from app.services.text_preprocessor import TextPreprocessor
from app.repositories.model_repository import ModelRepository

class SimilarityChecker:
    def __init__(self, model, config: Config):
        """
        Initialize the SimilarityChecker with the pre-trained model and configuration.
        :param model: The pre-trained model for text similarity.
        :param config: Configuration object containing model-related settings.
        """
        self.model = model
        self.config = config
        self.text_preprocessor = TextPreprocessor()

    def check_similarity(self, text1: str, text2: str) -> float:
        """
        Check the similarity between two texts.
        :param text1: The first text sample.
        :param text2: The second text sample.
        :return: Similarity score between 0 and 1.
        """
        # Preprocess the texts
        preprocessed_text1 = self.text_preprocessor.preprocess(text1)
        preprocessed_text2 = self.text_preprocessor.preprocess(text2)

        self.text_preprocessor.fit_vectorizer([preprocessed_text1, preprocessed_text2])
        
        # Get the embeddings for both texts
        embedding1 = self.text_preprocessor.get_text_embedding(preprocessed_text1)
        embedding2 = self.text_preprocessor.get_text_embedding(preprocessed_text2)
        
        # Ensure both embeddings are of the same size (e.g., using padding/truncation or averaging)
        embedding1_tensor = self._adjust_embedding_size(torch.tensor(embedding1))
        embedding2_tensor = self._adjust_embedding_size(torch.tensor(embedding2))
        
        # Calculate the similarity score using cosine similarity
        similarity_score = self._cosine_similarity(embedding1_tensor, embedding2_tensor)
        
        return similarity_score

    def _adjust_embedding_size(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Adjust the embedding size to ensure consistency across all embeddings.
        This could involve padding or truncating the embedding to a fixed size.
        """
        desired_size = 512  # Adjust to the desired size (based on your model's output size)
        
        if embedding.size(0) > desired_size:
            # Truncate if the embedding is too large
            embedding = embedding[:desired_size]
        elif embedding.size(0) < desired_size:
            # Pad if the embedding is too small (with zeros or a similar value)
            padding = torch.zeros(desired_size - embedding.size(0))
            embedding = torch.cat((embedding, padding))
        
        return embedding

    def _cosine_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Calculate cosine similarity between two embeddings.
        :param embedding1: The first text embedding.
        :param embedding2: The second text embedding.
        :return: Cosine similarity score between 0 and 1.
        """
        # Normalize embeddings to unit vectors to ensure cosine similarity is correct
        embedding1 = torch.nn.functional.normalize(embedding1, p=2, dim=0)
        embedding2 = torch.nn.functional.normalize(embedding2, p=2, dim=0)

        cosine_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)

        return max(0.0, min(1.0, cosine_sim.item()))
