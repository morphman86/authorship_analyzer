import unittest
from unittest.mock import patch, MagicMock
from app.services.similarity_checker import SimilarityChecker
from app.utils.config import Config
from app.repositories.model_repository import ModelRepository
from app.services.text_preprocessor import TextPreprocessor
import torch

class TestSimilarityChecker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Setup class-level resources like model and similarity checker instance.
        """
        cls.config = Config()
        cls.model_repository = ModelRepository(cls.config.model_save_path)
        model_class = "torch.nn.Module"
        model_name = "TestModel"
        
         # Mock the load_model method to return a dummy model
        with patch.object(cls.model_repository, 'load_model', return_value=MagicMock()) as mock_load_model:
            cls.model = cls.model_repository.load_model('DummyModelClass', 'DummyModelName')
            # Now cls.model is a mock object instead of a real model.

            # You can also verify the mock call if needed:
            mock_load_model.assert_called_once_with('DummyModelClass', 'DummyModelName')

        cls.model.to(torch.device('cpu'))  # Ensure the model is on the CPU
        cls.similarity_checker = SimilarityChecker(cls.model, cls.config)

    def test_similarity_check(self):
        """
        Test the similarity checker between two texts.
        """
        text1 = "The cat sat on the mat."
        text2 = "A cat is sitting on a mat."
        
        similarity_score = self.similarity_checker.check_similarity(text1, text2)
        
        self.assertGreaterEqual(similarity_score, 0)  # Similarity score should be between 0 and 1
        self.assertLessEqual(similarity_score, 1)

    def test_similarity_identical_texts(self):
        """
        Test the similarity checker between identical texts.
        """
        text1 = "Hello world!"
        text2 = "Hello world!"
        
        similarity_score = self.similarity_checker.check_similarity(text1, text2)
        
        self.assertEqual(similarity_score, 1.0)  # Identical texts should have a similarity score of 1

    def test_similarity_non_matching_texts(self):
        """
        Test the similarity checker with completely different texts.
        """
        text1 = "The quick brown fox jumped over the lazy dog."
        text2 = "This is a completely different sentence."
        
        similarity_score = self.similarity_checker.check_similarity(text1, text2)
        
        self.assertLess(similarity_score, 0.5)  # Completely different texts should have low similarity

if __name__ == '__main__':
    unittest.main()
