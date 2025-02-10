import unittest
import numpy as np
from app.services.feature_extraction import FeatureExtractor

class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        """Initialize test data and FeatureExtractor instance before each test"""
        self.extractor = FeatureExtractor()
        self.sample_text_1 = "This is a simple sentence."
        self.sample_text_2 = "Another, slightly different, example text!"
        self.sample_text_3 = "A short one."

    ### === STANDARD TESTS === ###

    def test_preprocessing(self):
        """Test if text preprocessing (e.g., lowercasing, tokenization) works correctly."""
        processed_text = self.extractor.preprocess_text(self.sample_text_1)
        self.assertIsInstance(processed_text, list)  # Should return a list of words
        self.assertGreater(len(processed_text), 0)  # Ensure non-empty output
        self.assertNotIn(".", processed_text)  # Ensure punctuation is removed

    def test_lexical_features(self):
        """Test extraction of lexical features like sentence length, unique words, etc."""
        features = self.extractor.extract_lexical_features(self.sample_text_1)
        self.assertIsInstance(features, dict)
        self.assertIn("avg_word_length", features)
        self.assertIn("unique_word_ratio", features)
        self.assertGreater(features["avg_word_length"], 0)
        self.assertGreater(features["unique_word_ratio"], 0)

    def test_syntactic_features(self):
        """Test extraction of syntactic features like punctuation and POS distribution."""
        features = self.extractor.extract_syntactic_features(self.sample_text_2)
        self.assertIsInstance(features, dict)
        self.assertIn("punctuation_count", features)
        self.assertIn("pos_distribution", features)
        self.assertGreater(features["punctuation_count"], 0)  # Sentence has commas & exclamation mark

    def test_stylometric_embedding(self):
        """Test that text embeddings (e.g., TF-IDF, Word2Vec, BERT) are generated correctly."""
        embedding = self.extractor.get_text_embedding(self.sample_text_3)
        self.assertIsInstance(embedding, np.ndarray)  # Should return a NumPy array
        self.assertGreater(len(embedding), 0)  # Ensure embeddings are generated

    def test_feature_vector_consistency(self):
        """Test if the final feature vector has a consistent length across texts."""
        vector1 = self.extractor.get_feature_vector(self.sample_text_1)
        vector2 = self.extractor.get_feature_vector(self.sample_text_2)
        self.assertEqual(len(vector1), len(vector2))  # Feature vectors should have a fixed length

    ### === EDGE CASE TESTS === ###

    def test_empty_string(self):
        """Test how feature extraction handles an empty string."""
        empty_text = ""
        vector = self.extractor.get_feature_vector(empty_text)
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(len(vector), self.extractor.vector_size)  # Should still return a fixed-size vector

    def test_only_punctuation(self):
        """Test handling of a string containing only punctuation."""
        punct_text = "!.,;:?!"
        vector = self.extractor.get_feature_vector(punct_text)
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(len(vector), self.extractor.vector_size)

    def test_single_word(self):
        """Test handling of a single-word input."""
        single_word = "Hello"
        features = self.extractor.extract_lexical_features(single_word)
        self.assertIsInstance(features, dict)
        self.assertEqual(features["unique_word_ratio"], 1.0)  # Only one word means ratio = 1

    def test_repetitive_text(self):
        """Test how the extractor handles repetitive text."""
        repetitive_text = "hello hello hello hello hello"
        features = self.extractor.extract_lexical_features(repetitive_text)
        self.assertLess(features["unique_word_ratio"], 0.5)  # Should detect low lexical diversity

    def test_numeric_text(self):
        """Test handling of text with mostly numbers."""
        numeric_text = "1234 5678 91011 1213"
        vector = self.extractor.get_feature_vector(numeric_text)
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(len(vector), self.extractor.vector_size)

    def test_long_text(self):
        """Test processing of a very long text input."""
        long_text = " ".join(["word"] * 10_000)  # 10,000 words
        vector = self.extractor.get_feature_vector(long_text)
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(len(vector), self.extractor.vector_size)

if __name__ == "__main__":
    unittest.main()
