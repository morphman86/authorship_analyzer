import re
import numpy as np
import nltk
import warnings
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure required resources are available
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class FeatureExtractor:
    def __init__(self, corpus=None):
        """Initialize feature extractor with TF-IDF vectorizer"""
        self.vectorizer = TfidfVectorizer(max_features=100)  # Adjust as needed
        self.vector_size = 110  # Fixed-size feature vector (10 lexical + syntactic + 100 TF-IDF)

        if corpus:
            self.vectorizer.fit(corpus)  # Fit on the provided corpus to avoid retraining on each text

    def preprocess_text(self, text):
        """Tokenize, lowercase, and remove punctuation"""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        return word_tokenize(text)

    def extract_lexical_features(self, text):
        """Extract lexical features: average word length, unique word ratio, etc."""
        words = self.preprocess_text(text)
        num_words = len(words)
        num_sentences = len(sent_tokenize(text))

        if num_words == 0:  # Avoid division by zero
            return {"avg_word_length": 0, "unique_word_ratio": 0, "avg_sentence_length": 0}

        avg_word_length = sum(len(word) for word in words) / num_words
        unique_word_ratio = len(set(words)) / num_words
        avg_sentence_length = num_words / max(num_sentences, 1)  # Avoid division by zero

        return {
            "avg_word_length": avg_word_length,
            "unique_word_ratio": unique_word_ratio,
            "avg_sentence_length": avg_sentence_length
        }

    def extract_syntactic_features(self, text):
        """Extract syntactic features: punctuation count, part-of-speech distribution"""
        punctuation_count = sum(1 for char in text if char in ".,;:?!")
        words = self.preprocess_text(text)
        pos_tags = nltk.pos_tag(words)  # Use nltk.pos_tag for POS tagging
        pos_counts = Counter(tag for _, tag in pos_tags)

        # Normalize POS counts
        total_words = max(len(words), 1)  # Avoid division by zero
        pos_distribution = {tag: count / total_words for tag, count in pos_counts.items()}

        return {
            "punctuation_count": punctuation_count,
            "pos_distribution": pos_distribution
        }

    def get_text_embedding(self, text):
        """Generate TF-IDF embedding for the text"""
        # Check if the text is empty
        if not text.strip():  # .strip() to avoid issues with spaces
            warnings.warn("Empty input text received. Returning zero vector.", UserWarning)
            return np.zeros((1, self.vectorizer.get_feature_names_out().shape[0]))  # Empty vector
        
        # Transform using the pre-fitted vectorizer
        vectorized_text = self.vectorizer.transform([text]).toarray()
        
        # If the resulting vector has fewer features than expected, pad with zeros
        if vectorized_text.shape[1] < 100:
            vectorized_text = np.pad(vectorized_text, ((0, 0), (0, 100 - vectorized_text.shape[1])), mode='constant')
        
        return vectorized_text


    def get_feature_vector(self, text):
        """Combine all extracted features into a single vector"""
        lexical_features = self.extract_lexical_features(text)
        syntactic_features = self.extract_syntactic_features(text)
        text_embedding = self.get_text_embedding(text)

        # Lexical features (3)
        lexical_vector = [
            lexical_features.get("avg_word_length", 0),
            lexical_features.get("unique_word_ratio", 0),
            lexical_features.get("avg_sentence_length", 0)
        ]

        # Syntactic features (1 + 6 POS tags)
        syntactic_vector = [
            syntactic_features.get("punctuation_count", 0)
        ]
        
        # Ensure the POS vector has 6 features, pad if necessary
        pos_vector = np.zeros(6)
        pos_tags = ["NN", "VB", "JJ", "RB", "IN", "DT"]
        for i, tag in enumerate(pos_tags):
            pos_vector[i] = syntactic_features["pos_distribution"].get(tag, 0)
        
        # Add POS features to syntactic vector
        syntactic_vector.extend(pos_vector)

        # TF-IDF embedding (100)
        embedding = text_embedding.flatten() if text_embedding.size > 0 else np.zeros(100)

        # Concatenate all features
        final_vector = np.concatenate([lexical_vector, syntactic_vector, embedding])

        return final_vector
