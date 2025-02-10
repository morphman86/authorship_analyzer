import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import download

# Download NLTK resources
download('punkt')
download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer()

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.lower()

    def tokenize(self, text):
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    def stem(self, tokens):
        return [self.stemmer.stem(word) for word in tokens]

    def preprocess(self, text):
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        tokens_without_stopwords = self.remove_stopwords(tokens)
        stemmed_tokens = self.stem(tokens_without_stopwords)
        return " ".join(stemmed_tokens)

    def fit_vectorizer(self, texts):
        """
        Fit the TF-IDF vectorizer on a corpus of texts.
        This should be called before using get_text_embedding().
        """
        processed_texts = [self.preprocess(text) for text in texts]
        self.vectorizer.fit(processed_texts)

    def get_text_embedding(self, text):
        """
        Transform text using the pre-fitted TF-IDF vectorizer.
        Returns a dense vector representation.
        """
        processed_text = self.preprocess(text)
        tfidf_matrix = self.vectorizer.transform([processed_text])  # Use transform, NOT fit_transform
        return tfidf_matrix.toarray().flatten()
