# app/domain/text_sample.py

class TextSample:
    def __init__(self, text, embedding, author=None):
        self.text = text
        self.embedding = embedding
        self.author = author
