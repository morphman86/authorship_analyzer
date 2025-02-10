import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.utils.config import Config
from app.services.model_prediction import ModelPrediction  # Importing model inference service

def load_text_data(file_path, text_column):
    """
    Load text data from a CSV file.
    :param file_path: Path to the CSV file.
    :param text_column: The column in the CSV that contains the text data.
    :return: A list of texts.
    """
    import pandas as pd
    df = pd.read_csv(file_path)
    return df[text_column].tolist()

def main():
    config = Config()
    
    # Load trained model (for inference)
    print(f"Loading trained model {config.checkpoint_model_name} from {config.model_save_path}")
    model_prediction = ModelPrediction(config)
    
    # User input for the two texts to compare
    text_1 = input("Enter first text: ")
    text_2 = input("Enter second text: ")
    
    # Predict similarity between the two texts
    print("Predicting similarity...")
    similarity_score = model_prediction.predict_similarity(text_1, text_2)
    
    # Output similarity score (probability of having same author)
    print(f"Similarity score: {similarity_score:.4f}")
    if similarity_score > 0.5:
        print("The texts are likely from the same author.")
    else:
        print("The texts are likely from different authors.")

if __name__ == "__main__":
    main()
