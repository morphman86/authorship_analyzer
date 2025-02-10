import os
import torch

class ModelRepository:
    def __init__(self, model_dir="models"):
        """Initialize the model repository."""
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def save_model(self, model, model_name):
        """
        Saves the model to the model directory.
        Args:
            model: PyTorch model to be saved.
            model_name: The name for the model file.
        """
        if not model_name.endswith('.pth'):
            model_name += '.pth'
        model_path = os.path.join(self.model_dir, model_name)  # FIXED LINE
        model_path = os.path.normpath(model_path)
        torch.save(model.state_dict(), model_path)

    def load_model(self, model_class, model_name):
        """
        Loads the model from the model directory.
        Args:
            model_class: The class of the model to be loaded.
            model_name: The name of the model file to load.
        Returns:
            The loaded model.
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found in {self.model_dir}")

        model = model_class()  # Initialize the model class
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        print(f"Model {model_name} loaded from {model_path}")
        return model

    def save_embeddings(self, embeddings, embeddings_name):
        """
        Save precomputed embeddings to the directory.
        Args:
            embeddings: The embeddings to be saved.
            embeddings_name: The name for the embeddings file.
        """
        embeddings_path = os.path.join(self.model_dir, f"{embeddings_name}.pkl")
        torch.save(embeddings, embeddings_path)
        print(f"Embeddings saved to {embeddings_path}")

    def load_embeddings(self, embeddings_name):
        """
        Loads precomputed embeddings from the directory.
        Args:
            embeddings_name: The name of the embeddings file to load.
        Returns:
            The loaded embeddings.
        """
        embeddings_path = os.path.join(self.model_dir, f"{embeddings_name}.pkl")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings {embeddings_name} not found in {self.model_dir}")

        embeddings = torch.load(embeddings_path)
        print(f"Embeddings {embeddings_name} loaded from {embeddings_path}")
        return embeddings
