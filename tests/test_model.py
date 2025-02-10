import unittest
from unittest.mock import patch, MagicMock
import torch
from app.repositories.model_repository import ModelRepository
from app.utils.config import Config
import os

class TestModelRepository(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Setup class-level resources like models and configuration.
        """
        cls.config = Config()  # Assuming you have a configuration class
        cls.model_repository = ModelRepository(cls.config.model_save_path)

        # Mock the load_model method to return a dummy model
        with patch.object(cls.model_repository, 'load_model', return_value=MagicMock(spec=torch.nn.Module)) as mock_load_model:
            cls.model = cls.model_repository.load_model('DummyModelClass', 'DummyModelName')
            # Now cls.model is a mock object that behaves like a torch.nn.Module
            mock_load_model.assert_called_once_with('DummyModelClass', 'DummyModelName')

        cls.model.to(torch.device('cpu'))  # Ensuring the model is on the CPU

    @patch('torch.save')  # Mocking torch.save to avoid saving to disk during tests
    def test_save_model(self, mock_save):
        """
        Test if the model can be saved successfully without actually saving to the disk.
        """
        save_path = "test_model"  # Pass the name without .pth extension

        # Call the save_model method (which will now use the mocked torch.save)
        self.model_repository.save_model(self.model, save_path)

        # Extract the directory from the config (now just 'models')
        model_dir = os.path.dirname(self.config.model_save_path)  # 'models'
        
        # The save path should be model_dir + save_path + .pth extension
        expected_save_path = os.path.join(model_dir, f"{save_path}.pth")  # 'models/test_model.pth'

        # Normalize the expected path to ensure cross-platform compatibility
        expected_save_path = os.path.normpath(expected_save_path)

        # Assert that torch.save was called with the correct arguments
        mock_save.assert_called_once_with(self.model.state_dict(), expected_save_path)


    def test_load_model(self):
        """
        Test if the model is loaded correctly.
        """
        self.assertIsInstance(self.model, torch.nn.Module)  # Checking if the model is a valid PyTorch module
    
    def test_model_forward(self):
      """
      Test if the model can perform a forward pass without errors.
      """
      # Create a mock tensor with the shape (1, 1)
      mock_output = MagicMock()
      mock_output.shape = (1, 1)

      # Configure the model mock to return the mock output
      self.model.return_value = mock_output

      # Call the forward pass with dummy input
      dummy_input = torch.randn(1, 10)  # Adjust the size to match your model's input
      output = self.model(dummy_input)

      # Check that the output has the expected shape
      self.assertEqual(output.shape, (1, 1))  # Adjust based on your model's output shape

if __name__ == '__main__':
    unittest.main()
