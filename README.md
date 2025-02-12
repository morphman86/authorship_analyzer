# Authorship Analyzer

**Authorship Analyzer** is a deep learning-based application designed for authorship verification. By analyzing textual features, the application determines if two texts were likely written by the same author. It utilizes state-of-the-art **Siamese Network** architecture with **BERT embeddings** for robust and accurate text similarity detection.

## Why Siamese Network with BERT Embeddings?

The choice of **Siamese Network** with **BERT embeddings** is driven by their ability to efficiently compare pairs of texts. The Siamese architecture is particularly well-suited for this task because it is designed to compare two inputs by using shared weights, which allows the model to learn whether they are similar or not. **BERT embeddings** capture nuanced, contextual information from the text, which improves the ability to detect subtle stylistic differences and similarities in writing. This combination allows the **Authorship Analyzer** to handle a variety of writing styles and detect whether different texts share the same author.

## Folder Structure

The project is organized into several directories and files to maintain modularity and clarity:

``` files
AuthorshipAnalyzer/
│── 📂 README.md                     # Project overview and usage instructions
│── 📂 requirements.txt              # Python dependencies for the project
│── 📂 app/
│   │── 📂 repositories/
│   │   └── model_repository.py          # Handles model saving and loading
│   │
│   │── 📂 services/
│   │   ├── model_prediction.py         # Service for making predictions on text similarity
│   │   ├── model_training.py           # Service for training the Siamese network model
│   │   ├── similarity_checker.py       # Utility for computing similarity between texts
│   │   └── text_preprocessor.py        # Utility for preprocessing text data
│   │
│   │── 📂 utils/
│   │   ├── config.py                   # Configuration settings for the project
│   │   ├── siamese_network.py          # Defines the Siamese Network architecture
│   │
│── 📂 scripts/
│   ├── generate_bert_embeddings.py     # Script to generate BERT embeddings for text data
│   ├── predict_similarity.py           # Script for predicting similarity between two texts
│   ├── test_checkpoint.py              # Script to test if a saved model checkpoint is valid
│   └── train_model.py                  # Script to train the Siamese network model
```

### **Key Directories and Files**

1. **`app/repositories/`**
   - **model_repository.py**: This file is responsible for handling the saving and loading of model checkpoints. It ensures that trained models can be persisted and restored for later use.

2. **`app/services/`**
   - **model_prediction.py**: Provides services for performing inference. This script is used to predict whether two texts are from the same author.
   - **model_training.py**: Handles the training of the Siamese network model on text data.
   - **similarity_checker.py**: A utility to compute the similarity between two text samples based on the trained model.
   - **text_preprocessor.py**: Contains utilities for preprocessing and tokenizing the text data before it is fed into the model.

3. **`app/utils/`**
   - **config.py**: Holds configuration parameters for the project, such as model names, batch sizes, and file paths.
   - **siamese_network.py**: Defines the architecture of the Siamese network, which is based on BERT embeddings and a fully connected layer for similarity comparison.

4. **`scripts/`**
   - **generate_bert_embeddings.py**: A script for generating BERT embeddings for a given set of text data.
   - **predict_similarity.py**: A script to input two texts and predict whether they were written by the same author based on a trained model.
   - **test_checkpoint.py**: Allows testing of model checkpoints to ensure they can be loaded correctly.
   - **train_model.py**: The script for training the Siamese network model on labeled text data, using precomputed embeddings.

## Installation and Setup

### Prerequisites

Ensure you have the following installed:

- Python 3.6+
- PyTorch (compatible version for your system)
- Transformers library (`pip install transformers`)
- tqdm (`pip install tqdm`)
- pandas (`pip install pandas`)

### Clone the Repository

```bash
git clone https://github.com/your-username/AuthorshipAnalyzer.git
cd AuthorshipAnalyzer
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generating BERT Embeddings

To generate BERT embeddings for your dataset (in CSV format), use the `generate_bert_embeddings.py` script. This will create embeddings that are later used for training or similarity prediction.

```bash
python scripts/generate_bert_embeddings.py
```

Make sure your input CSV file is structured with a `text` column containing the text samples.

### 2. Training the Model

To train the Siamese network model, use the `train_model.py` script. The training process requires labeled text pairs (same author or different authors).

```bash
python scripts/train_model.py
```

This script will train the Siamese network on the provided text data and save the model checkpoint.

### 3. Predicting Text Similarity

Once the model is trained, you can use the `predict_similarity.py` script to compare two texts and predict their similarity (whether they are from the same author).

```bash
python scripts/predict_similarity.py
```

The script will output a similarity score. A score closer to 1 indicates that the texts are more likely to be from the same author, while a score closer to 0 suggests they are from different authors.

### 4. Testing a Model Checkpoint

To check if your saved model checkpoint is valid and can be loaded properly, use the `test_checkpoint.py` script.

```bash
python scripts/test_checkpoint.py
```

### Configuration

Configuration settings can be found in the `app/utils/config.py` file. You can adjust parameters like the model name, batch size, and input/output file paths as needed.

```python
class Config:
    # Example configuration settings
    bert_base_model_name = "bert-base-uncased"
    max_seq_length = 512
    input_csv = "data/texts.csv"
    output_embeddings_path = "data/embeddings.npy"
    model_save_path = "models/saved_model.pth"
    epochs = 10
    batch_size = 32
```

## Contributing

If you would like to contribute to the project, feel free to fork the repository, make your changes, and submit a pull request. Make sure to follow the coding standards and add appropriate tests for your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
