import sys
import os
import torch
from transformers import BertModel, BertTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.utils.siamese_network import SiameseNetwork  # Import your SiameseNetwork model

checkpoint_path = "models/checkpoint_epoch_10.pth"
model_name = "bert-base-uncased"  # Must match the model used during training

# Ensure checkpoint exists
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' does not exist.")

print("Starting model loading process...")

# Step 1: Initialize the model architecture (matching training setup)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading BERT model architecture: {model_name}")
model = SiameseNetwork(model_name=model_name).to(device)   # Match your Siamese architecture
tokenizer = BertTokenizer.from_pretrained(model_name)

# Step 2: Load the checkpoint
print(f"Attempting to load checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Step 3: Load state dict with potential mismatches handled
missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

# Step 4: Debug missing/unexpected keys
if missing_keys:
    print(f"Warning: The following keys were missing from the checkpoint and not loaded:\n{missing_keys}")
if unexpected_keys:
    print(f"Warning: The following keys were unexpected and not used:\n{unexpected_keys}")

print("Checkpoint loaded successfully!")

# Example test to make sure the model is working:
# Test with a pair of sentences
text1 = "This is the first text."
text2 = "This is the second text."

# Tokenize and create attention masks
inputs1 = tokenizer(text1, padding=True, truncation=True, return_tensors="pt")
inputs2 = tokenizer(text2, padding=True, truncation=True, return_tensors="pt")

# Move inputs to device
inputs1 = {key: val.to(device) for key, val in inputs1.items()}
inputs2 = {key: val.to(device) for key, val in inputs2.items()}

# Forward pass with the loaded model
model.eval()
with torch.no_grad():
    output1, output2 = model(inputs1['input_ids'], inputs1['attention_mask'], inputs2['input_ids'], inputs2['attention_mask'])
    print(f"Output for text1: {output1}")
    print(f"Output for text2: {output2}")
