import torch
import torch.nn as nn
from transformers import BertModel

class SiameseNetwork(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_dim=128):
        super(SiameseNetwork, self).__init__()
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name)
        self.bert_hidden_size = self.bert.config.hidden_size  # Typically 768 for BERT

        # Fully connected layers for similarity scoring
        self.fc1 = nn.Linear(self.bert_hidden_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward_once(self, input_ids, attention_mask):
        # Extract embeddings from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding

        # Pass through the network
        x = torch.relu(self.fc1(cls_embedding))
        x = torch.relu(self.fc2(x))
        return x

    def forward(self, input1, input2, mask1, mask2):
        # Process both inputs through BERT + Siamese layers
        output1 = self.forward_once(input1, mask1)
        output2 = self.forward_once(input2, mask2)

        return output1, output2
