import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TransformerModel(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate=0.3):
        super(TransformerModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
        
        return loss, logits
    
    def save_model(self, path):
        """Save model to path"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path):
        """Load model from path"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        return self