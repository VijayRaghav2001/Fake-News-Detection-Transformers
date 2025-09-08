import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time
import os

class ModelTrainer:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.train_config = config['training']
        
    def create_data_loader(self, encodings, labels, batch_size, shuffle=True):
        """Create PyTorch DataLoader from tokenized data"""
        dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(labels)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train(self, train_loader, val_loader=None):
        """Train the model"""
        # Set up optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config['learning_rate'],
            weight_decay=self.train_config['weight_decay']
        )
        
        total_steps = len(train_loader) * self.train_config['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.train_config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(self.train_config['epochs']):
            print(f"Epoch {epoch+1}/{self.train_config['epochs']}")
            
            # Training phase
            self.model.train()
            total_loss = 0
            predictions, true_labels = [], []
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                loss, logits = self.model(input_ids, attention_mask, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                
                # Store predictions and labels for metrics
                preds = torch.argmax(logits, dim=1).flatten()
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
            
            # Calculate training metrics
            train_loss = total_loss / len(train_loader)
            train_acc = accuracy_score(true_labels, predictions)
            train_f1 = f1_score(true_labels, predictions, average='weighted')
            
            print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
            
            # Validation phase
            if val_loader:
                val_loss, val_acc, val_f1 = self.evaluate(val_loader)
                print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Save model checkpoint
            if self.train_config['save_strategy'] == 'epoch':
                checkpoint_path = os.path.join(
                    self.config['paths']['checkpoints'], 
                    f"checkpoint_epoch_{epoch+1}.pt"
                )
                self.model.save_model(checkpoint_path)
        
        return self.model
    
    def evaluate(self, data_loader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                loss, logits = self.model(input_ids, attention_mask, labels)
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1).flatten()
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def predict(self, data_loader):
        """Make predictions"""
        self.model.eval()
        predictions, probabilities = [], []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids, attention_mask, _ = [b.to(self.device) for b in batch]
                _, logits = self.model(input_ids, attention_mask)
                
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).flatten()
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return predictions, probabilities