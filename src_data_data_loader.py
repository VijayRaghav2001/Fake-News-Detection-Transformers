import pandas as pd
from sklearn.model_selection import train_test_split
from . import config

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.text_column = config['data']['text_column']
        self.label_column = config['data']['label_column']
    
    def load_data(self, file_path):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            return df
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return None
    
    def split_data(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        # First split: separate test data
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df[self.label_column]
        )
        
        # Second split: separate validation from train
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=train_val_df[self.label_column]
        )
        
        return train_df, val_df, test_df
    
    def get_class_weights(self, df):
        """Calculate class weights for imbalanced datasets"""
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        labels = df[self.label_column].values
        classes = np.unique(labels)
        class_weights = compute_class_weight('balanced', classes=classes, y=labels)
        return dict(zip(classes, class_weights))