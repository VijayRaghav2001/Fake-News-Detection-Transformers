import re
import pandas as pd
from transformers import AutoTokenizer

class DataPreprocessor:
    def __init__(self, model_name, max_length=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, texts, labels=None):
        """Tokenize texts for transformer models"""
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        if labels is not None:
            return encodings, labels.tolist()
        return encodings
    
    def preprocess_data(self, df, text_column, label_column):
        """Preprocess the entire dataset"""
        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        # Tokenize
        encodings, labels = self.tokenize(df['cleaned_text'], df[label_column])
        
        return encodings, labels, df