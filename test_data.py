import unittest
import pandas as pd
import os
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.config import load_config

class TestData(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.data_loader = DataLoader(self.config)
        self.preprocessor = DataPreprocessor(self.config['model']['name'])
        
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'text': [
                'This is a real news article about important events.',
                'Fake news alert! This is completely made up.',
                'Another legitimate piece of journalism here.',
                'https://example.com This is spam with a URL'
            ],
            'label': [1, 0, 1, 0]
        })
    
    def test_data_loading(self):
        df = self.data_loader.load_data('data/raw/train.csv')
        self.assertIsNotNone(df)
        self.assertTrue('text' in df.columns)
        self.assertTrue('label' in df.columns)
    
    def test_data_splitting(self):
        train_df, val_df, test_df = self.data_loader.split_data(self.sample_data)
        
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(self.sample_data))
        self.assertAlmostEqual(len(test_df) / len(self.sample_data), 0.2, delta=0.1)
    
    def test_text_cleaning(self):
        dirty_text = "Check this out: https://example.com #trending @user123!!!"
        cleaned_text = self.preprocessor.clean_text(dirty_text)
        
        self.assertNotIn('http', cleaned_text)
        self.assertNotIn('#', cleaned_text)
        self.assertNotIn('@', cleaned_text)
        self.assertNotIn('!', cleaned_text)
    
    def test_tokenization(self):
        texts = self.sample_data['text'].head(2)
        encodings = self.preprocessor.tokenize(texts)
        
        self.assertIn('input_ids', encodings)
        self.assertIn('attention_mask', encodings)
        self.assertEqual(encodings['input_ids'].shape[0], 2)

if __name__ == '__main__':
    unittest.main()