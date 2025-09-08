import unittest
import torch
from src.models.model import TransformerModel
from src.config import load_config

class TestModels(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.model = TransformerModel(
            self.config['model']['name'],
            self.config['model']['num_labels'],
            self.config['model']['dropout_rate']
        )
        
        # Create sample input
        self.batch_size = 2
        self.seq_length = 16
        self.input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_length))
        self.attention_mask = torch.ones((self.batch_size, self.seq_length))
        self.labels = torch.tensor([0, 1])
    
    def test_model_forward(self):
        loss, logits = self.model(self.input_ids, self.attention_mask, self.labels)
        
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(logits.shape[0], self.batch_size)
        self.assertEqual(logits.shape[1], self.config['model']['num_labels'])
    
    def test_model_save_load(self):
        # Save model
        self.model.save_model('test_model.pt')
        
        # Load model
        new_model = TransformerModel(
            self.config['model']['name'],
            self.config['model']['num_labels'],
            self.config['model']['dropout_rate']
        )
        new_model.load_model('test_model.pt')
        
        # Test that loaded model produces same output
        with torch.no_grad():
            _, original_logits = self.model(self.input_ids, self.attention_mask)
            _, loaded_logits = new_model(self.input_ids, self.attention_mask)
        
        self.assertTrue(torch.allclose(original_logits, loaded_logits, atol=1e-6))
        
        # Clean up
        import os
        os.remove('test_model.pt')

if __name__ == '__main__':
    unittest.main()