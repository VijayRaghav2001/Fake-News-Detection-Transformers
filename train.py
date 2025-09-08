import torch
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.models.model import TransformerModel
from src.models.model_trainer import ModelTrainer
from src.utils.helpers import set_seed, plot_training_history, save_results
from src.config import load_config
import os

def main():
    # Load configuration
    config = load_config()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_loader = DataLoader(config)
    train_df = data_loader.load_data(config['data']['train_path'])
    val_df = data_loader.load_data(config['data']['val_path'])
    
    if train_df is None or val_df is None:
        print("Error loading data. Please check file paths.")
        return
    
    # Preprocess data
    preprocessor = DataPreprocessor(
        config['model']['name'],
        max_length=config['training']['max_seq_length']
    )
    
    print("Preprocessing training data...")
    train_encodings, train_labels, _ = preprocessor.preprocess_data(
        train_df, config['data']['text_column'], config['data']['label_column']
    )
    
    print("Preprocessing validation data...")
    val_encodings, val_labels, _ = preprocessor.preprocess_data(
        val_df, config['data']['text_column'], config['data']['label_column']
    )
    
    # Create data loaders
    trainer = ModelTrainer(None, device, config)  # Temporary instance for creating data loaders
    
    train_loader = trainer.create_data_loader(
        train_encodings, train_labels, 
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    val_loader = trainer.create_data_loader(
        val_encodings, val_labels,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Initialize model
    model = TransformerModel(
        config['model']['name'],
        config['model']['num_labels'],
        config['model']['dropout_rate']
    ).to(device)
    
    # Initialize trainer with model
    trainer = ModelTrainer(model, device, config)
    
    # Train model
    print("Starting training...")
    model = trainer.train(train_loader, val_loader)
    
    # Save final model
    model_path = os.path.join(config['paths']['saved_models'], "final_model.pt")
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate on validation set
    val_loss, val_acc, val_f1 = trainer.evaluate(val_loader)
    print(f"Final Validation Results - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
    
    # Save results
    results = {
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'val_f1': val_f1
    }
    
    results_path = os.path.join(config['paths']['saved_models'], "training_results.json")
    save_results(results, results_path)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()