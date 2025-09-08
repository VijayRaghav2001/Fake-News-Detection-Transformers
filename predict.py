import torch
import pandas as pd
from src.data.data_preprocessor import DataPreprocessor
from src.models.model import TransformerModel
from src.config import load_config
import argparse
import os

def predict_text(model, preprocessor, text, device):
    """Predict class for a single text"""
    # Clean and tokenize text
    cleaned_text = preprocessor.clean_text(text)
    encodings = preprocessor.tokenizer(
        [cleaned_text],
        truncation=True,
        padding=True,
        max_length=preprocessor.max_length,
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        _, logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    
    return pred, confidence, probs.cpu().numpy()[0]

def main():
    parser = argparse.ArgumentParser(description="Predict fake news from text")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--input_file", type=str, help="Path to file containing texts to classify")
    parser.add_argument("--output_file", type=str, default="predictions.csv", help="Output file for predictions")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    
    args = parser.parse_args()
    
    if not args.text and not args.input_file:
        parser.error("Either --text or --input_file must be provided")
    
    # Load configuration
    config = load_config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = TransformerModel(
        config['model']['name'],
        config['model']['num_labels'],
        config['model']['dropout_rate']
    )
    
    model = model.load_model(args.model_path)
    model.to(device)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        config['model']['name'],
        max_length=config['training']['max_seq_length']
    )
    
    # Make predictions
    if args.text:
        pred, confidence, probabilities = predict_text(model, preprocessor, args.text, device)
        class_name = "Fake" if pred == 0 else "Real"
        print(f"Prediction: {class_name}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities: [Fake: {probabilities[0]:.4f}, Real: {probabilities[1]:.4f}]")
    
    if args.input_file:
        # Read input file
        if args.input_file.endswith('.csv'):
            df = pd.read_csv(args.input_file)
            texts = df[config['data']['text_column']].tolist()
        else:
            with open(args.input_file, 'r') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
        
        # Make predictions
        predictions = []
        confidences = []
        
        for text in texts:
            pred, confidence, _ = predict_text(model, preprocessor, text, device)
            predictions.append("Fake" if pred == 0 else "Real")
            confidences.append(confidence)
        
        # Save results
        results_df = pd.DataFrame({
            'text': texts,
            'prediction': predictions,
            'confidence': confidences
        })
        
        results_df.to_csv(args.output_file, index=False)
        print(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()