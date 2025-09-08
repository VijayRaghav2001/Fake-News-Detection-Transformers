from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def calculate_metrics(true_labels, predictions):
    """Calculate various evaluation metrics"""
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def calculate_class_metrics(true_labels, predictions, class_names):
    """Calculate metrics for each class"""
    results = {}
    for i, class_name in enumerate(class_names):
        # Create binary labels for this class
        binary_true = [1 if label == i else 0 for label in true_labels]
        binary_pred = [1 if pred == i else 0 for pred in predictions]
        
        precision = precision_score(binary_true, binary_pred, zero_division=0)
        recall = recall_score(binary_true, binary_pred, zero_division=0)
        f1 = f1_score(binary_true, binary_pred, zero_division=0)
        
        results[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    return results