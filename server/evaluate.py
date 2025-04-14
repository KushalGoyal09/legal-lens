import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class ClauseRiskDataset(Dataset):
    """Dataset for clause risk assessment using BERT."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text clauses
            labels: List of risk labels (0 or 1)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def evaluate(model, dataloader, device):
    """
    Evaluate the model on a given dataloader.
    
    Args:
        model: BERT model for sequence classification
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on (CPU or CUDA)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Get predictions
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            
            # Store predictions and actual labels
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    # Calculate metrics
    accuracy = accuracy_score(actual_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_labels, predictions, average='binary', zero_division=0
    )
    
    # Create confusion matrix
    cm = confusion_matrix(actual_labels, predictions)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'loss': total_loss / len(dataloader)
    }
    
    return results


def plot_confusion_matrix(cm, output_path='confusion_matrix.png'):
    """
    Plot and save a confusion matrix.
    
    Args:
        cm: Confusion matrix array
        output_path: Path to save the figure
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['No Risk', 'Risk'],
        yticklabels=['No Risk', 'Risk']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()


def main():
    # Configuration
    model_path = 'bert_risk_model'
    dataset_path = 'dataset.csv'
    max_length = 128
    batch_size = 16
    random_seed = 42
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    df = pd.read_csv(dataset_path)
    texts = df['clause'].values
    labels = df['risk'].values
    
    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    
    # Create dataset
    dataset = ClauseRiskDataset(texts, labels, tokenizer, max_length)
    
    # Split dataset
    test_size = int(0.15 * len(dataset))
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - test_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_metrics = evaluate(model, test_loader, device)
    
    # Print results
    print(f"Test - Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test - Precision: {test_metrics['precision']:.4f}")
    print(f"Test - Recall: {test_metrics['recall']:.4f}")
    print(f"Test - F1 Score: {test_metrics['f1']:.4f}")
    
    # Plot and save confusion matrix
    plot_confusion_matrix(test_metrics['confusion_matrix'])
    print("Confusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    main()