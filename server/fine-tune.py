import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Load the dataset
df = pd.read_csv('training_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(df.head())

# Check for class balance
print("\nClass distribution:")
class_distribution = df['risk'].value_counts()
print(class_distribution)

# Calculate class weights for balanced training
total_samples = len(df)
num_classes = 2
class_weights = torch.tensor([total_samples / (num_classes * count) for count in class_distribution], dtype=torch.float)
print(f"Class weights: {class_weights}")

# Tokenizer setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128  # Reduced to 128 as contract clauses are often shorter

# Create a custom dataset
class ClauseRiskDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
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

# Split the dataset into training, validation and test sets (70/15/15)
texts = df['clause'].values
labels = df['risk'].values

# Create the dataset
dataset = ClauseRiskDataset(texts, labels, tokenizer, max_length)

# Split dataset
test_size = int(0.15 * len(dataset))
val_size = int(0.15 * len(dataset))
train_size = len(dataset) - test_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, 
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(seed)
)

print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")
print(f"Testing samples: {test_size}")

# Create data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,  # Binary classification
    output_attentions=False,
    output_hidden_states=False,
    problem_type="single_label_classification"
)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)
class_weights = class_weights.to(device)

# Set up the optimizer with better learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)  # Reduced learning rate for stability

# Create learning rate scheduler
# Calculate total training steps
epochs = 50  # Defined here for clarity
total_steps = len(train_loader) * epochs
warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Loss function with class weights
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# Training function
def train_epoch(model, dataloader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels  # Added labels to get loss directly from the model
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Get predictions
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        scheduler.step()
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Evaluation function
def evaluate(model, dataloader, device, loss_fn=None):
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
                labels=labels  # Added labels to get loss directly
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

# Improved Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, monitor='loss', mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode  # 'min' for loss, 'max' for metrics like accuracy
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score, model=None):
        if self.best_score is None:
            self.best_score = score
            if model is not None:
                self.best_model_state = model.state_dict().copy()
        elif (self.mode == 'min' and score > self.best_score - self.min_delta) or \
             (self.mode == 'max' and score < self.best_score + self.min_delta):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if ((self.mode == 'min' and score < self.best_score) or 
                (self.mode == 'max' and score > self.best_score)):
                self.best_score = score
                if model is not None:
                    self.best_model_state = model.state_dict().copy()
            self.counter = 0

# Training loop with validation
epochs = 50  # Set a reasonable number of epochs
early_stopping = EarlyStopping(patience=5, min_delta=0.0, monitor='accuracy', mode='max')  # More patient early stopping
train_metrics_history = []
val_metrics_history = []

best_val_accuracy = 0
best_model_state = None

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    
    # Train
    train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, loss_fn)
    train_metrics_history.append(train_metrics)
    
    print(f"Training - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
          f"Prec: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, "
          f"F1: {train_metrics['f1']:.4f}")
    
    # Validate
    val_metrics = evaluate(model, val_loader, device)
    val_metrics_history.append(val_metrics)
    
    print(f"Validation - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
          f"Prec: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
          f"F1: {val_metrics['f1']:.4f}")
    
    # Save the best model based on validation accuracy
    if val_metrics['accuracy'] > best_val_accuracy:
        best_val_accuracy = val_metrics['accuracy']
        best_model_state = model.state_dict().copy()
        print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")
    
    # Check if early stopping criteria is met
    early_stopping(val_metrics['accuracy'], model)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

# Load the best model for final evaluation
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Loaded best model with validation accuracy: {best_val_accuracy:.4f}")

# Final evaluation on test set
print("\nFinal Evaluation on Test Set:")
test_metrics = evaluate(model, test_loader, device)

print(f"Test - Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Test - Precision: {test_metrics['precision']:.4f}")
print(f"Test - Recall: {test_metrics['recall']:.4f}")
print(f"Test - F1 Score: {test_metrics['f1']:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    test_metrics['confusion_matrix'], 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=['No Risk', 'Risk'],
    yticklabels=['No Risk', 'Risk']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Plot training history
plt.figure(figsize=(12, 8))
epochs_range = range(1, len(train_metrics_history) + 1)

plt.subplot(2, 2, 1)
plt.plot(epochs_range, [m['loss'] for m in train_metrics_history], 'b-', label='Training Loss')
plt.plot(epochs_range, [m['loss'] for m in val_metrics_history], 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs_range, [m['accuracy'] for m in train_metrics_history], 'b-', label='Training Accuracy')
plt.plot(epochs_range, [m['accuracy'] for m in val_metrics_history], 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs_range, [m['precision'] for m in train_metrics_history], 'b-', label='Training Precision')
plt.plot(epochs_range, [m['precision'] for m in val_metrics_history], 'r-', label='Validation Precision')
plt.title('Training and Validation Precision')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs_range, [m['f1'] for m in train_metrics_history], 'b-', label='Training F1')
plt.plot(epochs_range, [m['f1'] for m in val_metrics_history], 'r-', label='Validation F1')
plt.title('Training and Validation F1 Score')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# Save the model
model_save_path = 'bert_risk_model'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"\nModel saved to {model_save_path}")

# Function to make predictions on new text
def predict_risk(text, model, tokenizer, device, max_length=128):
    model.eval()
    
    # Tokenize the text
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        _, preds = torch.max(logits, dim=1)
    
    risk_score = probabilities[0][1].item()  # Probability of the "Risk" class
    prediction = preds.item()
    
    return {
        'prediction': prediction,
        'risk_score': risk_score,
        'label': 'Risk' if prediction == 1 else 'No Risk'
    }

# Test prediction on sample clauses
sample_clauses = [
    "All information shared during this meeting shall be treated as confidential.",
    "The supplier shall indemnify the company against all liabilities arising from defective products.",
    "Payment terms are net 30 days from the date of invoice.",
    "This agreement may be terminated by either party with 30 days written notice."
]

print("\nSample predictions:")
for clause in sample_clauses:
    result = predict_risk(clause, model, tokenizer, device)
    print(f"\nClause: '{clause}'")
    print(f"Prediction: {result['label']} (Score: {result['risk_score']:.4f})")

print("\nTraining and evaluation complete!")