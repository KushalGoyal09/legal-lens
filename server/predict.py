import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Path to your saved model
model_path = 'bert_risk_model'

# Load the saved model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

def predict_risk(clause_text, model, tokenizer, device):
    """
    Predict risk for a given clause text
    
    Args:
        clause_text (str): The legal clause text
        model: The fine-tuned BERT model
        tokenizer: The BERT tokenizer
        device: The compute device (CPU/GPU)
    
    Returns:
        dict: Prediction results including class and probability
    """
    # Set model to evaluation mode
    model.eval()
    
    # Tokenize the input text
    encoding = tokenizer(
        clause_text,
        add_special_tokens=True,
        max_length=256,
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
        
        # Get probabilities using softmax
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Get the predicted class (0: No Risk, 1: Risk)
        _, predicted_class = torch.max(logits, dim=1)
        
    return {
        'predicted_class': predicted_class.item(),
        'class_name': 'Risk' if predicted_class.item() == 1 else 'No Risk',
        'confidence': probabilities[0][predicted_class.item()].item()
    }


def get_risk_prediction(clause_text):
    """
    Accepts a text clause and returns the class name and probability.
        
    Args:
        clause_text (str): The legal clause text
        
        Returns:
            tuple: Class name and confidence probability
    """
    result = predict_risk(clause_text, model, tokenizer, device)
    return result['class_name'], result['confidence']