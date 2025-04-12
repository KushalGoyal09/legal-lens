import re

def clean_text(text):
    text = re.sub(r'\n\d+\n', '\n', text)  # Remove page numbers
    text = re.sub(r'(?m)^.*Confidential.*$', '', text)  # Remove headers/footers
    return text.strip()