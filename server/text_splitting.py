import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab')

def split_into_clauses(text):
    sentences = sent_tokenize(text)
    return sentences 