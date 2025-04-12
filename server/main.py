from predict import get_risk_prediction
from download_pdf_from_s3 import download_pdf_from_s3
from extract_text_from_pdf import extract_text_from_pdf
from text_splitting import split_into_clauses
from text_preprocessing import clean_text

def analyze_pdf_risk(url: str):
    pdf_stream = download_pdf_from_s3(url)
    text = extract_text_from_pdf(pdf_stream)
    preprocessed_text = clean_text(text)
    clauses = split_into_clauses(preprocessed_text)

    results = []
    for clause in clauses:
        prediction = get_risk_prediction(clause)
        results.append({
            "clause": clause,
            "risk_category": prediction["class_name"],
            "risk_probability": round(prediction["confidence"], 2)
        })

    return results
