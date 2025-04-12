import fitz

def extract_text_from_pdf(pdf_stream):
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text