import boto3
import io
from dotenv import load_dotenv
import os

load_dotenv()

def download_pdf_from_s3(s3_url: str):
    s3 = boto3.client(
        's3',
        aws_access_key_id= os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )

    bucket_name, key = s3_url.replace("s3://", "").split("/", 1)
    pdf_stream = io.BytesIO()
    s3.download_fileobj(bucket_name, key, pdf_stream)
    pdf_stream.seek(0)

    return pdf_stream