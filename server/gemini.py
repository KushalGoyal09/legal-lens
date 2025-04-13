from dotenv import load_dotenv
load_dotenv()
import os
from google import genai
from google.genai import types
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def generate(input: str):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=input,
        config=types.GenerateContentConfig(
            max_output_tokens=1,
            temperature=0.1,
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text="""you will be given with clauses from a legal documents. please tell weather the clause is risky or not. output 0 (not risky) or 1 (risky/ needs attention)"""),
            ],
        ),
    )
    return response.text

# Load the dataset
df = pd.read_csv('dataset.csv')


# Initialize lists to store predictions and actual values
predictions = []
actuals = df['risk'].tolist()

# Iterate over each clause and get predictions
i = 0
for clause in df['clause']:
    print(i)
    i += 1
    predicted_risk = int(generate(clause).strip())  # Ensure the output is converted to an integer
    predictions.append(predicted_risk)

# Calculate metrics
accuracy = accuracy_score(actuals, predictions)
precision = precision_score(actuals, predictions)
recall = recall_score(actuals, predictions)
f1 = f1_score(actuals, predictions)

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

