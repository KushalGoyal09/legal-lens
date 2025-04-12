import pandas as pd

# Load the TSV file
df = pd.read_csv('merged_output.tsv', sep='\t')

# Keep only the relevant columns
df = df[['text', 'answer']]

# Rename columns
df = df.rename(columns={'text': 'clause', 'answer': 'risk'})

# Convert risk labels to binary
df['risk'] = df['risk'].apply(lambda x: 1 if x.strip().lower() == 'risky' else 0)

# Remove duplicates based on the 'clause' column
df = df.drop_duplicates(subset='clause')

# Save to CSV
df.to_csv('dataset.csv', index=False)

