import pandas as pd
import numpy as np

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

# Set random seed for reproducibility
np.random.seed(42)

# Load the full dataset
print("Loading full dataset...")
full_dataset = pd.read_csv('dataset.csv')
print(f"Full dataset shape: {full_dataset.shape}")

# Check class distribution
class_distribution = full_dataset['risk'].value_counts()
print("\nClass distribution in full dataset:")
print(class_distribution)
print(f"Class 0 (No Risk): {class_distribution.get(0, 0)} clauses")
print(f"Class 1 (Risk): {class_distribution.get(1, 0)} clauses")

# Calculate text length statistics to ensure diversity
full_dataset['text_length'] = full_dataset['clause'].apply(len)
print("\nText length statistics in full dataset:")
print(f"Min length: {full_dataset['text_length'].min()} characters")
print(f"Max length: {full_dataset['text_length'].max()} characters")
print(f"Mean length: {full_dataset['text_length'].mean():.2f} characters")
print(f"Median length: {full_dataset['text_length'].median()} characters")

# Function to create a balanced subset
def create_balanced_subset(df, target_size=500):
    """
    Create a balanced subset of data with representation from different text lengths
    
    Args:
        df: DataFrame containing the data
        target_size: Target size of the balanced subset
    
    Returns:
        DataFrame: Balanced subset of data
    """
    # Split by risk class
    risk_0 = df[df['risk'] == 0]
    risk_1 = df[df['risk'] == 1]
    
    # Calculate how many samples to take from each class
    # Aim for 50/50 balance
    samples_per_class = target_size // 2
    
    # If one class has fewer samples than needed, take all of them
    samples_class_0 = min(len(risk_0), samples_per_class)
    samples_class_1 = min(len(risk_1), samples_per_class)
    
    # Adjust if necessary to reach target size
    if samples_class_0 + samples_class_1 < target_size:
        if len(risk_0) > samples_class_0:
            samples_class_0 = min(len(risk_0), target_size - samples_class_1)
        elif len(risk_1) > samples_class_1:
            samples_class_1 = min(len(risk_1), target_size - samples_class_0)
    
    print(f"\nSelecting {samples_class_0} samples from Class 0 (No Risk)")
    print(f"Selecting {samples_class_1} samples from Class 1 (Risk)")
    
    # For each class, stratify by text length to ensure diversity
    # Create bins for text length
    for df_class in [risk_0, risk_1]:
        df_class['length_bin'] = pd.qcut(df_class['text_length'], 
                                         q=5, 
                                         labels=False, 
                                         duplicates='drop')
    
    # Sample from each class with stratification by length bin
    sampled_risk_0 = risk_0.groupby('length_bin', group_keys=False).apply(
        lambda x: x.sample(min(len(x), int(np.ceil(samples_class_0 * len(x) / len(risk_0)))))
    )
    
    sampled_risk_1 = risk_1.groupby('length_bin', group_keys=False).apply(
        lambda x: x.sample(min(len(x), int(np.ceil(samples_class_1 * len(x) / len(risk_1)))))
    )
    
    # Combine the samples
    balanced_subset = pd.concat([sampled_risk_0, sampled_risk_1])
    
    # If we have more samples than needed, take a random subset
    if len(balanced_subset) > target_size:
        balanced_subset = balanced_subset.sample(target_size)
    
    # Drop the auxiliary columns
    balanced_subset = balanced_subset.drop(columns=['text_length', 'length_bin'])
    
    return balanced_subset

# Create balanced subset
print("\nCreating balanced subset...")
balanced_dataset = create_balanced_subset(full_dataset, target_size=2500)

# Check the resulting dataset
print(f"\nBalanced dataset shape: {balanced_dataset.shape}")
balanced_class_distribution = balanced_dataset['risk'].value_counts()
print("\nClass distribution in balanced dataset:")
print(balanced_class_distribution)
print(f"Class 0 (No Risk): {balanced_class_distribution.get(0, 0)} clauses")
print(f"Class 1 (Risk): {balanced_class_distribution.get(1, 0)} clauses")

# Calculate text length statistics for the balanced dataset
balanced_dataset['text_length'] = balanced_dataset['clause'].apply(len)
print("\nText length statistics in balanced dataset:")
print(f"Min length: {balanced_dataset['text_length'].min()} characters")
print(f"Max length: {balanced_dataset['text_length'].max()} characters")
print(f"Mean length: {balanced_dataset['text_length'].mean():.2f} characters")
print(f"Median length: {balanced_dataset['text_length'].median()} characters")

# Remove the text_length column before saving
balanced_dataset = balanced_dataset.drop(columns=['text_length'])

# Save the balanced dataset
output_file = 'training_dataset.csv'
balanced_dataset.to_csv(output_file, index=False)
print(f"\nBalanced dataset saved to {output_file}")

# Show a few examples from each class
print("\nSample clauses from Class 0 (No Risk):")
print(balanced_dataset[balanced_dataset['risk'] == 0]['clause'].head(2).to_string())
print("\nSample clauses from Class 1 (Risk):")
print(balanced_dataset[balanced_dataset['risk'] == 1]['clause'].head(2).to_string())

print("\nDataset creation complete!")