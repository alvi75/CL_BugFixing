import os
import pandas as pd

# Split dataset
def split_into_batches(file_path, output_folder, shuffle=True):
    # Load the dataset with complexity scores
    df = pd.read_csv(file_path)

    # Compute quartiles
    q1 = df['Complexity'].quantile(0.25)
    median = df['Complexity'].quantile(0.50)
    q3 = df['Complexity'].quantile(0.75)

    # Define complexity classes
    df['Complexity Class'] = df['Complexity'].apply(
        lambda x: 'Low' if x <= q1 else
                  'Low-Medium' if x <= median else
                  'Medium-High' if x <= q3 else
                  'High'
    )

    # Create cumulative batches
    low = df[df['Complexity Class'] == 'Low']
    low_medium = df[df['Complexity Class'].isin(['Low', 'Low-Medium'])]
    medium_high = df[df['Complexity Class'].isin(['Low', 'Low-Medium', 'Medium-High'])]
    high = df  # All instances

    # Optionally shuffle each batch
    if shuffle:
        low = low.sample(frac=1, random_state=42).reset_index(drop=True)
        low_medium = low_medium.sample(frac=1, random_state=42).reset_index(drop=True)
        medium_high = medium_high.sample(frac=1, random_state=42).reset_index(drop=True)
        high = high.sample(frac=1, random_state=42).reset_index(drop=True)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save batches to files
    low.to_csv(f"{output_folder}/batch_1_low.csv", index=False)
    low_medium.to_csv(f"{output_folder}/batch_2_low_medium.csv", index=False)
    medium_high.to_csv(f"{output_folder}/batch_3_medium_high.csv", index=False)
    high.to_csv(f"{output_folder}/batch_4_high.csv", index=False)

    print(f"Batches saved to {output_folder}")

# Paths for datasets and output folders
datasets = {
    "small": {
        "file_path": "data/small/train_with_complexity.csv",
        "output_folder": "data/small/batches",
    },
    "medium": {
        "file_path": "data/medium/train_medium_with_complexity.csv",
        "output_folder": "data/medium/batches",
    },
    "small+medium": {
        "file_path": "data/small+medium/train_combined_with_complexity.csv",
        "output_folder": "data/small+medium/batches",
    },
}

# Process each dataset
for dataset_type, paths in datasets.items():
    print(f"Processing {dataset_type} dataset for curriculum learning...")
    split_into_batches(paths["file_path"], paths["output_folder"], shuffle=True)
