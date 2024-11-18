import pandas as pd
import re
import Levenshtein

def tokenize_code(code):
    tokens = re.findall(r'\w+|[^\s\w]', code)
    return tokens

def compute_complexity(buggy_code, fixed_code):
    buggy_tokens = tokenize_code(buggy_code)
    fixed_tokens = tokenize_code(fixed_code)
    buggy_str = ' '.join(buggy_tokens)
    fixed_str = ' '.join(fixed_tokens)
    return Levenshtein.distance(buggy_str, fixed_str)

def add_complexity_column(file_path, save_path):
    df = pd.read_csv(file_path)

    # Computes complexity for each row
    df['Complexity'] = df.apply(lambda row: compute_complexity(row['Buggy Code'], row['Fixed Code']), axis=1)

    # Save the dataset with complexity scores
    df.to_csv(save_path, index=False)
    print(f"Complexity scores added and saved to {save_path}")

datasets = {
    "small": [
        ("data/small/train.csv", "data/small/train_with_complexity.csv"),
        ("data/small/valid.csv", "data/small/valid_with_complexity.csv"),
        ("data/small/test.csv", "data/small/test_with_complexity.csv"),
    ],
    "medium": [
        ("data/medium/train_medium.csv", "data/medium/train_medium_with_complexity.csv"),
        ("data/medium/valid_medium.csv", "data/medium/valid_medium_with_complexity.csv"),
        ("data/medium/test_medium.csv", "data/medium/test_medium_with_complexity.csv"),
    ],
    "small+medium": [
        ("data/small+medium/train_combined.csv", "data/small+medium/train_combined_with_complexity.csv"),
        ("data/small+medium/valid_combined.csv", "data/small+medium/valid_combined_with_complexity.csv"),
        ("data/small+medium/test_combined.csv", "data/small+medium/test_combined_with_complexity.csv"),
    ]
}

# Process each dataset
for dataset_type, paths in datasets.items():
    print(f"Processing {dataset_type} dataset...")
    for input_file, output_file in paths:
        add_complexity_column(input_file, output_file)
