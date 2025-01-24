import os
import subprocess
import pandas as pd
from tqdm import tqdm

# Path to the GumTree JAR file
GUMTREE_JAR_PATH = "tools/gumtree.jar"

# Function to compute AST-based complexity using GumTree
def compute_ast_complexity(buggy_code, fixed_code):
    try:
        # Save buggy and fixed code as temporary files
        with open("buggy_temp.py", "w") as f:
            f.write(buggy_code)
        with open("fixed_temp.py", "w") as f:
            f.write(fixed_code)

        # Run GumTree textdiff as a subprocess
        result = subprocess.run(
            ["java", "-jar", GUMTREE_JAR_PATH, "textdiff", "buggy_temp.py", "fixed_temp.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Check for errors
        if result.returncode != 0:
            print(f"GumTree error: {result.stderr}")
            return float('inf')  # Assign high complexity if there's an error

        # Parse the output and count actions as complexity
        actions = [line for line in result.stdout.splitlines() if line.startswith("UPD") or line.startswith("DEL") or line.startswith("INS")]
        return len(actions)

    except Exception as e:
        print(f"Error computing complexity: {e}")
        return float('inf')

    finally:
        # Clean up temporary files
        os.remove("buggy_temp.py")
        os.remove("fixed_temp.py")

# Add complexity column to the dataset
def add_complexity_column(file_path, save_path):
    df = pd.read_csv(file_path)

    # Initialize tqdm for progress tracking
    tqdm_iter = tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(file_path)}", unit="row")
    
    # Compute complexity for each row
    complexities = []
    for _, row in tqdm_iter:
        complexity = compute_ast_complexity(row['Buggy Code'], row['Fixed Code'])
        complexities.append(complexity)
    df['Complexity'] = complexities

    # Save the updated dataset
    df.to_csv(save_path, index=False)
    print(f"Complexity scores added and saved to {save_path}")

# Define datasets for all types
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

# Process all datasets
for dataset_type, paths in datasets.items():
    print(f"Processing {dataset_type} dataset...")
    for input_file, output_file in paths:
        add_complexity_column(input_file, output_file)
