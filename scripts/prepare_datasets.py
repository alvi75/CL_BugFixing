import pandas as pd

# File paths for small dataset
small_train_buggy = "data/small/train.buggy-fixed.buggy"
small_train_fixed = "data/small/train.buggy-fixed.fixed"
small_valid_buggy = "data/small/valid.buggy-fixed.buggy"
small_valid_fixed = "data/small/valid.buggy-fixed.fixed"
small_test_buggy = "data/small/test.buggy-fixed.buggy"
small_test_fixed = "data/small/test.buggy-fixed.fixed"

# File paths for medium dataset
medium_train_buggy = "data/medium/train.buggy-fixed.buggy"
medium_train_fixed = "data/medium/train.buggy-fixed.fixed"
medium_valid_buggy = "data/medium/valid.buggy-fixed.buggy"
medium_valid_fixed = "data/medium/valid.buggy-fixed.fixed"
medium_test_buggy = "data/medium/test.buggy-fixed.buggy"
medium_test_fixed = "data/medium/test.buggy-fixed.fixed"

# Load buggy and fixed code into a DataFrame
def load_buggy_fixed(buggy_file, fixed_file):
    with open(buggy_file, 'r') as buggy_f, open(fixed_file, 'r') as fixed_f:
        buggy_lines = buggy_f.readlines()
        fixed_lines = fixed_f.readlines()
    return pd.DataFrame({'Buggy Code': buggy_lines, 'Fixed Code': fixed_lines})

# Load small dataset
small_train = load_buggy_fixed(small_train_buggy, small_train_fixed)
small_valid = load_buggy_fixed(small_valid_buggy, small_valid_fixed)
small_test = load_buggy_fixed(small_test_buggy, small_test_fixed)

# Save small dataset to CSV
small_train.to_csv("data/small/train.csv", index=False)
small_valid.to_csv("data/small/valid.csv", index=False)
small_test.to_csv("data/small/test.csv", index=False)

print("Small dataset saved as CSV files.")

# Load medium dataset
medium_train = load_buggy_fixed(medium_train_buggy, medium_train_fixed)
medium_valid = load_buggy_fixed(medium_valid_buggy, medium_valid_fixed)
medium_test = load_buggy_fixed(medium_test_buggy, medium_test_fixed)

# Save medium dataset to CSV
medium_train.to_csv("data/medium/train_medium.csv", index=False)
medium_valid.to_csv("data/medium/valid_medium.csv", index=False)
medium_test.to_csv("data/medium/test_medium.csv", index=False)

print("Medium dataset saved as CSV files.")

# Combine small and medium datasets for "small+medium"
combined_train = pd.concat([small_train, medium_train], ignore_index=True)
combined_valid = pd.concat([small_valid, medium_valid], ignore_index=True)
combined_test = pd.concat([small_test, medium_test], ignore_index=True)

# Save combined datasets to CSV
combined_train.to_csv("data/small+medium/train_combined.csv", index=False)
combined_valid.to_csv("data/small+medium/valid_combined.csv", index=False)
combined_test.to_csv("data/small+medium/test_combined.csv", index=False)

print("Combined dataset saved as CSV files.")
