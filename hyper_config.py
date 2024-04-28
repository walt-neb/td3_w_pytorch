import os
import pandas as pd

def read_hyp_files(directory):
    hyp_files = [f for f in os.listdir(directory) if f.startswith("hyp_") and f.endswith(".txt")]
    hyperparameters = []

    for file in hyp_files:
        hypes = {}
        with open(os.path.join(directory, file), "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    hypes[key] = value

        hypes["file_name"] = file
        hyperparameters.append(hypes)

    return pd.DataFrame(hyperparameters)

# Usage
directory = "."  # Replace with the directory containing the hyp_xxxx.txt files
hyperparameter_table = read_hyp_files(directory)
print(hyperparameter_table)

# Print the transposed DataFrame
print("\nTransposed DataFrame:")
print(hyperparameter_table.T)

# Reorder the columns in the transposed DataFrame
transposed_table = hyperparameter_table.T
transposed_table.insert(0, "file_name", hyperparameter_table["file_name"])
cols = transposed_table.columns.tolist()

# Output transposed DataFrame to CSV file
csv_file = "hyperparameters_transposed.csv"
transposed_table.to_csv(csv_file, index_label="Parameter", columns=cols)
print(f"\nTransposed data saved to CSV file: {csv_file}")

# Output transposed DataFrame to Excel file (requires openpyxl library)
try:
    import openpyxl
    excel_file = "hyperparameters_transposed.xlsx"
    transposed_table.to_excel(excel_file, index_label="Parameter", columns=cols)
    print(f"\nTransposed data saved to Excel file: {excel_file}")
except ImportError:
    print("\nNote: openpyxl library is required to save data to Excel format.")