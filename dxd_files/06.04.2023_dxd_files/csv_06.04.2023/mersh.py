import pandas as pd
from openpyxl import load_workbook

# Specify the file paths for your input and output files
input_file1 = 'J298MCL001 - Strain Measurements First Setup.xlsx'
input_file2 = 'merged_data_second_third.xlsx'
output_file = 'J298MCL001 - Strain Measurements.xlsx'

# Load the first Excel file
xls1 = pd.read_excel(input_file1, sheet_name=None)

# Load the second Excel file
xls2 = pd.read_excel(input_file2, sheet_name=None)

# Combine the dictionaries containing the sheets from both files
combined_sheets = {**xls1, **xls2}

# Create a new Excel file with the combined sheets
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for sheet_name, df in combined_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Files combined successfully!")
