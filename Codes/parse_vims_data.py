
import pandas as pd
import os 


# sheet_names = ['2023-03-22','2023-03-23','2023-03-24','2023-03-25','2023-03-26','2023-03-27','2023-03-28','2023-03-29','2023-03-30']  # Add more sheet names as needed
# strain_measurements = pd.read_excel("Strain Measurements First Setup.xlsx", sheet_name=sheet_names, parse_dates=True)
# vims = pd.read_excel("vims.xlsx", sheet_name="Hoja1", parse_dates=['ReadTime'])


# # Convert the date formats in both DataFrames to be consistent
# vims['ReadTime'] = pd.to_datetime(vims['ReadTime'])

# # Iterate through each sheet in the strain_measurements DataFrame
# for sheet_name, df in strain_measurements.items():
#     # Convert the Date column to datetime
#     df['Date'] = pd.to_datetime(df['Date'])

#     # Create a new column 'Cycle' in the strain_measurements DataFrame
#     df['Cycle'] = 0

#     # Initialize the matching cycles set
#     matching_cycles = set()

#     # Assign the corresponding cycle values based on the vims DataFrame
#     for i in range(len(vims) - 1):
#         cycle_start, cycle_end = vims.iloc[i]['ReadTime'], vims.iloc[i + 1]['ReadTime']
#         cycle = vims.iloc[i]['Cycle']
#         mask = (df['Date'] >= cycle_start) & (df['Date'] < cycle_end)
#         df.loc[mask, 'Cycle'] = cycle

#         # Add the cycle to the matching_cycles set if it's present in the current sheet
#         if mask.any():
#             matching_cycles.add(cycle)

#     # Print the matching cycles for the current sheet
#     print(f"Matching cycles for sheet {sheet_name}: {matching_cycles}")

#     # Save the modified DataFrame to a new Excel file
#     df.to_excel(f"Strain Measuremets First SetUp_modified_{sheet_name}.xlsx", index=False)

def generate_vims_data(file_name):
    file_path= os.path.join('..', 'Reference',file_name)
    df = pd.read_excel(file_path, engine='openpyxl')
    # print(df.columns)
    # return df
    df2 = df[['ReadTime','Cycle']]
    file_out_path= os.path.join('..', 'Reference','vims.xlsx')
    print(file_out_path)
    # writer = pd.ExcelWriter(file_out_path , engine='xlsxwriter')
    df2.to_excel(file_out_path, index=False)
    # print('vims file  was generated')
    return df2

df  = generate_vims_data('Datos C429 - (pivoted) Ciclos.xlsx')