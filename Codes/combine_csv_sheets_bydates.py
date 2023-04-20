import os
import pandas as pd
import glob

# set the path to the folder containing the CSV files
csv_folder = os.path.join('..', 'dxd_files', '06.04.2023_dxd_files','csv_06.04.2023')

# get a list of all the CSV files in the folder
csv_files = glob.glob(os.path.join(csv_folder, 'resampled_data_*_rosette_*.csv')) + \
            glob.glob(os.path.join(csv_folder, 'resampled_data_1 2.*.dxd_rosette_1.csv'))

output_name =  'merged_data.xlsx'
# create an empty dictionary to store the data frames for each sheet
dfs = {}

# loop over the CSV files and read them into a dictionary of data frames
for csv_file in csv_files:
    print(csv_file)
    # read the CSV into a data frame
    df = pd.read_csv(csv_file)
    
    # get the name of the first sheet from the first timestamp in the Date column
    sheet_name = pd.to_datetime(df['Date'][0]).strftime('%Y-%m-%d')


    # If the sheet_name already exists in the dictionary, append the new data to the existing data frame
    if sheet_name in dfs:
        dfs[sheet_name] = dfs[sheet_name].append(df)
    # Otherwise, add the new data frame to the dictionary with the sheet name as the key
    else:
        dfs[sheet_name] = df

# create an Excel writer object to write the data frames to an Excel file
output_file = os.path.join(csv_folder, output_name)

writer = pd.ExcelWriter(output_file , engine='xlsxwriter')

# loop over the dictionary of data frames and write each one to a new sheet
for sheet_name, df in dfs.items():
    df.to_excel(writer, sheet_name=sheet_name, index=False)

# save the Excel file
writer.save()
