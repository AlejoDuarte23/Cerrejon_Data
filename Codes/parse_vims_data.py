                                                                                                                            
import pandas as pd
import os 


# sheet_names = ['2023-03-22','2023-03-23','2023-03-24','2023-03-25','2023-03-26','2023-03-27','2023-03-28','2023-03-29','2023-03-30']  # Add more sheet names as needed
# strain_measurements = pd.read_excel("Strain Measurements First Setup.xlsx", sheet_name=sheet_names, parse_dates=True)
# vims = pd.read_excel("vims.xlsx", sheet_name="Hoja1", parse_dates=['ReadTime'])



def generate_vims_data(file_name):
    file_path= os.path.join('..', 'Reference',file_name)
    df = pd.read_excel(file_path, engine='openpyxl')
    # print(df.columns)
    # return df
    df2 = df[['ReadTime','Cycle']]
    file_out_path= os.path.join('..', 'Reference','vims_28_feb.xlsx')
    print(file_out_path)
    # writer = pd.ExcelWriter(file_out_path , engine='xlsxwriter')
    df2.to_excel(file_out_path, index=False)
    # print('vims file  was generated')
    return df2

# df  = generate_vims_data('ciclos50h.xlsx')



def Match_Cicles(file__path,file__path_vims,sheet_names):
    vims = pd.read_excel(file__path_vims, sheet_name="Sheet1", parse_dates=['ReadTime'])
    
    strain_measurements = pd.read_excel(file__path, sheet_name=sheet_names, parse_dates=True)
    vims = pd.read_excel(file__path_vims, sheet_name="Sheet1", parse_dates=['ReadTime'])
    # Convert the date formats in both DataFrames to be consistent
    vims['ReadTime'] = pd.to_datetime(vims['ReadTime'])
    
    # Iterate through each sheet in the strain_measurements DataFrame
    for sheet_name, df in strain_measurements.items():
        # Convert the Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
    
        # Create a new column 'Cycle' in the strain_measurements DataFrame
        df['Cycle'] = 0
    
        # Initialize the matching cycles set
        matching_cycles = set()
    
        # Assign the corresponding cycle values based on the vims DataFrame
        for i in range(len(vims) - 1):
            cycle_start, cycle_end = vims.iloc[i]['ReadTime'], vims.iloc[i + 1]['ReadTime']
            cycle = vims.iloc[i]['Cycle']
            mask = (df['Date'] >= cycle_start) & (df['Date'] < cycle_end)
            df.loc[mask, 'Cycle'] = cycle
    
            # Add the cycle to the matching_cycles set if it's present in the current sheet
            if mask.any():
                matching_cycles.add(cycle)
    
        # Print the matching cycles for the current sheet
        print(f"Matching cycles for sheet {sheet_name}: {matching_cycles}")
        out_file= os.path.join('..', 'Reference',f"Strain Measuremets First SetUp_modified_{sheet_name}.xlsx")
        # Save the modified DataFrame to a new Excel file
        df.to_excel(out_file, index=False)
        
#  Run april 
# sheet_names = ['2023-04-06','2023-04-07','2023-04-08','2023-04-10','2023-04-13','2023-04-14','2023-04-15','2023-04-16','2023-04-17','2023-04-18','2023-04-19']
# file__path_vims = os.path.join('..', 'Reference','vims_2_abril.xlsx')

# file__path_vims = os.path.join('..', 'Reference','vims_2_abril.xlsx')
        # 
# Match_Cicles(file__path,file__path_vims,sheet_names)

# Run february

sheet_names = ['2023-02-28','2023-03-01']
file__path_vims = os.path.join('..', 'Reference','vims_28_feb.xlsx')
file__path = os.path.join('..', 'Reference', 'Strain Measurements 28.xlsx')
Match_Cicles(file__path,file__path_vims,sheet_names)

def read_location_data(file_name):
    file_path = os.path.join('..', 'Reference', file_name)
    df = pd.read_excel(file_path, engine='openpyxl', parse_dates=['Timestamp'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

# location_file =  os.path.join('..', 'Reference', 'GPS_Cluster_second_phase.xlsx')
# location_data = read_location_data(location_file )

def assign_location_id(file__path, location_data, sheet_names):
    strain_measurements = pd.read_excel(file__path, sheet_name=sheet_names, parse_dates=True)
    
    for sheet_name, df in strain_measurements.items():
        df['Date'] = pd.to_datetime(df['Date'])
        df['location_id'] = ''  # Create a new column 'location_id' in the strain_measurements DataFrame

        for i in range(len(location_data) - 1):
            time_start, time_end = location_data.iloc[i]['Timestamp'], location_data.iloc[i + 1]['Timestamp']
            location_id = location_data.iloc[i]['location_id']
            mask = (df['Date'] >= time_start) & (df['Date'] < time_end)
            df.loc[mask, 'location_id'] = location_id

        out_file = os.path.join('..', 'Reference', f"Strain_Measurements_GPS_modified_{sheet_name}.xlsx")
        df.to_excel(out_file, index=False)

# sheet_names = ['2023-04-06', '2023-04-07', '2023-04-08', '2023-04-10', '2023-04-13', '2023-04-14', '2023-04-15', '2023-04-16', '2023-04-17', '2023-04-18', '2023-04-19']
# file__path = os.path.join('..', 'Reference', 'J298MCL002 - Strain measurements second setup.xlsx')

# assign_location_id(file__path, location_data, sheet_names)


# folder_path = os.path.join('..', 'Reference','GPS')

# output_file = 'Strain_Measurements_GPS_modified_set_up2.xlsx'# os.path.join('..', 'Reference', 'GPS',)

def combine_excel_w_location_id(folder_path ,output_file):
    # List all files in the folder
    all_files = os.listdir(folder_path)
    
    # Filter out non-Excel files
    excel_files = [file for file in all_files if file.endswith('.xlsx') or file.endswith('.xls')]
    
    # Create a new Excel file to store the combined data
    writer = pd.ExcelWriter(os.path.join(folder_path, output_file), engine='openpyxl')
    
    # Read and write each Excel file's sheet(s) to the new Excel file
    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        
        # Read the Excel file
        xls = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
        
        # Write each sheet to the new Excel file
        for sheet_name, df in xls.items():
            # Create a unique sheet name by appending the file name to avoid conflicts
            new_sheet_name = sheet_name
            df.to_excel(writer, sheet_name=new_sheet_name, index=False)
    
    # Save the new Excel file
    writer.save()
    

# combine_excel_w_location_id(folder_path ,output_file)