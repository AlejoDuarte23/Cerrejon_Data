# import datetime
# import glob
# import re

# def process_file(input_file, output_file):
#     with open(input_file, "r") as f:
#         lines = f.readlines()

#     with open(output_file, "w") as f:
#         f.write("Date, Gauge 111, Gauge 112, Gauge 113, Gauge 331,Gauge 332,Gauge 333, Gauge 441,Gauge 442,Gauge 443,Gauge 991,Gauge 992,Gauge 993,Gauge 1011, Gauge 1012, Gauge 1013\n")

#         for line in lines[1:]:
#             if line.strip() == "}":
#                 break

#             date, time, *values = line.split(",")

#             dt = datetime.datetime.strptime(date + " " + time, "%d/%m/%Y %H:%M:%S.%f")
#             formatted_dt = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

#             f.write(f"{formatted_dt},{','.join(values)}")

# # Find all matching CSV files in the directory
# input_files = glob.glob("Set up 1 Cerrejon _*.csv")

# for input_file in input_files:
#     idd = re.search(r"_([\d]+)", input_file).group(1)
#     output_file = f"resampled_data_1 Cerrejon _{idd.zfill(4)}.dxd_rosette_1.csv"
#     process_file(input_file, output_file)

# import os
# import pandas as pd

# # set the input file name
# input_filename = 'Set up 1 Cerrejon _0013.csv'

# # read the input CSV file into a pandas dataframe
# df = pd.read_csv(input_filename, sep=',')

# # combine the 'Date' and 'Time (-)' columns into a single 'Datetime' column
# df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time (-)'])

# # extract the relevant columns and rename them
# df = df[['Datetime', 'AI A-1; AVE (um/m)', 'AI A-2; AVE (um/m)', 'AI A-3; AVE (um/m)', 'AI B-1; AVE (um/m)', 'AI B-2; AVE (um/m)', 'AI B-3; AVE (um/m)']]
# df = df.rename(columns={'AI A-1; AVE (um/m)': 'Gauge 111', 'AI A-2; AVE (um/m)': 'Gauge 112', 'AI A-3; AVE (um/m)': 'Gauge 113', 'AI B-1; AVE (um/m)': 'Gauge 331', 'AI B-2; AVE (um/m)': 'Gauge 332', 'AI B-3; AVE (um/m)': 'Gauge 333'})

# # format the datetime column as ISO 8601
# df['Datetime'] = df['Datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')

# # extract the filename prefix and suffix
# filename_prefix, filename_suffix = os.path.splitext(input_filename)

# # construct the output filename using the filename prefix
# filename_parts = filename_prefix.split('_')
# if len(filename_parts) >= 3:
#     output_filename = f"resampled_data_{filename_parts[2].replace(' ', '_')}.dxd_rosette_1.csv"
# else:
#     output_filename = "resampled_data.dxd_rosette_1.csv"

# # export the modified dataframe to a new CSV file
# df.to_csv(output_filename, index=False)

# import os
# import glob
# import pandas as pd

# # find all matching CSV files in the directory
# input_files = glob.glob("Set up 1 Cerrejon _*.csv")

# for input_filename in input_files:
#     print(f"Processing file {input_filename}...")

#     # read the input CSV file into a pandas dataframe
#     df = pd.read_csv(input_filename, sep=',')

#     # combine the 'Date' and 'Time (-)' columns into a single 'Datetime' column
#     df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time (-)'])

#     # extract the relevant columns and rename them
#     df = df[['Datetime', 'AI A-1; AVE (um/m)', 'AI A-2; AVE (um/m)', 'AI A-3; AVE (um/m)', 'AI B-1; AVE (um/m)', 'AI B-2; AVE (um/m)', 'AI B-3; AVE (um/m)']]
#     df = df.rename(columns={'AI A-1; AVE (um/m)': 'Gauge 111', 'AI A-2; AVE (um/m)': 'Gauge 112', 'AI A-3; AVE (um/m)': 'Gauge 113', 'AI B-1; AVE (um/m)': 'Gauge 331', 'AI B-2; AVE (um/m)': 'Gauge 332', 'AI B-3; AVE (um/m)': 'Gauge 333'})

#     # format the datetime column as ISO 8601
#     df['Datetime'] = df['Datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')

#     # export the modified dataframe to a new CSV file with the same structure as the input filename
#     output_filename_parts = input_filename.split('_')
#     if len(output_filename_parts) >= 3:
#         output_filename = 'resampled_data_' + output_filename_parts[2].replace(' ', '_') + '.dxd_rosette_1.csv'
#         df.to_csv(output_filename, index=False, float_format='%.6f')
#     else:
#         print(f"Could not generate output filename for {input_filename}")

# # import os
# # import pandas as pd
# # import glob
# # import re

# # # Find all matching CSV files in the directory
# # input_files = glob.glob("Set up 1 Cerrejon _*.csv")

# # # Loop through each input file
# # for input_file in input_files:

# #     # Read the input CSV file into a pandas dataframe
# #     df = pd.read_csv(input_file, sep=',')

# #     # Combine the 'Date' and 'Time (-)' columns into a single 'Date' column
# #     df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time (-)'])

# #     # Extract the relevant columns and rename them
# #     df = df[['Date', 'AI A-1; AVE (um/m)', 'AI A-2; AVE (um/m)', 'AI A-3; AVE (um/m)', 'AI B-1; AVE (um/m)', 'AI B-2; AVE (um/m)', 'AI B-3; AVE (um/m)']]
# #     df = df.rename(columns={'AI A-1; AVE (um/m)': 'Gauge 111', 'AI A-2; AVE (um/m)': 'Gauge 112', 'AI A-3; AVE (um/m)': 'Gauge 113', 'AI B-1; AVE (um/m)': 'Gauge 331', 'AI B-2; AVE (um/m)': 'Gauge 332', 'AI B-3; AVE (um/m)': 'Gauge 333'})

# #     # Format the Date column as ISO 8601
# #     df['Date'] = df['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')

# #     # Export the modified dataframe to a new CSV file
# #     idd = re.search(r"_([\d]+)", input_file).group(1)
# #     output_file = f"resampled_data_1 Cerrejon _{idd.zfill(4)}.dxd_rosette_1.csv"
# #     df.to_csv(output_file, index=False)


# import os
# import pandas as pd
# import glob
# import re

# # Find all matching CSV files in the directory
# input_files = glob.glob("Set up 1 Cerrejon _*.csv")

# # Loop through the input files and process each one
# for input_filename in input_files:

#     # read the input CSV file into a pandas dataframe
#     df = pd.read_csv(input_filename, sep=',')

#     df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time (-)'])
#     df = df.drop(['Time (-)'], axis=1)
    
#     # rename relevant columns
#     df = df.rename(columns={'AI A-1; AVE (um/m)': 'Gauge 221', 
#                             'AI A-2; AVE (um/m)': 'Gauge 222', 
#                             'AI A-3; AVE (um/m)': 'Gauge 223', 
#                             'AI A-4; AVE (um/m)': 'Gauge 551', 
#                             'AI A-5; AVE (um/m)': 'Gauge 552', 
#                             'AI A-6; AVE (um/m)': 'Gauge 553',
#                             'AI A-7; AVE (um/m)': 'Gauge 771',
#                             'AI A-8; AVE (um/m)': 'Gauge 772',
#                             'AI B-1; AVE (um/m)': 'Gauge 773',
#                             'AI B-2; AVE (um/m)': 'Gauge 881',
#                             'AI B-3; AVE (um/m)': 'Gauge 882',
#                             'AI B-4; AVE (um/m)': 'Gauge 883',
#                             'AI B-5; AVE (um/m)': 'Gauge 661',
#                             'AI B-6; AVE (um/m)': 'Gauge 662',
#                             'AI B-7; AVE (um/m)': 'Gauge 663'})
    
# #     # extract the relevant columns 
#     df = df[['Date', 'Gauge 221', 'Gauge 222', 'Gauge 223', 'Gauge 551', 
#               'Gauge 552', 'Gauge 553', 'Gauge 771', 'Gauge 772', 'Gauge 773', 
#               'Gauge 881', 'Gauge 882', 'Gauge 883', 'Gauge 661', 'Gauge 662', 
#               'Gauge 663']]

#     # format the datetime column as ISO 8601
#     df['Date'] = df['Date'].dt.strftime('%Y-%d-%mT%H:%M:%S.%f')
    
#     # export the modified dataframe to a new CSV file
#     idd = re.search(r"_([\d]+)", input_filename).group(1)
#     output_filename = f"resampled_data_1 Cerrejon _{idd.zfill(4)}.dxd_rosette_1.csv"
#     df.to_csv(output_filename, index=False)
# import os
# import pandas as pd
# import glob
# import re

# # Find all matching CSV files in the directory
# input_files = glob.glob("Set up 1 Cerrejon _*.csv")

# # Loop through the input files and process each one
# for input_filename in input_files:

#     # read the input CSV file into a pandas dataframe
#     df = pd.read_csv(input_filename, sep=',')

#     df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time (-)'])
#     df = df.drop(['Time (-)'], axis=1)
    
#     # rename relevant columns
#     df = df.rename(columns={'AI A-1; AVE (um/m)': 'Gauge 221', 
#                             'AI A-2; AVE (um/m)': 'Gauge 222', 
#                             'AI A-3; AVE (um/m)': 'Gauge 223', 
#                             'AI A-4; AVE (um/m)': 'Gauge 551', 
#                             'AI A-5; AVE (um/m)': 'Gauge 552', 
#                             'AI A-6; AVE (um/m)': 'Gauge 553',
#                             'AI A-7; AVE (um/m)': 'Gauge 771',
#                             'AI A-8; AVE (um/m)': 'Gauge 772',
#                             'AI B-1; AVE (um/m)': 'Gauge 773',
#                             'AI B-2; AVE (um/m)': 'Gauge 881',
#                             'AI B-3; AVE (um/m)': 'Gauge 882',
#                             'AI B-4; AVE (um/m)': 'Gauge 883',
#                             'AI B-5; AVE (um/m)': 'Gauge 661',
#                             'AI B-6; AVE (um/m)': 'Gauge 662',
#                             'AI B-7; AVE (um/m)': 'Gauge 663'})

#     # extract the relevant columns 
#     df = df[['Date', 'Gauge 221', 'Gauge 222', 'Gauge 223', 'Gauge 551', 
#               'Gauge 552', 'Gauge 553', 'Gauge 771', 'Gauge 772', 'Gauge 773', 
#               'Gauge 881', 'Gauge 882', 'Gauge 883', 'Gauge 661', 'Gauge 662', 
#               'Gauge 663']]

#     # format the datetime column as ISO 8601
#     df['Date'] = df['Date'].dt.strftime('%Y-%d-%mT%H:%M:%S.%f')
    
#     # export the modified dataframe to a new CSV file
#     idd = re.search(r"_([\d]+)", input_filename).group(1)
#     output_filename = f"resampled_data_1 Cerrejon _{idd.zfill(4)}.dxd_rosette_1.csv"
#     df.to_csv(output_filename, index=False, sep=', ')  # Add a space after the comma in the sep parameter


# # "Date, Gauge 221, Gauge 222, Gauge 223, Gauge 551,Gauge 552,Gauge 553, Gauge 771,Gauge 772,Gauge 773,Gauge 881,Gauge 882,Gauge 883,Gauge 661, Gauge 662, Gauge 663"
# #  este es 
import os
import pandas as pd
import glob
import re

# Find all matching CSV files in the directory
input_files = glob.glob("Set up 1 Cerrejon _*.csv")

# Loop through the input files and process each one
for input_filename in input_files:

    df = pd.read_csv(input_filename, sep=',')

    df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time (-)'])
    df = df.drop(['Time (-)'], axis=1)
    
    
    # rename relevant columns
    df = df.rename(columns={'AI A-1; AVE (um/m)': ' Gauge 221', 
                            'AI A-2; AVE (um/m)': ' Gauge 222', 
                            'AI A-3; AVE (um/m)': ' Gauge 223', 
                            'AI A-4; AVE (um/m)': ' Gauge 551', 
                            'AI A-5; AVE (um/m)': 'Gauge 552', 
                            'AI A-6; AVE (um/m)': 'Gauge 553',
                            'AI A-7; AVE (um/m)': ' Gauge 771',
                            'AI A-8; AVE (um/m)': 'Gauge 772',
                            'AI B-1; AVE (um/m)': 'Gauge 773',
                            'AI B-2; AVE (um/m)': 'Gauge 881',
                            'AI B-3; AVE (um/m)': 'Gauge 882',
                            'AI B-4; AVE (um/m)': 'Gauge 883',
                            'AI B-5; AVE (um/m)': 'Gauge 661',
                            'AI B-6; AVE (um/m)': ' Gauge 662',
                            'AI B-7; AVE (um/m)': ' Gauge 663'})

    # extract the relevant columns 
    df = df[['Date', ' Gauge 221', ' Gauge 222', ' Gauge 223', ' Gauge 551', 
              'Gauge 552', 'Gauge 553', ' Gauge 771', 'Gauge 772', 'Gauge 773', 
              'Gauge 881', 'Gauge 882', 'Gauge 883', 'Gauge 661', ' Gauge 662', 
              ' Gauge 663']]
    # format the datetime column as ISO 8601
    df['Date'] = df['Date'].dt.strftime('%Y-%d-%mT%H:%M:%S.%f')
    # Add a space after the comma in the column names
    # df.columns = [col.replace(',', ', ') for col in df.columns]


    
    # export the modified dataframe to a new CSV file
    idd = re.search(r"_([\d]+)", input_filename).group(1)
    output_filename = f"resampled_data_1 Cerrejon _{idd.zfill(4)}.dxd_rosette_1.csv"
    df.to_csv(output_filename, index=False)






# import os
# import pandas as pd
# import glob
# import re

# # Find all matching CSV files in the directory
# input_files = glob.glob("Set up 1 Cerrejon _*.csv")

# # Loop through the input files and process each one
# for input_filename in input_files:

#     # read the input CSV file into a pandas dataframe
#     df = pd.read_csv(input_filename, sep=',')

#     df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time (-)'])
#     df = df.drop(['Time (-)'], axis=1)
    
#     # rename relevant columns
#     df = df.rename(columns={'AI A-1; AVE (um/m)': 'Gauge 221', 
#                             'AI A-2; AVE (um/m)': 'Gauge 222', 
#                             'AI A-3; AVE (um/m)': 'Gauge 223', 
#                             'AI A-4; AVE (um/m)': 'Gauge 551', 
#                             'AI A-5; AVE (um/m)': 'Gauge 552', 
#                             'AI A-6; AVE (um/m)': 'Gauge 553',
#                             'AI A-7; AVE (um/m)': 'Gauge 771',
#                             'AI A-8; AVE (um/m)': 'Gauge 772',
#                             'AI B-1; AVE (um/m)': 'Gauge 773',
#                             'AI B-2; AVE (um/m)': 'Gauge 881',
#                             'AI B-3; AVE (um/m)': 'Gauge 882',
#                             'AI B-4; AVE (um/m)': 'Gauge 883',
#                             'AI B-5; AVE (um/m)': 'Gauge 661',
#                             'AI B-6; AVE (um/m)': 'Gauge 662',
#                             'AI B-7; AVE (um/m)': 'Gauge 663'})
    
#     # extract the relevant columns 
#     df = df[['Date', 'Gauge 221', 'Gauge 222', 'Gauge 223', 'Gauge 551', 
#               'Gauge 552', 'Gauge 553', 'Gauge 771', 'Gauge 772', 'Gauge 773', 
#               'Gauge 881', 'Gauge 882', 'Gauge 883', 'Gauge 661', 'Gauge 662', 
#               'Gauge 663']]

#     # format the datetime column as ISO 8601
#     df['Date'] = df['Date'].dt.strftime('%Y-%d-%mT%H:%M:%S.%f')
    
#     # export the modified dataframe to a new CSV file
#     idd = re.search(r"_([\d]+)", input_filename).group(1)
#     output_filename = f"resampled_data_1 Cerrejon _{idd.zfill(4)}.dxd_rosette_1.csv"
#     df.to_csv(output_filename, index=False)














# import os
# import pandas as pd
# import glob
# import re

# # Find all matching CSV files in the directory
# input_files = glob.glob("Set up 1 Cerrejon _*.csv")

# # Loop through the input files and process each one
# for input_filename in input_files:

#     # read the input CSV file into a pandas dataframe
#     df = pd.read_csv(input_filename, sep=',')

#     df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time (-)'])
#     df = df.drop(['Time (-)'], axis=1)
    
#     rename relevant columns
#     df = df.rename(columns={'AI A-1; AVE (um/m)': 'Gauge 221', 
#                             'AI A-2; AVE (um/m)': 'Gauge 222', 
#                             'AI A-3; AVE (um/m)': 'Gauge 223', 
#                             'AI A-4; AVE (um/m)': 'Gauge 551', 
#                             'AI A-5; AVE (um/m)': 'Gauge 552', 
#                             'AI A-6; AVE (um/m)': 'Gauge 553',
#                             'AI A-7; AVE (um/m)': 'Gauge 771',
#                             'AI A-8; AVE (um/m)': 'Gauge 772',
#                             'AI B-1; AVE (um/m)': 'Gauge 773',
#                             'AI B-2; AVE (um/m)': 'Gauge 881',
#                             'AI B-3; AVE (um/m)': 'Gauge 882',
#                             'AI B-4; AVE (um/m)': 'Gauge 883',
#                             'AI B-5; AVE (um/m)': 'Gauge 661',
#                             'AI B-6; AVE (um/m)': 'Gauge 662',
#                             'AI B-7; AVE (um/m)': 'Gauge 663'})
    
#     # extract the relevant columns 
#     df = df[['Date', 'Gauge 221', 'Gauge 222', 'Gauge 223', 'Gauge 551', 
#               'Gauge 552', 'Gauge 553', 'Gauge 771', 'Gauge 772', 'Gauge 773', 
#               'Gauge 881', 'Gauge 882', 'Gauge 883', 'Gauge 661', 'Gauge 662', 
#               'Gauge 663']]

#     # format the datetime column as ISO 8601
#     df['Date'] = df['Date'].dt.strftime('%Y-%d-%mT%H:%M:%S.%f')
    
#     # export the modified dataframe to a new CSV file
#     idd = re.search(r"_([\d]+)", input_filename).group(1)
#     output_filename = f"resampled_data_1 Cerrejon _{idd.zfill(4)}.dxd_rosette_1.csv"
#     df.to_csv(output_filename, index=False)














# import pandas as pd
# import numpy as np

# # Load the data from the Excel file
# data = pd.read_csv('resampled_data_1 Cerrejon _0097.dxd_rosette_1.csv')

# # Create a new DataFrame with hourly timestamps
# start_time = pd.to_datetime(data.iloc[0, 0])
# end_time = pd.to_datetime(data.iloc[-1, 0])
# new_index = pd.date_range(start=start_time, end=end_time, freq='1H')
# new_data = pd.DataFrame(index=new_index)

# # Interpolate the data for each column
# for col in data.columns[1:]:
#     values = data[col].values
#     # Use linear interpolation
#     interp_func = np.interp(new_data.index.values, data.iloc[:, 0].values.astype(float), values.astype(float))
#     new_data[col] = interp_func

# # Write the interpolated data to a new Excel file
# new_data.index.name = 'Date'
# new_data.to_excel('interpolated_data.xlsx')
