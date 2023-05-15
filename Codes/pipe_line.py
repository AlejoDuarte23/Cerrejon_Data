# %% General porpous pickle 
import numpy as np 
import pickle

#%% Dewesoft Imports 
from win32com.client import Dispatch
import sys
import time
import pandas as pd 
import pickle

from sklearn.cluster import DBSCAN
import numpy as np


#%% Get Data frame from DXD
# Start Dewesoft 

try: 
    with open("data.pickle", "rb") as file:
        data = pickle.load(file)
        
except:
    dw = Dispatch("Dewesoft.App")
    sys.stdout.flush()
    dw.Init()
    dw.Enabled = 1
    dw.Visible = 1
    dw.Top = 0
    dw.Left = 0
    dw.Width = 1024 
    dw.Height = 768
    
    # Import function from dw_read_dxd.py 
    from dw_read_dxd import  file_nameautomate , datasections_engine, get_alldata
    from dw_read_dxd import  resample_data
    # Route to the dxd files 
    _dir = r"C:\Users\aleja\Documents\Cerrejon Data Analyis\dxd_files\06.04.2023_dxd_files\Set up 1 Cerrejon _00{}.dxd"
    #  Generate 71 dxd file name to be read by dewesoft 
    a,b = 1,72
    file_names = file_nameautomate(_dir, a, b)
    #  Get channels for rossete 10
    channels  = [12,13,14]
    data = np.array([],
                    dtype=[('timestamp', '<M8[ns]'),
                           ('data0', '<f8'), ('data1', '<f8'), ('data2', '<f8')])
    
    for i in range(len(file_names)):
        try:
            #  Loading dxd file into dw 
            dw.LoadFile(file_names[i])
            #  Exporting the data from dw and resample the data into a df 
            data_section = datasections_engine()
            stacked_data =  get_alldata(channels,data_section)
            data_rs1 = resample_data(stacked_data, 1000, 1)
            data = np.append(data,data_rs1)
        except:
            print(file_names[i], "not working ")
    #%%  save data in a pickle 
    with open("data.pickle", "wb") as file:
        pickle.dump(data, file)
#%% detrend data 
from _preprocesing import detrend_data,convert_to_df,get_rosette_data, rosette_info_engine,solve_strain_system,plot_principal_stresses
dtrend_array = detrend_data(data)
#%% Convert to df 
df = convert_to_df(dtrend_array)

# %% Preprocesing 
rosette_name = 'ROSETTE10'
columns = ['data0', 'data1', 'data2']
_rosette_data = rosette_info_engine()

rosette_data = _rosette_data[rosette_name]
#  Calculate stress 
sigmas = solve_strain_system(df, columns, rosette_data['theta1'],
                              rosette_data['theta2'],
                              rosette_data['theta3'],
                              E=210000,
                              v=0.285,
                              G=0.37619047*210000,
                              Kt = rosette_data['Kt'])
# formating the results 
result_df = pd.DataFrame({
    'Timestamp': df['timestamp'],
    'sigma_1': sigmas[0],
    'sigma_2': sigmas[1],
    'tau' : sigmas[2],
    'theta_p': sigmas[3]
})

# plot_principal_stresses(result_df, 'sigma_1', 'sigma_2','tau','theta_p')
from _preprocesing import  merge_dataframes,plot_principal_stresses_clustered

# rosette_data = get_rosette_data(rosette_name,Set_up1)


main_df = result_df
slave_df = 'Vims_clustered.xlsx'
main_ts_col = 'Timestamp'
slave_ts_col = 'ReadTime'
slave_cols_interest = ['Left Front Suspension Cylinder', 'Left Rear Suspension Cylinder', 'Payload',
                        'Right Front Suspension Cylinder', 'Right Rear Suspension Cylinder', 'Cycle', 'Cluster','Ground Speed']
output_df  = merge_dataframes(main_df, slave_df, main_ts_col, slave_ts_col, slave_cols_interest)


# output_df = merge_dataframes(main_df, slave_df, main_ts_col, slave_ts_col, slave_cols_interest)
# # with open("parse_data.pickle", "wb") as file:
# #     pickle.dump(output_df, file)
# # plot_principal_stresses_clustered(output_df, 'sigma_1', 'sigma_2', 'tau', 'theta_p')


# #%% adding cords to df 
# with open("parse_data.pickle", "rb") as file:
#     data_parsed=  pickle.load(file)
sw = 7890
if  sw == 789:
# ##############################################
    main_df =  result_df#data_parsed
    slave_df = 'cluster_mincka_way.xlsx'
    main_ts_col = 'Timestamp'
    slave_ts_col = 'Timestamp'
    slave_cols_interest = ['Cluster','PositionX','PositionY','Cycle']
    output_df_cords  = merge_dataframes(main_df, slave_df, main_ts_col, slave_ts_col, slave_cols_interest)
    #  ####################################
    from map_cerrejon_with_geopandas import plot_specific_subcluster_bycycle
    from _preprocesing import  plot_cycle_data2
    cols = [ 'PositionX']
    units = [ '-']
    plot_cycle_data2(output_df_cords, 530, cols, units)

main_df =  output_df  
slave_df = 'fd_subcluster_output.xlsx'
main_ts_col = 'Timestamp'
slave_ts_col = 'Timestamp'
slave_cols_interest = ['PositionX','PositionY']
output_df_cords  = merge_dataframes(main_df, slave_df, main_ts_col, slave_ts_col, slave_cols_interest)
#  ####################################
from map_cerrejon_with_geopandas import plot_specific_subcluster_bycycle
from _preprocesing import  plot_cycle_data2
cols = [ 'PositionX','Payload','Ground Speed']
units = [ '-','Ton','-']
plot_cycle_data2(output_df_cords, 447, cols, units)

# Use the function for a specific cycle
# ####################################################
# plot_cycle_data(output_df_cords, 478, cols, units)

# def adjust_cluster_ids(df):
#     eps = 0.001

#     for cycle in df['Cycle'].unique():
#         cycle_data = df[df['Cycle'] == cycle].copy()

#         group1 = cycle_data[cycle_data['Cluster'].isin(['travelling_empty', 'loading_process'])]
#         group2 = cycle_data[cycle_data['Cluster'].isin(['travelling_full', 'dumping_process'])]
    
#         for group in [group1, group2]:
#             # Drop rows containing NaN values in the 'PositionX' and 'PositionY' columns
#             group = group.dropna(subset=['PositionX', 'PositionY'])

#             dbscan = DBSCAN(eps=eps)
#             labels = dbscan.fit_predict(group[['PositionX', 'PositionY']])

#             main_label = np.argmax(np.bincount(labels))

#             is_main_area = (labels == main_label)
#             if 'loading_process' in group['Cluster'].values:
#                 group.loc[is_main_area, 'Cluster'] = 'loading_process'
#                 group.loc[~is_main_area, 'Cluster'] = 'travelling_full'
#             else:
#                 group.loc[is_main_area, 'Cluster'] = 'dumping_process'
#                 group.loc[~is_main_area, 'Cluster'] = 'travelling_empty'

#             df.update(group)

#     return df


# array([429, 430, 431, 444, 445, 446, 447, 448, 449, 450, 452, 453, 454,
#        455, 459, 460, 461, 462, 463, 464, 465, 467, 468, 469, 471, 472,
#        473, 474, 475, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486,
#        487, 488, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500,
#        503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515,
#        516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528,
#        529, 530, 531, 532, 533, 556, 557, 686, 687, 688], dtype=object)