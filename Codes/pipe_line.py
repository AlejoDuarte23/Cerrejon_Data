# %% General porpous pickle 
import numpy as np 
import pickle

#%% Dewesoft Imports 
from win32com.client import Dispatch
import sys
import time

#%% Get Data frame from DXD
# Start Dewesoft 
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
from _preprocesing import detrend_data,convert_to_df,get_rosette_data, rosette_info_engine,solve_strain_system
dtrend_array = detrend_data(data)
#%% Convert to df 
df = convert_to_df(data)

# %% Preprocesing 
rosette_name = 'ROSETTE10'
columns = ['data0', 'data1', 'data2']
_rosette_data = rosette_info_engine()

rosette_data = _rosette_data[rosette_name]

sigmas = solve_strain_system(df, columns, rosette_data['theta1'],
                              rosette_data['theta2'],
                              rosette_data['theta3'],
                              E=210000,
                              v=0.285,
                              G=0.37619047*210000,
                              Kt = rosette_data['Kt'])


result_df = pd.DataFrame({
    'Timestamp': df['timestamp'],
    'sigma_1': sigmas[0],
    'sigma_2': sigmas[1],
    'tau' : sigmas[2],
    'theta_p': sigmas[3]
})


# rosette_data = get_rosette_data(rosette_name,Set_up1)


