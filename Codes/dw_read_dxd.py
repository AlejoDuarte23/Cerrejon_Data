from win32com.client import Dispatch
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
import psycopg2
import pandas  as pd
from tqdm import tqdm
from sqlalchemy import create_engine
from scipy.signal import resample

import math 
 
def stored_time_engine():
    stored_time = dw.Data.StartStoreTime
    dt = datetime(stored_time.year, stored_time.month,
                  stored_time.day, stored_time.hour,
                  stored_time.minute, stored_time.second,
                  stored_time.microsecond, stored_time.tzinfo)
    dt64 = np.datetime64(dt)
    time_delta_seconds = dw.Eventlist.Item(0).TimeStamp
    time_delta_ns = np.timedelta64(int(time_delta_seconds * 1e9), 'ns')
    dt64 = dt64 + time_delta_ns

    return dt64
    

def time_stamps_engine(dt64):
    total_samples = dw.LoadEngine.DataSections.Item(0).DataCount
    fs = dw.Data.SampleRate
    interval_ns = int(1e9 / fs)
    timestamps = np.arange(dt64, dt64 + np.timedelta64(interval_ns * total_samples, 'ns'), np.timedelta64(interval_ns, 'ns'))
    # timestamps = timestamps.reshape(-1, 1)
    return timestamps


def datasections_engine():
    data_sections = dw.LoadEngine.DataSections
    data_section = data_sections.Item(0)
    return data_section

def get_channel_data(c_id,data_section):
    ch = dw.Data.UsedChannels.Item(c_id)
    data , _ =data_section.ReadData(ch)
    return data

def time_stamp_engine_array():
    _dt64 = stored_time_engine()
    _timestamps = time_stamps_engine(_dt64)
    return _timestamps

def get_channelwtime(c_id):

    _timestamps = time_stamp_engine_array()
    data_section = datasections_engine()
    _data = get_channel_data(c_id,data_section)
    _data = np.array(_data)
    stacked_data = np.empty(len(_data), dtype=[('timestamp', 'datetime64[ns]'), ('data', 'float64')])
    stacked_data['timestamp'] = _timestamps
    stacked_data['data'] = _data
    return stacked_data


def get_alldata(selected_chanels,data_section):
    num_columns = len(selected_chanels)
    _data = []
    _timestamps = time_stamp_engine_array()
    for i in range(num_columns):
        _datai = get_channel_data(selected_chanels[i-1],data_section)
        _data.append(_datai)
    dtype = [('timestamp', 'datetime64[ns]')] + [('data' + str(i), 'float64') for i in range(num_columns)]
    stacked_data = np.empty(len(_datai), dtype=dtype)
    stacked_data['timestamp'] = _timestamps
    for i, _datai in enumerate(_data):
        data_array = np.array(_datai)
        stacked_data['data' + str(i)] = data_array
    return stacked_data


def ploting_chanels(selected_chanels,stacked_data):
   
    plt.figure(figsize=(3*1080/300, 3*720/300))
    timestamps = stacked_data['timestamp']
    for i in range(len(selected_chanels)):
        data = stacked_data['data' + str(i)]
        plt.plot(timestamps, data, label=f'Data {i+1}')
        plt.legend()
        plt.xlabel('TimeStamp')
        plt.ylabel('Strain ("\u00B5")m/m')
        
def file_nameautomate(_dir,a,b):
    # a initial  dxd file id
    # b last dxd file id
    file_names = []
    base_file_path = _dir
    numbers = range(a,b)    
    for num in numbers:
        file_path = base_file_path.format(str(num).zfill(2))    
        file_names.append(file_path)
    return file_names

def resample_data(data, old_fs, new_fs):
    num_columns = len(data.dtype.names) - 1
    old_len = len(data)
    new_len = int(old_len * new_fs / old_fs)
    resampled_data = np.zeros((new_len, num_columns))
    
    for i in range(num_columns):
        resampled_data[:, i] = resample(data['data' + str(i)], new_len)
    
    resampled_timestamps = pd.date_range(start=data['timestamp'][0], periods=new_len, freq=f'{1 / new_fs * 1e6:.0f}U')
    dtype = [('timestamp', 'datetime64[ns]')] + [('data' + str(i), 'float64') for i in range(num_columns)]
    resampled_stacked_data = np.empty(len(resampled_data), dtype=dtype)
    resampled_stacked_data['timestamp'] = resampled_timestamps
    for i in range(num_columns):
        resampled_stacked_data['data' + str(i)] = resampled_data[:, i]
    
    return resampled_stacked_data


def plot_all_measurements():

    _file_names = file_nameautomate()
    rosettes = [[1,2,3],[4,5,6],[7,8,9],[10,11,11],[13,14,15]]  
    # rosettes = [[1,2,3]]  
    
    for j in range(len(_file_names)):
        try:
            dw.LoadFile(_file_names[j])
            time.sleep(15)
        
            for i in range(len(rosettes)):
                selected_chanels = rosettes[i]
                data_section = datasections_engine()
                data_123 = get_alldata(selected_chanels,data_section)
                resampled_data_123 = resample_data(data_123, 1000, 50)
                np.savetxt(f"resampled_data_{_file_names[j][-20:]}_rosette_{i}.txt", resampled_data_123, fmt="%s", delimiter=",")
                ploting_chanels(selected_chanels, resampled_data_123)
                title_fid = 'Gauges ' + str(selected_chanels) + _file_names[j][-20:]
                plt.title('Gauges ' + str(selected_chanels) + _file_names[j][-20:])
                plt.savefig(title_fid+'.png', dpi=300)
                plt.close('all')
		
        except:
                print('skiped',' _file_names[j]')

    return data_123


def plt_all_measurements2():
    _file_names = file_nameautomate()
    rosettes = [[1,2,3],[4,5,6],[7,8,9],[10,11,11],[13,14,15]]  
 # rosettes = [[1,2,3]]  
 
    for j in range(len(_file_names)):
        dw.LoadFile(_file_names[j])
        time.sleep(10)
        try:
            for i in range(len(rosettes)):
                
                selected_chanels = rosettes[i]
                data_section = datasections_engine()
                data_123 = get_alldata(selected_chanels,data_section)
                resampled_data_123 = resample_data(data_123, 1000, 1)
                _header = ["Date, Gauge 111, Gauge 112, Gauge 113, Gauge 331,Gauge 332,Gauge 333, Gauge 441,Gauge 442,Gauge 443"]
                
                output_path = folder_output + f"resampled_data_{_file_names[j][-20:]}_rosette_{i+1}.csv"

                np.savetxt(output_path, resampled_data_123,
                           fmt="%s", delimiter=",", header=_header, comments="")
                ploting_chanels(selected_chanels, resampled_data_123)
                title_fid = 'Gauges ' + str(selected_chanels) + _file_names[j][-20:]
                plt.title('Gauges ' + str(selected_chanels) + _file_names[j][-20:])
                plt.savefig(title_fid+'.png', dpi=300)
                plt.close('all')
            
        except:
                print('skiped',' _file_names[j]')
 

def plt_all_measurements3(file_names,folder_output, PLT = False):
    _file_names = file_names
    rosettes = [[1,2,3,4,5,6,7,8,9,10,11,11,13,14,15]]  
    for j in range(len(_file_names)):
        dw.LoadFile(_file_names[j])
        time.sleep(10)
        try:
            for i in range(len(rosettes)):
                
                selected_chanels = rosettes[i]
                data_section = datasections_engine()
                data_123 = get_alldata(selected_chanels,data_section)
                resampled_data_123 = resample_data(data_123, 1000, 1)
                resampled_data_123['data12'] *= math.sqrt(2)/2
                # plt.plot(resampled_data_123)
                
                output_path = folder_output + f"resampled_data_{_file_names[j][-20:]}_rosette_{i+1}.csv"
                _header = "Date, Gauge 111, Gauge 112, Gauge 113, Gauge 331,Gauge 332,Gauge 333, Gauge 441,Gauge 442,Gauge 443,Gauge 991,Gauge 992,Gauge 993,Gauge 1011, Gauge 1012, Gauge 1013"
                np.savetxt(output_path, resampled_data_123,
                           fmt="%s", delimiter=",", header=_header, comments="")
                if PLT == True:    
                    ploting_chanels(selected_chanels, resampled_data_123)
                    title_fid = 'Gauges ' + str(selected_chanels) + _file_names[j][-20:]
                    plt.title('Gauges ' + str(selected_chanels) + _file_names[j][-20:])
                    plt.savefig(title_fid+'.png', dpi=300)
                    plt.close('all')
            
        except:
                output_path = folder_output + f"resampled_data_{_file_names[0][-20:]}_rosette_{i+1}.csv"
                print(output_path)
 



def BB(dump):
    columns = ['Timestamp', 'AI A-1', 'AI A-2', 'AI A-3', 'AI A-4', 'AI A-5', 'AI A-6', 'AI A-7', 'AI A-8', 'AI B-1', 'AI B-2', 'AI B-3', 'AI B-4', 'AI B-5', 'AI B-6', 'AI B-7']
    df = pd.DataFrame.from_records(dump, columns=columns)
    df.columns = columns
    conn = psycopg2.connect(
        database='postgres',
        user='postgres',
        password='minckA.2023',
        host='cerrejondb.ckensqtixcpt.ap-southeast-2.rds.amazonaws.com',
        port=5432
    )

    cursor = conn.cursor()
    create_query = """
        CREATE TABLE IF NOT EXISTS "50h" (
            "Timestamp" TIMESTAMP PRIMARY KEY,
            "AI A-1" DOUBLE PRECISION,
            "AI A-2" DOUBLE PRECISION,
            "AI A-3" DOUBLE PRECISION,
            "AI A-4" DOUBLE PRECISION,
            "AI A-5" DOUBLE PRECISION,
            "AI A-6" DOUBLE PRECISION,
            "AI A-7" DOUBLE PRECISION,
            "AI A-8" DOUBLE PRECISION,
            "AI B-1" DOUBLE PRECISION,
            "AI B-2" DOUBLE PRECISION,
            "AI B-3" DOUBLE PRECISION,
            "AI B-4" DOUBLE PRECISION,
            "AI B-5" DOUBLE PRECISION,
            "AI B-6" DOUBLE PRECISION,
            "AI B-7" DOUBLE PRECISION
        );
    """
    cursor.execute(create_query)


    insert_query = """INSERT INTO "50h" ("Timestamp", "AI A-1", "AI A-2", "AI A-3", "AI A-4", "AI A-5", "AI A-6", "AI A-7", "AI A-8", "AI B-1", "AI B-2", "AI B-3", "AI B-4", "AI B-5", "AI B-6", "AI B-7") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT ("Timestamp") DO UPDATE SET "AI A-1" = EXCLUDED."AI A-1", "AI A-2" = EXCLUDED."AI A-2", "AI A-3" = EXCLUDED."AI A-3", "AI A-4" = EXCLUDED."AI A-4", "AI A-5" = EXCLUDED."AI A-5", "AI A-6" = EXCLUDED."AI A-6", "AI A-7" = EXCLUDED."AI A-7", "AI A-8" = EXCLUDED."AI A-8", "AI B-1" = EXCLUDED."AI B-1", "AI B-2" = EXCLUDED."AI B-2", "AI B-3" = EXCLUDED."AI B-3", "AI B-4" = EXCLUDED."AI B-4", "AI B-5" = EXCLUDED."AI B-5", "AI B-6" = EXCLUDED."AI B-6", "AI B-7" = EXCLUDED."AI B-7";"""

    # for i, row in dump:
        # timestamp = pd.to_datetime(row['timestamp'])
        # record_to_insert = (timestamp, row['data0'], row['data1'], row['data2'], row['data3'], row['data4'], row['data5'], row['data6'], row['data7'], row['data8'], row['data9'], row['data10'], row['data11'], row['data12'], row['data13'], row['data14'])
        # cursor.execute(insert_query, record_to_insert)
    record_to_insert = [(pd.to_datetime(row['timestamp']), row['data0'], row['data1'], row['data2'], row['data3'], row['data4'], row['data5'], row['data6'], row['data7'], row['data8'], row['data9'], row['data10'], row['data11'], row['data12'], row['data13'], row['data14']) for row in dump]
    start_time = time.time()
    cursor.executemany(insert_query, record_to_insert)


    conn.commit()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    cursor.close()
    conn.close()


def BB2(dump):
    columns = ['Timestamp', 'AI A-1', 'AI A-2', 'AI A-3', 'AI A-4', 'AI A-5', 'AI A-6', 'AI A-7', 'AI A-8', 'AI B-1', 'AI B-2', 'AI B-3', 'AI B-4', 'AI B-5', 'AI B-6', 'AI B-7']
    df = pd.DataFrame.from_records(dump, columns=columns)
    df.columns = columns
    
    engine = create_engine('postgresql://postgres:minckA.2023@cerrejondb.ckensqtixcpt.ap-southeast-2.rds.amazonaws.com:5432/postgres')
    
    create_query = """
        CREATE TABLE IF NOT EXISTS "50h" (
            "Timestamp" TIMESTAMP PRIMARY KEY,
            "AI A-1" DOUBLE PRECISION,
            "AI A-2" DOUBLE PRECISION,
            "AI A-3" DOUBLE PRECISION,
            "AI A-4" DOUBLE PRECISION,
            "AI A-5" DOUBLE PRECISION,
            "AI A-6" DOUBLE PRECISION,
            "AI A-7" DOUBLE PRECISION,
            "AI A-8" DOUBLE PRECISION,
            "AI B-1" DOUBLE PRECISION,
            "AI B-2" DOUBLE PRECISION,
            "AI B-3" DOUBLE PRECISION,
            "AI B-4" DOUBLE PRECISION,
            "AI B-5" DOUBLE PRECISION,
            "AI B-6" DOUBLE PRECISION,
            "AI B-7" DOUBLE PRECISION
        );
    """
    with engine.connect() as conn:
        conn.execute(create_query)
        
        start_time = time.time()
        df.to_sql('50h', con=engine, if_exists='append', index=False)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

# create the Dewesoft DCOM object
dw = Dispatch("Dewesoft.App")
sys.stdout.flush()
dw.Init()
dw.Enabled = 1
dw.Visible = 1
dw.Top = 0
dw.Left = 0
dw.Width = 1024 
dw.Height = 768

_dir = r"C:\Users\aleja\Documents\Cerrejon Data Analyis\dxd_files\06.04.2023_dxd_files\Set up 1 Cerrejon _00{}.dxd"
a,b = 1,5
file_names = file_nameautomate(_dir, a, b)
folder_output = file_names[0][:78] + "csv_06.04.2023/"
data123 = plt_all_measurements3(file_names,folder_output)




        