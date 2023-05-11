from scipy.signal import spectrogram, periodogram
import numpy as np 
import sqlalchemy as db
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend



rosettes_info = {
    'ROSETTE1': {'theta1': 0, 'theta2': 42.807, 'theta3': 89.773, 'Kt': 1.035},
    'ROSETTE2': {'theta1': 0, 'theta2': 46.75, 'theta3': 92.009, 'Kt': 1.035},
    'ROSETTE3': {'theta1': 0, 'theta2': 44.09, 'theta3': 88.378, 'Kt': 1.035},
    'ROSETTE4': {'theta1': 0, 'theta2': 42.999, 'theta3': 85.933, 'Kt': 1.035},
    'ROSETTE5': {'theta1': 0, 'theta2': 41.589, 'theta3': 84.251, 'Kt': 1.035},
    'ROSETTE6': {'theta1': 0, 'theta2': 42.5403, 'theta3': 83.877, 'Kt': 1.04},
    'ROSETTE7': {'theta1': 0, 'theta2': 45, 'theta3': 90, 'Kt': 1.04},
    'ROSETTE8': {'theta1': 0, 'theta2': 45, 'theta3': 90, 'Kt': 1.04},
    'ROSETTE9': {'theta1': 0, 'theta2': 45, 'theta3': 90, 'Kt': 1.04},
    'ROSETTE10': {'theta1': 0, 'theta2': 45, 'theta3': 90, 'Kt': 1.04}
}



Set_up1 = {
    'ROSETTE1': {
        'SG_IDs': ['AI A-1', 'AI A-2', 'AI A-3'],
        'angles': {'theta1': 0, 'theta2': 42.807, 'theta3': 89.773},
        'Kt': 1.035,
    },
    'ROSETTE2': {
        'SG_IDs': ['AI A-4', 'AI A-5', 'AI A-6'],
        'angles': {'theta1': 0, 'theta2': 46.75, 'theta3': 92.009},
        'Kt': 1.035,
    },
    'ROSETTE3': {
        'SG_IDs': ['AI A-7', 'AI A-8', 'AI B-1'],
        'angles': {'theta1': 0, 'theta2': 44.09, 'theta3': 88.378},
        'Kt': 1.035,
    },
    'ROSETTE4': {
        'SG_IDs': ['AI B-2', 'AI B-3', 'AI B-4'],
        'angles': {'theta1': 0, 'theta2': 42.999, 'theta3': 85.933},
        'Kt': 1.035,
    },
    'ROSETTE5': {
        'SG_IDs': ['AI B-5', 'AI B-6', 'AI B-7'],
        'angles': {'theta1': 0, 'theta2': 41.589, 'theta3': 84.251},
        'Kt': 1.035,
    }}



def rosette_info_engine():
    rosettes_info = {
        'ROSETTE1': {'theta1': 0, 'theta2': 42.807, 'theta3': 89.773, 'Kt': 1.035},
        'ROSETTE2': {'theta1': 0, 'theta2': 46.75, 'theta3': 92.009, 'Kt': 1.035},
        'ROSETTE3': {'theta1': 0, 'theta2': 44.09, 'theta3': 88.378, 'Kt': 1.035},
        'ROSETTE4': {'theta1': 0, 'theta2': 42.999, 'theta3': 85.933, 'Kt': 1.035},
        'ROSETTE5': {'theta1': 0, 'theta2': 41.589, 'theta3': 84.251, 'Kt': 1.035},
        'ROSETTE6': {'theta1': 0, 'theta2': 42.5403, 'theta3': 83.877, 'Kt': 1.04},
        'ROSETTE7': {'theta1': 0, 'theta2': 45, 'theta3': 90, 'Kt': 1.04},
        'ROSETTE8': {'theta1': 0, 'theta2': 45, 'theta3': 90, 'Kt': 1.04},
        'ROSETTE9': {'theta1': 0, 'theta2': 45, 'theta3': 90, 'Kt': 1.04},
        'ROSETTE10': {'theta1': 0, 'theta2': 45, 'theta3': 90, 'Kt': 1.04}
    }
    return rosettes_info

def plot_subplots(result_df, fs, fo, fi, color, start_date=None, end_date=None):
    # Filter the data based on the specified date range
    if start_date:
        result_df = result_df[result_df['Timestamp'] >= start_date]
    if end_date:
        result_df = result_df[result_df['Timestamp'] <= end_date]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.delaxes(axs[0, 1])  # Remove the 1x2 subplot

    # 1x1: Time series
    axs[0, 0].plot(result_df['Timestamp'], result_df['tau'], color=color)
    axs[0, 0].set_title('Stress (MPa) vs Time')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Stress (MPa)')

    # 2x1: Spectrogram
    f, t, Sxx = spectrogram(result_df['tau'], fs, scaling='spectrum', nperseg=1024)
    cmap = plt.get_cmap('jet')
    axs[1, 0].pcolormesh(t, f, Sxx, shading='gouraud', cmap=cmap)
    axs[1, 0].set_title('Spectrogram')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Frequency (Hz)')
    axs[1, 0].set_ylim(fo, fi)

    # 2x2: PSD (rotated)
    f, Pxx = periodogram(result_df['tau'], fs)
    valid_indices = np.where((f >= fo) & (f <= fi))
    axs[1, 1].plot(Pxx[valid_indices], f[valid_indices], color=color)
    axs[1, 1].set_title('Power Spectral Density (PSD)')
    axs[1, 1].set_xlabel('Power')
    axs[1, 1].set_ylabel('Frequency (Hz)')
    axs[1, 1].set_ylim(fo, fi)
    axs[1, 1].set_xlim(0, np.max(Pxx[valid_indices]))

    plt.tight_layout()
    plt.show()
        
    
def get_data_between_dates(start_date, end_date, columns):
    # To do : make table number as a input of the function
    engine = db.create_engine('postgresql://postgres:minckA.2023@cerrejondb.ckensqtixcpt.ap-southeast-2.rds.amazonaws.com:5432/postgres')
    conn = engine.connect()
    metadata = db.MetaData()
    table = db.Table('50h', metadata, autoload=True, autoload_with=engine)
    query = db.select([table.columns[col] for col in columns] + [table.columns['Timestamp']]).where(table.columns['Timestamp'].between(start_date, end_date))
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
    
   
def get_rosette_data(rosette_id, setup):
    data = setup[rosette_id]
    SG_IDs = data['SG_IDs']
    angles = data['angles']
    Kt = data['Kt']
    return SG_IDs, angles, Kt


# def get_rosette_data(rosette_id):
#     data = setup[rosette_id]
#     SG_IDs = data['SG_IDs']
#     angles = data['angles']
#     Kt = data['Kt']
#     return SG_IDs, angles, Kt


def resample_data2(df, rule='10L', columns=['AI A-1', 'AI A-2', 'AI A-3']):
    if not isinstance(df.index, pd.DatetimeIndex):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
    df_resampled = df.resample(rule).mean()
    df_resampled.reset_index(inplace=True)
    return df_resampled[columns + ['Timestamp']]


def solve_strain_system(df, columns, theta1, theta2, theta3, E, v, G, Kt,T = 'On'):
    # Extract the strain values from the DataFrame
    if  T == 'On':
        print('Data Scaled')
        _eps1, _eps2, _eps3 = [Kt*df[col]/1e6 for col in columns]
    
        # Calculate the transformed strains
        eps1 = (1 - v*0.01) * (_eps1 - 0.01*_eps2)
        eps2 = (1 - v*0.01) * (_eps2 - 0.01*_eps1)
        eps3 = (1 - v*0.01) * (_eps3 - 0.01*(_eps1 + _eps2 - _eps3))
    else:
        eps1, eps2, eps3 = [df[col]/1e6 for col in columns]
    # eps1 =   _eps1#, (1 - v*0.01) * (_eps1 - 0.01*_eps2)
    # eps2 =  _eps2#(1 - v*0.01) * (_eps2 - 0.01*_eps1)
    # eps3 =  _eps3#,(1 - v*0.01) * (_eps3 - 0.01*(_eps1 + _eps2 - _eps3))
        
    
    
    theta1 = np.deg2rad(theta1)
    theta2 = np.deg2rad(theta2)
    theta3 = np.deg2rad(theta3)
    # Solve the system of equations
    A = np.array([[np.cos(theta1)**2, np.sin(theta1)**2, np.sin(theta1)*np.cos(theta1)],
                  [np.cos(theta2)**2, np.sin(theta2)**2, np.sin(theta2)*np.cos(theta2)],
                  [np.cos(theta3)**2, np.sin(theta3)**2, np.sin(theta3)*np.cos(theta3)]])
    
    b = np.array([eps1, eps2, eps3])
    
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    
    epsx, epsy, gammaxy = x
    
    # Calculate principal strains and maximum shear strain
    eps_max = (epsx + epsy)/2 + np.sqrt(((epsx - epsy)/2)**2 + (gammaxy/2)**2)
    eps_min = (epsx + epsy)/2 - np.sqrt(((epsx - epsy)/2)**2 + (gammaxy/2)**2)
    gamma_max = 2*np.sqrt(((epsx - epsy)/2)**2 + (gammaxy/2)**2)
    
    # Calculate stress components
    sigma1 = (E/(1-v**2))*(eps_max + v*eps_min)
    sigma2 = (E/(1-v**2))*(eps_min + v*eps_max)
    tau = G*gamma_max
    tan2_theta_p = 2*tau/(sigma1 - sigma2)
    theta_p = np.rad2deg(np.arctan(tan2_theta_p)/2)

    
    return sigma1, sigma2, tau , theta_p 

def plot_principal_stresses(df, sigma_1_col, sigma_2_col, tau,theta):
    fig, (ax1, ax2,ax3,ax4) = plt.subplots(4, 1, figsize=(10, 6), sharex=True)

    ax1.plot(df['Timestamp'], df[sigma_1_col], label='Sigma 1')
    ax1.set_ylabel('Sigma 1 [Mpa]')
    ax1.legend()

    ax2.plot(df['Timestamp'], df[sigma_2_col], label='Sigma 2', color='red')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Sigma 2 [Mpa]')
    ax2.legend()
    
    
    ax3.plot(df['Timestamp'], df[tau], label='Tau', color='green')
    ax3.set_xlabel('Timestamp')
    ax3.set_ylabel('Tau [Mpa]')
    ax3.legend()
    
    ax4.plot(df['Timestamp'], df[theta], label='theta_p', color='magenta')
    ax4.set_xlabel('Timestamp')
    ax4.set_ylabel('Thetha_p [°]')
    ax4.legend()

    plt.show()


def detrend_data(data):
    # Initialize an empty structured array for detrended data
    detrended_data = np.empty(data.shape, dtype=data.dtype)

    # Copy the timestamp column
    detrended_data['timestamp'] = data['timestamp']

    # Loop through the data columns and detrend each column
    for column in data.dtype.names:
        if column != 'timestamp':
            detrended_data[column] = detrend(data[column])

    return detrended_data

def convert_to_df(data):
    data_df = pd.DataFrame(data)

    # Convert the timestamp column to datetime64 dtype
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    return data_df


def merge_dataframes(main, slave, main_ts_col, slave_ts_col, slave_cols_interest):
    # Load data if the input is xlsx file
    if isinstance(main, str):
        main = pd.read_excel(main)
    if isinstance(slave, str):
        slave = pd.read_excel(slave)

    # Ensure that the timestamp columns are in datetime format
    main[main_ts_col] = pd.to_datetime(main[main_ts_col])
    slave[slave_ts_col] = pd.to_datetime(slave[slave_ts_col])

    # Initialize the result dataframe
    result_df = main.copy()

    # Iterate over the columns of interest in the slave dataframe
    for col in slave_cols_interest:
        # Create a temporary column in the result dataframe to store the interpolated data
        result_df[col] = None

        # Perform forward filling for the slave column
        slave[col] = slave[col].fillna(method='ffill')

        # Iterate over the result dataframe and assign the corresponding value from the slave dataframe
        for index, row in result_df.iterrows():
            slave_row = slave[(slave[slave_ts_col] <= row[main_ts_col])].iloc[-1]
            result_df.at[index, col] = slave_row[col]

    return result_df


def plot_principal_stresses_clustered(df, sigma_1_col, sigma_2_col, tau, theta):
    color_mapping = {
        'loading_process': 'red',
        'dumping_process': 'blue',
        'travelling_empty': 'green',
        'travelling_full': 'purple'
    }
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
    labels = ['Sigma 1', 'Sigma 2', 'Tau', 'Theta_p']
    ylabels = ['Sigma 1 [Mpa]', 'Sigma 2 [Mpa]', 'Tau [Mpa]', 'Theta_p [°]']
    
    for i, col in enumerate([sigma_1_col, sigma_2_col, tau, theta]):
        for cluster, color in color_mapping.items():
            mask = df['Cluster'] == cluster
            axes[i].scatter(df.loc[mask, 'Timestamp'], df.loc[mask, col], label=cluster, color=color, s=10)
        axes[i].set_ylabel(ylabels[i])
        axes[i].legend()
    
    axes[-1].set_xlabel('Timestamp')
    plt.show()

# Example usage



# Example usage
# main_df = 'result_df.xlsx'  # Replace with your main dataframe file or dataframe
# slave_df = 'Vims_clustered.xlsx'
# main_ts_col = 'Timestamp'
# slave_ts_col = 'ReadTime'
# slave_cols_interest = ['Left Front Suspension Cylinder', 'Left Rear Suspension Cylinder', 'Payload',
#                        'Right Front Suspension Cylinder', 'Right Rear Suspension Cylinder', 'Cycle', 'Cluster']

# output_df = merge_dataframes(main_df, slave_df, main_ts_col, slave_ts_col, slave_cols_interest)
# print(output_df)

# start_date = '2023-02-28 11:20:00.000'
# end_date = '2023-02-28 11:36:44.000'
# start_date = '2023-02-28 18:10:000'
# end_date = '2023-02-28 18:45:000'
# columns = ['AI A-1', 'AI A-2', 'AI A-3']



# _df = get_data_between_dates(start_date, end_date, columns)

# # df = get_data_between_dates(start_date, end_date, columns)
# # _df = get_data_between_dates(start_date, end_date, columns)
# _df = _df.sort_values(by='Timestamp')

# df_resampled = resample_data2(_df, rule='50L', columns=columns)

# rosette_name = 'ROSETTE1'  # replace with the name of the rosette you want to analyze
# rosette_data = get_rosette_data(rosette_name,Set_up1)

# sigmas = solve_strain_system(df_resampled, rosette_data[0], rosette_data[1]['theta1'],
#                               rosette_data[1]['theta2'],
#                               rosette_data[1]['theta3'],
#                               E=210000,
#                               v=0.285,
#                               G=0.37619047*210000,
#                               Kt = rosette_data[2])


    
    
# print(df_resampled)
# result_df = pd.DataFrame({
#     'Timestamp': df_resampled['Timestamp'],  # Replace 'Timestamp' with the appropriate column name for the timestamp in your DataFrame
#     'sigma_1': sigmas[0],
#     'sigma_2': sigmas[1],
#     'tau' : sigmas[2],
#     'theta_p': sigmas[3]
# })
# plot_principal_stresses(result_df, 'sigma_1', 'sigma_2','tau','theta_p')



