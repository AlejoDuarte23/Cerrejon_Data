# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:57:25 2023

@author: aleja
"""

import pandas as pd
import numpy as np 
import os 
from _preprocesing import  merge_dataframes
import matplotlib.pyplot as plt

# %% Geting Cycles slave
file_name = 'Datos C429 - Ciclos pivoted abril.xlsx'
file_path= os.path.join('..', 'Reference',file_name)
slave = pd.read_excel(file_path, engine='openpyxl')

# %% main Data frame 
#%%
file_name = 'Datos GPS equipo 022-429 - Abril 06-21 2023.xlsx'
file_path= os.path.join('..', 'Reference',file_name)
sheet_names = '20230406-20230421'
df = pd.read_excel(file_path,sheet_name=sheet_names, engine='openpyxl',skiprows=1)

main_df = df
slave_df = slave
main_ts_col = 'Timestamp'
slave_ts_col = 'ReadTime'
slave_cols_interest = ['Cycle']
df_cord_cycles  = merge_dataframes(main_df, slave_df, main_ts_col, slave_ts_col, slave_cols_interest)


#  ####################################
# def calculate_distance(df):
#     # Calculate cartesian distance
#     df['Distance'] = np.sqrt(df['PositionX'].diff()**2 + df['PositionY'].diff()**2)
#     # Calculate difference in position
#     df['Position_Diff'] = df['Distance'].diff()
#     # Default movement to False
#     df['Movement'] = False
#     # If distance is more than 20, set movement to True
#     df.loc[df['Distance'] > 100, 'Movement'] = True

    # return df

def calculate_distance(df):
    # Calculate cartesian distance
    df['Distance'] = np.sqrt(df['PositionX'].diff()**2 + df['PositionY'].diff()**2)
    # Calculate difference in position
    df['Position_Diff'] = df['Distance'].diff()
    # Default movement to False
    df['Movement'] = False
    # If distance is more than 20, set movement to True
    df.loc[df['Distance'] >100, 'Movement'] = True

    return df
from itertools import groupby
# def assign_clusters(df, min_stop_length=3):
#     df['Cluster'] = ''

#     for name, group in df.groupby('Cycle'):
#         clusters = []
#         cluster = ''

#         for i, (index, row) in enumerate(group.iterrows()):
#             if row['Movement']:
#                 if cluster in ['', 'dumping_process']:
#                     cluster = 'travelling_empty'
#                 elif cluster == 'loading_process':
#                     cluster = 'travelling_full'
#             else:
#                 if cluster == 'travelling_empty':
#                     cluster = 'loading_process'
#                 elif cluster == 'travelling_full':
#                     cluster = 'dumping_process'
#             clusters.append(cluster)

#         # If no 'loading_process' or 'dumping_process' found, iteratively relax the stop criteria
#         while 'loading_process' not in clusters and 'dumping_process' not in clusters and min_stop_length > 0:
#             clusters = []
#             cluster = ''
#             stop_counter = 0

#             for i, (index, row) in enumerate(group.iterrows()):
#                 if row['Movement']:
#                     stop_counter = 0
#                     if cluster in ['', 'dumping_process']:
#                         cluster = 'travelling_empty'
#                     elif cluster == 'loading_process':
#                         cluster = 'travelling_full'
#                 else:
#                     stop_counter += 1
#                     if stop_counter >= min_stop_length:
#                         if cluster == 'travelling_empty':
#                             cluster = 'loading_process'
#                         elif cluster == 'travelling_full':
#                             cluster = 'dumping_process'
#                 clusters.append(cluster)
            
#             min_stop_length -= 1

#         # Find the largest continuous 'loading_process' and 'dumping_process'
#         groups = [(c, len(list(g))) for c, g in groupby(clusters)]
#         loading_process = max(((i, c, l) for i, (c, l) in enumerate(groups) if c == 'loading_process'), key=lambda x: x[2], default=(None, '', 0))
#         dumping_process = max(((i, c, l) for i, (c, l) in enumerate(groups) if c == 'dumping_process'), key=lambda x: x[2], default=(None, '', 0))

#         # Reassign clusters
#         start_loading = sum(l for c, l in groups[:loading_process[0]]) if loading_process[0] is not None else None
#         end_loading = start_loading + loading_process[2] if loading_process[0] is not None else None
#         start_dumping = sum(l for c, l in groups[:dumping_process[0]]) if dumping_process[0] is not None else None
#         end_dumping = start_dumping + dumping_process[2] if dumping_process[0] is not None else None

#         for i in range(len(clusters)):
#             if start_loading is not None and start_dumping is not None:
#                 if i < start_loading or i > end_dumping:
#                     clusters[i] = 'travelling_empty'
#                 elif start_loading <= i < end_loading:
#                     clusters[i] = 'loading_process'
#                 elif end_loading <= i < start_dumping:
#                     clusters[i] = 'travelling_full'
#                 elif start_dumping <= i < end_dumping:
#                     clusters[i] = 'dumping_process'
#             else:
#                 clusters[i] = 'travelling_empty' if clusters[i] in ['', 'dumping_process'] else 'travelling_full'

#         # Assign clusters to the DataFrame
#         df.loc[group.index, 'Cluster'] = clusters

#     return df
from itertools import groupby

def assign_clusters(df, min_stop_length=1):
    df['Cluster'] = ''

    for name, group in df.groupby('Cycle'):
        clusters = []
        cluster = ''

        for i, (index, row) in enumerate(group.iterrows()):
            if row['Movement']:
                if cluster in ['', 'dumping_process']:
                    cluster = 'travelling_empty'
                elif cluster == 'loading_process':
                    cluster = 'travelling_full'
            else:
                if cluster == 'travelling_empty':
                    cluster = 'loading_process'
                elif cluster == 'travelling_full':
                    cluster = 'dumping_process'
            clusters.append(cluster)

        # Find the largest continuous 'loading_process' and 'dumping_process'
        groups = [(c, len(list(group))) for c, group in groupby(clusters)]
        loading_process = max(((i, c, l) for i, (c, l) in enumerate(groups) if c == 'loading_process'), key=lambda x: x[2], default=(None, '', 0))
        dumping_process = max(((i, c, l) for i, (c, l) in enumerate(groups) if c == 'dumping_process'), key=lambda x: x[2], default=(None, '', 0))

        # If no 'loading_process' or 'dumping_process' found, assign 'loading_process' to the first large group of False in Movement, and 'dumping_process' to the second
        if loading_process[0] is None or dumping_process[0] is None:
            false_groups = [(i, len(list(group))) for i, (c, group) in enumerate(groupby([not m for m in group['Movement']])) if not c]
            false_groups.sort(key=lambda x: x[1], reverse=True)
            if len(false_groups) >= 2:
                loading_process = (false_groups[0][0], 'loading_process', false_groups[0][1])
                dumping_process = (false_groups[1][0], 'dumping_process', false_groups[1][1])

        # Reassign clusters
        start_loading = sum(l for c, l in groups[:loading_process[0]]) if loading_process[0] is not None else None
        end_loading = start_loading + loading_process[2] if loading_process[0] is not None else None
        start_dumping = sum(l for c, l in groups[:dumping_process[0]]) if dumping_process[0] is not None else None
        end_dumping = start_dumping + dumping_process[2] if dumping_process[0] is not None else None

        for i in range(len(clusters)):
            if start_loading is not None and start_dumping is not None:
                if i < start_loading or i > end_dumping:
                    clusters[i] = 'travelling_empty'
                elif start_loading <= i < end_loading:
                    clusters[i] = 'loading_process'
                elif end_loading <= i < start_dumping:
                    clusters[i] = 'travelling_full'
                elif start_dumping <= i < end_dumping:
                    clusters[i] = 'dumping_process'
            else:
                clusters[i] = 'travelling_empty' if clusters[i] in ['', 'dumping_process'] else 'travelling_full'

        # Assign clusters to the DataFrame
        df.loc[group.index, 'Cluster'] = clusters

    return df

# # 

# def assign_clusters(df):
#     # Initialize cluster
#     df['Cluster'] = ''

#     # Iterate over DataFrame, grouped by Cycle
#     for name, group in df.groupby('Cycle'):
#         # List to store the clusters
#         clusters = []
#         # Initialize cluster
#         cluster = ''
#         # Iterate over the group
#         for i, (index, row) in enumerate(group.iterrows()):
#             if row['Movement']:
#                 if cluster in ['', 'dumping_process']:
#                     cluster = 'travelling_empty'
#                 elif cluster == 'loading_process':
#                     cluster = 'travelling_full'
#             else:
#                 if cluster == 'travelling_empty':
#                     cluster = 'loading_process'
#                 elif cluster == 'travelling_full':
#                     cluster = 'dumping_process'
#             clusters.append(cluster)

#         # Find the largest continuous 'loading_process' and 'dumping_process'
#         groups = [(c, len(list(g))) for c, g in groupby(clusters)]
#         loading_process = max(((i, c, l) for i, (c, l) in enumerate(groups) if c == 'loading_process'), key=lambda x: x[2], default=(None, '', 0))
#         dumping_process = max(((i, c, l) for i, (c, l) in enumerate(groups[loading_process[0]+1:]) if c == 'dumping_process' and loading_process[0] is not None), key=lambda x: x[2], default=(None, '', 0))
#         dumping_process = (dumping_process[0] + loading_process[0] + 1, dumping_process[1], dumping_process[2]) if dumping_process[0] is not None and loading_process[0] is not None else (None, '', 0)

#         # Reassign clusters
#         start_loading = sum(l for c, l in groups[:loading_process[0]]) if loading_process[0] is not None else None
#         end_loading = start_loading + loading_process[2] if loading_process[0] is not None else None
#         start_dumping = sum(l for c, l in groups[:dumping_process[0]]) if dumping_process[0] is not None else None
#         end_dumping = start_dumping + dumping_process[2] if dumping_process[0] is not None else None

#         for i in range(len(clusters)):
#             if start_loading is not None and end_loading is not None and start_dumping is not None and end_dumping is not None:
#                 if i < start_loading:
#                     clusters[i] = 'travelling_empty'
#                 elif start_loading <= i < end_loading:
#                     clusters[i] = 'loading_process'
#                 elif end_loading <= i < start_dumping:
#                     clusters[i] = 'travelling_full'
#                 elif start_dumping <= i < end_dumping:
#                     clusters[i] = 'dumping_process'
#                 elif i >= end_dumping:
#                     clusters[i] = 'travelling_empty'
#             else:
#                 clusters[i] = 'travelling_empty' if clusters[i] in ['', 'dumping_process'] else 'travelling_full'

#         # Assign clusters to the DataFrame
#         df.loc[group.index, 'Cluster'] = clusters

#     return df


def plot_cycle(df, cycle_number):
    # Select the data for the given cycle
    df_cycle = df[df['Cycle'] == cycle_number]

    # Define color mapping
    color_mapping = {
        'loading_process': 'red',
        'dumping_process': 'blue',
        'travelling_empty': 'yellow',
        'travelling_full': 'purple'
    }

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Iterate through each row and plot the point with the corresponding color
    for i, row in df_cycle.iterrows():
        plt.scatter(row['Timestamp'], row['PositionX'], color=color_mapping.get(row['Cluster'], 'black'))

    # Set title and labels
    plt.title(f'PositionX vs Timestamp for cycle {cycle_number}')
    plt.xlabel('Timestamp')
    plt.ylabel('PositionX')

    # Show the plot
    plt.show()

# Call the function with your DataFrame and the cycle number




df_cord_cycles = calculate_distance(df_cord_cycles)
df_cord_cycles = assign_clusters(df_cord_cycles)
plot_cycle(df_cord_cycles, 452)

# df_cord_cycles.to_excel('cluster_mincka_way.xlsx',index = False )
