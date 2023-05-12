

import pandas as pd
import numpy as np 
import fatpack
import pickle 
from collections import Counter

def calculate_AMPS(df):
    # Ensure the specific columns are numeric
    df['sigma_1'] = pd.to_numeric(df['sigma_1'], errors='coerce')
    df['sigma_2'] = pd.to_numeric(df['sigma_2'], errors='coerce')

    # Calculate the AMPS using NumPy's vectorized operations
    df['AMPS'] = df[['sigma_1', 'sigma_2']].apply(lambda x: x[np.argmax(np.abs(x))], axis=1)

    return df

import pandas as pd
import fatpack


def perform_rainflow(df: pd.DataFrame) -> dict:
    # Create an empty dictionary to store all results
    all_results = {}

    # Loop over each unique 'Cycle' value
    for cycle in df['Cycle'].unique():
        # Filter the DataFrame for the current cycle
        cycle_df = df[df['Cycle'] == cycle]

        # Create an empty DataFrame to store the results for this cycle
        cycle_results = pd.DataFrame()

        # Loop over each unique 'Cluster' within this cycle
        for cluster in cycle_df['Cluster'].unique():
            # Filter the DataFrame for the current cluster
            cluster_df = cycle_df[cycle_df['Cluster'] == cluster]

            # Perform rainflow counting
            stresses = cluster_df['AMPS'].values
            reversals, indices = fatpack.find_rainflow_cycles(stresses)

            # Calculate rainflow cycles
            cycles = len(indices) / 2  # As each cycle includes a reversal and return
            

            # Calculate stress range
            if len(reversals) > 0:  # Only calculate ptp if there are reversals
                stress_range = np.ptp(reversals)  # Using numpy.ptp (Peak-To-Peak) to calculate the range
            else:  # If there are no reversals, set stress_range to None (or any other value that makes sense in your case)
                stress_range = None

            # Calculate stress range
            # stress_range = np.ptp(reversals)  # Using numpy.ptp (Peak-To-Peak) to calculate the range

            # Create a DataFrame to hold the results for this cluster
            cluster_results = pd.DataFrame({
                'Cycle': [cycle],
                'Cluster': [cluster],
                'Timestamp': [cluster_df['Timestamp'].values[0]],
                'Ni': [cycles],  # Ensure this is a list
                'Stress range': [stress_range],  # Take the maximum stress range
            })

            # Append these results to the cycle results
            cycle_results = cycle_results.append(cluster_results, ignore_index=True)

        # Add the cycle results to the overall results
        all_results[cycle] = cycle_results

    return all_results


def perform_rainflow2(df: pd.DataFrame) -> dict:
    # Create an empty dictionary to store all results
    all_results = {}

    # Loop over each unique 'Cycle' value
    for cycle in df['Cycle'].unique():
        # Filter the DataFrame for the current cycle
        cycle_df = df[df['Cycle'] == cycle]

        # Loop over each unique 'Cluster' within this cycle
        for cluster in cycle_df['Cluster'].unique():
            # Filter the DataFrame for the current cluster
            cluster_df = cycle_df[cycle_df['Cluster'] == cluster]

            # Perform rainflow counting
            stresses = cluster_df['AMPS'].values
            reversals, indices = fatpack.find_rainflow_cycles(stresses)

            # Calculate rainflow cycles
            cycles = len(indices) / 2  # As each cycle includes a reversal and return

            # Calculate stress range
            if len(reversals) > 0:  # Only calculate ptp if there are reversals
                stress_range = np.ptp(reversals)  # Using numpy.ptp (Peak-To-Peak) to calculate the range
            else:  # If there are no reversals, set stress_range to None (or any other value that makes sense in your case)
                stress_range = None

            # Add the results for this cluster to the cycle results
            all_results[(cycle, cluster)] = {'Timestamp': cluster_df['Timestamp'].values[0],
                                              'Ni': cycles, 
                                              'Stress range': stress_range}

    return all_results



def perform_rainflow3(df: pd.DataFrame) -> dict:
    # Create an empty dictionary to store all results
    all_results = {}

    # Loop over each unique 'Cycle' value
    for cycle in df['Cycle'].unique():
        # Filter the DataFrame for the current cycle
        cycle_df = df[df['Cycle'] == cycle]

        # Create an empty dictionary to store the results for this cycle
        cycle_results = {}

        # Loop over each unique 'Cluster' within this cycle
        for cluster in cycle_df['Cluster'].unique():
            # Filter the DataFrame for the current cluster
            cluster_df = cycle_df[cycle_df['Cluster'] == cluster]

            # Perform rainflow counting
            stresses = cluster_df['AMPS'].values
            reversals, indices = fatpack.find_rainflow_cycles(stresses)

            # Calculate rainflow cycles
            cycles = len(indices) / 2  # As each cycle includes a reversal and return

            # Calculate stress range for each cycle
            stress_ranges = np.ptp(reversals, axis=1)  # Using numpy.ptp (Peak-To-Peak) to calculate the range for each cycle
            
            # Count the occurrences of each unique stress range
            stress_range_counts = Counter(stress_ranges)

            # Store the results for this cluster
            cycle_results[cluster] = {
                'Timestamp': cluster_df['Timestamp'].values[0],
                'Ni': cycles,  # The total number of cycles
                'Stress range counts': stress_range_counts,  # The counts of each unique stress range
            }

        # Add the cycle results to the overall results
        all_results[cycle] = cycle_results

    return all_results





with open("data_stress.pickle", "rb") as file:
    df_stress = pickle.load(file)
    

output_df_amps = calculate_AMPS(df_stress)

# from _preprocesing import  plot_cycle_data2
# cols = ['AMPS','Left Front Suspension Cylinder' ,'Right Front Suspension Cylinder','Payload']
# units = ['Mpa', '-','-', 'Ton','Mpa']

# # Use the function for a specific cycle
# plot_cycle_data2(output_df_amps, 478, cols, units)


def perform_rainflow10(df: pd.DataFrame) -> dict:
    results = {}

    # Loop over each unique 'Cycle' value
    for cycle in df['Cycle'].unique():
        # Filter the DataFrame for the current cycle
        cycle_df = df[df['Cycle'] == cycle]

        # Loop over each unique 'Cluster' within this cycle
        for cluster in cycle_df['Cluster'].unique():
            # Filter the DataFrame for the current cluster
            cluster_df = cycle_df[cycle_df['Cluster'] == cluster]

            # Check if 'AMPS' series has at least 3 points (needed for rainflow analysis)
            if len(cluster_df['AMPS']) < 3:
                print(f"Insufficient data for Cycle {cycle}, Cluster {cluster}")
                continue  # Skip to the next cluster or cycle

            # Perform Rainflow analysis on 'AMPS' values within this cluster
            try:
                cycles = fatpack.find_rainflow_ranges(cluster_df['AMPS'])
            except IndexError:
                print(f"Error performing rainflow analysis on Cycle {cycle}, Cluster {cluster}")
                continue  # Skip to the next cluster or cycle

            # Check if any cycles were found
            if len(cycles) == 0:
                print(f"No cycles found for Cycle {cycle}, Cluster {cluster}")
                continue  # Skip to the next cluster or cycle

            # Calculate the range, mean, start and end of each cycle
            for i, cycle_range in enumerate(cycles):
                results[f'Cycle_{cycle}_{cluster}_{i}'] = {
                    'range': cycle_range,
                    'mean': np.mean(cluster_df['AMPS']),
                    'start': cluster_df['AMPS'].iloc[0],
                    'end': cluster_df['AMPS'].iloc[-1]
                }

    return results



def perform_rainflow11(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Loop over each unique 'Cycle' value
    for cycle in df['Cycle'].unique():
        # Filter the DataFrame for the current cycle
        cycle_df = df[df['Cycle'] == cycle]

        # Loop over each unique 'Cluster' within this cycle
        for cluster in cycle_df['Cluster'].unique():
            # Filter the DataFrame for the current cluster
            cluster_df = cycle_df[cycle_df['Cluster'] == cluster]

            # Check if 'AMPS' series has at least 3 points (needed for rainflow analysis)
            if len(cluster_df['AMPS']) < 3:
                print(f"Insufficient data for Cycle {cycle}, Cluster {cluster}")
                continue  # Skip to the next cluster or cycle

            # Perform Rainflow analysis on 'AMPS' values within this cluster
            try:
                cycles = fatpack.find_rainflow_ranges(cluster_df['AMPS'])
            except IndexError:
                print(f"Error performing rainflow analysis on Cycle {cycle}, Cluster {cluster}")
                continue  # Skip to the next cluster or cycle

            # Check if any cycles were found
            if len(cycles) == 0:
                print(f"No cycles found for Cycle {cycle}, Cluster {cluster}")
                continue  # Skip to the next cluster or cycle

            # Calculate the range, mean, start and end of each cycle
            for i, cycle_range in enumerate(cycles):
                row = {
                    'Cycle': cycle,
                    'Cluster': cluster,
                    'Ni': i,
                    'start_date': cluster_df['Timestamp'].iloc[0],
                    'end_date': cluster_df['Timestamp'].iloc[-1],
                    'range': cycle_range,
                    'mean': np.mean(cluster_df['AMPS'])
                }
                rows.append(row)

    # Create DataFrame from list of rows
    results_df = pd.DataFrame(rows)
    return results_df

rainflow_df = perform_rainflow11(output_df_amps)



def calculate_Nfi(row, C_o, SD, d, m):
    C = 10**(np.log10(C_o) - d * SD)
    delta_sigma = row['range']
    N_fi = C / (delta_sigma ** m)
    return N_fi

def add_Nfi_D_columns(df, C_o, SD, d, m):
    df['Nfi'] = df.apply(calculate_Nfi, args=(C_o, SD, d, m), axis=1)
    df['D'] = 1 / df['Nfi']
    return df


C_o = 3.988*10**12
SD = 0.295
d = 2
m = 3

rainflow_df = add_Nfi_D_columns(rainflow_df, C_o, SD, d, m)


