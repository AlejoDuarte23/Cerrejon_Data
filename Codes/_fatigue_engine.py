
import math
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

import numpy as np

def compute_delta_a(df, A=5.21e-13, m=4, a_prev=10**-4):
    # Calculate Δσ_eq_AMPS_i
    df['delta_sigma_eq_AMPS_i'] = df['range']

    # Calculate ΔK
    df['delta_K'] = 1.12 * df['delta_sigma_eq_AMPS_i'] * np.sqrt(np.pi * a_prev)

    # Calculate Δa_i
    df['delta_a_i'] = np.where(df['delta_K'] > 2, A * (df['delta_K']**m) * df['Ni'], 0)

    return df


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

def create_summary_df(df):
    summary_df = df.groupby(['Cycle', 'Cluster']).agg({
        'Nfi': 'sum',
        'D': 'sum',
        'mean': 'first',
        'start_date': 'first',
        'end_date': 'last'
    }).reset_index()

    # Flatten the MultiIndex columns
    summary_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in summary_df.columns.values]

    # Calculate duration in hours
    summary_df['duration_hours'] = (summary_df['end_date'] - summary_df['start_date']).dt.total_seconds() / 3600

    return summary_df


# Example usage:
# Assuming 'rainflow_df' is already defined and contains 'Cycle', 'Cluster', 'Nfi', 'D', 'mean', 'start_date', and 'end_date' columns
summary_df = create_summary_df(rainflow_df)
print(summary_df)





def calculate_af(K_IC, sigma_eq_AMPS_max):
    af = (1/math.pi) * ((K_IC / (1.12 * sigma_eq_AMPS_max)) ** 2)
    return af

# usage:
K_IC = 54  # replace with your value
sigma_eq_AMPS_max = output_df_amps['AMPS'].abs().max()
af = calculate_af(K_IC, sigma_eq_AMPS_max)
af = 5*af
print(f"The calculated value of af is {af}")


rainflow_df = compute_delta_a(rainflow_df)
ao = 10**-4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt






def compute_delta_a(df, A=5.21e-13, m=4, a_prev=10**-4 , af = 0.1086):
    # Initialize a list to store delta_a_i and a_i values
    delta_a_i_values = []
    a_values = []
    a_i = a_prev

    i = 0
    while a_i < af:
        # Calculate Δσ_eq_AMPS_i
        delta_sigma_eq_AMPS_i = df.loc[i, 'range']

        # Calculate ΔK
        delta_K = 1.12 * delta_sigma_eq_AMPS_i * np.sqrt(np.pi * a_prev)

        # Calculate Δa_i and a_i
        if delta_K > 2:
            delta_a_i = A * (delta_K**m)
        else:
            delta_a_i = 0

        a_i = delta_a_i + a_prev

        # Update a_prev for next iteration
        a_prev = a_i

        # Add current delta_a_i and a_i to the lists
        delta_a_i_values.append(delta_a_i)
        a_values.append(a_i)

        # Go to next row or loop back to first row if we're at the end
        i = (i + 1) % len(df)
        print(a_i)

    # Create a new DataFrame for the results
    result_df = pd.DataFrame({
        'delta_a_i': delta_a_i_values,
        'a_i': a_values,
    })
    
    # Plot a_i values
    plt.plot(result_df['a_i'])
    plt.xlabel('Index')
    plt.ylabel('a_i')
    plt.show()

    return result_df
