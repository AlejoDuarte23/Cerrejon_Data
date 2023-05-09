import pandas as pd 
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
import pyautocad
from shapely.wkt import loads
import os 
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
import numpy as np 
import math 
from scipy import stats
import matplotlib.patches as mpatches

def plot_corrdinates_web_mercator(df, PRT=False):
    geometry = [Point(xy) for xy in zip(df['PositionX'], df['PositionY'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.crs = 'EPSG:3116'
    gdf_web_mercator = gdf.to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_web_mercator.plot(ax=ax, color='yellow', markersize=5, marker='x', edgecolor='black')
    
    # Add the basemap without attribution text
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, attribution="")
    
    # Set the plot limits to focus on the region of interest
    ax.set_xlim(gdf_web_mercator.geometry.bounds.minx.min() - 50,
                gdf_web_mercator.geometry.bounds.maxx.max() + 50)
    ax.set_ylim(gdf_web_mercator.geometry.bounds.miny.min() - 50,
                gdf_web_mercator.geometry.bounds.maxy.max() + 50)
    
    plt.show()
    
    if PRT == True:
        output_folder = 'images'
        os.makedirs(output_folder, exist_ok=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        name = df['Timestamp'][0].strftime('%d-%m-%Y')
        output_file = os.path.join(output_folder, f'{name}.png')
        plt.savefig(output_file, dpi=500)
        plt.close()
        

def perform_clustering(df, n_clusters, n_subclusters):
    # Perform KMeans clustering
    coords = df[['PositionX', 'PositionY']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
    df['Cluster'] = kmeans.labels_

    # Initialize an empty list for subcluster labels
    df['Subcluster'] = -1

    for cluster_id in range(n_clusters):
        cluster_data = df[df['Cluster'] == cluster_id]

        # Perform subclustering
        coords_subcluster = cluster_data[['PositionX', 'PositionY']].values
        kmeans_subcluster = KMeans(n_clusters=n_subclusters, random_state=0).fit(coords_subcluster)
        subcluster_labels = kmeans_subcluster.labels_

        # Assign subcluster labels to the corresponding data points in the original DataFrame
        df.loc[cluster_data.index, 'Subcluster'] = subcluster_labels

    return df

def remove_outliers(df, cluster_id, threshold=3):
    cluster_data = df[df['Cluster'] == cluster_id]
    z_scores = np.abs(stats.zscore(cluster_data[['PositionX', 'PositionY']]))
    
    # Get indices of the data points with z-scores above the threshold
    outlier_indices = np.where(z_scores > threshold)[0]
    
    # Drop the outliers from the DataFrame
    df_no_outliers = df.drop(cluster_data.iloc[outlier_indices].index)
    
    return df_no_outliers


def plot_main_clusters(df, n_clusters):
    geometry = [Point(xy) for xy in zip(df['PositionX'], df['PositionY'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.crs = 'EPSG:3116'
    gdf_web_mercator = gdf.to_crs(epsg=3857)
    
    cols = 2
    rows = math.ceil(n_clusters / cols)
    
    fig, axs = plt.subplots(rows, cols, figsize=(10 * cols, 10 * rows), constrained_layout=True)
    
    for cluster_id in range(n_clusters):
        ax = axs.flatten()[cluster_id]
        gdf_cluster = gdf_web_mercator[gdf_web_mercator['Cluster'] == cluster_id]
        gdf_cluster.plot(ax=ax, markersize=5, marker='x', edgecolor='black', color='yellow')
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, attribution="")
        
        ax.set_xlim(gdf_cluster.geometry.bounds.minx.min() - 50, gdf_cluster.geometry.bounds.maxx.max() + 50)
        ax.set_ylim(gdf_cluster.geometry.bounds.miny.min() - 50, gdf_cluster.geometry.bounds.maxy.max() + 50)
        ax.set_title(f'Cluster {cluster_id}')

    # If the number of clusters is odd, put the last empty subplot in the middle
    if n_clusters % 2 != 0:
        axs.flatten()[-1].axis('off')

    plt.show()
    
def plot_specific_subcluster(df, cluster_id, subcluster_id):
    geometry = [Point(xy) for xy in zip(df['PositionX'], df['PositionY'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.crs = 'EPSG:3116'
    gdf_web_mercator = gdf.to_crs(epsg=3857)

    gdf_cluster = gdf_web_mercator[(gdf_web_mercator['Cluster'] == cluster_id) & (gdf_web_mercator['Subcluster'] == subcluster_id)]

    # Define color mapping for subcluster_ids
    color_mapping = {
        'loading_process': 'red',
        'dumping_process': 'blue',
        'travelling_empty': 'green',
        'travelling_full': 'purple'
    }
    
    # Assign colors based on subcluster_id
    gdf_cluster['color'] = gdf_cluster['Subcluster'].map(color_mapping)

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_cluster.plot(ax=ax, markersize=5, marker='x', edgecolor='black', c=gdf_cluster['color'])

    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, attribution="")
    ax.set_xlim(gdf_cluster.geometry.bounds.minx.min() - 50, gdf_cluster.geometry.bounds.maxx.max() + 50)
    ax.set_ylim(gdf_cluster.geometry.bounds.miny.min() - 50, gdf_cluster.geometry.bounds.maxy.max() + 50)
    ax.set_title(f'Cluster {cluster_id} - Subcluster {subcluster_id}')
    plt.show()

       

def plot_all_main_clusters(df, n_clusters):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'lime', 'navy']
    
    geometry = [Point(xy) for xy in zip(df['PositionX'], df['PositionY'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.crs = 'EPSG:3116'
    gdf_web_mercator = gdf.to_crs(epsg=3857)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for cluster_id in range(n_clusters):
        gdf_cluster = gdf_web_mercator[gdf_web_mercator['Cluster'] == cluster_id]
        gdf_cluster.plot(ax=ax, markersize=5, marker='x', edgecolor='black', color=colors[cluster_id % len(colors)])
        
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, attribution="")
    ax.set_xlim(gdf_web_mercator.geometry.bounds.minx.min() - 50, gdf_web_mercator.geometry.bounds.maxx.max() + 50)
    ax.set_ylim(gdf_web_mercator.geometry.bounds.miny.min() - 50, gdf_web_mercator.geometry.bounds.maxy.max() + 50)
    plt.show()
    return fig
    


def plot_all_subclusters(df, cluster_id):
    subclusters = ['loading_process', 'dumping_process', 'travelling_empty', 'travelling_full']
    subcluster_colors = {'loading_process': 'red', 'dumping_process': 'blue', 'travelling_empty': 'green', 'travelling_full': 'orange'}

    geometry = [Point(xy) for xy in zip(df['PositionX'], df['PositionY'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.crs = 'EPSG:3116'
    gdf_web_mercator = gdf.to_crs(epsg=3857)
    
    gdf_cluster = gdf_web_mercator[gdf_web_mercator['Cluster'] == cluster_id]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for subcluster in subclusters:
        gdf_subcluster = gdf_cluster[gdf_cluster['Subcluster'] == subcluster]
        gdf_subcluster.plot(ax=ax, markersize=5, marker='x', edgecolor='black', color=subcluster_colors[subcluster])

    # Create legend elements
    legend_elements = [mpatches.Patch(color=subcluster_colors[subcluster], label=subcluster) for subcluster in subclusters]

    # Add legend to the plot
    ax.legend(handles=legend_elements, loc='upper left')

    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, attribution="")
    ax.set_xlim(gdf_cluster.geometry.bounds.minx.min() - 50, gdf_cluster.geometry.bounds.maxx.max() + 50)
    ax.set_ylim(gdf_cluster.geometry.bounds.miny.min() - 50, gdf_cluster.geometry.bounds.maxy.max() + 50)
    plt.show()
    return fig 

    
def assign_clusters(file_path, look_ahead=2):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Initialize the 'Cluster' column with empty strings
    df['Cluster'] = ''

    # Iterate through the rows to assign cluster labels
    for i in range(1, len(df)):
        prev_payload = df.loc[i - 1, 'Payload']
        curr_payload = df.loc[i, 'Payload']
        future_payloads = df.loc[i + 1 : i + look_ahead, 'Payload']

        if curr_payload > prev_payload and curr_payload > 0:
            df.loc[i, 'Cluster'] = 'loading_process'
        elif any(curr_payload > future_payload for future_payload in future_payloads) and curr_payload > 0:
            df.loc[i, 'Cluster'] = 'dumping_process'
        elif curr_payload == 0:
            df.loc[i, 'Cluster'] = 'travelling_empty'
        elif curr_payload > 0:
            df.loc[i, 'Cluster'] = 'travelling_full'

    return df



def assign_sub_clusters(df, vm_cluster):
    # Create a new column 'Subcluster' in the df DataFrame
    df['Subcluster'] = ''

    # Iterate through the rows in the df DataFrame
    for i, row in df.iterrows():
        timestamp = row['Timestamp']
        
        # Find the closest timestamp in the vm_cluster DataFrame
        closest_idx = (vm_cluster['ReadTime'] - timestamp).abs().idxmin()
        
        # Get the cluster label from the vm_cluster DataFrame
        cluster_label = vm_cluster.loc[closest_idx, 'Cluster']
        
        # Assign the cluster label to the corresponding row in the df DataFrame
        df.loc[i, 'Subcluster'] = cluster_label

    return df



#%%
# # Example usage
# plot_all_main_clusters(df_clustered, n_clusters)


# file_name = 'Datos GPS equipo 022-429 - Mayo-22 hasta Abril-5 2023.xlsx'
# file_path= os.path.join('..', 'Reference',file_name)
# sheet_names = ['20230322',
#                '20230323',
#                '20230324',
#                '20230325',
#                '20230326',
#                '20230327',
#                '20230328',
#                '20230329',
#                '20230330']
# df = df[sheet_names[]]
# print(df.head())
# for sheet_names in df:
#     plot_corrdinates_web_mercator(df[sheet_names],PRT = True)
    

#%%
file_name = 'Datos GPS equipo 022-429 - Abril 06-21 2023.xlsx'
file_path= os.path.join('..', 'Reference',file_name)
sheet_names = '20230406-20230421'
df = pd.read_excel(file_path,sheet_name=sheet_names, engine='openpyxl',skiprows=1)

n_clusters = 5
n_subclusters = 10

df = perform_clustering(df, n_clusters, n_subclusters)
# cluster_id = 1
df_no_outliers = remove_outliers(df, 1)
# plot_main_clusters(df_no_outliers, n_clusters)
# plot_all_subclusters(df_no_outliers, 1)

#%%
file_name = 'Datos C429 - Ciclos pivoted abril.xlsx'
file_path= os.path.join('..', 'Reference',file_name)
vm_cluster = assign_clusters(file_path)
fd_subcluster = assign_sub_clusters(df_no_outliers, vm_cluster)
#%%
plot_specific_subcluster(fd_subcluster, 1,  'loading_process')
# plot_all_main_clusters(df_no_outliers, n_clusters)
if not os.path.exists('images'):
    os.makedirs('images')
for i in [0, 1, 2, 3, 4]:
    try:
        fig = plot_all_subclusters(fd_subcluster, i)
        fig.savefig(f'images/cluster_{i}.svg')
    except:
        print(i, "somethin happend")


fig = plot_all_main_clusters(fd_subcluster,5)
fig.savefig(f'images/all_cluster_{i}.svg')



    