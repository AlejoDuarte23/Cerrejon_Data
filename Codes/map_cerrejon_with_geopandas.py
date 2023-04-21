import pandas as pd 
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
import pyautocad
from shapely.wkt import loads
import os 
from matplotlib.backends.backend_pdf import PdfPages


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
file_name = 'Datos GPS equipo 022-429 - Mayo-22 hasta Abril-5 2023.xlsx'
file_path= os.path.join('..', 'Reference',file_name)
sheet_names = ['20230322',
               '20230323',
               '20230324',
               '20230325',
               '20230326',
               '20230327',
               '20230328',
               '20230329',
               '20230330']


df = pd.read_excel(file_path,sheet_name=sheet_names, engine='openpyxl',skiprows=1)
# print(df.head())
for sheet_names in df:
    plot_corrdinates_web_mercator(df[sheet_names],PRT = True)
    
    