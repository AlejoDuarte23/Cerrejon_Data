import pandas as pd 
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
import pyautocad
from shapely.wkt import loads
df = pd.read_excel('cords.xlsx', engine='openpyxl')
print(df.head())
# Create a GeoDataFrame with the coordinates
# Create a GeoDataFrame with the coordinates

# Create a GeoDataFrame with the coordinates
geometry = [Point(xy) for xy in zip(df['PositionX'], df['PositionY'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Set the coordinate reference system (CRS) to Magna-Sirgas / Colombia West (EPSG:3116)
gdf.crs = 'EPSG:3116'

# Convert the GeoDataFrame to Web Mercator (EPSG:3857)
gdf_web_mercator = gdf.to_crs(epsg=3857)

# Create an empty plot with the desired figure size
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the points on the empty plot
gdf_web_mercator.plot(ax=ax, color='red', markersize=5, marker='o', edgecolor='black')

# Add the satellite basemap
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

# Set the plot limits to focus on the region of interest
ax.set_xlim(gdf_web_mercator.geometry.bounds.minx.min() - 5000,
            gdf_web_mercator.geometry.bounds.maxx.max() + 5000)
ax.set_ylim(gdf_web_mercator.geometry.bounds.miny.min() - 5000,
            gdf_web_mercator.geometry.bounds.maxy.max() + 5000)

plt.show()



# acad = pyautocad.Autocad()

# for i, row in df.iterrows():
#     point = loads(row['geometry'])
#     point_wkt = point.wkt
#     x, y = point.x, point.y
#     center = pyautocad.APoint(x, y, 0)
#     radius = 100  # set the radius of the circle
#     circle = acad.model.AddCircle(center, radius)

# # Save the changes
# acad.doc.Save()



# import pandas as pd
# from geopy.distance import geodesic
# import numpy as np

# # Set the distance threshold in meters
# distance_threshold = 10

# # Calculate the distance between consecutive points
# df['distance'] = np.nan
# for i in range(1, len(df)):
#     coord1 = (df.loc[i - 1, 'PositionY'], df.loc[i - 1, 'PositionX'])
#     coord2 = (df.loc[i, 'PositionY'], df.loc[i, 'PositionX'])
#     distance = geodesic(coord1, coord2).meters
#     df.loc[i, 'distance'] = distance

# # Identify time stamps where the truck was stopped
# df['stopped'] = df['distance'] <= distance_threshold

# # Extract the stopped time stamps
# stopped_timestamps = df.loc[df['stopped'], 'Timestamp']

# # Calculate the total distance traveled
# total_distance = df['distance'].sum()

# print("Stopped time stamps:")
# print(stopped_timestamps)
# print("\nTotal distance traveled (meters):", total_distance)

# import pandas as pd
# import numpy as np

# def haversine(coord1, coord2):
#     R = 6371000  # Earth's radius in meters
#     lat1, lon1, lat2, lon2 = np.radians(coord1[0]), np.radians(coord1[1]), np.radians(coord2[0]), np.radians(coord2[1])
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
#     return R * c

# # Set the distance threshold in meters
# distance_threshold = 10

# # Calculate the distance between consecutive points
# df['distance'] = np.nan
# for i in range(1, len(df)):
#     coord1 = (df.loc[i - 1, 'PositionY'], df.loc[i - 1, 'PositionX'])
#     coord2 = (df.loc[i, 'PositionY'], df.loc[i, 'PositionX'])
#     distance = haversine(coord1, coord2)
#     df.loc[i, 'distance'] = distance

# # Identify time stamps where the truck was stopped
# df['stopped'] = df['distance'] <= distance_threshold

# # Extract the stopped time stamps
# stopped_timestamps = df.loc[df['stopped'], 'Timestamp']

# # Calculate the total distance traveled
# total_distance = df['distance'].sum()

# print("Stopped time stamps:")
# print(stopped_timestamps)
# print("\nTotal distance traveled (meters):", total_distance)


# import pandas as pd
# import numpy as np

# def euclidean(coord1, coord2):
#     return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

# # Set the distance threshold in meters
# distance_threshold = 10

# # Calculate the distance between consecutive points
# df['distance'] = np.nan
# for i in range(1, len(df)):
#     coord1 = (df.loc[i - 1, 'PositionY'], df.loc[i - 1, 'PositionX'])
#     coord2 = (df.loc[i, 'PositionY'], df.loc[i, 'PositionX'])
#     distance = euclidean(coord1, coord2)
#     df.loc[i, 'distance'] = distance

# # Filter out outliers by setting an upper distance threshold (e.g., 1000 meters)
# upper_distance_threshold = 1000
# df_filtered = df[df['distance'] <= upper_distance_threshold]

# # Identify time stamps where the truck was stopped
# df_filtered['stopped'] = df_filtered['distance'] <= distance_threshold

# # Extract the stopped time stamps
# stopped_timestamps = df_filtered.loc[df_filtered['stopped'], 'Timestamp']

# # Calculate the total distance traveled
# total_distance = df_filtered['distance'].sum()

# print("Stopped time stamps:")
# print(stopped_timestamps)
# print("\nTotal distance traveled (meters):", total_distance)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def euclidean(coord1, coord2):
#     return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

# # Set the distance threshold in meters
# distance_threshold = 10

# # Calculate the distance between consecutive points
# filtered_['distance'] = np.nan
# for i in range(1, len(df)):
#     coord1 = (df.loc[i - 1, 'PositionY'], df.loc[i - 1, 'PositionX'])
#     coord2 = (df.loc[i, 'PositionY'], df.loc[i, 'PositionX'])
#     distance = euclidean(coord1, coord2)
#     df.loc[i, 'distance'] = distance

# Plot a bar chart of the distances between consecutive points
# plt.figure(figsize=(10, 5))
# plt.bar(range(1, len(df)), df['distance'][1:])
# plt.xlabel('Index')
# plt.ylabel('Distance (m)')
# plt.title('Distances Between Consecutive Points')
# plt.show()


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def euclidean(coord1, coord2):
#     return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

# # Convert the 'Timestamp' column to datetime objects
# df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# # Calculate the distance and time difference between consecutive points
# df['distance'] = np.nan
# df['time_diff'] = np.nan
# for i in range(1, len(df)):
#     coord1 = (df.loc[i - 1, 'PositionY'], df.loc[i - 1, 'PositionX'])
#     coord2 = (df.loc[i, 'PositionY'], df.loc[i, 'PositionX'])
#     distance = euclidean(coord1, coord2)
#     time_diff = (df.loc[i, 'Timestamp'] - df.loc[i - 1, 'Timestamp']).total_seconds()
#     df.loc[i, 'distance'] = distance
#     df.loc[i, 'time_diff'] = time_diff

# # Calculate velocity (m/s)
# df['velocity'] = df['distance'] / df['time_diff']

# # Set the speed threshold in m/s (50 km/h = 13.89 m/s)
# speed_threshold = 13.89

# # Filter out the rows with velocities greater than the speed threshold
# filtered_df = df[df['velocity'] <= speed_threshold]

# # Reset the index of the filtered DataFrame
# filtered_df.reset_index(drop=True, inplace=True)

# def plt_distance(_df):

#     # Plot a bar chart of the distances between consecutive points
#     plt.figure(figsize=(10, 5))
#     plt.bar(range(1, len(_df)), _df['distance'][1:])
#     plt.xlabel('Index')
#     plt.ylabel('velocity (m)')
#     plt.title('Distances Between Consecutive Points')
#     plt.show()

# # Print the filtered DataFrame before plotting
# # print(filtered_df)

# plt_distance(filtered_df)


